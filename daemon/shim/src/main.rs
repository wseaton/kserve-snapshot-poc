//! PID 1 of a workload container. It starts the server when the node daemon
//! says to, or stays out of the way while the daemon restores a checkpoint
//! into this pod, and then looks after whichever server it ends up with.
//!
//! ```text
//! node daemon ──(shim.sock, through /proc/<pid>/root)──► shim (PID 1)
//!   Start            spawn the server in its own session
//!   Release          the server is about to be checkpointed; its exit is expected
//!   Adopt { pid }    the server is now this restored process
//!   Serving          open the readiness port
//!   State            what has happened so far, for a daemon that restarted
//!   Fail             exit, so the container starts over
//! ```
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use rustix::process::{Pid, Signal, WaitOptions};
use snapshotd_shim::{Reply, Request, SERVER_LOG, SOCKET};

const TICK: Duration = Duration::from_millis(100);
const PROBE: Duration = Duration::from_millis(200);
const READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

#[derive(Debug, thiserror::Error)]
enum ShimError {
    #[error("the server was already started")]
    AlreadyStarted,
}

#[derive(Parser)]
#[command(name = "snapshotd-shim")]
struct Args {
    /// Where the node daemon connects.
    #[arg(long, default_value = SOCKET)]
    socket: PathBuf,
    /// Where the server's output goes; it is copied to this process's output.
    #[arg(long, default_value = SERVER_LOG)]
    log: PathBuf,
    /// Port opened once the server answers requests. It accepts connections
    /// and answers any HTTP request with 200, for a TCP or an HTTP probe.
    #[arg(long, default_value_t = 8001)]
    ready_port: u16,
    /// Port the server listens on; used only when no node daemon shows up.
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Seconds to wait for the node daemon before starting the server without it.
    #[arg(long, default_value_t = 30)]
    daemon_wait_s: u64,
    /// The server's command line.
    #[arg(last = true, required = true)]
    command: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Server {
    NotStarted,
    Running(Pid),
    /// Stopped for a checkpoint; a restored one follows.
    Released,
}

struct Daemon {
    stream: UnixStream,
    unread: Vec<u8>,
}

struct Shim {
    args: Args,
    listener: UnixListener,
    daemon: Option<Daemon>,
    daemon_seen: bool,
    server: Server,
    ready: Option<TcpListener>,
    log: Option<File>,
    began: Instant,
    failed: bool,
}

fn termination_signals() -> libc::sigset_t {
    // SAFETY: sigemptyset and sigaddset initialize and edit a set this
    // function owns.
    unsafe {
        let mut set = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGINT);
        set
    }
}

fn terminated_within(tick: Duration) -> bool {
    let timeout = libc::timespec {
        tv_sec: tick.as_secs() as _,
        tv_nsec: tick.subsec_nanos() as libc::c_long,
    };
    // SAFETY: the set and the timeout live across the call, and a null info
    // pointer is allowed.
    unsafe { libc::sigtimedwait(&termination_signals(), std::ptr::null_mut(), &timeout) > 0 }
}

impl Shim {
    fn new(args: Args) -> Result<Self> {
        for path in [&args.socket, &args.log] {
            if let Some(directory) = path.parent() {
                fs::create_dir_all(directory)
                    .with_context(|| format!("create {}", directory.display()))?;
            }
        }
        match fs::remove_file(&args.socket) {
            Err(error) if error.kind() != ErrorKind::NotFound => {
                return Err(error).context("remove a stale socket");
            }
            _ => {}
        }
        let listener = UnixListener::bind(&args.socket)
            .with_context(|| format!("listen on {}", args.socket.display()))?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            args,
            listener,
            daemon: None,
            daemon_seen: false,
            server: Server::NotStarted,
            ready: None,
            log: None,
            began: Instant::now(),
            failed: false,
        })
    }

    fn start(&mut self) -> Result<Pid> {
        if self.server != Server::NotStarted {
            return Err(ShimError::AlreadyStarted.into());
        }
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.args.log)
            .with_context(|| format!("open {}", self.args.log.display()))?;
        let mut command = Command::new(&self.args.command[0]);
        command
            .args(&self.args.command[1..])
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log);
        // SAFETY: setsid and sigprocmask are async-signal-safe and act only
        // on the child between fork and exec.
        unsafe {
            command.pre_exec(|| {
                rustix::process::setsid()?;
                if libc::sigprocmask(
                    libc::SIG_UNBLOCK,
                    &termination_signals(),
                    std::ptr::null_mut(),
                ) != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command
            .spawn()
            .with_context(|| format!("start {:?}", self.args.command))?;
        let pid = Pid::from_child(&child);
        self.server = Server::Running(pid);
        println!("shim: started the server as process {}", child.id());
        Ok(pid)
    }

    fn handle(&mut self, request: Request) -> Reply {
        let outcome = match request {
            Request::Start => self.start().map(|pid| Reply::Started {
                pid: pid.as_raw_nonzero().get(),
            }),
            Request::Release => {
                self.server = Server::Released;
                Ok(Reply::Done)
            }
            Request::Adopt { pid } => Pid::from_raw(pid)
                .ok_or_else(|| anyhow!("{pid} is not a process ID"))
                .map(|pid| {
                    self.server = Server::Running(pid);
                    Reply::Done
                }),
            Request::Serving => self.serve().map(|()| Reply::Done),
            Request::State => Ok(Reply::State {
                started: self.server != Server::NotStarted,
                serving: self.ready.is_some(),
            }),
            Request::Fail => {
                self.failed = true;
                Ok(Reply::Done)
            }
        };
        outcome.unwrap_or_else(|error| Reply::Refused {
            reason: format!("{error:#}"),
        })
    }

    fn serve(&mut self) -> Result<()> {
        if self.ready.is_none() {
            let address = SocketAddr::from((Ipv4Addr::UNSPECIFIED, self.args.ready_port));
            let ready = TcpListener::bind(address)
                .with_context(|| format!("open the readiness port {address}"))?;
            ready.set_nonblocking(true)?;
            self.ready = Some(ready);
            println!("shim: serving");
        }
        Ok(())
    }

    fn talk_to_daemon(&mut self) -> Result<()> {
        match self.listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(true)?;
                self.daemon = Some(Daemon {
                    stream,
                    unread: Vec::new(),
                });
                self.daemon_seen = true;
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            Err(error) => return Err(error).context("accept the node daemon"),
        }
        let Some(mut daemon) = self.daemon.take() else {
            return Ok(());
        };
        let mut chunk = [0; 4096];
        let open = loop {
            match daemon.stream.read(&mut chunk) {
                Ok(0) => break false,
                Ok(read) => daemon.unread.extend_from_slice(&chunk[..read]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => break true,
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(_) => break false,
            }
        };
        while let Some(end) = daemon.unread.iter().position(|&byte| byte == b'\n') {
            let line: Vec<u8> = daemon.unread.drain(..=end).collect();
            let reply = match serde_json::from_slice::<Request>(&line) {
                Ok(request) => self.handle(request),
                Err(error) => Reply::Refused {
                    reason: format!("unreadable request: {error}"),
                },
            };
            let mut line = serde_json::to_vec(&reply)?;
            line.push(b'\n');
            daemon.stream.set_nonblocking(false)?;
            let written = daemon.stream.write_all(&line);
            daemon.stream.set_nonblocking(true)?;
            if written.is_err() {
                return Ok(());
            }
        }
        if open {
            self.daemon = Some(daemon);
        }
        Ok(())
    }

    /// Reap every exited child. Returns the server's exit code if it was the
    /// server and it was not released.
    fn reap(&mut self) -> Result<Option<i32>> {
        loop {
            match rustix::process::wait(WaitOptions::NOHANG) {
                Ok(Some((pid, status))) => {
                    if self.server == Server::Running(pid) {
                        let code = status
                            .exit_status()
                            .or_else(|| status.terminating_signal().map(|signal| 128 + signal))
                            .unwrap_or(1);
                        return Ok(Some(code));
                    }
                }
                Ok(None) => return Ok(None),
                Err(rustix::io::Errno::CHILD) => return Ok(None),
                Err(rustix::io::Errno::INTR) => {}
                Err(error) => return Err(error).context("reap children"),
            }
        }
    }

    fn without_a_daemon(&mut self) -> Result<()> {
        if self.daemon_seen {
            return Ok(());
        }
        if self.server == Server::NotStarted
            && self.began.elapsed() >= Duration::from_secs(self.args.daemon_wait_s)
        {
            println!("shim: no node daemon, starting the server without it");
            self.start()?;
        }
        if matches!(self.server, Server::Running(_)) && self.ready.is_none() {
            let server = SocketAddr::from((Ipv4Addr::LOCALHOST, self.args.port));
            if TcpStream::connect_timeout(&server, TICK).is_ok() {
                self.serve()?;
            }
        }
        Ok(())
    }

    fn answer_probes(&self) {
        let Some(ready) = &self.ready else { return };
        while let Ok((mut probe, _)) = ready.accept() {
            let _ = probe.set_nonblocking(false);
            let _ = probe.set_read_timeout(Some(PROBE));
            let _ = probe.set_write_timeout(Some(PROBE));
            let _ = probe.read(&mut [0; 1024]);
            let _ = probe.write_all(READY);
        }
    }

    fn copy_log(&mut self) -> Result<()> {
        if self.log.is_none() {
            match File::open(&self.args.log) {
                Ok(log) => self.log = Some(log),
                Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
                Err(error) => return Err(error).context("open the server's log"),
            }
        }
        let Some(log) = &mut self.log else {
            return Ok(());
        };
        let position = log.stream_position()?;
        if log.metadata()?.len() < position {
            log.seek(SeekFrom::Start(0))?;
        }
        std::io::copy(log, &mut std::io::stdout().lock()).context("copy the server's log")?;
        Ok(())
    }

    fn stop_server(&self) {
        if let Server::Running(pid) = self.server {
            let _ = rustix::process::kill_process_group(pid, Signal::TERM);
        }
    }

    fn run(mut self) -> Result<i32> {
        let mut stopping = false;
        loop {
            if terminated_within(TICK) && !stopping {
                println!("shim: terminating");
                stopping = true;
                self.stop_server();
                if !matches!(self.server, Server::Running(_)) {
                    return Ok(0);
                }
            }
            self.talk_to_daemon()?;
            let exited = self.reap()?;
            self.copy_log()?;
            if self.failed {
                println!("shim: the node daemon gave up on this container");
                self.stop_server();
                return Ok(1);
            }
            if let Some(code) = exited {
                println!("shim: the server exited with {code}");
                return Ok(if stopping { 0 } else { code });
            }
            self.without_a_daemon()?;
            self.answer_probes();
        }
    }
}

fn run(args: Args) -> Result<i32> {
    // SAFETY: blocking signals for this thread before any other exists; the
    // set lives across the call.
    if unsafe {
        libc::sigprocmask(
            libc::SIG_BLOCK,
            &termination_signals(),
            std::ptr::null_mut(),
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error()).context("block termination signals");
    }
    rustix::process::set_child_subreaper(rustix::process::getpid().into())
        .context("become the reaper of restored processes")?;
    Shim::new(args)?.run()
}

fn main() -> Result<()> {
    std::process::exit(run(Args::parse())?)
}
