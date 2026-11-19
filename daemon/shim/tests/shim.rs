//! The shim, run as the real binary: a test stands in for the node daemon on
//! its socket, and the servers are real processes.
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Shim {
    child: Child,
    dir: PathBuf,
    ready_port: u16,
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

impl Shim {
    fn start(name: &str, extra: &[&str], command: &[&str]) -> Self {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-shim-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let ready_port = free_port();
        let child = Command::new(env!("CARGO_BIN_EXE_snapshotd-shim"))
            .arg("--socket")
            .arg(dir.join("shim.sock"))
            .arg("--log")
            .arg(dir.join("server.log"))
            .args(["--ready-port", &ready_port.to_string()])
            .args(extra)
            .arg("--")
            .args(command)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let shim = Self {
            child,
            dir,
            ready_port,
        };
        wait_until("the shim's socket", || shim.dir.join("shim.sock").exists());
        shim
    }

    fn daemon(&self) -> Daemon {
        let stream = UnixStream::connect(self.dir.join("shim.sock")).unwrap();
        Daemon {
            reader: BufReader::new(stream.try_clone().unwrap()),
            stream,
        }
    }

    fn ready(&self) -> bool {
        TcpStream::connect(("127.0.0.1", self.ready_port)).is_ok()
    }

    fn exit_code(mut self) -> (i32, String) {
        let deadline = Instant::now() + Duration::from_secs(20);
        let status = loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "the shim did not exit");
            std::thread::sleep(Duration::from_millis(50));
        };
        let mut output = String::new();
        self.child
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut output)
            .unwrap();
        fs::remove_dir_all(&self.dir).unwrap();
        (status.code().unwrap(), output)
    }
}

struct Daemon {
    stream: UnixStream,
    reader: BufReader<UnixStream>,
}

impl Daemon {
    fn call(&mut self, request: &str) -> serde_json::Value {
        writeln!(self.stream, "{request}").unwrap();
        let mut reply = String::new();
        self.reader.read_line(&mut reply).unwrap();
        serde_json::from_str(&reply).unwrap()
    }

    fn start(&mut self) -> i32 {
        self.call(r#""Start""#)["Started"]["pid"].as_i64().unwrap() as i32
    }
}

fn alive(pid: i32) -> bool {
    Path::new(&format!("/proc/{pid}/stat")).exists()
        && !fs::read_to_string(format!("/proc/{pid}/stat"))
            .unwrap_or_default()
            .contains(") Z ")
}

#[test]
fn the_server_starts_only_when_told_and_its_exit_code_is_the_shims() {
    let shim = Shim::start(
        "exit",
        &[],
        &["sh", "-c", "echo from the server; sleep 0.3; exit 7"],
    );
    let mut daemon = shim.daemon();
    assert_eq!(
        daemon.call(r#""State""#),
        serde_json::json!({"State": {"started": false, "serving": false}})
    );
    let pid = daemon.start();
    assert!(pid > 0);
    assert_eq!(
        daemon.call(r#""State""#),
        serde_json::json!({"State": {"started": true, "serving": false}})
    );
    assert!(daemon.call(r#""Start""#)["Refused"]["reason"].is_string());
    let (code, output) = shim.exit_code();
    assert_eq!(code, 7);
    assert!(output.contains("from the server"), "{output}");
}

#[test]
fn the_server_runs_in_its_own_session() {
    let shim = Shim::start("session", &[], &["sleep", "30"]);
    let mut daemon = shim.daemon();
    let pid = daemon.start();
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).unwrap();
    let fields: Vec<&str> = stat.rsplit(") ").next().unwrap().split(' ').collect();
    assert_eq!(fields[2], pid.to_string(), "process group");
    assert_eq!(fields[3], pid.to_string(), "session");
    assert_eq!(daemon.call(r#""Fail""#), serde_json::json!("Done"));
    let (code, _) = shim.exit_code();
    assert_eq!(code, 1);
    wait_until("the server to stop", || !alive(pid));
}

#[test]
fn serving_opens_the_readiness_port() {
    let shim = Shim::start("ready", &[], &["sleep", "30"]);
    let mut daemon = shim.daemon();
    daemon.start();
    assert!(!shim.ready());
    assert_eq!(daemon.call(r#""Serving""#), serde_json::json!("Done"));
    assert!(shim.ready());
    assert!(shim.ready());
    assert_eq!(
        daemon.call(r#""State""#),
        serde_json::json!({"State": {"started": true, "serving": true}})
    );
    daemon.call(r#""Fail""#);
    shim.exit_code();
}

#[test]
fn a_released_server_may_exit_and_an_adopted_one_takes_its_place() {
    let pidfile =
        std::env::temp_dir().join(format!("snapshotd-shim-orphan-{}", std::process::id()));
    let shim = Shim::start(
        "adopt",
        &[],
        &[
            "sh",
            "-c",
            "sleep 2 & echo $! > \"$0\"; sleep 0.5; exit 3",
            pidfile.to_str().unwrap(),
        ],
    );
    let mut daemon = shim.daemon();
    daemon.start();
    assert_eq!(daemon.call(r#""Release""#), serde_json::json!("Done"));
    wait_until("the orphan's PID", || {
        fs::read_to_string(&pidfile).is_ok_and(|pid| pid.ends_with('\n'))
    });
    let orphan: i32 = fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    std::thread::sleep(Duration::from_millis(800));
    assert!(shim.dir.join("shim.sock").exists());
    assert_eq!(
        daemon.call(&format!(r#"{{"Adopt":{{"pid":{orphan}}}}}"#)),
        serde_json::json!("Done")
    );
    let (code, output) = shim.exit_code();
    assert_eq!(code, 0, "{output}");
    assert!(output.contains("the server exited with 0"), "{output}");
    fs::remove_file(pidfile).unwrap();
}

#[test]
fn an_unreadable_request_is_refused_and_the_shim_goes_on() {
    let shim = Shim::start("garbage", &[], &["sleep", "30"]);
    let mut daemon = shim.daemon();
    assert!(daemon.call("not json")["Refused"]["reason"].is_string());
    assert!(daemon.call(r#"{"Adopt":{"pid":0}}"#)["Refused"]["reason"].is_string());
    assert!(daemon.start() > 0);
    daemon.call(r#""Fail""#);
    shim.exit_code();
}

#[test]
fn a_second_daemon_connection_replaces_the_first() {
    let shim = Shim::start("reconnect", &[], &["sleep", "30"]);
    let mut first = shim.daemon();
    first.start();
    drop(first);
    let mut second = shim.daemon();
    assert_eq!(
        second.call(r#""State""#),
        serde_json::json!({"State": {"started": true, "serving": false}})
    );
    second.call(r#""Fail""#);
    shim.exit_code();
}

#[test]
fn without_a_daemon_the_server_starts_and_is_ready_once_it_listens() {
    let port = free_port();
    let listen = format!(
        "import socket, time; s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); time.sleep(0.5); s.bind(('127.0.0.1', {port})); s.listen(); time.sleep(30)"
    );
    let shim = Shim::start(
        "alone",
        &["--daemon-wait-s", "0", "--port", &port.to_string()],
        &["python3", "-c", &listen],
    );
    assert!(!shim.ready());
    wait_until("readiness", || shim.ready());
    // SAFETY: sending a signal to a child this test owns.
    assert_eq!(
        unsafe { libc::kill(shim.child.id() as i32, libc::SIGTERM) },
        0
    );
    let (code, output) = shim.exit_code();
    assert_eq!(code, 0, "{output}");
    assert!(output.contains("no node daemon"), "{output}");
}

#[test]
fn termination_stops_the_server_and_exits_cleanly() {
    let shim = Shim::start("term", &[], &["sleep", "30"]);
    let mut daemon = shim.daemon();
    let pid = daemon.start();
    // SAFETY: sending a signal to a child this test owns.
    assert_eq!(
        unsafe { libc::kill(shim.child.id() as i32, libc::SIGTERM) },
        0
    );
    let (code, output) = shim.exit_code();
    assert_eq!(code, 0, "{output}");
    assert!(!alive(pid));
}
