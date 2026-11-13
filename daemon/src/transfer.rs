//! Copy a checkpoint directory to another node over RDMA with NIXL.
//!
//! ```text
//! sender                                             receiver
//! local disk ─pread─► slot k ══ NIXL write (UCX) ══► slot k ─pwrite─► local disk
//!                     └── one huge-page region, registered once on each side ──┘
//! ```
//!
//! Each side maps one anonymous region, asks for transparent huge pages,
//! touches it and registers it with NIXL once. The region is cut into slots.
//! A slot carries one chunk of one file at a time: a sender worker reads the
//! chunk into its slot, the NIXL thread writes it into the receiver's slot of
//! the same index, and the receiver writes it to disk and acknowledges, which
//! frees the slot on both sides. Control messages are JSON lines on one TCP
//! connection.
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions, Permissions};
use std::io::{BufRead, BufReader, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::{FileExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use nixl_sys::{
    Agent, Backend, MemType, MemoryRegion, NixlRegistration, OptArgs, SystemStorage, XferDescList,
    XferOp, XferRequest,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::image::{self, Arena, Extent, HugeRegion, huge_page_fraction};
use crate::trace::{self, FILE_TRACKS, Trace};
use walkdir::WalkDir;

const BACKEND: &str = "UCX";
const RECEIVER: &str = "snapshotd-receive";
const SENDER: &str = "snapshotd-send";

/// How the registered region is cut up.
#[derive(Clone, Copy, Debug, clap::Args, Serialize, Deserialize)]
pub struct Pool {
    /// Chunks in flight at once.
    #[arg(long, default_value_t = 16)]
    pub slots: usize,
    /// Size of one chunk.
    #[arg(long, default_value_t = 256)]
    pub slot_mib: usize,
}

impl Pool {
    fn slot_bytes(self) -> usize {
        self.slot_mib << 20
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct FileEntry {
    path: PathBuf,
    size: u64,
    mode: u32,
}

#[derive(Debug, Serialize, Deserialize)]
enum Message {
    Hello {
        pool: Pool,
        metadata: String,
        files: Vec<FileEntry>,
    },
    Ready {
        metadata: String,
        slot_addresses: Vec<usize>,
        register_s: f64,
        huge_page_fraction: f64,
    },
    /// A file whose size is not known yet; its chunks follow.
    Open {
        file: usize,
        path: PathBuf,
    },
    Chunk {
        slot: usize,
        file: usize,
        offset: u64,
        len: usize,
    },
    Ack {
        slot: usize,
    },
    Done,
    Stats {
        write_s: f64,
        sync_s: f64,
    },
}

fn send(stream: &Mutex<TcpStream>, message: &Message) -> Result<()> {
    let mut line = serde_json::to_vec(message)?;
    line.push(b'\n');
    stream
        .lock()
        .map_err(|_| anyhow!("control connection lock poisoned"))?
        .write_all(&line)
        .context("write control message")
}

fn receive(reader: &mut impl BufRead) -> Result<Message> {
    let mut line = String::new();
    ensure!(
        reader
            .read_line(&mut line)
            .context("read control message")?
            > 0,
        "control connection closed"
    );
    serde_json::from_str(&line).with_context(|| format!("parse control message {line:?}"))
}

/// The registered region and the NIXL objects that use it. Every NIXL call
/// happens on the thread that owns this.
struct Endpoint {
    agent: Agent,
    backend: Backend,
    storage: SystemStorage<HugeRegion>,
    pool: Pool,
    register: Duration,
    huge_page_fraction: f64,
}

impl Endpoint {
    fn new(name: &str, pool: Pool) -> Result<Self> {
        let len = pool
            .slots
            .checked_mul(pool.slot_bytes())
            .context("pool size overflows")?;
        ensure!(len > 0, "the pool is empty");
        let mut region = HugeRegion::new(len)?;
        region.as_mut().fill(0);
        let huge_page_fraction = huge_page_fraction(&[(region.address(), region.len())]);

        let agent = Agent::new(name).map_err(|e| anyhow!("create NIXL agent: {e}"))?;
        let (_, params) = agent
            .get_plugin_params(BACKEND)
            .map_err(|e| anyhow!("{BACKEND} plugin: {e}"))?;
        let backend = agent
            .create_backend(BACKEND, &params)
            .map_err(|e| anyhow!("create {BACKEND} backend: {e}"))?;
        // SAFETY: a HugeRegion keeps its address and length until it is
        // dropped, and `storage` owns it for as long as it is registered.
        let mut storage = unsafe { SystemStorage::from_buffer(region) };
        let start = Instant::now();
        storage
            .register(&agent, Some(&backend_args(&backend)?))
            .map_err(|e| anyhow!("register {len} bytes: {e}"))?;
        Ok(Self {
            agent,
            backend,
            storage,
            pool,
            register: start.elapsed(),
            huge_page_fraction,
        })
    }

    fn slots(&self) -> Slots {
        // SAFETY: the registration keeps the region mapped at this address.
        let base = unsafe { self.storage.as_ptr() } as usize;
        Slots {
            base,
            slot_bytes: self.pool.slot_bytes(),
            count: self.pool.slots,
        }
    }

    fn metadata(&self) -> Result<String> {
        let metadata = self
            .agent
            .get_local_md()
            .map_err(|e| anyhow!("local NIXL metadata: {e}"))?;
        Ok(hex::encode(metadata))
    }

    fn load_peer(&self, metadata: &str) -> Result<String> {
        let metadata = hex::decode(metadata).context("peer NIXL metadata")?;
        self.agent
            .load_remote_md(&metadata)
            .map_err(|e| anyhow!("load peer NIXL metadata: {e}"))
    }
}

fn backend_args(backend: &Backend) -> Result<OptArgs> {
    let mut args = OptArgs::new().map_err(|e| anyhow!("NIXL arguments: {e}"))?;
    args.add_backend(backend)
        .map_err(|e| anyhow!("select the backend: {e}"))?;
    Ok(args)
}

/// The slots of a registered region. Each slot is used by one worker at a
/// time, which is what makes handing out `&mut` slices sound.
#[derive(Clone, Copy)]
struct Slots {
    base: usize,
    slot_bytes: usize,
    count: usize,
}

impl Slots {
    fn address(self, slot: usize) -> usize {
        self.base + slot * self.slot_bytes
    }

    /// # Safety
    /// No other reference to slot `slot` may be live, and the region must
    /// still be mapped.
    unsafe fn slice_mut<'a>(self, slot: usize, len: usize) -> &'a mut [u8] {
        debug_assert!(slot < self.count && len <= self.slot_bytes);
        // SAFETY: in bounds of the mapping by the assertion; exclusive and
        // mapped by the caller's contract.
        unsafe { std::slice::from_raw_parts_mut(self.address(slot) as *mut u8, len) }
    }
}

fn list_files(dir: &Path) -> Result<Vec<FileEntry>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(dir).sort_by_file_name() {
        let entry = entry.with_context(|| format!("walk {}", dir.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let metadata = entry.metadata()?;
        files.push(FileEntry {
            path: entry.path().strip_prefix(dir)?.to_path_buf(),
            size: metadata.len(),
            mode: metadata.permissions().mode() & 0o7777,
        });
    }
    ensure!(!files.is_empty(), "{} holds no files", dir.display());
    Ok(files)
}

#[derive(Clone, Copy)]
struct Job {
    file: usize,
    offset: u64,
    len: usize,
}

fn seconds(nanos: &AtomicU64) -> f64 {
    nanos.load(Ordering::Relaxed) as f64 / 1e9
}

fn add(nanos: &AtomicU64, since: Instant) {
    nanos.fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Send every file under `dir` to a receiver listening at `to`.
pub fn send_directory(dir: &Path, to: &str, pool: Pool, trace: Option<PathBuf>) -> Result<()> {
    let trace = Trace::new(trace, "sender");
    let files = list_files(dir)?;
    let total: u64 = files.iter().map(|f| f.size).sum();
    let handles: Vec<File> = files
        .iter()
        .map(|f| File::open(dir.join(&f.path)).with_context(|| format!("open {:?}", f.path)))
        .collect::<Result<_>>()?;
    let mut jobs = Vec::new();
    for (file, entry) in files.iter().enumerate() {
        let mut offset = 0;
        while offset < entry.size {
            let len = (entry.size - offset).min(pool.slot_bytes() as u64) as usize;
            jobs.push(Job { file, offset, len });
            offset += len as u64;
        }
    }

    let endpoint = Endpoint::new(SENDER, pool)?;
    let slots = endpoint.slots();
    let stream = TcpStream::connect(to).with_context(|| format!("connect {to}"))?;
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let stream = Mutex::new(stream);
    send(
        &stream,
        &Message::Hello {
            pool,
            metadata: endpoint.metadata()?,
            files,
        },
    )?;
    let Message::Ready {
        metadata,
        slot_addresses,
        register_s,
        huge_page_fraction,
    } = receive(&mut reader)?
    else {
        bail!("the receiver did not answer with Ready");
    };
    ensure!(
        slot_addresses.len() == pool.slots,
        "the receiver has {} slots, not {}",
        slot_addresses.len(),
        pool.slots
    );
    let peer = endpoint.load_peer(&metadata)?;

    let jobs = Mutex::new(jobs.into_iter());
    let read_ns = AtomicU64::new(0);
    let rdma_ns = AtomicU64::new(0);
    let (post_tx, post_rx) = mpsc::channel::<(usize, usize)>();
    let mut written_tx = Vec::new();
    let mut ack_tx = Vec::new();
    let mut worker_ends = Vec::new();
    for _ in 0..pool.slots {
        let (wtx, wrx) = mpsc::channel::<()>();
        let (atx, arx) = mpsc::channel::<()>();
        written_tx.push(wtx);
        ack_tx.push(atx);
        worker_ends.push((wrx, arx));
    }
    let (stats_tx, stats_rx) = mpsc::channel::<Message>();
    let start = Instant::now();

    let outcome = std::thread::scope(|scope| -> Result<()> {
        scope.spawn(move || {
            while let Ok(message) = receive(&mut reader) {
                match message {
                    Message::Ack { slot } => {
                        if ack_tx.get(slot).is_none_or(|tx| tx.send(()).is_err()) {
                            break;
                        }
                    }
                    other => {
                        let _ = stats_tx.send(other);
                        break;
                    }
                }
            }
        });
        let workers: Vec<_> = worker_ends
            .into_iter()
            .enumerate()
            .map(|(slot, (written, acked))| {
                let (jobs, handles, stream, post_tx) = (&jobs, &handles, &stream, post_tx.clone());
                let trace = &trace;
                let (read_ns, rdma_ns) = (&read_ns, &rdma_ns);
                scope.spawn(move || -> Result<()> {
                    loop {
                        let next = jobs
                            .lock()
                            .map_err(|_| anyhow!("job queue lock poisoned"))?
                            .next();
                        let Some(job) = next else { return Ok(()) };
                        let read = Instant::now();
                        let read_at = trace::now();
                        // SAFETY: this worker is the only user of its slot,
                        // and the endpoint outlives the scope.
                        let buffer = unsafe { slots.slice_mut(slot, job.len) };
                        handles[job.file]
                            .read_exact_at(buffer, job.offset)
                            .with_context(|| format!("read file {} at {}", job.file, job.offset))?;
                        add(read_ns, read);
                        let args =
                            json!({ "file": job.file, "offset": job.offset, "bytes": job.len });
                        trace.span("disk read", slot as u64, read_at, args.clone());
                        let rdma = Instant::now();
                        let rdma_at = trace::now();
                        post_tx
                            .send((slot, job.len))
                            .map_err(|_| anyhow!("the NIXL thread stopped"))?;
                        written
                            .recv()
                            .map_err(|_| anyhow!("the NIXL thread stopped"))?;
                        add(rdma_ns, rdma);
                        trace.span("rdma write", slot as u64, rdma_at, args.clone());
                        let sent_at = trace::now();
                        send(
                            stream,
                            &Message::Chunk {
                                slot,
                                file: job.file,
                                offset: job.offset,
                                len: job.len,
                            },
                        )?;
                        acked
                            .recv()
                            .map_err(|_| anyhow!("the receiver stopped acknowledging"))?;
                        trace.span("receiver write and ack", slot as u64, sent_at, args);
                    }
                })
            })
            .collect();
        drop(post_tx);

        let posted = drive_transfers(
            &endpoint,
            &peer,
            slots,
            &slot_addresses,
            post_rx,
            written_tx,
        );
        let mut result = posted;
        for worker in workers {
            let joined = worker
                .join()
                .unwrap_or_else(|_| Err(anyhow!("a transfer worker panicked")));
            if result.is_ok() {
                result = joined;
            }
        }
        result?;
        send(&stream, &Message::Done)
    });
    outcome?;
    let Message::Stats { write_s, sync_s } = stats_rx
        .recv()
        .context("the receiver closed before reporting")?
    else {
        bail!("the receiver did not answer with Stats");
    };
    let elapsed = start.elapsed().as_secs_f64();
    let workers = pool.slots as f64;
    println!(
        "TRANSFER {}",
        serde_json::json!({
            "bytes": total,
            "gb": total as f64 / 1e9,
            "seconds": elapsed,
            "gbps": total as f64 * 8.0 / 1e9 / elapsed,
            "mb_per_s": total as f64 / 1e6 / elapsed,
            "slots": pool.slots,
            "slot_mib": pool.slot_mib,
            "sender_register_s": endpoint.register.as_secs_f64(),
            "sender_huge_page_fraction": endpoint.huge_page_fraction,
            "receiver_register_s": register_s,
            "receiver_huge_page_fraction": huge_page_fraction,
            "read_s_per_worker": seconds(&read_ns) / workers,
            "rdma_s_per_worker": seconds(&rdma_ns) / workers,
            "write_s_per_worker": write_s / workers,
            "receiver_sync_s": sync_s,
        })
    );
    trace.write()
}

/// Post each chunk a worker hands over and tell the worker when it has
/// landed. Returns when every worker is done and nothing is in flight.
fn drive_transfers(
    endpoint: &Endpoint,
    peer: &str,
    slots: Slots,
    remote: &[usize],
    posts: Receiver<(usize, usize)>,
    written: Vec<Sender<()>>,
) -> Result<()> {
    let args = backend_args(&endpoint.backend)?;
    let mut in_flight: BTreeMap<usize, XferRequest> = BTreeMap::new();
    let mut workers_done = false;
    loop {
        let mut progressed = false;
        loop {
            match posts.try_recv() {
                Ok((slot, len)) => {
                    let mut source = XferDescList::new(MemType::Dram)
                        .map_err(|e| anyhow!("source descriptors: {e}"))?;
                    source.add_desc(slots.address(slot), len, 0);
                    let mut target = XferDescList::new(MemType::Dram)
                        .map_err(|e| anyhow!("target descriptors: {e}"))?;
                    target.add_desc(remote[slot], len, 0);
                    let request = endpoint
                        .agent
                        .create_xfer_req(XferOp::Write, &source, &target, peer, Some(&args))
                        .map_err(|e| anyhow!("create the transfer for slot {slot}: {e}"))?;
                    endpoint
                        .agent
                        .post_xfer_req(&request, Some(&args))
                        .map_err(|e| anyhow!("post the transfer for slot {slot}: {e}"))?;
                    in_flight.insert(slot, request);
                    progressed = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    workers_done = true;
                    break;
                }
            }
        }
        let mut landed = Vec::new();
        for (&slot, request) in &in_flight {
            let status = endpoint
                .agent
                .get_xfer_status(request)
                .map_err(|e| anyhow!("transfer for slot {slot}: {e}"))?;
            if status.is_success() {
                landed.push(slot);
            }
        }
        for slot in landed {
            in_flight.remove(&slot);
            written[slot]
                .send(())
                .map_err(|_| anyhow!("the worker for slot {slot} stopped"))?;
            progressed = true;
        }
        if workers_done && in_flight.is_empty() {
            return Ok(());
        }
        if !progressed {
            std::thread::sleep(Duration::from_micros(50));
        }
    }
}

const DIRECT_ALIGNMENT: usize = 4096;

/// A received file, with a second handle that bypasses the page cache when
/// direct I/O is on.
enum Destination {
    Disk {
        file: File,
        unbuffered: Option<File>,
        /// The length so far, when chunks are copied in through mappings.
        mapped: Option<Mutex<u64>>,
    },
    /// Chunks copied into the arena, keyed by their offset in the file.
    Memory(Mutex<BTreeMap<u64, Vec<Extent>>>),
}

impl Destination {
    /// Create `path` under `dir`, at its size and mode when they are known up front.
    fn create(dir: &Path, path: &Path, known: Option<(u64, u32)>, direct: bool) -> Result<Self> {
        ensure!(
            path.is_relative()
                && !path
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir)),
            "refusing path {path:?}"
        );
        let path = dir.join(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("create {}", path.display()))?;
        if let Some((size, mode)) = known {
            file.set_len(size)?;
            file.set_permissions(Permissions::from_mode(mode))?;
        }
        let unbuffered = direct
            .then(|| {
                OpenOptions::new()
                    .write(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)
                    .with_context(|| format!("open {} for direct I/O", path.display()))
            })
            .transpose()?;
        Ok(Self::Disk {
            file,
            unbuffered,
            mapped: None,
        })
    }

    /// Copy chunks in through shared mappings instead of `write`, which holds
    /// the file's lock and so lets one chunk in at a time. For RAM-backed
    /// files, where the copy is all there is.
    fn mapped(mut self) -> Self {
        if let Self::Disk { mapped, .. } = &mut self {
            *mapped = Some(Mutex::new(0));
        }
        self
    }

    /// Slots are 2 MiB aligned and chunks start on slot-size offsets, so only
    /// the tail of a file's last chunk needs the buffered handle.
    fn write(&self, buffer: &[u8], offset: u64, arena: Option<&Arena>) -> Result<()> {
        match self {
            Self::Disk {
                file,
                mapped: Some(length),
                ..
            } => {
                if buffer.is_empty() {
                    return Ok(());
                }
                let end = offset + buffer.len() as u64;
                {
                    let mut length = length
                        .lock()
                        .map_err(|_| anyhow!("file length lock poisoned"))?;
                    if end > *length {
                        file.set_len(end)?;
                        *length = end;
                    }
                }
                // SAFETY: a fresh shared mapping of a range that exists in the
                // file; nothing else in this process maps it.
                unsafe {
                    let mapping = rustix::mm::mmap(
                        std::ptr::null_mut(),
                        buffer.len(),
                        rustix::mm::ProtFlags::READ | rustix::mm::ProtFlags::WRITE,
                        rustix::mm::MapFlags::SHARED,
                        file,
                        offset,
                    )
                    .context("map a received file")?;
                    std::ptr::copy_nonoverlapping(buffer.as_ptr(), mapping.cast(), buffer.len());
                    rustix::mm::munmap(mapping, buffer.len()).context("unmap a received file")?;
                }
                Ok(())
            }
            Self::Disk {
                file, unbuffered, ..
            } => {
                let Some(unbuffered) = unbuffered else {
                    return Ok(file.write_all_at(buffer, offset)?);
                };
                let aligned = buffer.len() - buffer.len() % DIRECT_ALIGNMENT;
                unbuffered.write_all_at(&buffer[..aligned], offset)?;
                Ok(file.write_all_at(&buffer[aligned..], offset + aligned as u64)?)
            }
            Self::Memory(chunks) => {
                let arena = arena.context("a memory destination without an arena")?;
                let mut extents = Vec::new();
                let mut copied = 0;
                while copied < buffer.len() {
                    let (extent, address) = arena.block(buffer.len() - copied)?;
                    // SAFETY: the arena gave this block to this thread alone,
                    // and it stays mapped while the arena lives.
                    let block =
                        unsafe { std::slice::from_raw_parts_mut(address as *mut u8, extent.len) };
                    block.copy_from_slice(&buffer[copied..copied + extent.len]);
                    copied += extent.len;
                    extents.push(extent);
                }
                chunks
                    .lock()
                    .map_err(|_| anyhow!("chunk map lock poisoned"))?
                    .insert(offset, extents);
                Ok(())
            }
        }
    }
}

/// Keep streamed image files in huge-page memory instead of writing them.
#[derive(Clone, Copy, Debug, clap::Args)]
pub struct Hold {
    /// Keep streamed files in memory and serve `criu restore --stream` from
    /// them, on the socket in the destination directory.
    #[arg(long)]
    pub hold: bool,
    /// Copy streamed files in through shared mappings; for a RAM-backed
    /// destination, where plain writes go in one at a time.
    #[arg(long)]
    pub mmap: bool,
    /// Memory to map and fault in before the sender connects.
    #[arg(long, default_value_t = 0)]
    pub reserve_gib: usize,
    /// Size of each huge-page segment.
    #[arg(long, default_value_t = 4)]
    pub segment_gib: usize,
}

/// Accept one sender on `listen` and write what it sends under `dir`.
pub fn receive_directory(
    dir: &Path,
    listen: &str,
    direct: bool,
    hold: Hold,
    trace: Option<PathBuf>,
) -> Result<()> {
    let trace = Trace::new(trace, "receiver");
    let written = AtomicU64::new(0);
    let reserve = Instant::now();
    let arena = hold
        .hold
        .then(|| Arena::new(hold.segment_gib << 30, hold.reserve_gib << 30))
        .transpose()?;
    let reserve_s = reserve.elapsed().as_secs_f64();
    let names = Mutex::new(Vec::<String>::new());
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let listener = TcpListener::bind(listen).with_context(|| format!("listen on {listen}"))?;
    println!("listening on {}", listener.local_addr()?);
    let (stream, from) = listener.accept().context("accept the sender")?;
    drop(listener);
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let stream = Arc::new(Mutex::new(stream));
    let Message::Hello {
        pool,
        metadata,
        files,
    } = receive(&mut reader)?
    else {
        bail!("{from} did not start with Hello");
    };
    let handles = files
        .iter()
        .map(|entry| {
            Destination::create(dir, &entry.path, Some((entry.size, entry.mode)), direct)
                .map(Arc::new)
        })
        .collect::<Result<Vec<_>>>()?;
    let handles = RwLock::new(handles);
    let destination = |file: usize| -> Result<Arc<Destination>> {
        handles
            .read()
            .map_err(|_| anyhow!("destination lock poisoned"))?
            .get(file)
            .cloned()
            .with_context(|| format!("no file {file}"))
    };

    let endpoint = Endpoint::new(RECEIVER, pool)?;
    let slots = endpoint.slots();
    endpoint.load_peer(&metadata)?;
    send(
        &stream,
        &Message::Ready {
            metadata: endpoint.metadata()?,
            slot_addresses: (0..pool.slots).map(|slot| slots.address(slot)).collect(),
            register_s: endpoint.register.as_secs_f64(),
            huge_page_fraction: endpoint.huge_page_fraction,
        },
    )?;

    let write_ns = AtomicU64::new(0);
    std::thread::scope(|scope| -> Result<()> {
        let mut writers = Vec::new();
        let mut queues = Vec::new();
        for slot in 0..pool.slots {
            let (tx, rx) = mpsc::channel::<Job>();
            queues.push(tx);
            let (destination, stream, write_ns) = (&destination, Arc::clone(&stream), &write_ns);
            let (trace, written, arena) = (&trace, &written, arena.as_ref());
            writers.push(scope.spawn(move || -> Result<()> {
                let run = || -> Result<()> {
                    for job in rx {
                        let write = Instant::now();
                        let write_at = trace::now();
                        // SAFETY: the sender does not reuse a slot before its
                        // acknowledgement, so nothing else touches it now.
                        let buffer = unsafe { slots.slice_mut(slot, job.len) };
                        destination(job.file)?
                            .write(buffer, job.offset, arena)
                            .with_context(|| {
                                format!("write file {} at {}", job.file, job.offset)
                            })?;
                        add(write_ns, write);
                        trace.span(
                            if arena.is_some() {
                                "copy to memory"
                            } else {
                                "disk write"
                            },
                            slot as u64,
                            write_at,
                            json!({ "file": job.file, "offset": job.offset, "bytes": job.len }),
                        );
                        trace.counter(
                            "bytes written",
                            written.fetch_add(job.len as u64, Ordering::Relaxed) + job.len as u64,
                        );
                        send(&stream, &Message::Ack { slot })?;
                    }
                    Ok(())
                };
                let outcome = run();
                if outcome.is_err()
                    && let Ok(stream) = stream.lock()
                {
                    // The sender waits for an acknowledgement that will not
                    // come; closing the connection tells it.
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                }
                outcome
            }));
        }
        loop {
            match receive(&mut reader)? {
                Message::Chunk {
                    slot,
                    file,
                    offset,
                    len,
                } => {
                    ensure!(
                        slot < pool.slots && len <= pool.slot_bytes(),
                        "chunk out of range"
                    );
                    queues[slot]
                        .send(Job { file, offset, len })
                        .map_err(|_| anyhow!("the writer for slot {slot} stopped"))?;
                }
                Message::Open { file, path } => {
                    let opened = Arc::new(if arena.is_some() {
                        Destination::Memory(Mutex::new(BTreeMap::new()))
                    } else {
                        let created = Destination::create(dir, &path, None, direct)?;
                        if hold.mmap { created.mapped() } else { created }
                    });
                    names
                        .lock()
                        .map_err(|_| anyhow!("name lock poisoned"))?
                        .push(path.to_string_lossy().into_owned());
                    let mut handles = handles
                        .write()
                        .map_err(|_| anyhow!("destination lock poisoned"))?;
                    ensure!(file == handles.len(), "file {file} opened out of order");
                    handles.push(opened);
                }
                Message::Done => break,
                other => bail!("unexpected {other:?}"),
            }
        }
        drop(queues);
        for writer in writers {
            writer
                .join()
                .unwrap_or_else(|_| Err(anyhow!("a writer panicked")))?;
        }
        Ok(())
    })?;
    let sync = Instant::now();
    let handles = handles
        .into_inner()
        .map_err(|_| anyhow!("destination lock poisoned"))?;
    let names = names
        .into_inner()
        .map_err(|_| anyhow!("name lock poisoned"))?;
    let mut held = BTreeMap::new();
    let declared = handles.len() - names.len();
    for (index, destination) in handles.iter().enumerate() {
        match destination.as_ref() {
            Destination::Disk { file, .. } => file.sync_all().context("sync a received file")?,
            Destination::Memory(chunks) => {
                let chunks = chunks
                    .lock()
                    .map_err(|_| anyhow!("chunk map lock poisoned"))?;
                let extents: Vec<Extent> = chunks.values().flatten().copied().collect();
                held.insert(names[index - declared].clone(), extents);
            }
        }
    }
    send(
        &stream,
        &Message::Stats {
            write_s: seconds(&write_ns),
            sync_s: sync.elapsed().as_secs_f64(),
        },
    )?;
    trace.write()?;
    let Some(arena) = arena else { return Ok(()) };
    let store = arena.finish(held)?;
    println!(
        "HOLD {}",
        serde_json::json!({
            "gb": store.manifest.bytes() as f64 / 1e9,
            "files": store.manifest.files.len(),
            "segments": store.segments.len(),
            "reserve_s": reserve_s,
            "huge_page_fraction": huge_page_fraction(&store.ranges()),
        })
    );
    image::serve_forever(&store, &image::serve_socket(dir)?)
}

/// Take one `criu dump --stream` and send it to a receiver as it arrives:
/// every image file's pipe is read into free slots, and each slot goes out
/// over RDMA as soon as it is full. Nothing is written on this node.
pub fn stream_capture(
    images_dir: &Path,
    to: &str,
    pool: Pool,
    trace: Option<PathBuf>,
) -> Result<()> {
    let trace = Trace::new(trace, "sender");
    let endpoint = Endpoint::new(SENDER, pool)?;
    let slots = endpoint.slots();
    let stream = TcpStream::connect(to).with_context(|| format!("connect {to}"))?;
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let stream = Mutex::new(stream);
    send(
        &stream,
        &Message::Hello {
            pool,
            metadata: endpoint.metadata()?,
            files: Vec::new(),
        },
    )?;
    let Message::Ready {
        metadata,
        slot_addresses,
        register_s,
        huge_page_fraction,
    } = receive(&mut reader)?
    else {
        bail!("the receiver did not answer with Ready");
    };
    ensure!(
        slot_addresses.len() == pool.slots,
        "the receiver has {} slots, not {}",
        slot_addresses.len(),
        pool.slots
    );
    let peer = endpoint.load_peer(&metadata)?;

    let listener = image::capture_socket(images_dir)?;
    println!("ready");
    let (mut criu, _) = listener.accept().context("accept criu dump")?;
    let start = Instant::now();

    let (post_tx, post_rx) = mpsc::channel::<(usize, usize)>();
    let (free_tx, free_rx) = mpsc::channel::<usize>();
    let free_rx = Mutex::new(free_rx);
    let unused_tx = free_tx.clone();
    let mut written_tx = Vec::new();
    let mut written_rx = Vec::new();
    for slot in 0..pool.slots {
        let (tx, rx) = mpsc::channel::<()>();
        written_tx.push(tx);
        written_rx.push(Mutex::new(rx));
        free_tx
            .send(slot)
            .map_err(|_| anyhow!("slot pool closed"))?;
    }
    let (stats_tx, stats_rx) = mpsc::channel::<Message>();
    let sent_at: Vec<AtomicU64> = (0..pool.slots).map(|_| AtomicU64::new(0)).collect();
    let bytes = AtomicU64::new(0);
    let files = AtomicU64::new(0);
    let rdma_ns = AtomicU64::new(0);

    std::thread::scope(|scope| -> Result<()> {
        let (trace, sent_at) = (&trace, &sent_at);
        scope.spawn(move || {
            while let Ok(message) = receive(&mut reader) {
                match message {
                    Message::Ack { slot } => {
                        if let Some(sent) = sent_at.get(slot) {
                            trace.span(
                                "receiver write and ack",
                                slot as u64,
                                sent.load(Ordering::Relaxed),
                                json!({}),
                            );
                        }
                        if free_tx.send(slot).is_err() {
                            break;
                        }
                    }
                    other => {
                        let _ = stats_tx.send(other);
                        break;
                    }
                }
            }
        });
        let (stream, free_rx, written_rx) = (&stream, &free_rx, &written_rx);
        let (bytes, files, rdma_ns) = (&bytes, &files, &rdma_ns);
        let capture = scope.spawn(move || -> Result<()> {
            std::thread::scope(|readers| -> Result<()> {
                let mut handles = Vec::new();
                let mut file = 0;
                while let Some(filename) = image::read_request(&mut criu)? {
                    let mut pipe = image::receive_pipe(&criu)?;
                    send(
                        stream,
                        &Message::Open {
                            file,
                            path: PathBuf::from(&filename),
                        },
                    )?;
                    let (post_tx, unused_tx) = (post_tx.clone(), unused_tx.clone());
                    handles.push(readers.spawn(move || -> Result<()> {
                        let mut offset = 0u64;
                        loop {
                            let wait_at = trace::now();
                            let slot = free_rx
                                .lock()
                                .map_err(|_| anyhow!("slot pool lock poisoned"))?
                                .recv()
                                .map_err(|_| anyhow!("the receiver stopped acknowledging"))?;
                            trace.span("wait for a slot", FILE_TRACKS + file as u64, wait_at, json!({}));
                            let read_at = trace::now();
                            // SAFETY: a slot taken from the pool belongs to
                            // this thread until the receiver acknowledges it.
                            let buffer = unsafe { slots.slice_mut(slot, slots.slot_bytes) };
                            let mut filled = 0;
                            while filled < buffer.len() {
                                match pipe.read(&mut buffer[filled..]) {
                                    Ok(0) => break,
                                    Ok(read) => filled += read,
                                    Err(error) if error.kind() == ErrorKind::Interrupted => {}
                                    Err(error) => return Err(error).context("read an image pipe"),
                                }
                            }
                            if filled == 0 {
                                unused_tx
                                    .send(slot)
                                    .map_err(|_| anyhow!("slot pool closed"))?;
                                return Ok(());
                            }
                            let args = json!({ "file": file, "name": filename, "offset": offset, "bytes": filled });
                            trace.span("pipe read", slot as u64, read_at, args.clone());
                            let rdma = Instant::now();
                            let rdma_at = trace::now();
                            post_tx
                                .send((slot, filled))
                                .map_err(|_| anyhow!("the NIXL thread stopped"))?;
                            written_rx[slot]
                                .lock()
                                .map_err(|_| anyhow!("slot lock poisoned"))?
                                .recv()
                                .map_err(|_| anyhow!("the NIXL thread stopped"))?;
                            add(rdma_ns, rdma);
                            trace.span("rdma write", slot as u64, rdma_at, args);
                            sent_at[slot].store(trace::now(), Ordering::Relaxed);
                            send(
                                stream,
                                &Message::Chunk {
                                    slot,
                                    file,
                                    offset,
                                    len: filled,
                                },
                            )?;
                            trace.counter(
                                "bytes sent",
                                bytes.fetch_add(filled as u64, Ordering::Relaxed) + filled as u64,
                            );
                            offset += filled as u64;
                            if filled < slots.slot_bytes {
                                return Ok(());
                            }
                        }
                    }));
                    file += 1;
                }
                files.store(file as u64, Ordering::Relaxed);
                drop(post_tx);
                handles.into_iter().try_for_each(|handle| {
                    handle
                        .join()
                        .unwrap_or_else(|_| Err(anyhow!("a reader panicked")))
                })
            })
        });
        let posted = drive_transfers(
            &endpoint,
            &peer,
            slots,
            &slot_addresses,
            post_rx,
            written_tx,
        );
        let captured = capture
            .join()
            .unwrap_or_else(|_| Err(anyhow!("the capture thread panicked")));
        posted.and(captured)?;
        send(stream, &Message::Done)
    })?;
    let Message::Stats { write_s, sync_s } = stats_rx
        .recv()
        .context("the receiver closed before reporting")?
    else {
        bail!("the receiver did not answer with Stats");
    };
    let elapsed = start.elapsed().as_secs_f64();
    let total = bytes.load(Ordering::Relaxed);
    println!(
        "STREAM {}",
        serde_json::json!({
            "gb": total as f64 / 1e9,
            "files": files.load(Ordering::Relaxed),
            "seconds": elapsed,
            "gbps": total as f64 * 8.0 / 1e9 / elapsed,
            "mb_per_s": total as f64 / 1e6 / elapsed,
            "slots": pool.slots,
            "slot_mib": pool.slot_mib,
            "sender_register_s": endpoint.register.as_secs_f64(),
            "sender_huge_page_fraction": endpoint.huge_page_fraction,
            "receiver_register_s": register_s,
            "receiver_huge_page_fraction": huge_page_fraction,
            "rdma_s_total": seconds(&rdma_ns),
            "write_s_total": write_s,
            "receiver_sync_s": sync_s,
        })
    );
    trace.write()
}
