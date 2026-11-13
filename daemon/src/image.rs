//! Hold a CRIU checkpoint in huge-page memory instead of in files.
//!
//! With `--stream`, CRIU does not write image files. It connects to a UNIX
//! socket in the images directory and, for every image file, sends the file
//! name and one end of a pipe:
//!
//! ```text
//! dump     criu ──name, pipe──► streamer-capture.sock   pipe ─read─► arena
//! restore  criu ──name────────► streamer-serve.sock
//!               ◄─exists──────
//!               ──pipe────────►                          arena ─write─► pipe
//! ```
//!
//! This module is the other end of both sockets. Captured bytes land in an
//! arena of anonymous, 2 MiB-aligned segments mapped with `MADV_HUGEPAGE`,
//! so the segments can later be registered for RDMA cheaply and served back
//! to a restore without touching a disk.
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, IoSliceMut, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::OwnedFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use prost::Message;
use rustix::mm::{Advice, MapFlags, ProtFlags};
use rustix::net::{RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags};
use serde::{Deserialize, Serialize};

pub const CAPTURE_SOCKET: &str = "streamer-capture.sock";
pub const SERVE_SOCKET: &str = "streamer-serve.sock";
const HUGE_PAGE: usize = 2 << 20;
const FIRST_BLOCK: usize = 64 << 10;
const LARGEST_BLOCK: usize = 16 << 20;
const PIPE_BYTES: usize = 64 << 20;

#[derive(Clone, PartialEq, Message)]
struct FileRequest {
    #[prost(string, required, tag = "1")]
    filename: String,
}

#[derive(Clone, PartialEq, Message)]
struct FileReply {
    #[prost(bool, required, tag = "1")]
    exists: bool,
}

/// An anonymous mapping aligned to 2 MiB and advised for transparent huge
/// pages. Its address and length never change.
pub struct HugeRegion {
    base: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is plain memory owned by this value.
unsafe impl Send for HugeRegion {}
// SAFETY: shared access only reads the address and length; callers that write
// through it hold `&mut` or follow the arena's exclusive-block rule.
unsafe impl Sync for HugeRegion {}

impl HugeRegion {
    pub fn new(len: usize) -> Result<Self> {
        ensure!(len > 0, "a region cannot be empty");
        let len = len.next_multiple_of(HUGE_PAGE);
        let span = len + HUGE_PAGE;
        // SAFETY: a fresh anonymous private mapping, not placed over anything.
        let raw = unsafe {
            rustix::mm::mmap_anonymous(
                std::ptr::null_mut(),
                span,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::PRIVATE,
            )
        }
        .with_context(|| format!("map {span} bytes"))?;
        let start = raw as usize;
        let aligned = start.next_multiple_of(HUGE_PAGE);
        let head = aligned - start;
        let tail = span - head - len;
        // SAFETY: both ranges lie inside the mapping made above and are not
        // part of the aligned region that is kept.
        unsafe {
            if head > 0 {
                rustix::mm::munmap(raw, head).context("trim the mapping")?;
            }
            if tail > 0 {
                rustix::mm::munmap((aligned + len) as *mut _, tail).context("trim the mapping")?;
            }
            rustix::mm::madvise(aligned as *mut _, len, Advice::LinuxHugepage)
                .context("ask for huge pages")?;
        }
        let base = NonNull::new(aligned as *mut u8).context("mapping at address zero")?;
        Ok(Self { base, len })
    }

    pub fn address(&self) -> usize {
        self.base.as_ptr() as usize
    }

    /// Fault every page in now, so later writes do not pay for it.
    pub fn populate(&mut self) -> Result<()> {
        // SAFETY: the range is exactly this mapping.
        unsafe {
            rustix::mm::madvise(
                self.base.as_ptr().cast(),
                self.len,
                Advice::LinuxPopulateWrite,
            )
        }
        .context("populate a region")
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for HugeRegion {
    fn drop(&mut self) {
        // SAFETY: this value owns exactly this mapping.
        let _ = unsafe { rustix::mm::munmap(self.base.as_ptr().cast(), self.len) };
    }
}

impl fmt::Debug for HugeRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HugeRegion({:#x}, {} bytes)", self.address(), self.len)
    }
}

impl AsRef<[u8]> for HugeRegion {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the mapping is readable for `len` bytes while `self` lives.
        unsafe { std::slice::from_raw_parts(self.base.as_ptr(), self.len) }
    }
}

impl AsMut<[u8]> for HugeRegion {
    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.base.as_ptr(), self.len) }
    }
}

/// Share of `[address, address + len)` backed by transparent huge pages.
pub fn huge_page_fraction(ranges: &[(usize, usize)]) -> f64 {
    let Ok(smaps) = fs::read_to_string("/proc/self/smaps") else {
        return 0.0;
    };
    let total: usize = ranges.iter().map(|(_, len)| len).sum();
    let mut inside = false;
    let mut huge_kib = 0u64;
    for line in smaps.lines() {
        let mut fields = line.split_ascii_whitespace();
        let Some(first) = fields.next() else { continue };
        if let Some((start, end)) = first.split_once('-')
            && let (Ok(start), Ok(end)) = (
                usize::from_str_radix(start, 16),
                usize::from_str_radix(end, 16),
            )
        {
            inside = ranges
                .iter()
                .any(|&(address, len)| start < address + len && end > address);
        } else if inside && first == "AnonHugePages:" {
            huge_kib += fields.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        }
    }
    if total == 0 {
        0.0
    } else {
        (huge_kib * 1024) as f64 / total as f64
    }
}

/// Filled bytes of one image file, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extent {
    pub segment: usize,
    pub offset: usize,
    pub len: usize,
}

/// Where every image file's bytes are, and how much of each segment is used.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub segment_bytes: usize,
    pub used: Vec<usize>,
    pub files: BTreeMap<String, Vec<Extent>>,
}

impl Manifest {
    pub fn bytes(&self) -> u64 {
        self.files
            .values()
            .flatten()
            .map(|extent| extent.len as u64)
            .sum()
    }
}

struct Bump {
    segments: Vec<HugeRegion>,
    used: Vec<usize>,
}

/// Segments handed out in blocks. A block belongs to the one thread that
/// asked for it, which is what lets threads fill different blocks at once.
pub struct Arena {
    segment_bytes: usize,
    bump: Mutex<Bump>,
}

impl Arena {
    /// An arena with enough segments for `reserve_bytes` already mapped and
    /// faulted in, every segment on its own thread. It grows past that one
    /// unfaulted segment at a time.
    pub fn new(segment_bytes: usize, reserve_bytes: usize) -> Result<Self> {
        let segment_bytes = segment_bytes.next_multiple_of(HUGE_PAGE);
        let mut segments = (0..reserve_bytes.div_ceil(segment_bytes))
            .map(|_| HugeRegion::new(segment_bytes))
            .collect::<Result<Vec<_>>>()?;
        std::thread::scope(|scope| {
            let faults: Vec<_> = segments
                .iter_mut()
                .map(|segment| scope.spawn(|| segment.populate()))
                .collect();
            faults.into_iter().try_for_each(|fault| {
                fault
                    .join()
                    .unwrap_or_else(|_| Err(anyhow!("a populate thread panicked")))
            })
        })?;
        Ok(Self {
            segment_bytes,
            bump: Mutex::new(Bump {
                segments,
                used: Vec::new(),
            }),
        })
    }

    /// A block of up to `want` bytes: its place and its address.
    pub fn block(&self, want: usize) -> Result<(Extent, usize)> {
        let mut bump = self
            .bump
            .lock()
            .map_err(|_| anyhow!("arena lock poisoned"))?;
        let full = bump
            .used
            .last()
            .is_none_or(|&used| used == self.segment_bytes);
        if full {
            if bump.segments.len() == bump.used.len() {
                bump.segments.push(HugeRegion::new(self.segment_bytes)?);
            }
            bump.used.push(0);
        }
        let segment = bump.used.len() - 1;
        let offset = bump.used[segment];
        let len = want.min(self.segment_bytes - offset);
        bump.used[segment] = offset + len;
        Ok((
            Extent {
                segment,
                offset,
                len,
            },
            bump.segments[segment].address() + offset,
        ))
    }

    pub fn finish(self, files: BTreeMap<String, Vec<Extent>>) -> Result<Store> {
        let mut bump = self
            .bump
            .into_inner()
            .map_err(|_| anyhow!("arena lock poisoned"))?;
        bump.segments.truncate(bump.used.len());
        Ok(Store {
            manifest: Manifest {
                segment_bytes: self.segment_bytes,
                used: bump.used,
                files,
            },
            segments: bump.segments,
        })
    }
}

/// A checkpoint held in memory.
pub struct Store {
    pub manifest: Manifest,
    pub segments: Vec<HugeRegion>,
}

impl Store {
    pub fn ranges(&self) -> Vec<(usize, usize)> {
        self.segments
            .iter()
            .map(|segment| (segment.address(), segment.len()))
            .collect()
    }

    fn slice(&self, extent: Extent) -> Result<&[u8]> {
        self.segments
            .get(extent.segment)
            .and_then(|segment| {
                segment
                    .as_ref()
                    .get(extent.offset..extent.offset + extent.len)
            })
            .with_context(|| format!("{extent:?} is outside the store"))
    }
}

pub fn read_request(stream: &mut UnixStream) -> Result<Option<String>> {
    let mut size = [0u8; 4];
    match stream.read_exact(&mut size) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error).context("read a file request"),
    }
    let mut payload = vec![0u8; u32::from_le_bytes(size) as usize];
    stream
        .read_exact(&mut payload)
        .context("read a file request")?;
    let request = FileRequest::decode(payload.as_slice()).context("decode a file request")?;
    Ok(Some(request.filename))
}

/// Receive the pipe end CRIU sends after a request: one byte carrying one
/// descriptor.
pub fn receive_pipe(stream: &UnixStream) -> Result<File> {
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut ancillary = RecvAncillaryBuffer::new(&mut space);
    let mut byte = [0u8; 1];
    let received = rustix::net::recvmsg(
        stream,
        &mut [IoSliceMut::new(&mut byte)],
        &mut ancillary,
        RecvFlags::CMSG_CLOEXEC,
    )
    .context("receive a pipe")?;
    ensure!(
        received.bytes == 1,
        "the peer closed instead of sending a pipe"
    );
    let pipe: Option<OwnedFd> = ancillary.drain().find_map(|message| match message {
        RecvAncillaryMessage::ScmRights(mut fds) => fds.next(),
        _ => None,
    });
    let pipe = pipe.context("no descriptor came with the pipe message")?;
    let _ = rustix::pipe::fcntl_setpipe_size(&pipe, PIPE_BYTES);
    Ok(File::from(pipe))
}

fn drain(mut pipe: File, arena: &Arena) -> Result<Vec<Extent>> {
    let mut extents = Vec::new();
    let mut want = FIRST_BLOCK;
    loop {
        let (mut extent, address) = arena.block(want)?;
        // SAFETY: the arena gave this block to this thread alone, and its
        // segment stays mapped until the arena is finished.
        let block = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, extent.len) };
        let mut filled = 0;
        while filled < block.len() {
            match pipe.read(&mut block[filled..]) {
                Ok(0) => break,
                Ok(read) => filled += read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error).context("read an image pipe"),
            }
        }
        let complete = filled == extent.len;
        extent.len = filled;
        if filled > 0 {
            extents.push(extent);
        }
        if !complete {
            return Ok(extents);
        }
        want = (want * 2).min(LARGEST_BLOCK);
    }
}

fn listen(images_dir: &Path, name: &str) -> Result<UnixListener> {
    let path = images_dir.join(name);
    match fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("remove {}", path.display())),
    }
    UnixListener::bind(&path).with_context(|| format!("listen on {}", path.display()))
}

/// The capture socket, bound before CRIU is started.
pub struct Capture {
    listener: UnixListener,
    arena: Arena,
}

impl Capture {
    pub fn bind(images_dir: &Path, segment_bytes: usize, reserve_bytes: usize) -> Result<Self> {
        let start = Instant::now();
        let arena = Arena::new(segment_bytes, reserve_bytes)?;
        println!(
            "RESERVE {}",
            serde_json::json!({
                "seconds": start.elapsed().as_secs_f64(),
                "gib": reserve_bytes >> 30,
            })
        );
        Ok(Self {
            listener: listen(images_dir, CAPTURE_SOCKET)?,
            arena,
        })
    }

    /// Take one `criu dump --stream` into memory.
    pub fn run(self) -> Result<Store> {
        let (mut stream, _) = self.listener.accept().context("accept criu dump")?;
        let start = Instant::now();
        let arena = self.arena;
        let files = std::thread::scope(|scope| -> Result<BTreeMap<String, Vec<Extent>>> {
            let mut readers = Vec::new();
            while let Some(filename) = read_request(&mut stream)? {
                let pipe = receive_pipe(&stream)?;
                let arena = &arena;
                readers.push((filename, scope.spawn(move || drain(pipe, arena))));
            }
            let mut files = BTreeMap::new();
            for (filename, reader) in readers {
                let extents = reader
                    .join()
                    .unwrap_or_else(|_| Err(anyhow!("a reader panicked")))
                    .with_context(|| format!("capture {filename}"))?;
                ensure!(
                    files.insert(filename.clone(), extents).is_none(),
                    "criu sent {filename} twice"
                );
            }
            Ok(files)
        })?;
        let store = arena.finish(files)?;
        println!(
            "CAPTURE {}",
            serde_json::json!({
                "seconds": start.elapsed().as_secs_f64(),
                "gb": store.manifest.bytes() as f64 / 1e9,
                "files": store.manifest.files.len(),
                "segments": store.segments.len(),
                "segment_gib": store.manifest.segment_bytes >> 30,
                "huge_page_fraction": huge_page_fraction(&store.ranges()),
            })
        );
        Ok(store)
    }
}

/// Answer one `criu restore --stream` from `store`.
pub fn serve(store: &Store, listener: &UnixListener) -> Result<()> {
    let (mut stream, _) = listener.accept().context("accept criu restore")?;
    let start = Instant::now();
    let served = std::thread::scope(|scope| -> Result<u64> {
        let mut writers = Vec::new();
        while let Some(filename) = read_request(&mut stream)? {
            let extents = store.manifest.files.get(&filename);
            let mut reply = FileReply {
                exists: extents.is_some(),
            }
            .encode_to_vec();
            let mut message = (reply.len() as u32).to_le_bytes().to_vec();
            message.append(&mut reply);
            stream
                .write_all(&message)
                .context("answer a file request")?;
            let Some(extents) = extents else { continue };
            let mut pipe = receive_pipe(&stream)?;
            writers.push(scope.spawn(move || -> Result<u64> {
                let mut written = 0;
                for &extent in extents {
                    match pipe.write_all(store.slice(extent)?) {
                        Ok(()) => written += extent.len as u64,
                        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => break,
                        Err(error) => return Err(error).context("write an image pipe"),
                    }
                }
                Ok(written)
            }));
        }
        let mut served = 0;
        for writer in writers {
            served += writer
                .join()
                .unwrap_or_else(|_| Err(anyhow!("a writer panicked")))?;
        }
        Ok(served)
    })?;
    println!(
        "SERVE {}",
        serde_json::json!({
            "seconds": start.elapsed().as_secs_f64(),
            "gb": served as f64 / 1e9,
        })
    );
    Ok(())
}

/// Capture one dump into memory, then answer restores from it until killed.
pub fn hold(images_dir: &Path, segment_bytes: usize, reserve_bytes: usize) -> Result<()> {
    let capture = Capture::bind(images_dir, segment_bytes, reserve_bytes)?;
    let restores = listen(images_dir, SERVE_SOCKET)?;
    println!("ready");
    let store = capture.run()?;
    serve_forever(&store, &restores)
}

pub fn serve_forever(store: &Store, restores: &UnixListener) -> Result<()> {
    loop {
        if let Err(error) = serve(store, restores) {
            bail!("serve a restore: {error:#}");
        }
    }
}

pub fn serve_socket(images_dir: &Path) -> Result<UnixListener> {
    listen(images_dir, SERVE_SOCKET)
}

pub fn capture_socket(images_dir: &Path) -> Result<UnixListener> {
    listen(images_dir, CAPTURE_SOCKET)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::os::fd::AsFd;
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;

    use prost::Message;
    use rustix::net::{SendAncillaryBuffer, SendAncillaryMessage, SendFlags};

    use crate::image::{
        Arena, CAPTURE_SOCKET, Capture, FileReply, FileRequest, HUGE_PAGE, HugeRegion,
        SERVE_SOCKET, serve, serve_socket,
    };

    fn images_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-image-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn request(stream: &mut UnixStream, filename: &str) {
        let payload = FileRequest {
            filename: filename.to_string(),
        }
        .encode_to_vec();
        stream
            .write_all(&(payload.len() as u32).to_le_bytes())
            .unwrap();
        stream.write_all(&payload).unwrap();
    }

    fn send_pipe(stream: &UnixStream, end: impl AsFd) {
        let fds = [end.as_fd()];
        let mut space = [std::mem::MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut ancillary = SendAncillaryBuffer::new(&mut space);
        assert!(ancillary.push(SendAncillaryMessage::ScmRights(&fds)));
        rustix::net::sendmsg(
            stream,
            &[std::io::IoSlice::new(b"*")],
            &mut ancillary,
            SendFlags::empty(),
        )
        .unwrap();
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31) ^ seed)
            .collect()
    }

    #[test]
    fn huge_region_is_aligned_zeroed_and_writable() {
        let mut region = HugeRegion::new(3 * HUGE_PAGE + 1).unwrap();
        assert_eq!(region.address() % HUGE_PAGE, 0);
        assert_eq!(region.len(), 4 * HUGE_PAGE);
        assert!(region.as_ref().iter().all(|&b| b == 0));
        region.as_mut()[4 * HUGE_PAGE - 1] = 7;
        assert_eq!(region.as_ref()[4 * HUGE_PAGE - 1], 7);
    }

    #[test]
    fn arena_blocks_never_overlap_and_roll_over_segments() {
        let arena = Arena::new(HUGE_PAGE, 0).unwrap();
        let mut seen = Vec::new();
        for want in [HUGE_PAGE / 2, HUGE_PAGE, 10] {
            let (extent, address) = arena.block(want).unwrap();
            assert!(extent.len <= want && extent.len > 0);
            seen.push((extent, address));
        }
        assert_eq!(seen[0].0.len, HUGE_PAGE / 2);
        assert_eq!(seen[1].0.len, HUGE_PAGE / 2);
        assert_eq!((seen[2].0.segment, seen[2].0.offset), (1, 0));
        let store = arena.finish(Default::default()).unwrap();
        assert_eq!(store.manifest.used, vec![HUGE_PAGE, 10]);
    }

    #[test]
    fn a_dump_is_captured_and_served_back_byte_for_byte() {
        let dir = images_dir("roundtrip");
        let big = pattern(5 * HUGE_PAGE + 12345, 1);
        let small = pattern(100, 2);
        let empty = Vec::new();
        let capture = Capture::bind(&dir, 2 * HUGE_PAGE, 3 * HUGE_PAGE).unwrap();
        let store = std::thread::scope(|scope| {
            let dump = scope.spawn(|| {
                let mut stream = UnixStream::connect(dir.join(CAPTURE_SOCKET)).unwrap();
                let mut writers = Vec::new();
                for (name, data) in [
                    ("pages-1.img", &big),
                    ("core-1.img", &small),
                    ("empty.img", &empty),
                ] {
                    let (read_end, write_end) = rustix::pipe::pipe().unwrap();
                    request(&mut stream, name);
                    send_pipe(&stream, &read_end);
                    drop(read_end);
                    writers.push(scope.spawn(move || {
                        std::fs::File::from(write_end).write_all(data).unwrap();
                    }));
                }
                for writer in writers {
                    writer.join().unwrap();
                }
            });
            let store = capture.run().unwrap();
            dump.join().unwrap();
            store
        });
        assert_eq!(store.manifest.files.len(), 3);
        assert_eq!(store.manifest.bytes(), (big.len() + small.len()) as u64);
        assert!(store.manifest.files["empty.img"].is_empty());

        let listener = serve_socket(&dir).unwrap();
        std::thread::scope(|scope| {
            let restore = scope.spawn(|| {
                let mut stream = UnixStream::connect(dir.join(SERVE_SOCKET)).unwrap();
                let mut fetch = |name: &str| -> Option<Vec<u8>> {
                    request(&mut stream, name);
                    let mut size = [0u8; 4];
                    stream.read_exact(&mut size).unwrap();
                    let mut payload = vec![0u8; u32::from_le_bytes(size) as usize];
                    stream.read_exact(&mut payload).unwrap();
                    if !FileReply::decode(payload.as_slice()).unwrap().exists {
                        return None;
                    }
                    let (read_end, write_end) = rustix::pipe::pipe().unwrap();
                    send_pipe(&stream, &write_end);
                    drop(write_end);
                    let mut data = Vec::new();
                    std::fs::File::from(read_end)
                        .read_to_end(&mut data)
                        .unwrap();
                    Some(data)
                };
                assert_eq!(fetch("pages-1.img").as_deref(), Some(big.as_slice()));
                assert_eq!(fetch("core-1.img").as_deref(), Some(small.as_slice()));
                assert_eq!(fetch("empty.img").as_deref(), Some(&[][..]));
                assert_eq!(fetch("missing.img"), None);
            });
            serve(&store, &listener).unwrap();
            restore.join().unwrap();
        });
    }
}
