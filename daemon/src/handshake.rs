//! What two agents say before one sends the other a checkpoint.
//!
//! ```text
//! asking replica                             replica with an image
//!   Pull { revision, nonce } ───────────────►
//!                            ◄─────────────── Challenge { nonce, proof }
//!   Proof { proof } ────────────────────────►
//!                            ◄─────────────── Accept | Refuse { reason }
//! ```
//!
//! Each proof is an HMAC-SHA256, under the key the replicas of one Deployment
//! share, of the prover's role, both nonces and the revision. Each side
//! proves it holds the key before the other acts on anything it said. Frames
//! are JSON lines.
use std::fmt;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use openssl::hash::MessageDigest;
use openssl::pkey::{PKey, Private};
use openssl::sign::Signer;
use serde::{Deserialize, Serialize};

const NONCE_BYTES: usize = 32;
const MIN_KEY_BYTES: usize = 32;
const MAX_FRAME_BYTES: usize = 1024;

/// The secret the replicas of one Deployment share.
pub struct Key(PKey<Private>);

impl Key {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        Self::from_bytes(&bytes).with_context(|| format!("key in {}", path.display()))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() >= MIN_KEY_BYTES,
            "{} bytes, need at least {MIN_KEY_BYTES}",
            bytes.len()
        );
        Ok(Self(PKey::hmac(bytes).context("load the HMAC key")?))
    }
}

/// What a checkpoint was taken from; an image restores only into a replica of
/// the same revision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Revision(String);

impl Revision {
    pub fn new(revision: impl Into<String>) -> Result<Self> {
        let revision = revision.into();
        ensure!(!revision.is_empty(), "the revision is empty");
        ensure!(revision.len() <= 512, "the revision is too long");
        Ok(Self(revision))
    }
}

impl std::str::FromStr for Revision {
    type Err = anyhow::Error;

    fn from_str(revision: &str) -> Result<Self> {
        Self::new(revision)
    }
}

impl fmt::Display for Revision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why a replica that holds the key will not send its image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Refusal {
    /// It has no finished image.
    NoImage,
    /// It is sending its image to another replica.
    Busy,
}

#[derive(Debug, Serialize, Deserialize)]
enum Frame {
    Pull { revision: Revision, nonce: String },
    Challenge { nonce: String, proof: String },
    Proof { proof: String },
    Accept,
    Refuse { reason: Refusal },
}

#[derive(Clone, Copy)]
enum Role {
    Asking,
    Holding,
}

impl Role {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::Asking => b"snapshotd pull v1 asking ",
            Self::Holding => b"snapshotd pull v1 holding",
        }
    }
}

type Nonce = [u8; NONCE_BYTES];

fn nonce() -> Result<Nonce> {
    let mut nonce = [0; NONCE_BYTES];
    openssl::rand::rand_bytes(&mut nonce).context("draw a nonce")?;
    Ok(nonce)
}

fn decode_nonce(encoded: &str) -> Result<Nonce> {
    let mut nonce = [0; NONCE_BYTES];
    hex::decode_to_slice(encoded, &mut nonce).context("decode the peer's nonce")?;
    Ok(nonce)
}

fn proof(
    key: &Key,
    role: Role,
    asking: &Nonce,
    holding: &Nonce,
    revision: &Revision,
) -> Result<Vec<u8>> {
    let mut signer = Signer::new(MessageDigest::sha256(), &key.0).context("start the HMAC")?;
    for part in [role.tag(), asking, holding, revision.0.as_bytes()] {
        signer.update(part).context("feed the HMAC")?;
    }
    signer.sign_to_vec().context("finish the HMAC")
}

fn proves(expected: &[u8], given: &str) -> bool {
    hex::decode(given)
        .is_ok_and(|given| given.len() == expected.len() && openssl::memcmp::eq(expected, &given))
}

fn write_frame(stream: &mut impl Write, frame: &Frame) -> Result<()> {
    let mut line = serde_json::to_vec(frame)?;
    line.push(b'\n');
    stream.write_all(&line).context("write a handshake frame")
}

fn read_frame(stream: &mut impl Read) -> Result<Frame> {
    let mut line = Vec::new();
    let mut byte = [0];
    loop {
        ensure!(
            stream.read(&mut byte).context("read a handshake frame")? == 1,
            "the peer closed the connection during the handshake"
        );
        if byte[0] == b'\n' {
            break;
        }
        ensure!(line.len() < MAX_FRAME_BYTES, "handshake frame too long");
        line.push(byte[0]);
    }
    serde_json::from_slice(&line).context("parse a handshake frame")
}

/// Ask the replica at the other end of `stream` for its image of `revision`.
/// `Ok(Ok(()))` means it holds the key and its image follows on `stream`.
pub fn ask(
    stream: &mut (impl Read + Write),
    key: &Key,
    revision: &Revision,
) -> Result<Result<(), Refusal>> {
    let asking = nonce()?;
    write_frame(
        stream,
        &Frame::Pull {
            revision: revision.clone(),
            nonce: hex::encode(asking),
        },
    )?;
    let Frame::Challenge {
        nonce: holding,
        proof: given,
    } = read_frame(stream)?
    else {
        bail!("the peer did not answer with a challenge");
    };
    let holding = decode_nonce(&holding)?;
    ensure!(
        proves(
            &proof(key, Role::Holding, &asking, &holding, revision)?,
            &given
        ),
        "the peer does not hold the key"
    );
    write_frame(
        stream,
        &Frame::Proof {
            proof: hex::encode(proof(key, Role::Asking, &asking, &holding, revision)?),
        },
    )?;
    match read_frame(stream)? {
        Frame::Accept => Ok(Ok(())),
        Frame::Refuse { reason } => Ok(Err(reason)),
        other => bail!("unexpected {other:?}"),
    }
}

/// Check that the replica at the other end of `stream` holds the key, and
/// return the revision it asks for. Follow with [`answer`].
pub fn authenticate(stream: &mut (impl Read + Write), key: &Key) -> Result<Revision> {
    let Frame::Pull {
        revision,
        nonce: asking,
    } = read_frame(stream)?
    else {
        bail!("the peer did not start with a pull");
    };
    let asking = decode_nonce(&asking)?;
    let holding = nonce()?;
    write_frame(
        stream,
        &Frame::Challenge {
            nonce: hex::encode(holding),
            proof: hex::encode(proof(key, Role::Holding, &asking, &holding, &revision)?),
        },
    )?;
    let Frame::Proof { proof: given } = read_frame(stream)? else {
        bail!("the peer did not answer the challenge");
    };
    ensure!(
        proves(
            &proof(key, Role::Asking, &asking, &holding, &revision)?,
            &given
        ),
        "the peer does not hold the key"
    );
    Ok(revision)
}

/// Tell an authenticated peer whether the image follows.
pub fn answer(stream: &mut impl Write, verdict: Result<(), Refusal>) -> Result<()> {
    write_frame(
        stream,
        &match verdict {
            Ok(()) => Frame::Accept,
            Err(reason) => Frame::Refuse { reason },
        },
    )
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::os::unix::net::UnixStream;
    use std::thread;

    use crate::handshake::{
        Frame, Key, MAX_FRAME_BYTES, Refusal, Revision, Role, answer, ask, authenticate,
        decode_nonce, nonce, proof, read_frame, write_frame,
    };

    fn key(fill: u8) -> Key {
        Key::from_bytes(&[fill; 32]).unwrap()
    }

    fn revision(name: &str) -> Revision {
        Revision::new(name).unwrap()
    }

    /// Run both ends; the holding end answers with `verdict` once the asking
    /// end has proved itself.
    fn exchange(
        asking_key: Key,
        holding_key: Key,
        verdict: Result<(), Refusal>,
    ) -> (
        anyhow::Result<Result<(), Refusal>>,
        anyhow::Result<Revision>,
    ) {
        let (mut asking, mut holding) = UnixStream::pair().unwrap();
        let holder = thread::spawn(move || {
            let asked = authenticate(&mut holding, &holding_key)?;
            answer(&mut holding, verdict)?;
            Ok(asked)
        });
        let asked = ask(&mut asking, &asking_key, &revision("7d9f8c6b54"));
        drop(asking);
        (asked, holder.join().unwrap())
    }

    #[test]
    fn replicas_with_the_same_key_agree() {
        let (asked, held) = exchange(key(1), key(1), Ok(()));
        assert_eq!(asked.unwrap(), Ok(()));
        assert_eq!(held.unwrap(), revision("7d9f8c6b54"));
    }

    #[test]
    fn a_refusal_reaches_the_asking_replica() {
        for reason in [Refusal::NoImage, Refusal::Busy] {
            let (asked, held) = exchange(key(1), key(1), Err(reason));
            assert_eq!(asked.unwrap(), Err(reason));
            held.unwrap();
        }
    }

    #[test]
    fn the_asking_replica_rejects_a_holder_with_another_key() {
        let (asked, held) = exchange(key(1), key(2), Ok(()));
        assert!(format!("{:#}", asked.unwrap_err()).contains("does not hold the key"));
        assert!(held.is_err());
    }

    #[test]
    fn the_holder_rejects_a_wrong_proof() {
        let (mut asking, mut holding) = UnixStream::pair().unwrap();
        let holder = thread::spawn(move || authenticate(&mut holding, &key(1)));
        write_frame(
            &mut asking,
            &Frame::Pull {
                revision: revision("r"),
                nonce: hex::encode(nonce().unwrap()),
            },
        )
        .unwrap();
        let Frame::Challenge { .. } = read_frame(&mut asking).unwrap() else {
            panic!("no challenge");
        };
        write_frame(
            &mut asking,
            &Frame::Proof {
                proof: hex::encode([0u8; 32]),
            },
        )
        .unwrap();
        let error = holder.join().unwrap().unwrap_err();
        assert!(format!("{error:#}").contains("does not hold the key"));
    }

    #[test]
    fn the_holder_rejects_its_own_proof_sent_back() {
        let (mut asking, mut holding) = UnixStream::pair().unwrap();
        let holder = thread::spawn(move || authenticate(&mut holding, &key(1)));
        write_frame(
            &mut asking,
            &Frame::Pull {
                revision: revision("r"),
                nonce: hex::encode(nonce().unwrap()),
            },
        )
        .unwrap();
        let Frame::Challenge { proof, .. } = read_frame(&mut asking).unwrap() else {
            panic!("no challenge");
        };
        write_frame(&mut asking, &Frame::Proof { proof }).unwrap();
        assert!(holder.join().unwrap().is_err());
    }

    #[test]
    fn a_proof_covers_the_role_both_nonces_and_the_revision() {
        let (a, b, c) = (nonce().unwrap(), nonce().unwrap(), nonce().unwrap());
        let base = proof(&key(1), Role::Asking, &a, &b, &revision("r")).unwrap();
        assert_eq!(base.len(), 32);
        assert_eq!(
            base,
            proof(&key(1), Role::Asking, &a, &b, &revision("r")).unwrap()
        );
        assert_ne!(
            base,
            proof(&key(2), Role::Asking, &a, &b, &revision("r")).unwrap()
        );
        assert_ne!(
            base,
            proof(&key(1), Role::Holding, &a, &b, &revision("r")).unwrap()
        );
        assert_ne!(
            base,
            proof(&key(1), Role::Asking, &c, &b, &revision("r")).unwrap()
        );
        assert_ne!(
            base,
            proof(&key(1), Role::Asking, &a, &c, &revision("r")).unwrap()
        );
        assert_ne!(
            base,
            proof(&key(1), Role::Asking, &a, &b, &revision("s")).unwrap()
        );
    }

    #[test]
    fn a_proof_matches_hmac_sha256_of_its_parts() {
        let (a, b) = ([1u8; 32], [2u8; 32]);
        let mut message = b"snapshotd pull v1 asking ".to_vec();
        message.extend_from_slice(&a);
        message.extend_from_slice(&b);
        message.extend_from_slice(b"r");
        let pkey = openssl::pkey::PKey::hmac(&[7u8; 32]).unwrap();
        let mut signer =
            openssl::sign::Signer::new(openssl::hash::MessageDigest::sha256(), &pkey).unwrap();
        signer.update(&message).unwrap();
        assert_eq!(
            proof(&key(7), Role::Asking, &a, &b, &revision("r")).unwrap(),
            signer.sign_to_vec().unwrap()
        );
    }

    #[test]
    fn the_role_tags_have_one_length() {
        assert_eq!(Role::Asking.tag().len(), Role::Holding.tag().len());
        assert_ne!(Role::Asking.tag(), Role::Holding.tag());
    }

    #[test]
    fn a_short_key_is_refused() {
        assert!(Key::from_bytes(&[1; 31]).is_err());
        assert!(Key::from_bytes(&[1; 32]).is_ok());
    }

    #[test]
    fn a_key_is_loaded_from_its_file() {
        let path = std::env::temp_dir().join(format!("snapshotd-key-{}", std::process::id()));
        std::fs::write(&path, [9u8; 32]).unwrap();
        let (a, b) = (nonce().unwrap(), nonce().unwrap());
        assert_eq!(
            proof(
                &Key::load(&path).unwrap(),
                Role::Asking,
                &a,
                &b,
                &revision("r")
            )
            .unwrap(),
            proof(&key(9), Role::Asking, &a, &b, &revision("r")).unwrap()
        );
        std::fs::write(&path, [9u8; 8]).unwrap();
        let error = Key::load(&path).err().unwrap();
        assert!(format!("{error:#}").contains(path.to_str().unwrap()));
        std::fs::remove_file(&path).unwrap();
        assert!(Key::load(&path).is_err());
    }

    #[test]
    fn an_empty_revision_is_refused() {
        assert!(Revision::new("").is_err());
        assert!(Revision::new("x".repeat(513)).is_err());
        assert_eq!(revision("abc").to_string(), "abc");
    }

    #[test]
    fn a_nonce_of_the_wrong_length_is_refused() {
        assert!(decode_nonce(&hex::encode([0u8; 31])).is_err());
        assert!(decode_nonce(&hex::encode([0u8; 33])).is_err());
        assert!(decode_nonce("not hex").is_err());
        assert_ne!(nonce().unwrap(), nonce().unwrap());
    }

    #[test]
    fn an_endless_frame_is_refused() {
        let (mut writer, mut reader) = UnixStream::pair().unwrap();
        writer.write_all(&vec![b'x'; MAX_FRAME_BYTES + 1]).unwrap();
        let error = read_frame(&mut reader).unwrap_err();
        assert!(format!("{error:#}").contains("too long"));
    }

    #[test]
    fn a_closed_connection_is_an_error() {
        let (writer, mut reader) = UnixStream::pair().unwrap();
        drop(writer);
        assert!(read_frame(&mut reader).is_err());
    }

    #[test]
    fn nothing_after_a_frame_is_consumed() {
        let (mut writer, mut reader) = UnixStream::pair().unwrap();
        write_frame(&mut writer, &Frame::Accept).unwrap();
        writer.write_all(b"next").unwrap();
        assert!(matches!(read_frame(&mut reader).unwrap(), Frame::Accept));
        let mut rest = [0; 4];
        std::io::Read::read_exact(&mut reader, &mut rest).unwrap();
        assert_eq!(&rest, b"next");
    }
}
