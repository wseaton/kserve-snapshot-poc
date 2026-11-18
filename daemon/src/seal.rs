//! AES-256-GCM over a checkpoint's chunks on their way between nodes.
use anyhow::{Context, Result, ensure};
use openssl::cipher::Cipher;
use openssl::cipher_ctx::CipherCtx;
use serde::{Deserialize, Serialize};

const NONCE_BYTES: usize = 12;
const TAG_BYTES: usize = 16;
const PIECE: usize = 1 << 24;

pub struct TransferKey([u8; Self::BYTES]);

#[derive(Clone, Copy, Debug)]
pub struct Place {
    pub file: usize,
    pub offset: u64,
    pub len: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Seal {
    counter: u64,
    tag: [u8; TAG_BYTES],
}

impl Place {
    fn bytes(self) -> [u8; 24] {
        let mut bytes = [0; 24];
        bytes[..8].copy_from_slice(&(self.file as u64).to_be_bytes());
        bytes[8..16].copy_from_slice(&self.offset.to_be_bytes());
        bytes[16..].copy_from_slice(&(self.len as u64).to_be_bytes());
        bytes
    }
}

fn run(context: &mut CipherCtx, data: &mut [u8]) -> Result<()> {
    for piece in data.chunks_mut(PIECE) {
        let len = piece.len();
        ensure!(
            context.cipher_update_inplace(piece, len)? == len,
            "the cipher held bytes back"
        );
    }
    Ok(())
}

impl TransferKey {
    pub const BYTES: usize = 32;

    pub fn generate() -> Result<Self> {
        let mut key = [0; Self::BYTES];
        openssl::rand::rand_bytes(&mut key).context("generate a transfer key")?;
        Ok(Self(key))
    }

    pub fn from_bytes(key: [u8; Self::BYTES]) -> Self {
        Self(key)
    }

    pub fn as_bytes(&self) -> &[u8; Self::BYTES] {
        &self.0
    }

    fn nonce(counter: u64) -> [u8; NONCE_BYTES] {
        let mut nonce = [0; NONCE_BYTES];
        nonce[NONCE_BYTES - 8..].copy_from_slice(&counter.to_be_bytes());
        nonce
    }

    /// `counter` must not repeat under this key.
    pub fn seal(&self, counter: u64, place: Place, data: &mut [u8]) -> Result<Seal> {
        ensure!(data.len() == place.len, "the chunk is not its stated size");
        let mut context = CipherCtx::new()?;
        context.encrypt_init(
            Some(Cipher::aes_256_gcm()),
            Some(&self.0),
            Some(&Self::nonce(counter)),
        )?;
        context.cipher_update(&place.bytes(), None)?;
        run(&mut context, data)?;
        context.cipher_final(&mut [])?;
        let mut tag = [0; TAG_BYTES];
        context.tag(&mut tag)?;
        Ok(Seal { counter, tag })
    }

    /// On an error `data` holds nothing that may be used.
    pub fn open(&self, seal: &Seal, place: Place, data: &mut [u8]) -> Result<()> {
        ensure!(data.len() == place.len, "the chunk is not its stated size");
        let mut context = CipherCtx::new()?;
        context.decrypt_init(
            Some(Cipher::aes_256_gcm()),
            Some(&self.0),
            Some(&Self::nonce(seal.counter)),
        )?;
        context.cipher_update(&place.bytes(), None)?;
        run(&mut context, data)?;
        context.set_tag(&seal.tag)?;
        context
            .cipher_final(&mut [])
            .ok()
            .with_context(|| format!("the chunk at {place:?} is not what the sender sealed"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::seal::{PIECE, Place, TransferKey};

    fn chunk(len: usize) -> (Place, Vec<u8>) {
        let place = Place {
            file: 3,
            offset: 1 << 33,
            len,
        };
        (place, (0..len).map(|i| (i * 31 % 251) as u8).collect())
    }

    #[test]
    fn a_sealed_chunk_opens_to_what_it_was() {
        let key = TransferKey::generate().unwrap();
        for len in [0, 1, 4096, PIECE + 12345] {
            let (place, plain) = chunk(len);
            let mut data = plain.clone();
            let seal = key.seal(7, place, &mut data).unwrap();
            assert!(len < 16 || data != plain, "{len} bytes left in the clear");
            let seal = serde_json::from_str(&serde_json::to_string(&seal).unwrap()).unwrap();
            let same_key = TransferKey::from_bytes(*key.as_bytes());
            same_key.open(&seal, place, &mut data).unwrap();
            assert_eq!(data, plain, "{len} bytes");
        }
    }

    #[test]
    fn a_chunk_opens_only_with_its_key_place_and_bytes() {
        let key = TransferKey::generate().unwrap();
        let (place, plain) = chunk(4096);
        let mut sealed = plain.clone();
        let seal = key.seal(1, place, &mut sealed).unwrap();

        let other = TransferKey::generate().unwrap();
        assert!(other.open(&seal, place, &mut sealed.clone()).is_err());

        for moved in [
            Place { file: 4, ..place },
            Place {
                offset: place.offset + 4096,
                ..place
            },
        ] {
            assert!(key.open(&seal, moved, &mut sealed.clone()).is_err());
        }
        assert!(
            key.open(
                &seal,
                Place { len: 4095, ..place },
                &mut sealed[..4095].to_vec()
            )
            .is_err()
        );

        let mut flipped = sealed.clone();
        flipped[2000] ^= 1;
        assert!(key.open(&seal, place, &mut flipped).is_err());

        let mut again = plain.clone();
        let later = key.seal(2, place, &mut again).unwrap();
        assert_ne!(again, sealed, "two counters gave one ciphertext");
        assert!(key.open(&later, place, &mut sealed.clone()).is_err());

        key.open(&seal, place, &mut sealed).unwrap();
        assert_eq!(sealed, plain);
    }

    #[test]
    fn a_chunk_of_the_wrong_size_is_not_sealed() {
        let key = TransferKey::generate().unwrap();
        let (place, mut data) = chunk(64);
        assert!(key.seal(1, place, &mut data[..63]).is_err());
    }
}
