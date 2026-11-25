//! The TLS link between a daemon and the kube-rbac-proxy in front of it.
//!
//! ```text
//! peer daemon ══ TLS ══► kube-rbac-proxy ══ mutual TLS, 127.0.0.1:7700 ══► daemon
//! ```
//!
//! Each pod makes its own authority before its containers start, signs a
//! serving certificate for the daemon and a client certificate for the
//! proxy, and keeps only the authority's certificate. The daemon answers no
//! connection that does not present the proxy's certificate.
use std::fs;
use std::net::{IpAddr, Ipv4Addr};
use std::os::unix::fs::{PermissionsExt, chown};
use std::path::Path;

use anyhow::{Context, Result};
use openssl::asn1::Asn1Time;
use openssl::bn::{BigNum, MsbOption};
use openssl::ec::{EcGroup, EcKey};
use openssl::hash::MessageDigest;
use openssl::nid::Nid;
use openssl::pkey::{PKey, Private};
use openssl::ssl::{SslAcceptor, SslFiletype, SslMethod, SslVerifyMode, SslVersion};
use openssl::x509::extension::{
    BasicConstraints, ExtendedKeyUsage, KeyUsage, SubjectAlternativeName,
};
use openssl::x509::{X509, X509Builder, X509Name, X509NameBuilder};

pub const AUTHORITY: &str = "ca.crt";
pub const DAEMON_CERTIFICATE: &str = "daemon.crt";
pub const DAEMON_KEY: &str = "daemon.key";
pub const PROXY_CERTIFICATE: &str = "proxy.crt";
pub const PROXY_KEY: &str = "proxy.key";
pub const DAEMON_ADDRESS: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const VALID_DAYS: u32 = 3650;

#[cfg(feature = "p2p")]
#[derive(clap::Args)]
pub struct Args {
    /// Directory to write the link's certificates to, shared by the daemon
    /// and the proxy.
    #[arg(long)]
    dir: std::path::PathBuf,
    /// The user the proxy runs as, given its key.
    #[arg(long, default_value_t = 65532)]
    proxy_uid: u32,
}

#[cfg(feature = "p2p")]
pub fn run(args: &Args) -> Result<()> {
    generate(&args.dir, Some(args.proxy_uid))
}

struct Signed {
    key: PKey<Private>,
    certificate: X509,
}

fn key() -> Result<PKey<Private>> {
    let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1)?;
    Ok(PKey::from_ec_key(EcKey::generate(&group)?)?)
}

#[derive(Clone, Copy)]
enum Role {
    Server,
    Client,
}

fn builder(name: &str, key: &PKey<Private>) -> Result<(X509Builder, X509Name)> {
    let mut subject = X509NameBuilder::new()?;
    subject.append_entry_by_text("CN", name)?;
    let subject = subject.build();
    let mut builder = X509::builder()?;
    builder.set_version(2)?;
    let mut serial = BigNum::new()?;
    serial.rand(127, MsbOption::MAYBE_ZERO, false)?;
    let serial = serial.to_asn1_integer()?;
    builder.set_serial_number(&serial)?;
    builder.set_subject_name(&subject)?;
    builder.set_pubkey(key)?;
    let (from, until) = (
        Asn1Time::days_from_now(0)?,
        Asn1Time::days_from_now(VALID_DAYS)?,
    );
    builder.set_not_before(&from)?;
    builder.set_not_after(&until)?;
    Ok((builder, subject))
}

fn authority() -> Result<Signed> {
    let key = key()?;
    let (mut builder, subject) = builder("snapshotd-link-ca", &key)?;
    builder.set_issuer_name(&subject)?;
    builder.append_extension(BasicConstraints::new().critical().ca().pathlen(0).build()?)?;
    builder.append_extension(
        KeyUsage::new()
            .critical()
            .key_cert_sign()
            .crl_sign()
            .build()?,
    )?;
    builder.sign(&key, MessageDigest::sha256())?;
    Ok(Signed {
        key,
        certificate: builder.build(),
    })
}

fn leaf(name: &str, role: Role, authority: &Signed) -> Result<Signed> {
    let key = key()?;
    let (mut builder, _) = builder(name, &key)?;
    builder.set_issuer_name(authority.certificate.subject_name())?;
    builder.append_extension(BasicConstraints::new().critical().build()?)?;
    builder.append_extension(KeyUsage::new().critical().digital_signature().build()?)?;
    let mut usage = ExtendedKeyUsage::new();
    match role {
        Role::Server => {
            usage.server_auth();
        }
        Role::Client => {
            usage.client_auth();
        }
    }
    builder.append_extension(usage.build()?)?;
    if let Role::Server = role {
        let names = SubjectAlternativeName::new()
            .ip(&DAEMON_ADDRESS.to_string())
            .build(&builder.x509v3_context(Some(&authority.certificate), None))?;
        builder.append_extension(names)?;
    }
    builder.sign(&authority.key, MessageDigest::sha256())?;
    Ok(Signed {
        key,
        certificate: builder.build(),
    })
}

fn write(path: &Path, bytes: &[u8], mode: u32, owner: Option<u32>) -> Result<()> {
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    if let Some(uid) = owner {
        chown(path, Some(uid), None).with_context(|| format!("chown {}", path.display()))?;
    }
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
        .with_context(|| format!("restrict {}", path.display()))
}

/// Write a new authority's certificate, and the daemon's and the proxy's
/// certificates and keys, to `dir`. The proxy's key goes to `proxy_uid`.
pub fn generate(dir: &Path, proxy_uid: Option<u32>) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let authority = authority()?;
    let daemon = leaf("snapshotd", Role::Server, &authority)?;
    let proxy = leaf("kube-rbac-proxy", Role::Client, &authority)?;
    write(
        &dir.join(AUTHORITY),
        &authority.certificate.to_pem()?,
        0o644,
        None,
    )?;
    for (signed, certificate, key, owner) in [
        (&daemon, DAEMON_CERTIFICATE, DAEMON_KEY, None),
        (&proxy, PROXY_CERTIFICATE, PROXY_KEY, proxy_uid),
    ] {
        write(
            &dir.join(certificate),
            &signed.certificate.to_pem()?,
            0o644,
            None,
        )?;
        write(
            &dir.join(key),
            &signed.key.private_key_to_pem_pkcs8()?,
            0o600,
            owner,
        )?;
    }
    Ok(())
}

/// The daemon's side of the link: its certificate, and a demand for one
/// the pod's authority signed.
pub fn acceptor(dir: &Path) -> Result<SslAcceptor> {
    let file = |name: &str| dir.join(name);
    let mut acceptor = SslAcceptor::mozilla_intermediate_v5(SslMethod::tls_server())?;
    acceptor.set_min_proto_version(Some(SslVersion::TLS1_2))?;
    acceptor
        .set_certificate_chain_file(file(DAEMON_CERTIFICATE))
        .with_context(|| format!("read {}", file(DAEMON_CERTIFICATE).display()))?;
    acceptor
        .set_private_key_file(file(DAEMON_KEY), SslFiletype::PEM)
        .with_context(|| format!("read {}", file(DAEMON_KEY).display()))?;
    acceptor.check_private_key()?;
    acceptor
        .set_ca_file(file(AUTHORITY))
        .with_context(|| format!("read {}", file(AUTHORITY).display()))?;
    acceptor.set_verify(SslVerifyMode::PEER | SslVerifyMode::FAIL_IF_NO_PEER_CERT);
    acceptor.set_alpn_select_callback(|_, offered| {
        openssl::ssl::select_next_proto(b"\x08http/1.1", offered)
            .ok_or(openssl::ssl::AlpnError::NOACK)
    });
    Ok(acceptor.build())
}

#[cfg(test)]
pub mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};

    use openssl::ssl::{SslConnector, SslFiletype, SslMethod};
    use openssl::x509::X509;

    use crate::link::{
        AUTHORITY, DAEMON_CERTIFICATE, DAEMON_KEY, PROXY_CERTIFICATE, PROXY_KEY, acceptor, generate,
    };

    pub fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("snapshotd-link-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        generate(&dir, None).unwrap();
        dir
    }

    /// The proxy's side of the link made from `dir`, presenting the
    /// certificate in `client`.
    pub fn connector(dir: &Path, client: &Path) -> SslConnector {
        let mut connector = SslConnector::builder(SslMethod::tls_client()).unwrap();
        connector.set_ca_file(dir.join(AUTHORITY)).unwrap();
        connector
            .set_certificate_file(client.join(PROXY_CERTIFICATE), SslFiletype::PEM)
            .unwrap();
        connector
            .set_private_key_file(client.join(PROXY_KEY), SslFiletype::PEM)
            .unwrap();
        connector.set_alpn_protos(b"\x08http/1.1").unwrap();
        connector.build()
    }

    #[test]
    fn a_link_is_an_authority_two_certificates_and_two_private_keys() {
        let dir = scratch("files");
        let read = |name: &str| X509::from_pem(&fs::read(dir.join(name)).unwrap()).unwrap();
        let authority = read(AUTHORITY);
        for name in [DAEMON_CERTIFICATE, PROXY_CERTIFICATE] {
            let certificate = read(name);
            assert!(
                certificate
                    .verify(&authority.public_key().unwrap())
                    .unwrap(),
                "{name}"
            );
            assert_eq!(
                certificate.issuer_name().to_der().unwrap(),
                authority.subject_name().to_der().unwrap()
            );
        }
        let daemon = read(DAEMON_CERTIFICATE);
        let addresses: Vec<Vec<u8>> = daemon
            .subject_alt_names()
            .unwrap()
            .iter()
            .filter_map(|name| name.ipaddress().map(<[u8]>::to_vec))
            .collect();
        assert_eq!(addresses, [vec![127, 0, 0, 1]]);
        for key in [DAEMON_KEY, PROXY_KEY] {
            let mode = fs::metadata(dir.join(key)).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600, "{key}");
        }
        let mode = fs::metadata(dir.join(AUTHORITY))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o644);
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            5,
            "the authority's key is never written"
        );
        acceptor(&dir).unwrap();
        fs::remove_dir_all(&dir).unwrap();
        assert!(acceptor(&dir).is_err());
    }

    #[test]
    fn every_link_has_its_own_authority() {
        let (one, two) = (scratch("one"), scratch("two"));
        let authority = |dir: &Path| fs::read(dir.join(AUTHORITY)).unwrap();
        assert_ne!(authority(&one), authority(&two));
        fs::remove_dir_all(one).unwrap();
        fs::remove_dir_all(two).unwrap();
    }
}
