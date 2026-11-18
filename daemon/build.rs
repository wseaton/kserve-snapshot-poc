fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?).join("proto");
    for (directory, file) in [
        ("vllm", "control.proto"),
        ("cri", "api.proto"),
        ("podresources", "api.proto"),
    ] {
        let include = root.join(directory);
        let proto = include.join(file);
        println!("cargo:rerun-if-changed={}", proto.display());
        tonic_prost_build::configure()
            .build_server(false)
            .compile_fds(protox::compile([&proto], [&include])?)?;
    }
    Ok(())
}
