FROM docker.io/library/rust:1-bookworm AS build
COPY daemon /src/daemon
RUN cd /src/daemon && cargo build --release --locked --bin snapshotd --example stand_in

FROM registry.fedoraproject.org/fedora:42
RUN dnf install -y --setopt=install_weak_deps=False criu util-linux openssl-libs \
    && dnf clean all
COPY --from=build /src/daemon/target/release/snapshotd /usr/local/bin/snapshotd
COPY --from=build /src/daemon/target/release/examples/stand_in /usr/local/bin/stand-in
RUN set -eux; \
    mkdir -p /opt/tools/lib /opt/tools/criu-plugins; \
    cp "$(command -v criu)" /opt/tools/criu; \
    /opt/tools/criu --version; \
    snapshotd --help > /dev/null; \
    stand-in --help > /dev/null
