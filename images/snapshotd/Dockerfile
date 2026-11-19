ARG FEDORA_VERSION=42

FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION} AS rust
RUN dnf install -y --setopt=install_weak_deps=False \
        gcc gcc-c++ make pkgconf-pkg-config git-core curl ca-certificates \
        openssl-devel binutils findutils which xz \
    && dnf clean all
ARG RUST_VERSION=1.100.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION} \
            --target "$(uname -m)-unknown-linux-musl"
ENV PATH="/root/.cargo/bin:${PATH}"

FROM rust AS plain
COPY daemon /src/daemon
RUN set -eux; cd /src/daemon; \
    cargo build --release --locked --bin snapshotd --example stand_in; \
    cargo build --release --locked -p snapshotd-shim --target "$(uname -m)-unknown-linux-musl"; \
    cp "target/$(uname -m)-unknown-linux-musl/release/snapshotd-shim" target/release/; \
    if readelf -d target/release/snapshotd-shim | grep NEEDED; then exit 1; fi; \
    objdump -T target/release/snapshotd | grep -oE '(GLIBC|OPENSSL)_[0-9.]+' | sort -Vu | tee /tmp/needs; \
    newest() { grep "^$1_" /tmp/needs | tail -1; }; \
    test "$(printf '%s\n' "$(newest GLIBC)" GLIBC_2.39 | sort -V | tail -1)" = GLIBC_2.39; \
    test "$(printf '%s\n' "$(newest OPENSSL)" OPENSSL_3.0.0 | sort -V | tail -1)" = OPENSSL_3.0.0

FROM rust AS nixl
RUN dnf install -y --setopt=install_weak_deps=False \
        cmake meson ninja-build autoconf automake libtool rdma-core rdma-core-devel \
        numactl-devel hwloc-devel libaio-devel liburing-devel libstdc++-devel \
        clang clang-devel llvm-devel python3 python3-devel pybind11-devel \
    && dnf clean all

ARG UCX_REF=v1.23.2
RUN git clone --depth 1 -b ${UCX_REF} https://github.com/openucx/ucx.git /src/ucx \
    && cd /src/ucx && ./autogen.sh \
    && ./contrib/configure-release-mt --prefix=/usr/local/ucx \
        --enable-shared --disable-static --disable-doxygen-doc \
        --enable-optimizations --without-avx --enable-cma \
        --without-cuda --without-rocm --with-verbs --with-rdmacm \
    && make -j"$(nproc)" && make install \
    && rm -rf /src/ucx

ENV PKG_CONFIG_PATH="/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:/usr/local/ucx/lib/pkgconfig:/usr/lib64/pkgconfig:/usr/share/pkgconfig"
ENV NIXL_PREFIX=/usr/local

ARG NIXL_REPO=https://github.com/wseaton/nixl
ARG NIXL_REF=70b07d16232c9654b7ce8597de2c6f9c12032425
COPY daemon/Cargo.toml /tmp/Cargo.toml
RUN grep -q "rev = \"${NIXL_REF}\"" /tmp/Cargo.toml \
    && git init -q /src/nixl && cd /src/nixl \
    && git fetch -q --depth 1 ${NIXL_REPO} ${NIXL_REF} && git checkout -q FETCH_HEAD \
    && meson setup build --prefix=/usr/local --buildtype=release \
        -Denable_plugins=UCX \
        -Ddisable_gds_backend=true \
        -Dbuild_tests=false \
        -Dbuild_examples=false \
        -Drust=false \
        -Ducx_path=/usr/local/ucx \
    && ninja -C build install \
    && rm -rf /src/nixl \
    && printf '%s\n' /usr/local/ucx/lib /usr/local/lib64 /usr/local/lib > /etc/ld.so.conf.d/nixl.conf \
    && ldconfig

COPY daemon /src/daemon
RUN cd /src/daemon && cargo build --release --locked --features nixl

FROM rust AS criu
RUN dnf install -y --setopt=install_weak_deps=False \
        libbsd-devel libcap-devel libnet-devel libnl3-devel protobuf-c-devel protobuf-devel \
        protobuf-compiler gnutls-devel nftables-devel libuuid-devel libselinux-devel python3 perl \
    && dnf clean all
COPY upstream.tsv /src/upstream.tsv
COPY patches/criu /src/patches/criu
RUN set -eux; \
    git init -q /src/criu; cd /src/criu; \
    git fetch -q --depth 1 $(awk '$1 == "criu" { print $2, $3 }' /src/upstream.tsv); \
    git checkout -q FETCH_HEAD; \
    git apply /src/patches/criu/*.patch; \
    make -j"$(nproc)" criu; \
    ldd criu/criu | grep -q libselinux

FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION} AS e2e
RUN dnf install -y --setopt=install_weak_deps=False \
        libbsd libcap libnet libnl3 protobuf-c gnutls nftables libuuid libselinux \
        util-linux openssl-libs \
    && dnf clean all
COPY --from=criu /src/criu/criu/criu /opt/tools/criu
COPY --from=plain /src/daemon/target/release/snapshotd /src/daemon/target/release/snapshotd-shim /usr/local/bin/
COPY --from=plain /src/daemon/target/release/examples/stand_in /usr/local/bin/stand-in
RUN set -eux; \
    mkdir -p /opt/tools/lib /opt/tools/criu-plugins; \
    /opt/tools/criu --version; \
    snapshotd --help > /dev/null; \
    snapshotd-shim --help > /dev/null; \
    stand-in --help > /dev/null

FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION} AS runtime
RUN dnf install -y --setopt=install_weak_deps=False \
        rdma-core libibverbs libibverbs-utils librdmacm libibumad infiniband-diags \
        numactl-libs hwloc-libs \
        libaio liburing libstdc++ python3 util-linux procps-ng findutils which \
    && dnf clean all
COPY --from=nixl /usr/local /usr/local
COPY --from=nixl /etc/ld.so.conf.d/nixl.conf /etc/ld.so.conf.d/nixl.conf
COPY --from=nixl /src/daemon/target/release/snapshotd /opt/snapshotd/bin/snapshotd-nixl
COPY --from=plain /src/daemon/target/release/snapshotd /src/daemon/target/release/snapshotd-shim /opt/snapshotd/bin/
COPY experiments/bench/restore-bench.sh experiments/bench/offload-check.sh experiments/criu/noiouring.py /opt/snapshotd/exp/
RUN set -eux; \
    ldconfig; \
    ln -s /opt/snapshotd/bin/snapshotd-nixl /usr/local/bin/snapshotd; \
    chmod 0755 /opt/snapshotd/exp/*; \
    if find /usr/local -name '*.so*' -type f -exec ldd {} + 2>/dev/null | grep 'not found'; then exit 1; fi; \
    snapshotd transfer --help >/dev/null; \
    /opt/snapshotd/bin/snapshotd cuda --help >/dev/null; \
    /opt/snapshotd/bin/snapshotd-shim --help >/dev/null
ENV PATH="/usr/local/ucx/bin:${PATH}"
ENV HOME=/tmp

FROM runtime AS e2e-peers
RUN dnf install -y --setopt=install_weak_deps=False \
        libbsd libcap libnet libnl3 protobuf-c gnutls nftables libuuid libselinux \
        curl openssl \
    && dnf clean all
COPY --from=criu /src/criu/criu/criu /opt/tools/criu
COPY --from=plain /src/daemon/target/release/examples/stand_in /usr/local/bin/stand-in
RUN set -eux; \
    ln -s /opt/snapshotd/bin/snapshotd-shim /usr/local/bin/snapshotd-shim; \
    mkdir -p /opt/tools/lib /opt/tools/criu-plugins; \
    /opt/tools/criu --version; \
    snapshotd node --help | grep -q topology-levels; \
    stand-in --help > /dev/null

FROM runtime
