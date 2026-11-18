ARG FEDORA_VERSION=42

FROM docker.io/library/rust:1-bookworm@sha256:ade39a710ef85f3506cffcfec3e8a24ba090badef1dff2fb798f71d878833fea AS plain
COPY daemon /src/daemon
RUN cd /src/daemon && cargo build --release --locked

FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION} AS nixl
RUN dnf install -y --setopt=install_weak_deps=False \
        gcc gcc-c++ make cmake meson ninja-build pkgconf-pkg-config git-core \
        autoconf automake libtool rdma-core rdma-core-devel \
        numactl-devel hwloc-devel libaio-devel liburing-devel libstdc++-devel \
        clang clang-devel llvm-devel python3 python3-devel pybind11-devel \
        curl ca-certificates findutils which xz \
    && dnf clean all

ARG RUST_VERSION=1.100.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}
ENV PATH="/root/.cargo/bin:${PATH}"

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

FROM registry.fedoraproject.org/fedora:${FEDORA_VERSION}
RUN dnf install -y --setopt=install_weak_deps=False \
        rdma-core libibverbs libibverbs-utils librdmacm numactl-libs hwloc-libs \
        libaio liburing libstdc++ python3 util-linux procps-ng findutils which \
    && dnf clean all
COPY --from=nixl /usr/local /usr/local
COPY --from=nixl /etc/ld.so.conf.d/nixl.conf /etc/ld.so.conf.d/nixl.conf
COPY --from=nixl /src/daemon/target/release/snapshotd /opt/snapshotd/bin/snapshotd-nixl
COPY --from=plain /src/daemon/target/release/snapshotd /opt/snapshotd/bin/snapshotd
COPY experiments/bench/restore-bench.sh experiments/bench/offload-check.sh experiments/criu/noiouring.py /opt/snapshotd/exp/
RUN set -eux; \
    ldconfig; \
    ln -s /opt/snapshotd/bin/snapshotd-nixl /usr/local/bin/snapshotd; \
    chmod 0755 /opt/snapshotd/exp/*; \
    if find /usr/local -name '*.so*' -type f -exec ldd {} + 2>/dev/null | grep 'not found'; then exit 1; fi; \
    snapshotd transfer --help >/dev/null; \
    /opt/snapshotd/bin/snapshotd cuda --help >/dev/null
ENV PATH="/usr/local/ucx/bin:${PATH}"
ENV HOME=/tmp
