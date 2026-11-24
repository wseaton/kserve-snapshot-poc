ARG BASE_IMAGE=vllm/vllm-openai:nightly-f2f9edcac30d128bc090a3c032e4ba125716a641@sha256:eaa63fd6ee3aac77f3158e80ed79ec425db3a53f7cd666bf1954e96abfaf19a7

FROM docker.io/library/rust:1.95-bookworm@sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1 AS frontend
ARG VLLM_REPO=wseaton/vllm
ARG FRONTEND_REF=b712f6f4aaaee90b39a185c2339d17b1221f798d
RUN set -eux; \
    git init -q /src/vllm; \
    cd /src/vllm; \
    git fetch -q --depth 1 "https://github.com/${VLLM_REPO}.git" "${FRONTEND_REF}"; \
    git checkout -q FETCH_HEAD; \
    cd rust; \
    cargo build --release --locked -p vllm-cmd --bin vllm-rs

FROM ${BASE_IMAGE}

ARG VLLM_REPO=wseaton/vllm
ARG VLLM_REF=b712f6f4aaaee90b39a185c2339d17b1221f798d
ARG VLLM_FILES="vllm/config/model.py vllm/device_allocator/__init__.py vllm/device_allocator/cumem.py vllm/device_allocator/sleep_mode_backend.py vllm/device_allocator/xpumem.py vllm/distributed/device_communicators/cuda_communicator.py vllm/distributed/device_communicators/pynccl.py vllm/distributed/kv_transfer/kv_connector/v1/base.py vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py vllm/engine/arg_utils.py vllm/v1/engine/core.py vllm/v1/engine/llm_engine.py vllm/v1/kv_offload/base.py vllm/v1/kv_offload/cpu/gpu_worker.py vllm/v1/kv_offload/cpu/shared_offload_region.py vllm/v1/kv_offload/cpu/spec.py vllm/v1/kv_offload/tiering/spec.py vllm/v1/worker/gpu/model_runner.py vllm/v1/worker/gpu_worker.py"
ARG MX_REPO=wseaton/modelexpress
ARG MX_REF=16ecf7a627d2a117f0c4f97c7a549923d6719529

RUN set -eux; \
    site="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent.parent)')"; \
    for f in ${VLLM_FILES}; do \
        curl -fsSL "https://raw.githubusercontent.com/${VLLM_REPO}/${VLLM_REF}/${f}" -o "${site}/${f}"; \
    done; \
    uv pip freeze --system | grep -E '^[A-Za-z0-9_.-]+==' | grep -v '^vllm==' > /tmp/base-constraints.txt; \
    uv pip install --system --no-cache -c /tmp/base-constraints.txt \
        "modelexpress @ https://github.com/${MX_REPO}/archive/${MX_REF}.tar.gz#subdirectory=modelexpress_client/python"; \
    rm /tmp/base-constraints.txt; \
    python3 -c "import modelexpress, nixl, vllm; from vllm.config import ModelConfig; assert 'checkpoint_release_weights' in ModelConfig.__dataclass_fields__; from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlConnector; assert 'supports_checkpoint' in vars(NixlConnector); from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion; assert hasattr(SharedOffloadRegion, 'release_pages'); from modelexpress import envs; assert envs.MX_POD_INFO_DIR == ''; print(vllm.__version__)"

ARG CUDA_COMPAT_VERSION=13.1
ARG CUDA_KEYRING_URL=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb

RUN set -eux; \
    pkg="cuda-compat-$(echo "${CUDA_COMPAT_VERSION}" | tr . -)"; \
    apt-get update; \
    if ! apt-cache show "${pkg}" >/dev/null 2>&1; then \
        apt-get install -y --no-install-recommends ca-certificates curl; \
        curl -fsSL -o /tmp/cuda-keyring.deb "${CUDA_KEYRING_URL}"; \
        dpkg -i /tmp/cuda-keyring.deb; \
        rm /tmp/cuda-keyring.deb; \
        apt-get update; \
    fi; \
    apt-get install -y --no-install-recommends "${pkg}"; \
    rm -rf /var/lib/apt/lists/*; \
    test -e "/usr/local/cuda-${CUDA_COMPAT_VERSION}/compat/libcuda.so.1"

COPY --from=frontend /src/vllm/rust/target/release/vllm-rs /tmp/vllm-rs
RUN set -eux; \
    target="$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent / "vllm-rs")')"; \
    install -m 0755 /tmp/vllm-rs "${target}"; \
    rm /tmp/vllm-rs; \
    "${target}" --version; \
    grep -q CheckpointPrepare "${target}"
