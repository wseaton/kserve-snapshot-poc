ARG BASE_IMAGE=vllm/vllm-openai:nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121@sha256:bd419f1669b420e7dfcfa2545a9f3b05c2312c858a215b1f287284d7398d3367
FROM ${BASE_IMAGE}

ARG VLLM_REPO=wseaton/vllm
ARG VLLM_REF=7fd5e5ba729af2b67ce140aea9221c6797de57e6
ARG VLLM_FILES="vllm/config/model.py vllm/engine/arg_utils.py vllm/model_executor/model_loader/__init__.py vllm/model_executor/model_loader/base_loader.py vllm/model_executor/model_loader/modelexpress_loader.py vllm/model_executor/model_loader/reload/inplace.py vllm/v1/engine/core.py vllm/v1/worker/gpu/model_runner.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/gpu_worker.py"
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
    python3 -c "import modelexpress, nixl, vllm; from vllm.config import ModelConfig; assert 'checkpoint_release_weights' in ModelConfig.__dataclass_fields__; from modelexpress import envs; assert envs.MX_POD_INFO_DIR == ''; print(vllm.__version__)"

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
