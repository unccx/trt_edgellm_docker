FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvcr.io/nvidia/pytorch:25.12-py3

ARG USERNAME=qcraft

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        ninja-build \
        gdb \
        clang-format \
        sudo \
        git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN groupadd --gid 1001 ${USERNAME} \
    && useradd --uid 1001 --gid 1001 -m ${USERNAME} -s /bin/bash \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /workspaces/trt_edgellm_docker
