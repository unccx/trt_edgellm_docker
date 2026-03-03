ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

ARG USERNAME=qcraft
ARG USER_UID=1001
ARG USER_GID=1001

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        cmake \
        build-essential \
        ninja-build \
        gdb \
        clang-format \
        sudo \
        git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

RUN USER_UID="${USER_UID:-1001}" \
    && USER_GID="${USER_GID:-1001}" \
    && if ! getent group "${USER_GID}" >/dev/null; then groupadd --gid "${USER_GID}" "${USERNAME}"; fi \
    && if ! id -u "${USERNAME}" >/dev/null 2>&1; then useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" -s /bin/bash; fi \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /workspaces/trt_edgellm_docker
