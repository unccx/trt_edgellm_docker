#!/usr/bin/env bash
set -euo pipefail

home_dir="${HOME:-}"
if [ -z "${home_dir}" ]; then
    echo "HOME is empty; cannot prepare host mount directories"
    exit 1
fi

required_dirs=(
    ".cache/pip"
    ".cache/uv"
    ".ccache"
    ".vscode-server"
    ".cursor-server"
)

for rel_dir in "${required_dirs[@]}"; do
    abs_dir="${home_dir}/${rel_dir}"
    if [ -d "${abs_dir}" ]; then
        echo "Host mount dir exists: ${abs_dir}"
        continue
    fi
    mkdir -p "${abs_dir}"
    echo "Created host mount dir: ${abs_dir}"
done
