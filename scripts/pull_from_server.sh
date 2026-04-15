#!/bin/bash
#
# Usage: ./pull_from_server.sh <gpu_ip> <username> [remote_path] [local_path]
# Example: ./pull_from_server.sh 192.168.1.100 dnx04 saved_models ./saved_models

GPU_IP="${1:?Usage: $0 <gpu_ip> <username> [remote_path] [local_path]}"
USER="${2:?Usage: $0 <gpu_ip> <username> [remote_path] [local_path]}"
REMOTE_PATH="${3:-saved_models}"
LOCAL_PATH="${4:-saved_models}"

echo "Pulling from ${USER}@${GPU_IP}:${REMOTE_PATH} -> ${LOCAL_PATH}"

scp -r "${USER}@${GPU_IP}:${REMOTE_PATH}" "${LOCAL_PATH}"

echo "Done."
