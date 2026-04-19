#!/bin/bash
#
# Usage: ./push_to_server.sh <gpu_ip> <username> [local_path] [remote_path]
# Example: ./push_to_server.sh 192.168.1.100 dnx04 saved_models ./saved_models

GPU_IP="${1:?Usage: $0 <gpu_ip> <username> [local_path] [remote_path]}"
USER="${2:?Usage: $0 <gpu_ip> <username> [local_path] [remote_path]}"
LOCAL_PATH="${3:-saved_models}"
REMOTE_PATH="${4:-saved_models}"

echo "Pushing from ${LOCAL_PATH} -> ${USER}@${GPU_IP}:${REMOTE_PATH}"

scp -r "${LOCAL_PATH}" "${USER}@${GPU_IP}:${REMOTE_PATH}"

echo "Done."
