#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"

unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
set -u

mapfile -t camera_packages < <(
  colcon list | awk '$2 ~ /^src\/camera_ws\/(depthai-ros|realsense-ros)(\/|$)/ {print $1}'
)

if ((${#camera_packages[@]} == 0)); then
  echo "No DepthAI or RealSense packages found under src/camera_ws." >&2
  exit 1
fi

colcon build --symlink-install \
  --packages-skip realsense2_gz_description fairino_hardware "${camera_packages[@]}" \
  "$@"
