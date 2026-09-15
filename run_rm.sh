#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mapfile -t camera_packages < <(
  colcon list | awk '$2 ~ /^src\/camera_ws\/(depthai-ros|realsense-ros)(\/|$)/ {print $1}'
)

if ((${#camera_packages[@]} == 0)); then
  echo "No DepthAI or RealSense packages found under src/camera_ws." >&2
  exit 1
fi

declare -A keep=([realsense2_gz_description]=1 [fairino_hardware]=1)
for package in "${camera_packages[@]}"; do
  keep["$package"]=1
done

clean_package_artifacts() {
  local dir=$1

  find "$dir" -mindepth 1 -maxdepth 1 -printf '%f\0' | while IFS= read -r -d '' name; do
    if [[ ${keep[$name]+_} ]]; then
      echo "keep $dir/$name"
    else
      rm -rf -- "$dir/$name"
    fi
  done
}

for dir in build install; do
  [ -d "$dir" ] || continue
  clean_package_artifacts "$dir"
done

for dir in log/*build; do
  [ -d "$dir" ] || continue
  clean_package_artifacts "$dir"
done
