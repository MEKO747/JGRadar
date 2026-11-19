#!/usr/bin/env bash
set -eo pipefail
JG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cmake_args=(-DRADAR_BUILD_FLOWGRAPH=ON)
if [[ $# == 1 && "$1" == --ros-only ]]; then
  cmake_args=(-DRADAR_BUILD_FLOWGRAPH=OFF)
elif (($#)); then
  echo 'Usage: bash tools/build_radio.sh [--ros-only]' >&2
  exit 2
fi
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
radio_build="$JG_ROOT/build/radio"
exec colcon --log-base "$radio_build/log" build \
  --base-paths "$JG_ROOT/radio/radar_parse_em_wave" \
  --build-base "$radio_build/build" --install-base "$radio_build/install" \
  --packages-select radar_parse_em_wave --cmake-args "${cmake_args[@]}"
