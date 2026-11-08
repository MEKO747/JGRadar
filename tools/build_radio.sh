#!/usr/bin/env bash
set -eo pipefail
JG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
radio_root="${RM_RADIO_ROOT:-$JG_ROOT/radio/RM_Radio}"
if [[ "${1:-}" == --ros-only && $# == 1 ]]; then
  source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
  exec colcon --log-base "$radio_root/log" build --base-paths "$radio_root/src" \
    --build-base "$radio_root/build/ros" --install-base "$radio_root/install" \
    --packages-select radio_interface radio_recive --parallel-workers 2 \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
fi
if (($#)); then echo 'Usage: bash tools/build_radio.sh [--ros-only]' >&2; exit 2; fi
exec bash "$radio_root/build.sh"
