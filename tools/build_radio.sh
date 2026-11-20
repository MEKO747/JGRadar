#!/usr/bin/env bash
set -eo pipefail
JG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# 本机 ~/.local 里装着 numpy 2.5.5，GNU Radio 的 Python 绑定是按 numpy 2.x 之前的 ABI
# 编译的，两者同时在 sys.path 上时 `import gnuradio` 直接抛 "ImportError: initialization
# failed"，表现为 colcon 在 grcc 生成流图那一步失败——而 C++ 部分编得好好的，报错信息
# 完全不指向 numpy，所以这个坑已经踩过不止一次。
#
# 只在没有显式设置时兜底：调用方如果自己设了 PYTHONNOUSERSITE=0 表示他想保留用户 site，
# 这里尊重他的选择，不再覆盖。
if [[ -z "${PYTHONNOUSERSITE:-}" ]]; then
  export PYTHONNOUSERSITE=1
fi

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
