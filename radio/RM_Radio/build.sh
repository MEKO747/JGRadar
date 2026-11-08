#!/usr/bin/env bash
set -eo pipefail
RM_RX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
cmake -S "$RM_RX_ROOT/gr-robomaster" -B "$RM_RX_ROOT/build/oot" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$RM_RX_ROOT/install/oot" \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DGR_PYTHON_DIR=lib/python3/dist-packages
cmake --build "$RM_RX_ROOT/build/oot" --parallel 2
cmake --install "$RM_RX_ROOT/build/oot"
colcon --log-base "$RM_RX_ROOT/log" build --base-paths "$RM_RX_ROOT/src" \
  --build-base "$RM_RX_ROOT/build/ros" --install-base "$RM_RX_ROOT/install" \
  --packages-select radio_interface radio_recive --parallel-workers 2
bash "$RM_RX_ROOT/check.sh"
