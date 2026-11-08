#!/usr/bin/env bash
# Source this file from the delivery's build environment.
RM_RX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -r "$RM_RX_ROOT/install/local_setup.bash" ]]; then
  echo 'RM_Radio has not been built; run bash build.sh first.' >&2
  return 1
fi
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash" || return
source "$RM_RX_ROOT/install/local_setup.bash" || return
export PYTHONPATH="$RM_RX_ROOT/install/oot/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RM_RX_ROOT/install/oot/lib:$RM_RX_ROOT/install/oot/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
