#!/usr/bin/env bash
set -eo pipefail
JG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != red && "${1:-}" != blue ]]; then
  echo 'Usage: bash tools/start_radio_vision.sh red|blue [--no-rx] [--sdr-uri URI] [runner options]' >&2
  exit 2
fi
team="$1"
shift
radio_install="$JG_ROOT/build/radio/install"
runner_python="${JG_RADAR_PYTHON:-$JG_ROOT/.venv/bin/python}"
with_rx=1
runner_args=()
rx_args=()
while (($#)); do
  case "$1" in
    --no-rx) with_rx=0; shift ;;
    --sdr-uri|--flowgraph-python)
      if (($# < 2)) || [[ "$2" == --* ]]; then echo "$1 needs a value" >&2; exit 2; fi
      rx_args+=("$1" "$2"); shift 2 ;;
    --sdr-uri=*|--flowgraph-python=*) rx_args+=("$1"); shift ;;
    *) runner_args+=("$1"); shift ;;
  esac
done
if [[ ! -x "$runner_python" ]]; then
  echo "Python not found: $runner_python; set JG_RADAR_PYTHON to your ROS-compatible vision environment." >&2
  exit 1
fi
if [[ ! -f "$radio_install/local_setup.bash" ]]; then
  echo 'Build the receiver first with bash tools/build_radio.sh.' >&2
  exit 1
fi
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source "$radio_install/local_setup.bash"
cd "$JG_ROOT"
"$runner_python" tools/run_radio_vision.py --team "$team" --check "${runner_args[@]}"
# --check itself never starts the receiver or camera.
for arg in "${runner_args[@]}"; do
  if [[ "$arg" == --check ]]; then exit 0; fi
done
if ((with_rx == 0)); then
  exec "$runner_python" -u tools/run_radio_vision.py --team "$team" "${runner_args[@]}"
fi
rx_prefix="$(ros2 pkg prefix radar_parse_em_wave)"
if [[ ! -f "$rx_prefix/share/radar_parse_em_wave/grc/RMUC2026_NanoSDR_RX.py" ]]; then
  echo 'GNU Radio flowgraph missing. Run bash tools/build_radio.sh without --ros-only.' >&2
  exit 1
fi
pids=()
cleanup() {
  trap - EXIT INT TERM
  if ((${#pids[@]})); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$rx_prefix/lib/radar_parse_em_wave/radar_parse_em_wave_node" "${rx_args[@]}" &
pids+=("$!")
"$runner_python" -u tools/run_radio_vision.py --team "$team" "${runner_args[@]}" &
pids+=("$!")
set +e
wait -n "${pids[@]}"
status=$?
set -e
exit "$status"
