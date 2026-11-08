#!/usr/bin/env bash
set -eo pipefail
JG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" != red && "${1:-}" != blue ]]; then
  echo 'Usage: bash tools/start_radio_vision.sh red|blue [--no-rx] [--noise-cutoff Hz] [runner options]' >&2
  exit 2
fi
team="$1"
shift
radio_root="${RM_RADIO_ROOT:-$JG_ROOT/radio/RM_Radio}"
runner_python="${JG_RADAR_PYTHON:-$JG_ROOT/.venv/bin/python}"
with_rx=1
runner_args=()
rx_args=()
while (($#)); do
  case "$1" in
    --no-rx) with_rx=0; shift ;;
    --noise-cutoff)
      if (($# < 2)); then echo '--noise-cutoff needs Hz' >&2; exit 2; fi
      rx_args+=("--noise-cutoff" "$2"); shift 2 ;;
    *) runner_args+=("$1"); shift ;;
  esac
done
if [[ ! -x "$runner_python" ]]; then
  echo "Python not found: $runner_python; set JG_RADAR_PYTHON to your Python 3.12 vision environment." >&2
  exit 1
fi
source "$radio_root/env.sh"
cd "$JG_ROOT"
"$runner_python" tools/run_radio_vision.py --team "$team" --check "${runner_args[@]}"
# --check itself never starts the receiver or camera.
for arg in "${runner_args[@]}"; do
  if [[ "$arg" == --check ]]; then exit 0; fi
done
if ((with_rx == 0)); then
  exec "$runner_python" -u tools/run_radio_vision.py --team "$team" "${runner_args[@]}"
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
bash "$radio_root/start.sh" "$team" "${rx_args[@]}" &
pids+=("$!")
"$runner_python" -u tools/run_radio_vision.py --team "$team" "${runner_args[@]}" &
pids+=("$!")
set +e
wait -n "${pids[@]}"
status=$?
set -e
exit "$status"
