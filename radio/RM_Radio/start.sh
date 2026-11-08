#!/usr/bin/env bash
set -eo pipefail
RM_RX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" != red && "${1:-}" != blue ]]; then
  echo 'Usage: bash start.sh red|blue [rx.py options, e.g. --noise-cutoff 270000]' >&2
  exit 2
fi
team="$1"
shift
source "$RM_RX_ROOT/env.sh"
node="$RM_RX_ROOT/install/radio_recive/lib/radio_recive/radio_recive_node"
if [[ ! -x "$node" ]]; then
  echo 'ROS receiver is missing; run bash build.sh first.' >&2
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
"$node" &
pids+=("$!")
/usr/bin/python3 -u "$RM_RX_ROOT/rx.py" --team "$team" "$@" &
pids+=("$!")
set +e
wait -n "${pids[@]}"
status=$?
set -e
echo "A receive process exited (status=$status); stopping this delivery's remaining process." >&2
if ((status == 0)); then status=1; fi
exit "$status"
