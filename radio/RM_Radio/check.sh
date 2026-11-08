#!/usr/bin/env bash
set -eo pipefail
RM_RX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RM_RX_ROOT/env.sh"
cd "$RM_RX_ROOT"
/usr/bin/python3 - <<'PY'
import rx
decoder = rx.load_decoder_module()
from gnuradio import iio, zeromq
from radio_interface.msg import Position, Hp, Fire, State, Buff, Password
print('Decoder:', decoder.__file__)
print('Blocks:', decoder.gfsk_receiver, decoder.gfsk_receiver_noise)
print('ROS messages: Position, Hp, Fire, State, Buff, Password')
PY
ros2 pkg executables radio_recive
/usr/bin/python3 "$RM_RX_ROOT/rx.py" --team red --describe
/usr/bin/python3 "$RM_RX_ROOT/rx.py" --team blue --describe
