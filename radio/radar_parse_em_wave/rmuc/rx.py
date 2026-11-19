"""读取 GNU Radio 的解调比特流并发布恢复出的业务包。"""

from collections.abc import Callable
import threading

import zmq

from .coder import AirFrameDecoder, BusinessDecoder, SerialFrameDecoder
from .constants import LOCAL_HOST, RX_INFO_DEFAULT_PORT, RX_INTERFERE_DEFAULT_PORT, WaveType
from .state import DemodConfig


class RxWorker:
    def __init__(
        self,
        get_config: Callable[[], DemodConfig],
        publish: Callable[[dict], None],
    ):
        self._get_config = get_config
        self._publish = publish
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rmuc-rx", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        with (
            zmq.Context() as context,
            context.socket(zmq.SUB) as interfere,
            context.socket(zmq.SUB) as info,
        ):
            poller = zmq.Poller()
            sources = {interfere: WaveType.INTERFERE, info: WaveType.INFO}
            for socket, port in (
                (interfere, RX_INTERFERE_DEFAULT_PORT),
                (info, RX_INFO_DEFAULT_PORT),
            ):
                socket.setsockopt(zmq.SUBSCRIBE, b"")
                socket.connect(f"tcp://{LOCAL_HOST}:{port}")
                poller.register(socket, zmq.POLLIN)

            current_config = None
            air = {wave: AirFrameDecoder(wave) for wave in WaveType}
            serial = {wave: SerialFrameDecoder() for wave in WaveType}
            while not self._stop.is_set():
                ready = poller.poll(timeout=50)
                config = self._get_config()
                if config != current_config:
                    for decoder in air.values():
                        decoder.bit_buffer.clear()
                    for decoder in serial.values():
                        decoder.buffer.clear()
                    current_config = config

                for socket, _ in ready:
                    wave = sources[socket]
                    bits = socket.recv()
                    if config is None or (wave == WaveType.INFO and config.interference_level != 3):
                        continue

                    for payload in air[wave].feed_unpacked_bytes(bits):
                        for frame in serial[wave].feed(payload):
                            decoded = BusinessDecoder.decode(frame)
                            if decoded is not None and config == self._get_config():
                                self._publish(decoded)
