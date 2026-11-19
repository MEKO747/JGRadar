"""同步接收参数，并按流图功率遥测调整 SDR 增益。"""

from collections.abc import Callable
from http.client import HTTPConnection
import math
import struct
import threading
import time
from xmlrpc.client import ServerProxy, Transport

import zmq

from .constants import LOCAL_HOST, RX_POWER_TELEMETRY_DEFAULT_PORT, XMLRPC_DEFAULT_PORT
from .state import DemodConfig


class _LocalTransport(Transport):
    def make_connection(self, host: str) -> HTTPConnection:
        connection = super().make_connection(host)
        connection.timeout = 1.0
        return connection


class SdrParameterController:
    def __init__(
        self,
        get_config: Callable[[], DemodConfig],
        warn: Callable[[str], None],
    ):
        self._get_config = get_config
        self._warn = warn
        self._generation = 0
        self._applied: tuple[int, DemodConfig] | None = None
        self._gain = 40
        self._next_gain_update = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rmuc-sdr-control", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def reset_flowgraph(self) -> None:
        """让新启动的流图重新接收整组参数。"""
        self._generation += 1

    def active_config(self) -> DemodConfig | None:
        config = self._get_config()
        return config if self._applied == (self._generation, config) else None

    def _run(self) -> None:
        endpoint = f"http://{LOCAL_HOST}:{XMLRPC_DEFAULT_PORT}"
        telemetry_endpoint = f"tcp://{LOCAL_HOST}:{RX_POWER_TELEMETRY_DEFAULT_PORT}"
        next_config_attempt = 0.0

        with (
            ServerProxy(endpoint, transport=_LocalTransport()) as proxy,
            zmq.Context() as context,
            context.socket(zmq.SUB) as telemetry,
        ):
            telemetry.setsockopt(zmq.SUBSCRIBE, b"")
            telemetry.setsockopt(zmq.CONFLATE, 1)
            telemetry.connect(telemetry_endpoint)
            while not self._stop.is_set():
                generation, config = desired = (self._generation, self._get_config())
                now = time.monotonic()
                if desired != self._applied and now >= next_config_attempt:
                    next_config_attempt = now + 0.5
                    restore_gain = self._applied is None or self._applied[0] != generation
                    self._applied = None
                    try:
                        self._apply_config(proxy, config, restore_gain)
                    except Exception as exc:
                        self._warn(f"SDR 解调配置同步失败，将重试：{exc}")
                    else:
                        self._applied = desired

                if telemetry.poll(timeout=50):
                    payload = telemetry.recv()
                    if self.active_config() is not None:
                        self._update_gain(proxy, payload)

    def _apply_config(self, proxy: ServerProxy, config: DemodConfig, restore_gain: bool) -> None:
        for name, value in config.sdr_parameters().items():
            if self._stop.is_set():
                return
            getattr(proxy, f"set_{name}")(value)
        if restore_gain and not self._stop.is_set():
            proxy.set_rx_gain(self._gain)

    def _update_gain(self, proxy: ServerProxy, payload: bytes) -> None:
        now = time.monotonic()
        if now < self._next_gain_update or len(payload) != 12:
            return

        powers_db = struct.unpack("<3f", payload)
        if not all(math.isfinite(value) for value in powers_db):
            return

        peak_db = max(powers_db)
        if -20.0 <= peak_db <= -10.0:
            return

        target_gain = max(0, min(60, self._gain + round(-15.0 - peak_db)))
        if target_gain == self._gain:
            return

        self._next_gain_update = now + 5.0
        try:
            proxy.set_rx_gain(target_gain)
        except Exception as exc:
            self._warn(f"SDR 增益更新失败：{exc}")
            return
        self._gain = target_gain
