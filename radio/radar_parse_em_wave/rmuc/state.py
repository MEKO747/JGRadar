"""把雷达主程序的解调配置映射为 GNU Radio 接收参数。"""

from dataclasses import dataclass

from .constants import RF_PROFILES, RX_CENTER_FREQ_MAP, Team, WaveType


@dataclass(frozen=True)
class DemodConfig:
    team: Team = Team.RED
    interference_level: int = 1

    def sdr_parameters(self) -> dict[str, int]:
        profiles = RF_PROFILES[self.team]
        interfere = profiles[self.interference_level]
        return {
            "rx_center_freq": RX_CENTER_FREQ_MAP[self.team][self.interference_level],
            "rx_interfere_channel_freq": interfere["freq_hz"],
            "rx_info_channel_freq": profiles[WaveType.INFO]["freq_hz"],
            "rx_int_bw": interfere["bandwidth_hz"],
        }
