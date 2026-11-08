# 来源与交付改动

- 上游：https://github.com/T-DT-Algorithm-2026/RM_sdr ，部署起点 `0b8aa69`。
- 本次整理来源：本机 `RM_sdr` 提交 `d134bfe`，2026-11-06。
- `gr-robomaster/` 的两个 C++ 解调块、Python 绑定及构建支持文件从原工程复制，未改动算法。
- `src/radio_recive/` 和 `src/interface/radio_interface/` 从原工程复制，保留包名、消息字段、CRC 和发布逻辑。
- `rx.py` 根据上游红蓝 GNU Radio 流图整理为无图形接收入口，合并重复配置，移除 Qt、IQ 录制和个人绝对路径；仅启用 RX2，启动时校验设备序列号。信道参数保留上游默认值，另提供实验用滤波收窄选项。
- 新增本地安装的构建脚本、进程启动/清理脚本及视觉联调说明。未引入 `info_wave_test.py`、`bin.py`、网页服务或任何发射入口。

根目录 `LICENSE` 为上游 MIT 文本；GNU Radio OOT、原流图和本包 `rx.py` 的 GPL 声明另见各文件及 `gr-robomaster/COPYING`。保留所有已有版权与许可证声明。

`SHA256SUMS` 列出本交付源文件的校验值。在本目录运行 `sha256sum -c SHA256SUMS` 可检查完整性。
