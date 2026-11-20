# JGRadar

雷达站视觉检测、定位、跟踪和无线电坐标融合。

无线电接收使用 `radio/radar_parse_em_wave/` 中的 JGRadio 赛场接收包。联合入口为 `tools/start_radio_vision.sh`。详见 [无线电与视觉集成说明](docs/radio_integration.md) 和 [视觉架构说明](ARCHITECTURE.md)。调参不靠猜：用 [离线回放与参数扫描](docs/radio_tuning.md) 在真实流图上对同一段 IQ 反复跑。
0x0305 该发什么、能拿多少 P：见 [被标记进度与过期策略](docs/marking_progress.md)。
真机上手先看 [真机链路验证清单](docs/hardware_checklist.md)：每步命令、期望输出、通过判据和失败先查什么。

```bash
bash tools/build_radio.sh
bash tools/start_radio_vision.sh red --no-rx --check
```

完整运行需要 GNU Radio、ROS 2、实际相机标定和模型。接收包使用 RMUC 2026 协议。
