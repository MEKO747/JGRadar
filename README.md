# JGRadar

雷达站视觉检测、定位、跟踪和无线电坐标融合。

无线电接收使用 `radio/radar_parse_em_wave/` 中的 JGRadio 赛场接收包。联合入口为 `tools/start_radio_vision.sh`。详见 [无线电与视觉集成说明](docs/radio_integration.md) 和 [视觉架构说明](ARCHITECTURE.md)。

```bash
bash tools/build_radio.sh
bash tools/start_radio_vision.sh red --no-rx --check
```

完整运行需要 GNU Radio、ROS 2、实际相机标定和模型。接收包使用 RMUC 2026 协议。
