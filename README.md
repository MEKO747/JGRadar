# JGRadar

雷达站视觉检测、定位、跟踪和无线电坐标融合。

无线电已接入本仓库：纯接收源码位于 `radio/RM_Radio`，联合入口为 `tools/start_radio_vision.sh`。构建、真实相机配置、话题接口和联调方式见 [无线电与视觉集成说明](docs/radio_integration.md)。原视觉流程及配置见 [架构说明](ARCHITECTURE.md)。

```bash
bash tools/build_radio.sh
bash tools/start_radio_vision.sh red --no-rx --check
```

运行完整视觉融合需要实际相机标定和模型；接收器需 GNU Radio 与 Jazzy。详见集成说明。当前无线电采用 T-DT 2026 协议，2027 新机制尚待适配。
