# JGRadio 与 JGRadar 联合运行

JGRadio `game` 的 ROS 2 接收包位于 `radio/radar_parse_em_wave/`，包名为 `radar_parse_em_wave`。Fish Ball SDR 接收官方信号，GNU Radio 解调，接收节点校验并发布 ROS 2 消息。JGRadar 将位置观测写入现有 Stage B 缓冲区，由原有仲裁逻辑和视觉跟踪结果共同确定 `/jg_radar/fused` 的输出。

## 环境与构建

目标环境为 Ubuntu 24.04、ROS 2 Jazzy、Python 3.12 和系统 GNU Radio。依赖列表见 [接收包说明](../radio/radar_parse_em_wave/README.md)。通过 `JG_RADAR_PYTHON` 指定能导入 `rclpy`、接收包消息和视觉依赖的 Python，默认用仓库内的 `.venv/bin/python`。ROS 2 发行版默认用 Jazzy，也可以先加载已安装 ROS 2 的 `setup.bash` 或设置 `ROS_DISTRO`。

```bash
bash tools/build_radio.sh
bash tools/start_radio_vision.sh red --no-rx --check
```

构建产物位于 `build/radio/`。使用 `bash tools/build_radio.sh --ros-only` 可构建 ROS 2 接口和 Python 包，适合软件联调。启动本地 SDR 接收需要完整构建生成的 GNU Radio 流图。`--check` 检查配置和消息类型，无需连接设备。

## 联合启动

首个参数是己方阵营。下例用 3 级接收位置，并将实测相机标定 YAML 放到配置列表最后。

```bash
bash tools/start_radio_vision.sh red --interference-level 3 \
  --config config/camera.yaml --config config/field.yaml \
  --config config/vision_check.yaml \
  --config /absolute/path/to/your_calibrated_camera.yaml
```

接收板默认使用 `ip:192.168.2.1`，可通过 `--sdr-uri` 指定地址。接收节点负责启动和重启 GNU Radio 流图。增加 `--no-rx` 可连接单独启动的接收节点，增加 `--radio-only` 可跳过相机。跨机器联调需要相同的 `ROS_DOMAIN_ID`、可达的 DDS 网络和同步的系统时钟。

JGRadar 每秒在 `/rmuc/demod_config` 发布已选的阵营和等级。默认 1 级，1/2 级接收干扰波密钥，3 级同时接收位置等信息波数据。现场若由外部裁判系统节点确定等级，使用 `--interference-level external`。此模式下外部节点应持续下发有效阵营和等级，融合入口收到配置后开始采纳无线电消息。阵营或等级改变时立即撤回之前的无线电观测。

```bash
bash tools/start_radio_vision.sh blue --radio-only --interference-level 3
bash tools/start_radio_vision.sh blue --no-rx --radio-only --interference-level external
```

`--seconds 10` 可限制运行时间，`--out out/radio_vision.jsonl` 可保存融合输出。

## 诊断：卡在帧同步还是压根没信号

`packet_0x0a0x` 话题没有数据时，两种原因的处理方式完全不同：空口没有信号，
和信号有但解不出来。`tools/radio_probe.py` 用接收包自己的解码器逐级计数，
把这两种情况分开。

```bash
source ~/.jg_env.zsh
python3 tools/radio_probe.py                # 旁听 10 秒，默认
python3 tools/radio_probe.py --seconds 30
python3 tools/radio_probe.py --mode watch   # 打印解出的业务包内容
```

`bits` 模式（默认）连接 GNU Radio 推出的两路比特流——53103 信息波、53104
干扰波——按 `比特 → 空口帧 → 序列帧 → 业务包` 四级统计，然后对每一路给出
一句结论：

| 现象 | 结论 |
| --- | --- |
| 没有任何比特 | 流图没跑，或端口不通 |
| 有比特、空口帧为 0 | 这一路是噪声，空口没有可用信号 |
| 空口帧偶发（< 1 次/秒） | 信号很弱或误码过高，撑不完一帧 |
| 有空口帧、序列帧为 0 | 帧头、长度或 CRC8 不过 |
| 有序列帧、业务包为 0 | CRC16 过了但命令字/长度与当前协议不符 |

它不启动接收节点、不占设备，可以和正在运行的实例同时用。注意空口帧的
Access Code 允许 5 bit 误码，所以「帧同步」这一级本身偏宽松；判断信号
好坏要看后两级的通过率。

## 调参：离线回放与参数扫描

`tools/iq_synth.py` 合成或 `tools/radio_replay.py --record` 录制一段 IQ，
`tools/radio_replay.py` 用**签入流图本身**（只换源块）回放，`tools/radio_sweep.py`
在同一段信号上扫 `loop_bw` / `bt`，输出 CSV、曲线和报告。原理、指标口径和
基线结论见 [解调参数调优](radio_tuning.md)。

## ROS 2 消息和融合坐标

| 接收话题 | 内容 | JGRadar 用途 |
| --- | --- | --- |
| `/rmuc/rx/packet_0x0a01` | 机器人位置 | 进入现有仲裁缓冲 |
| `/rmuc/rx/packet_0x0a02` | 血量 | `hp` 状态 |
| `/rmuc/rx/packet_0x0a03` | 弹量 | `fire` 状态 |
| `/rmuc/rx/packet_0x0a04` | 场地状态 | `state` 状态 |
| `/rmuc/rx/packet_0x0a05` | 机器人增益 | `buff` 状态 |
| `/rmuc/rx/packet_0x0a06` | 干扰波密钥 | `password` 收包状态，密钥值由该强类型话题提供 |

`hero`、`engineer`、`infantry3`、`infantry4`、`aerial`、`sentry` 依次映射为对方的 1、2、3、4、6、7 号槽位。2027 通信协议里只有**重装、3 号步兵、4 号步兵、哨兵**四处有效，工程和空中是保留位，接收端不会发布这两个槽位的坐标。使用厘米，按 2026 场地边界 2800 × 1500 校验。`(0,0)` 或越界值会撤回对应槽位的无线电坐标。默认按红方场地坐标解释。现场若确认源数据以己方视角为基准，使用 `--radio-frame own_team`，蓝方输入会绕场地中心旋转 180°。联调时用已知场地点确认坐标轴方向与原点。

消息 `header.stamp` 记录接收节点解包发布时刻。桥接将它换算到本机单调时钟，拒绝超过 0.5 秒的消息或明显超前的时钟。`--input-latency` 可填实测的空口到发布延迟，默认为 0。断流时，原有 `fuse` 配置和位置误差约束决定无线电数据何时退出仲裁。

`/jg_radar/fused` 和 `/jg_radar/radio_status` 都是 `std_msgs/String` JSON。融合输出沿用 `FrameOutput`。状态输出按消息分别计算 `age_s` 和 `valid`，有效期为 1 秒。状态字段以新接收包的 ROS 消息为准，密钥状态只包含时效信息。
