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

## 上行：把雷达坐标发给选手端（0x0305）

雷达站算出的位置要回给己方选手端的小地图，走的是 `0x0305`（48 字节，
雷达→服务器→己方所有选手端，**频率上限 5 Hz**）。2027 协议里每方只有
**重装(1)、3 号步兵(3)、4 号步兵(4)、哨兵(7)** 四个有效位，工程和空中是
保留位；`x = y = 0` 表示「未发送此机器人坐标」，不是「它在原点」。

发送端在 `jg_radar/integration/referee_tx.py`，接在融合主循环里：

```zsh
# 只统计不发送（没有串口设备时先这么验证节拍）
bash tools/start_radio_vision.sh red --radio-only --referee-tx

# 真发：接裁判系统串口
bash tools/start_radio_vision.sh red --radio-only --referee-tx --referee-serial /dev/ttyUSB0
```

要点：

- 默认 **4.7 Hz**（`referee_tx.DEFAULT_RATE_HZ`），`--referee-rate-hz` 可以写大，
  但代码会夹到 5 Hz 并在日志里说明。为什么不是 4.8：节拍按相位对齐后，4.8 Hz 下
  某些 1 秒窗口正好落满 5 帧，压在上限线上零余量；4.7 Hz 下最坏窗口仍是 5 帧
  （6 帧要求 5 个周期 ≤1 s，即 ≥5 Hz，结构上做不到），平均留出约 6% 余量，
  代价是累积到阈值的时间约 +2%，可以忽略。
- 统计里的 `max_frames_1s` 就是「最近 1 秒窗口最大帧数」，现场直接看它有没有贴线。
- `--referee-stale-policy drop`（默认）遇到过期坐标写 `(0,0)`；
  `send_stale` 则照发旧值并计入 `sent_stale`。默认选 `drop`：发过期坐标大概率
  被判 `d≥1.6m` 的错误，而错误会重置累计值，比不发更亏。这个结论要用回放里
  整条 P(t) 曲线验证，不能靠直觉。
- `--referee-stale-after` 是「多少秒算过期」，默认 0.5 秒，和仲裁器的
  `stale_mark_s` 对齐。
- `--referee-log out/referee_0305.jsonl` 把每一帧的内容（四个槽位、年龄、
  策略）落盘，用于对账和论文统计。
- 串口打不开时**不会拖垮视觉**：启动时打印 `referee_tx_error` 并退化成只计数；
  运行中拔线则记一次失败后停止尝试（`failed` 置位），统计里看得见。

线格式（`0xA5` 帧头 + CRC8 + CRC16）在 `radio/radar_parse_em_wave/rmuc/referee_frame.py`，
和接收方向共用同一份 CRC 表。统计字段：`frames_sent`、`effective_rate_hz`、
`dropped`（每台因无坐标或过期写成 `(0,0)` 的次数）、`sent_stale`、`last_max_age_s`。

联调验收：连续 10 秒无信号时真实发送 48 帧（4.806 Hz）、每台写 `(0,0)`；
断流 0.6 秒时 `drop` 下该台为 `(0,0)`、`send_stale` 下发出旧值。

## 2027 雷达侧接口（已核对协议原文，**尚未实现**）

除了无线链路的 `0x0A0x`，2027 协议还给雷达站开了三个常规链路接口。字段按
《2027 高校系列赛通信协议 V1.0.0》逐字节抄录，实现前请再对一遍原文。

| 命令 | 长度 | 方向 | 内容 |
| --- | --- | --- | --- |
| `0x020C` | 2B | 服务器→雷达 | 易伤/特殊标识位图 |
| `0x020E` | 8B | 服务器→雷达 | 自主决策信息同步 |
| `0x0121` | 13B | 雷达→（包在 `0x0301` 里） | 雷达自主决策指令 |

**`0x020C`**：bit0 对方重装、bit2 对方 3 号、bit3 对方 4 号、bit5 对方哨兵、
bit6 己方重装、bit8 己方 3 号、bit9 己方 4 号、bit11 己方哨兵，其余保留。
对方的判定是「被标记进度 ≥ 100」，己方是「≥ 50」。

**`0x020E`**：

| 偏移 | 字段 |
| --- | --- |
| byte0 bit0-1 | 雷达拥有触发双倍易伤的机会数（开局 0，至多 2） |
| byte0 bit2 | 对方是否正在被触发双倍易伤 |
| byte0 bit3-4 | **己方加密等级**（即对方干扰波难度，开局 1，最高 3） |
| byte0 bit5 | 当前是否可以修改密钥 |
| byte1 bit0 | 是否存在生效中的装甲加固区域 |
| byte1 bit1-3 | 剩余装甲加固机会数 |
| byte2-3 / byte4-5 / byte6-7 | 加固区域圆心 x、y 与半径，单位 mm |

`byte0 bit3-4` 就是现有文档里 `(radar_info >> 3) & 0x03` 取得加密等级的那两位——
2027 仍然成立，可以直接接到现有的 `interference_level` 通道。

**`0x0121`** 是 `0x0301`（机器人交互数据）的**子内容**，不是独立命令：
`0x0301` = 子内容 ID(2B) + 发送者 ID(2B) + 接收者 ID(2B) + 内容数据段(≤112B)。
它自己的 13 字节是：

| 偏移 | 字段 |
| --- | --- |
| 0 | 雷达是否确认触发双倍易伤：**单调递增、每次只能 +1** |
| 1-7 | 密钥指令：byte1=1 更新己方密钥；byte1=2 上报破解的对方密钥（10 秒冷却） |
| 8 | 雷达是否确认使用装甲加固机会：同样单调递增每次 +1 |
| 9-10 / 11-12 | 加固区域圆心 x、y，单位 mm |

那两条「单调递增、每次只能 +1」是协议原文的要求：重复发同一个值等于没发，
多加会判不合法。要接进代码就得配一个只增不减的计数器状态机，不能直接在
每帧里写 `1`。
