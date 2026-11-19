# JGRadio 赛场无线电接收

`radar_parse_em_wave` 是 ROS 2 接收包。Fish Ball SDR 接收官方发射源的信号，GNU Radio 解调后由本包校验、解包并发布 ROS 2 消息。雷达主程序下发己方阵营和干扰等级，订阅无线电坐标，在自己的跟踪与裁判系统链路中完成视觉融合和上报。

本包使用 RoboMaster 2026 超级对抗赛规则 V2.0.1 和高校系列赛通信协议 V2.0.0。接入 2027 赛季前需按当季规则重新核对射频参数与业务字段，尤其是 `0x0A01` 中重装机器人字段和工程、空中机器人的保留位。

## 部署

推荐沿用原工程的 Ubuntu 24.04、ROS 2 Jazzy 和系统 GNU Radio 3.10 环境。将整个 `game/` 目录放到雷达 ROS 2 工作空间的 `src/` 下，例如 `~/radar_ws/src/radar_parse_em_wave/`。工作空间中只放一份同名包。

在已安装 ROS 2 Jazzy 的主机上执行：

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install build-essential cmake gnuradio libiio-utils python3-zmq \
  python3-colcon-common-extensions ros-jazzy-rclpy ros-jazzy-std-msgs \
  ros-jazzy-ament-index-python ros-jazzy-ament-cmake-python \
  ros-jazzy-rosidl-default-generators ros-jazzy-rosidl-default-runtime
cd ~/radar_ws
colcon build --packages-select radar_parse_em_wave
source install/setup.bash
```

完整编译雷达主程序时直接在该工作空间执行 `colcon build`。本包在构建时调用 `grcc` 生成并安装流图。通过 USB 虚拟网卡连接 Fish Ball SDR，将接收天线接到 `RX1A`。按设备实际地址检查 IIO 访问，再启动节点：

```bash
iio_info -u ip:192.168.2.1
ros2 run radar_parse_em_wave radar_parse_em_wave_node
```

默认接收板 URI 为 `ip:192.168.2.1`。自定义地址时添加 `--sdr-uri ip:<接收板IP>`。节点用 `/usr/bin/python3` 自动启动无界面流图，流图异常退出后会重启并重新下发当前配置。GNU Radio 装在其他 Python 环境时，通过 `--flowgraph-python <解释器绝对路径>` 指定。流图和节点同机运行，使用本机 TCP `53103–53106`，分别用于信息波比特、干扰波比特、功率遥测和 XML-RPC。每台接收主机运行一个实例。

主程序通过 ROS 2 launch 启动本包时，使用 `package="radar_parse_em_wave"`、`executable="radar_parse_em_wave_node"`，并按需通过 `arguments=["--sdr-uri", "ip:192.168.2.1"]` 传入板卡地址。

## ROS 2 接口

所有话题均为绝对名，默认 QoS 为 **Reliable / Volatile / Keep Last(10)**。主程序订阅可用 `rclcpp::QoS(10)` 或 rclpy 的队列深度 `10`。两端使用相同的 `ROS_DOMAIN_ID` 并确认 DDS 发现互通。C++ 主程序在 `package.xml` 中添加 `<depend>radar_parse_em_wave</depend>`，在 CMake 中加入 `find_package(radar_parse_em_wave REQUIRED)`，并将它加到主程序目标的 `ament_target_dependencies`。

| 主程序方向 | 话题 | `radar_parse_em_wave/msg/` 下的类型 | 用途 |
| --- | --- | --- | --- |
| 发布 | `/rmuc/demod_config` | `RadarParseEmWaveDemodConfig` | 选择官方广播源阵营和干扰等级 |
| 订阅 | `/rmuc/rx/packet_0x0a01` | `RadarParseEmWave0A01RobotPosition` | 对方机器人二维坐标，融合的主要无线电输入 |
| 订阅 | `/rmuc/rx/packet_0x0a02` | `RadarParseEmWave0A02RobotHp` | 对方机器人血量 |
| 订阅 | `/rmuc/rx/packet_0x0a03` | `RadarParseEmWave0A03RobotAmmo` | 对方机器人弹量 |
| 订阅 | `/rmuc/rx/packet_0x0a04` | `RadarParseEmWave0A04FieldStatus` | 对方场地状态和金币 |
| 订阅 | `/rmuc/rx/packet_0x0a05` | `RadarParseEmWave0A05RobotBuff` | 对方机器人增益、哨兵姿态和主要状态 |
| 订阅 | `/rmuc/rx/packet_0x0a06` | `RadarParseEmWave0A06InterferenceKey` | 对方干扰波密钥，由裁判系统业务链路处理 |

字段以 [msg/](msg/) 为准。业务消息按实际收包逐条发布，CRC 或业务长度校验失败的帧会丢弃。节点启动前发布的配置不会被补发，节点启动时默认红方 1 级。

### 主程序下发解调配置

- `team` 填 **己方阵营** `red` 或 `blue`。例如本方是红方，红方基座上的官方发射源播发的是蓝方坐标，所以应填 `red`。
- `interference_level` 填 1、2 或 3。使用 2026 裁判系统 `0x020E` 的 `radar_info` 时，用 `(radar_info >> 3) & 0x03` 取己方加密等级。只在阵营、等级已知且有效时发布配置。
- 主程序在开局与配置变化时发送，并以约 1 Hz 重发当前配置，使无线电节点重启后能重新收到。当前实现 1、2 级提供干扰波密钥，3 级开放信息波业务包和干扰波密钥。主程序应在阵营、等级变更时清除缓存中的无线电位置，等待重新收到有效位置。

### 视觉融合

1. 订阅 `0x0A01`，按角色和对方阵营关联视觉轨迹。坐标字段用 `int16` 厘米表示。角色前缀和裁判系统机器人 ID 见下表。主程序内部用米时，每轴乘以 `0.01`。
2. 统一到视觉使用的赛场坐标系。先用已知场地点核对原点、轴向、比例和红蓝镜像约定，再配置所需坐标变换。`header.frame_id` 固定为 `rmuc_0a01`，是业务来源标识；节点无需发布 TF。每对坐标没有单独的有效位。结合参赛角色、场地边界和已确认的空值约定校验坐标，谨慎处理 `(0, 0)`。
3. `header.stamp` 是本机解包发布时刻。按接收时间做新鲜度检查，跨机部署时同步时钟。可从 **300 ms** 的无线电有效期起步，根据实测延迟及断包间隔调整。记录每个角色各自最后一次通过校验的无线电观测；旧消息或异常位置不得刷新有效期。
4. 视觉和无线电都新鲜时，按同一机器人 ID 做时间与距离门控，将通过门控的无线电观测送入现有跟踪器。视觉遮挡时可用仍有效的无线电坐标补位。无线电断流、降到 1/2 级或超过有效期时，回到视觉与现有跟踪失效策略。雷达主程序用融合后的结果接续现有裁判系统上报，按当季协议填写 `0x0305`。2026 协议使用厘米，双轴均填 0 表示未发送该机器人的坐标。

| `0x0A01` 字段 | 角色 | 对方为红方 ID | 对方为蓝方 ID |
| --- | --- | ---: | ---: |
| `hero_x / hero_y` | 英雄 | 1 | 101 |
| `engineer_x / engineer_y` | 工程 | 2 | 102 |
| `infantry3_x / infantry3_y` | 3 号步兵 | 3 | 103 |
| `infantry4_x / infantry4_y` | 4 号步兵 | 4 | 104 |
| `aerial_x / aerial_y` | 空中 | 6 | 106 |
| `sentry_x / sentry_y` | 哨兵 | 7 | 107 |

## 联调顺序

每个终端先加载 ROS 2 与工作空间的 `setup.bash`。接好 RX 板并启动节点后，确认主程序和无线电节点都出现在话题图里。使用官方信号时，在裁判系统报告为 3 级的情况下观察位置流：

```bash
ros2 topic info -v /rmuc/demod_config
ros2 topic info -v /rmuc/rx/packet_0x0a01
ros2 topic echo --once /rmuc/rx/packet_0x0a01
ros2 topic hz /rmuc/rx/packet_0x0a01
```

主程序暂未启动时，可按现场真实阵营和等级手动下发一次。以下以己方蓝色、3 级为例：

```bash
ros2 topic pub --once /rmuc/demod_config radar_parse_em_wave/msg/RadarParseEmWaveDemodConfig \
  '{team: blue, interference_level: 3}'
```

联调时在双方确认的位置放置一台已知 ID 的对方机器人，核对无线电与视觉轨迹的 ID、X/Y 方向和米制数值，再观察运动、视觉遮挡、等级下降或无线电断流时的融合输出。位置话题一直无消息时，先按顺序核对 IIO 连接、节点终端中的 GNU Radio/解调配置日志、实际阵营和等级，以及是否存在对应官方信号。话题已发现表示 ROS 2 链路建立，收到有效业务帧后才会有位置消息。
