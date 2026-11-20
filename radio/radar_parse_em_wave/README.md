# radar_parse_em_wave

JGRadar 的 ROS 2 无线电接收包。SDR 采样经 GNU Radio 解调，本包校验帧头、CRC 和业务长度，再发布 RoboMaster 业务消息。业务包同时支持 **2026**（超级对抗赛通信协议 V2.0.0）和 **2027**（高校系列赛通信协议 V1.0.0）两套布局，默认按载荷长度自动识别；接入其他赛季时，需按当季协议核对射频参数、机器人槽位和消息字段。

两代的差别集中在长度上：`0x0A04` 从 8 字节变 12 字节（末尾多一个地形跨越状态 `uint32`），`0x0A05` 从 36/41 字节变 47 字节（工程机器人变保留位、末尾多自定义装甲模块加固区域的圆心与半径，单位 mm）。`0x0A01`/`0x0A02`/`0x0A03` 长度不变，但 2027 里工程和空中变成保留位，不再下发。

接收节点用 `--protocol auto|2026|2027` 控制解释口径（默认 `auto`）。长度相同的命令在 `auto` 下按 2027 解释；回放 2026 季的录制时请显式指定 `--protocol 2026`，否则工程和空中两个槽位会被当成保留位丢掉。`tools/radio_probe.py` 有同名参数，并在结论里列出每个业务包属于哪个赛季。

- **无线电原始坐标**订阅 `/rmuc/rx/packet_0x0a01`，类型为 `radar_parse_em_wave/msg/RadarParseEmWave0A01RobotPosition`，单位为厘米。
- **JGRadar 融合坐标**订阅 `/jg_radar/fused`，类型为 `std_msgs/msg/String`。JSON 中的坐标在 `robots.<机器人ID>.position_cm`，`source` 表示该帧采用的来源。

## 克隆后构建

克隆 JGRadar 后，可以在仓库根目录直接构建。构建会从 `msg/` 生成 ROS 2 消息类型，从 `grc/` 生成 GNU Radio 接收流图，并安装接收节点。构建成功后，日常使用直接执行启动命令。更新无线电源码或重新建立环境后再构建。

已配置 JGRadar Dev Container 的工作区使用 Ubuntu 24.04、ROS 2 Jazzy 和 GNU Radio 3.10。首次在 VS Code 里打开容器时，初始化脚本会自动构建。也可以按下文的「原生 ROS 2 构建」在克隆目录里构建。需要在容器里手动重建时执行：

```bash
cd /workspaces/JGRadar
bash tools/build_radio.sh
```

## 随 JGRadar 启动无线电

### 检查板卡

将天线接到板卡的第一路 RX（Fish Ball SDR 标为 `RX1A`），在容器里查询 SDR：

```bash
iio_info -u ip:192.168.2.1
```

查询结果应包含 `ad9361-phy` 和可缓冲接收设备，例如 `cf-ad9361-lpc`。板卡使用其他 IP 时，按实际地址修改。

### 启动接收

**容器终端 1** 执行以下启动命令。示例使用己方红方、干扰 3 级和默认板卡地址，实际使用时按现场配置修改。

```bash
cd /workspaces/JGRadar
export ROS_DOMAIN_ID=184 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
bash tools/start_radio_vision.sh red --radio-only --interference-level 3 --sdr-uri ip:192.168.2.1
```

`--radio-only` 跳过相机，保留接收、解包和融合输出。完整视觉启动及模拟收包步骤见 [JGRadar 联调说明](../../docs/radio_integration.md)。

### 订阅无线电坐标

**容器终端 2** 先加载相同的 ROS 2 环境。`184` 是本机联调域号示例，两个终端使用同一域号。跨机器部署时，按实际网络设置 DDS 发现范围并同步系统时钟。

```bash
source /opt/ros/jazzy/setup.bash
source "$JG_RADAR_RADIO_BUILD_DIR/install/local_setup.bash"
export ROS_DOMAIN_ID=184 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

查看一帧坐标：

```bash
ros2 topic echo --once /rmuc/rx/packet_0x0a01
```

持续观察时运行 `ros2 topic hz /rmuc/rx/packet_0x0a01`。话题出现收包频率、终端 1 中 `radio_frames` 增加，表示位置包已解码并交给融合。`active_sources` 中的 `radio` 表示该槽位当前采用无线电坐标。

## ROS 2 接口

接收节点的话题使用 **Reliable / Volatile / Keep Last(10)**。每个通过校验的业务帧发布一条消息。完整定义见 [msg](msg/)，下表中的类型均位于 `radar_parse_em_wave/msg`。

| 方向（相对接收节点） | 话题 | 消息类型 | 内容 |
| --- | --- | --- | --- |
| 订阅 | `/rmuc/demod_config` | `RadarParseEmWaveDemodConfig` | 己方阵营和干扰等级 |
| 发布 | `/rmuc/rx/packet_0x0a01` | `RadarParseEmWave0A01RobotPosition` | 对方机器人二维坐标 |
| 发布 | `/rmuc/rx/packet_0x0a02` | `RadarParseEmWave0A02RobotHp` | 对方机器人血量 |
| 发布 | `/rmuc/rx/packet_0x0a03` | `RadarParseEmWave0A03RobotAmmo` | 对方机器人弹量 |
| 发布 | `/rmuc/rx/packet_0x0a04` | `RadarParseEmWave0A04FieldStatus` | 对方场地状态和金币 |
| 发布 | `/rmuc/rx/packet_0x0a05` | `RadarParseEmWave0A05RobotBuff` | 对方机器人增益、哨兵姿态和主要状态 |
| 发布 | `/rmuc/rx/packet_0x0a06` | `RadarParseEmWave0A06InterferenceKey` | 干扰波密钥 |

### 位置消息 `0x0A01`

每个角色有两个 `int16` 坐标字段，单位为厘米。`header.stamp` 是接收机解包发布的时间，`header.frame_id` 为业务来源标识 `rmuc_0a01`。坐标对没有单独的有效位，JGRadar 将 `(0, 0)` 和越界坐标当作无效。联调时用已知场地点确认原点、轴向和红蓝坐标约定。

| 字段 | 角色 | 对方红方 ID | 对方蓝方 ID |
| --- | --- | ---: | ---: |
| `hero_x`、`hero_y` | 英雄 | 1 | 101 |
| `engineer_x`、`engineer_y` | 工程 | 2 | 102 |
| `infantry3_x`、`infantry3_y` | 3 号步兵 | 3 | 103 |
| `infantry4_x`、`infantry4_y` | 4 号步兵 | 4 | 104 |
| `aerial_x`、`aerial_y` | 空中 | 6 | 106 |
| `sentry_x`、`sentry_y` | 哨兵 | 7 | 107 |

JGRadar 在己方为红方时将 `hero` 映射到 `B1`，己方为蓝方时映射到 `R1`。其他角色按相同规则映射。阵营或等级变化时，JGRadar 撤回旧无线电观测，再按新配置接收。

### 解调配置

- `team` 填己方 `red` 或 `blue`。己方为红方时，红方基座的官方发射源播发蓝方坐标。
- `interference_level` 填 `1`、`2` 或 `3`。1、2 级发布干扰波密钥，3 级同时发布信息波业务包。使用 2026 裁判系统 `0x020E` 的 `radar_info` 时，用 `(radar_info >> 3) & 0x03` 提取己方加密等级，确认有效后再发送。
- 接收节点启动默认红方 1 级。JGRadar 的固定等级模式每秒发送配置。裁判系统节点负责配置时，给 JGRadar 传入 `--interference-level external`，由外部节点持续发布当前有效配置。

仅运行接收节点时，可在同一 ROS 2 域用以下命令手动配置。例子为己方蓝方 3 级。

```bash
ros2 topic pub --rate 1 /rmuc/demod_config radar_parse_em_wave/msg/RadarParseEmWaveDemodConfig '{team: blue, interference_level: 3}'
```

## 原生 ROS 2 构建

在已安装 ROS 2 Jazzy 的 Ubuntu 24.04 主机上，先补齐构建和接收依赖：

```bash
sudo apt install build-essential cmake gnuradio libiio-utils python3-zmq \
  python3-colcon-common-extensions ros-jazzy-rclpy ros-jazzy-std-msgs \
  ros-jazzy-ament-index-python ros-jazzy-ament-cmake-python \
  ros-jazzy-rosidl-default-generators ros-jazzy-rosidl-default-runtime
```

进入克隆后的 JGRadar 根目录，执行构建：

```bash
bash tools/build_radio.sh
```

原生产物默认位于 `build/radio/`。如果其他 ROS 2 工作空间需要这个包，也可将本目录放入其 `src/`，从对应工作空间运行 `colcon build --packages-select radar_parse_em_wave`。

## 单独启动接收节点

在 JGRadar 根目录加载安装环境。下面的命令会选用当前环境的构建目录。独立 ROS 2 工作空间使用对应的 `install/local_setup.bash`。

```bash
source /opt/ros/jazzy/setup.bash
source "${JG_RADAR_RADIO_BUILD_DIR:-$PWD/build/radio}/install/local_setup.bash"
```

启动接收节点：

```bash
ros2 run radar_parse_em_wave radar_parse_em_wave_node --sdr-uri ip:192.168.2.1
```

单独启动时，使用上文的解调配置命令切到现场阵营和等级。接收节点负责启动流图，流图退出后会自动重启并重新应用解调配置。`--flowgraph-python` 可指定 GNU Radio 的 Python 解释器，默认为 `/usr/bin/python3`。流图和节点使用本机 TCP `53103` 至 `53106` 传输解调比特、功率遥测及控制请求，每个接收环境同时运行一个实例。

ROS 2 launch 使用 `package="radar_parse_em_wave"` 和 `executable="radar_parse_em_wave_node"`。其他 C++ 包可在 `package.xml` 声明 `<depend>radar_parse_em_wave</depend>`，在 CMake 中用 `find_package(radar_parse_em_wave REQUIRED)` 和 `ament_target_dependencies` 引入接口。

## 排查收包

| 现象 | 检查 |
| --- | --- |
| 宿主机能访问 SDR，容器里查询失败 | 从容器执行 `iio_info -u ip:192.168.2.1`，核对板卡地址和 USB 网卡路由。 |
| 启动时出现一次 `Connection refused` | 等待 GNU Radio 的本地 XML-RPC 服务开始响应。若持续出现，查看流图进程的首条报错和是否有其他接收实例占用端口。 |
| 发现位置话题，`ros2 topic hz` 一直无输出 | 确认两个终端使用同一 ROS 域，再检查己方阵营、3 级配置及对应的 RMUC 2026 信息波。需要时单独用 `ros2 topic hz /rmuc/rx/packet_0x0a06` 检查干扰波收包。 |
| 位置话题有数据，融合输出没有无线电来源 | 检查运行日志的 `rejected_radio_frames`、`radio_frames` 和 `active_sources`，核对消息时间戳、角色坐标和场地边界。 |
