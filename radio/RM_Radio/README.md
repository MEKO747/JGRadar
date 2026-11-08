# RM_Radio：纯 RX 接收与视觉联调交付包

交付目标：HAMGEEK RX2 → T-DT GFSK 解调 → ZeroMQ → CRC / 协议解析 → ROS 2 六类消息。没有 TX、网页服务、录波数据或固定 IQ 文件输出；运行不需要显示器或 Conda。

**协议范围为 T-DT 2026 开源实现，未完成 2027 协议或官方赛场波形验收。**红蓝参数沿用上游样例；上场前应按目标赛季和现场频点核对。当前已验证环境为 Ubuntu 24.04、ROS 2 Jazzy、GNU Radio 3.10.9.2、系统 Python 3.12。

## 目录与入口

```text
rx.py / rx.json       无图形 RX2 接收程序与配置
start.sh              同时启动接收流图和 ROS 2 解析节点
build.sh / env.sh     本地构建与环境加载，不修改系统安装的解调器
check.sh              检查本包的解调模块、消息和配置，不访问硬件
gr-robomaster/        原 C++ 信息波 / 干扰波解调器及构建绑定文件
src/radio_recive/     原协议重组、CRC、ROS 2 发布节点
src/interface/radio_interface/  六类原始 ROS 2 消息定义
```

交付的是源码。拿到包后在目标机器重新构建，不要复制另一台机器的 `build/` 或 `install/`。原始许可证见 `LICENSE`、`gr-robomaster/COPYING` 和各源文件声明，来源及改动见 `NOTICE.md`。

## 安装与构建

目标环境先装好 ROS 2 Jazzy。已有视觉环境如使用其他 ROS 2 发行版，需要统一版本后重新构建；本包没有验证跨发行版联调。

Ubuntu 24.04 / Jazzy 的额外依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake gnuradio gnuradio-dev \
  libiio-dev libiio-utils libzmq3-dev cppzmq-dev pybind11-dev \
  python3-dev python3-numpy python3-colcon-common-extensions \
  ros-jazzy-ament-cmake-auto ros-jazzy-rosidl-default-generators \
  ros-jazzy-rclcpp-components

# 进入本交付目录后执行
bash build.sh
bash check.sh
```

解调模块安装在本包 `install/oot`；ROS 2 包安装在本包 `install`。不需要 `sudo cmake --install`。`check.sh` 应显示来自本包 `install/oot` 的解调模块路径，以及 `radio_recive_node` 可执行程序。

## 硬件与启动

当前配置针对本项目已核对的 HAMGEEK：

- 天线接 **RX2**；仅启用第二接收链。
- RX URI：`ip:192.168.2.1`；电脑需能通过 USB 虚拟网卡或网口访问该地址。
- 预期序列号：`104473023196000bf5ff1a00aae12c3ca8`，启动时核对。不匹配会退出。更换接收板后修改 `rx.json`。
- 手动增益初值 40 dB；增益和 `manual / slow_attack / fast_attack` 模式在 `rx.json` 中配置。

先确认连接和参数：

```bash
iio_attr -u ip:192.168.2.1 -T 5000 -q -C hw_serial
python3 rx.py --team red --describe
```

选择**己方颜色**启动，一次只运行一套：

```bash
export ROS_DOMAIN_ID=0  # 与视觉主程序保持一致
bash start.sh red      # 己方红；己方蓝改为 bash start.sh blue
```

Ctrl+C 会停止本启动器的两个子进程。任一子进程退出时，另一进程也会被关闭；错误会显示在终端，不会静默重启或关闭其他人的进程。不要同时运行旧 RX 程序、网页硬件观测或另一份 `start.sh`，以免争用设备和端口。

仅检查硬件接收连接、不启动 ROS 节点时：

```bash
source env.sh
python3 rx.py --team red --seconds 3
```

此检查能确认流图启动，不代表收到 CRC 正确的业务帧。停止后再运行正式启动器。

## 频点与支路

采样率 2 MS/s，各支路抽取后 1 MS/s、47 SPS。保留原上游两个干扰通道的滤波配置。

| 己方 | RX 本振 | 信息波 | 干扰 1 | 干扰 2 |
| --- | --- | --- | --- | --- |
| 红 | 432.600 MHz | 433.200 MHz | 432.200 MHz | 432.500 MHz |
| 蓝 | 434.520 MHz | 433.920 MHz | 434.920 MHz | 434.620 MHz |

信息波送本机 ZeroMQ 5555，干扰 1 / 2 送 6666 / 6667。ROS 节点也订阅上游可选的 6668，本包没有启用第三条干扰支路。各端口由 ROS 节点分别重组缓存、执行 CRC，再发布消息。

**与本项目单 NanoSDR 双波测试源联调时**，可使用：

```bash
bash start.sh red --noise-cutoff 270000
```

该参数只将干扰支路的前端信道滤波收窄到 270 kHz，用于排除测试 TX 本振泄漏。默认 480 / 420 kHz 仍沿用上游流图；收窄参数针对已知测试波形，不能直接当作官方宽带干扰波配置。

## 给视觉负责人的接口说明

六类消息定义与 T-DT 雷达站 `radio_interface` 对应，包名和字段未改。以下话题在默认命名空间下发布；QoS 为默认 Reliable / Volatile，队列深度 10。

| 命令 | 话题 | 类型（均属于 `radio_interface/msg`） | 主要字段 |
| --- | --- | --- | --- |
| 0x0A01 | `/robot_position` | `Position` | `uint16[6] x, y`，单位 cm |
| 0x0A02 | `/radio_hp` | `Hp` | `uint16[6] hp` |
| 0x0A03 | `/radio_fire` | `Fire` | `uint16[5] fire` |
| 0x0A04 | `/radio_state` | `State` | 金币、场地状态位解析字段 |
| 0x0A05 | `/radio_buff` | `Buff` | 恢复、冷却、防御、攻击、姿态等数组 |
| 0x0A06 | `/key_usart_sender` | `Password` | `uint8[6] password` |

`/radio` 是原节点的字节流调试字符串，业务接入请使用上表的强类型消息。密钥最终发送所需的串口程序属于主程序侧，不在此 RX 包内。

视觉端另开终端，在本目录执行：

```bash
source env.sh
export ROS_DOMAIN_ID=0
ros2 topic list -t
ros2 topic echo /robot_position
# 查看其他类型时，将话题替换为 /radio_hp、/radio_fire、/radio_state、/radio_buff 或 /key_usart_sender
```

联调时核对：

1. 主程序与 RX 使用同一 ROS 2 发行版、相同 `ROS_DOMAIN_ID`、兼容 QoS；跨容器时确保 DDS 网络可达。本机 ZeroMQ 只在 RX 流图和协议节点之间使用，不作为视觉端接口。
2. 同一 colcon 工作空间只保留一份 `radio_interface`。主程序已有该包时比较六个 `.msg`，保持定义一致，不要重复放入两个同名包。
3. `/robot_position` 保留厘米值和原六个协议槽位。T-DT 融合节点会自行做厘米转米、红蓝坐标转换和敌方 ID 映射，避免重复转换。
4. 消息没有 `Header` 时间戳；业务侧记录接收时刻并处理超时。T-DT 原坐标融合逻辑为无线电有效时优先、超过 1 秒未更新回退雷达。其他业务各自维护时效，断流时不能把旧数值当作仍有效。
5. 当前测试源只发哪些命令，就只能收到哪些命令；“节点启动 / 话题已发现”不等同于“收到数据 / CRC 通过”。

## 验证范围

见 `VALIDATION.md`。赛前还需在目标机器、目标协议、现场频点和真实信号条件下完成最终联调。
