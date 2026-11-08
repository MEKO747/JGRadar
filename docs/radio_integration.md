# RM_Radio 与 JGRadar 联合运行

接入的是本机 2026-11-06 的 RM_Radio 纯接收源码，副本位于 `radio/RM_Radio/`，原始下载目录保留。解调器、CRC 和六类 ROS 消息保持交付包原样，来源和许可证见该目录 `NOTICE.md`、`LICENSE` 与 `gr-robomaster/COPYING`。本集成不表示完成 2027 协议、装甲加固指令或比赛现场验收。

## 数据路线

HAMGEEK RX2 → GNU Radio 三支路 → ZeroMQ → radio_recive → ROS 2 → RadioAdapter → StageB 最新值缓冲 → 原有仲裁 → `/jg_radar/fused`。

相机 → 原有三级检测、投影与跟踪 → 同一 StageB；视觉与无线电独立更新，视觉没有新帧时仍按时检查过期。融合是选择当前可用来源，沿用现有优先级、切换驻留及误差约束，不把两个坐标简单平均。位置有效期和新鲜阈值沿用 `config/default.yaml` 的 `fuse` 配置。

`/robot_position` 的六个槽位按敌方 1、2、3、4、6、7 映射，单位始终是 cm。输入默认已经是红方场地坐标，与 JGRadar 一致；不会因为己方蓝色再翻转一次。只有测试源明确发送己方视角坐标时才加 `--radio-frame own_team`。映射依据为上游 T-DT 2026 的 radio_callback / publish_timer_callback：
https://github.com/T-DT-Algorithm-2026/T-DT_Radar/blob/main/src/fusion/kalman_filter/src/kalman_filter.cpp

`(0,0)` 和 `(2800,1500)` 是上游无线电无效哨兵值；越界、非整数也撤回该槽旧坐标。单独 x=0 或 y=0 的合法边界点仍可使用。当前适配器只接受 2800×1500 cm、角色 1/2/3/4/6/7 的 2026 消息；2027 角色/场地/协议需单独适配。

## 环境与构建

目标为 Ubuntu 24.04、ROS 2 Jazzy、Python 3.12；视觉运行环境还需原有 ONNX/MobileNet 依赖。使用同一 ROS_DOMAIN_ID。当前本机 `.venv` 可以导入系统 Jazzy 的 rclpy（需先 source）。

```bash
cd /home/meko/JGRadar
# 在装有 GNU Radio 及交付包 README 中依赖的目标机，构建完整 RX：
bash tools/build_radio.sh
# 若只做 ROS 解析/接口联调（不构建 GNU Radio）：
bash tools/build_radio.sh --ros-only
# 检查，不访问 RX 或相机：
bash tools/start_radio_vision.sh red --no-rx --check
```

依赖安装要求见 `radio/RM_Radio/README.md`。`--ros-only` 不会让 GNU Radio 接收器可用。通过 `RM_RADIO_ROOT` 可改用其他已经构建的同版本交付目录；通过 `JG_RADAR_PYTHON` 可指定视觉 Python 3.12 环境。副本移动后应在新目录重新构建。

## 联合启动

先准备真实相机内外参 YAML（可包含 camera / transform / io 分组）。现有 `config/camera.yaml` 是占位标定，不能直接用于真实坐标融合。

```bash
cd /home/meko/JGRadar
export ROS_DOMAIN_ID=0
bash tools/start_radio_vision.sh red \
  --config config/camera.yaml --config config/field.yaml \
  --config config/vision_check.yaml \
  --config /absolute/path/to/your_calibrated_camera.yaml
```

`red/blue` 表示己方颜色，并同时传给 RX 与融合映射。默认启动本仓库副本的 RX 和协议节点，再启动实时视觉与 ROS 适配器；退出或任一子进程结束后停止本次启动器的其他进程。不会按进程名关闭别人的程序。仅由本启动器启动一套 RX，避免与旧接收器争抢设备。

已有 RM_Radio 在另一终端或同域其他机器运行时，增加 `--no-rx`。两端电脑的系统时间须同步，DDS 网络须可达。纯接收融合联调、不打开相机时：

```bash
bash tools/start_radio_vision.sh red --radio-only
# 接收器另行运行时：
bash tools/start_radio_vision.sh red --no-rx --radio-only
```

原测试源需要收窄干扰滤波时，加 `--noise-cutoff 270000`，其适用边界遵循 RX 包 README。`--seconds 10` 可限制融合进程运行时间；`--out out/radio_vision.jsonl` 可选保存融合业务输出，不保存 IQ。

## 输出和时效

- `/jg_radar/fused`：`std_msgs/String` JSON，沿用 FrameOutput，保留机器人来源、真实观测时间、发布时刻及预测状态。
- `/jg_radar/radio_status`：`std_msgs/String` JSON，独立报告 position/hp/fire/state/buff/password 的年龄与 valid；业务状态暂按 1 秒 TTL，过期数据仍可用于诊断，使用方必须检查 valid。密钥仅公开收到时间，不把值写入诊断输出。
- 原来的六类强类型话题全部保留；最终密钥发送继续订阅 `/key_usart_sender`。本项目此前没有裁判串口发送应用，当前集成也未建立实际发送链路。

消息原本没有 Header，无法还原真实 RF 采集时刻。本集成利用 DDS 发布时间估计该消息年龄，并转成本机 monotonic 时钟；超过 0.5 秒或发送方时钟明显超前的消息被拒绝，订阅队列只保留 1 条。`--input-latency 秒数` 可减去实测解调延迟；默认 0 表示尚未测得，输出会注明 timestamp_basis，不能把它当成空口采集时间已得到验证。没有新消息时，tick 不刷新任何无线电时间。录像回放使用录制时间，实时 ROS 接入入口只开放真实相机或 radio-only，防止混用时钟。

## 验收边界

单元验证覆盖红蓝敌方映射、厘米值、显式坐标翻转、无效槽撤回、乱序/旧包拒绝、断流回退与状态独立过期。ROS 联调使用隔离的 ROS_DOMAIN_ID，合成消息与输入字节不进入正式话题域。实际 RX、相机标定、空口误码、长时间稳定性及 2027 新协议需在设备就绪后验证。
