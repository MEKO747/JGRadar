# 无线电与视觉集成验证记录

验证日期：2026-11-08。位置：`/home/meko/JGRadar`。

- 原有视觉、跟踪和身份相关的未提交修改保持原样；本次新增 integration 模块、构建/启动工具、文档、回归用例，以及 `radio/RM_Radio` 源码副本。
- 副本与原交付包 SHA256SUMS 全部匹配，解调算法、CRC、六个消息定义未修改。
- 本机 ROS 2 Jazzy：radio_interface、radio_recive 成功构建；原代码 NULL 宏重定义警告仍存在，不影响编译。
- 新适配器及现有缓冲、仲裁、线程和回放相关回归：85 项通过。
- 真实 Jazzy 元信息与定长数组类型回归：另外 3 项通过。修复了回调元信息为 dict 及数组元素为 numpy 整数时的兼容问题。
- 原协议解析器端到端检查：测试字节流经本机 ZeroMQ → 未改动的真实 C++ CRC/协议解析节点 → 六类真实 ROS 消息 → 新适配器 → StageB → `/jg_radar/fused` 与状态输出。红、蓝均通过；六类字段、1/2/3/4/6/7 槽映射、厘米坐标、状态过期和无新视觉输入时的位置失效均符合预期。输入是合成协议数据，不是射频接收或 GNU Radio 解调。
- 原有 96 帧视觉 fixture 按 monotonic 时间线运行投影、跟踪与 ThreadedEventLoop，同时通过隔离 ROS 域发送无线电消息：本次处理 92 个视觉包、接收 28 帧无线电位置；已验证无线电优先及断流后自动回到视觉。fixture 预先提供检测结果，不代表重新验收真实相机/模型精度。
- 启动前检查 `bash tools/start_radio_vision.sh red --no-rx --check` 通过；检查不访问设备。脚本语法和非法阵营参数检查通过。

ROS 检查使用隔离域 198。验证进程结束后已关闭。构建产物位于 radio/RM_Radio 内被忽略的 build/install/log 目录。

本机未安装 GNU Radio，未执行完整 OOT 构建；本次没有打开 RX、相机或进行空口发射。完整运行仍需目标机安装交付包依赖、完成真实相机标定，并复核目标赛季协议。2027 装甲加固、串口发送和实战验收不在当前已完成范围。

可重复运行的安全回归（不访问硬件、无需 DDS 通信）：

```bash
cd /home/meko/JGRadar
source radio/RM_Radio/env.sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_rm_radio_integration.py tests/test_rm_radio_ros_types.py \
  tests/test_fuse_buffer.py tests/test_arbiter_accuracy.py tests/test_arbiter_hysteresis.py \
  tests/test_pipeline_mailbox.py tests/test_pipeline_replay.py
```
