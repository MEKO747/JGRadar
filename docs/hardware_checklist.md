# 真机链路验证清单

阶段 3（视觉与无线电链路验证）的现场作业单。每一步都写清：**命令、期望输出、通过判据、
失败先看什么**。照做的人不需要回头猜。

先记住三条定位：本清单里**只有第 1~8 步的现场数据算验证**；第 10 节是本机已经做过的
准备性验证，**不等于**视觉或无线电验证。

## 0. 开工前先记录（缺这一节，后面的数据都没法回溯）

```zsh
cd /home/meko/JGRadar && git rev-parse --short HEAD && date -Is
lsusb | grep -i 0456            # SDR 出厂号
ls -l /dev/ttyUSB* /dev/v4l/by-id/ 2>/dev/null
```

记录表（每轮验证填一行，交给研究员入台账）：

| 项 | 内容 |
| --- | --- |
| 时间（含时区） | |
| commit | |
| SDR 序列号 / URI | |
| 相机型号 / 序列号 / `/dev/videoN` | |
| 裁判串口设备 | |
| 使用的配置 YAML | |
| 原始数据落盘路径 | |

## 1. SDR 上电检查

```zsh
lsusb -t | grep -A2 -i pluto        # 必须是 Bus 001 根口，不能挂在 hub 下面
ip -br a show enx00e022ee4a3a       # 期望 192.168.2.10/24
ping -c 3 192.168.2.1
iio_info -u ip:192.168.2.1 2>&1 | head -20
```

**通过判据**：`lsusb -t` 里 Pluto 在 `Bus 001` 根口（历史故障：经 GenesysLogic hub 一开流
就 USB 复位）；`ping` 0% 丢包；`iio_info` 能列出 `ad9361-phy` 与 RX 通道。

**失败先看**：① 是否又插到 hub 上（`lsusb -t` 的树形层级）；② 网卡是不是 `192.168.2.10/24`
（DHCP 掉了会变成 169.254.x.x）；③ `dmesg | tail -30` 有没有 USB 复位。

## 2. 裁判系统串口自环（先验硬件，再验协议）

串口线 **TX 与 RX 短接**（USB-TTL 侧自环，不接机器人），然后：

```zsh
ls -l /dev/ttyUSB0
stty -F /dev/ttyUSB0 115200 raw -echo
```

用仓库封装做自环（`radio/radar_parse_em_wave/rmuc/referee_frame.py`）：

```zsh
zsh -c 'source ~/.jg_env.zsh; source /opt/ros/jazzy/setup.zsh; \
        source build/radio/install/local_setup.zsh; \
        python3 - <<"PY"
import serial, time
from rmuc.referee_frame import build_frame
from rmuc.coder import SerialFrameDecoder
port = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.5)
frame = build_frame(0x0305, bytes(48), seq=1)
port.write(frame); time.sleep(0.2)
print("回读", port.read(len(frame)).hex())
PY'
```

**通过判据**：回读字节与发出的一致（含 CRC8/CRC16）。这一条**不接机器人**，只证明「串口能
收发」。接上裁判系统后同一条命令的判据变成：机器人或服务器侧有反应（小地图出现坐标）。

**失败先看**：① 波特率（协议表 1-1 是 115200，图传那条才是 921600）；② `/dev/ttyUSB*` 有没有
出现（`dmesg | grep tty`）；③ 电平（裁判系统是 3.3V TTL，别接 RS232 电平）；④ 权限（`id` 是否在
`dialout` 组）。

## 3. 实时接收基线（空口 → 解调）

```zsh
bash tools/start_radio_vision.sh red --no-rx          # 只跑视觉侧入口，确认不炸
zsh -c 'source ~/.jg_env.zsh; source /opt/ros/jazzy/setup.zsh; \
        source build/radio/install/local_setup.zsh; \
        python3 tools/radio_probe.py --mode bits --seconds 10 --protocol 2027'
```

**通过判据**：干扰波稳定 ~10Hz、信息波业务包能解出 `0x0A01`；`包/kbit` 与之前真实信号基线
同量级（历史：2.43 包/kbit @ SNR 30dB 合成信号，真实信号按现场记录）。

**失败先看**：① `iio_info` 是否通（第 1 步）；② 解调等级——信息波业务包只在 **3 级**解
（`rmuc/state.py`），默认是 1 级，`--interference-level 3` 或 `--referee-level`；③ 频率/增益
（`--set` 覆盖）；④ 是否只是没人在发（业务包只在帧头+CRC+长度全过才发布）。

## 4. `0x0305` 实发与上限

```zsh
zsh -c 'source ~/.jg_env.zsh; source /opt/ros/jazzy/setup.zsh; \
        source build/radio/install/local_setup.zsh; \
        python3 tools/run_radio_vision.py --team red --radio-only \
        --referee-tx --referee-serial /dev/ttyUSB0 --referee-log out/referee_live.jsonl \
        --seconds 60'
```

**通过判据**（从收尾的 `referee_tx_final` 读）：

- `effective_rate_hz ≥ 4.7`
- `max_frames_1s ≤ 5`（协议上限 5Hz）
- 无连续 0.5 秒断流（看 `referee_tx_final` 与 `referee_live.jsonl` 相邻 `t` 的差 ≤0.5s）
- 串口侧抓包核对 48 字节布局：对方重装 0–3、对方 3/4 号 8–15、对方哨兵 20–23、己方 24–27 /
  32–39 / 44–47，保留位全 0

**失败先看**：① `referee_tx_error`（串口打不开只会退化计数，不报错——容易误判成"在发"）；
② 端口被占用（`fuser /dev/ttyUSB0`）；③ 服务器侧有没有真的收到（小地图是否变化）。

## 5. `0x0A01` 定位精度（当前**零样本**，这是阶段 3 的核心）

方法：机器人摆在 4~6 个已知点位（用卷尺/地贴标出，记录坐标），每点静止 30 秒 +
直线通过 2~3 趟，全程落盘：

```zsh
zsh -c 'source ~/.jg_env.zsh; source /opt/ros/jazzy/setup.zsh; \
        source build/radio/install/local_setup.zsh; \
        python3 tools/run_radio_vision.py --team red --radio-only --out out/points_run.jsonl \
        --seconds 300'
```

离线算 `d`（我们发出的坐标 vs 该点位真值），逐次判定：

**通过判据**：`d<0.8m` 的比例 **≥90%**（阶段 3 验收线）。同时报 `d` 的中位数与 P95 作参考，
但**不用 P95 判合格**。

**失败先看**：① 真值本身是否可靠（先把点位量准）；② 相机外参/场地点位（第 6 步）；③ 目标
是否在场地边缘（投影误差最大处）；④ 时间对齐（我们的时间戳 vs 真值采集时刻）。

## 6. 相机与标定

```zsh
zsh -c 'source ~/.jg_env.zsh; python3 tools/camera_preview.py --serial <SN> --save-frame /tmp/hik.png'
zsh -c 'source ~/.jg_env.zsh; python3 tools/validate_alignment.py --config config/default.yaml \
        --camera config/camera.yaml --field config/field.yaml --rms-max-px 1.0 --max-max-px 2.0'
```

**通过判据**：`camera_preview` 能出图（判据是**能出图**，不是好看）；`validate_alignment` 的
重投影 RMS 与最大误差在阈值内，往返误差 `--roundtrip-max-cm` 达标；场地 4 角 + 中央高地
在图像上的投影与实拍对得上。

**失败先看**：① 相机序列号选错（多台设备时 `--serial`）；② 曝光/增益（过曝会让角点提不准）；
③ 标定 YAML 是哪台相机、哪个机位——**换机位必须重标**；④ 镜头畸变模型（`dist` 长度）。

## 7. 融合切换（两路同时在线）

无线电与视觉同时跑，做两种扰动：拔掉 SDR 网线（模拟无线电断流）、用手/板挡住目标
（模拟视觉遮挡）。

**通过判据**：`tier_share` 里能按顺序看到 `radio_fresh → radio_stale → vision_*` 的迁移；
换源 ≤0.4s（0.2s 新鲜窗 + 0.2s 驻留）；同一槽位相邻帧跳变 ≤150cm（`switch_jump_cm`）；
`arbiter_switches` / `arbiter_vetoed` 计数与扰动次数对得上。

**失败先看**：① 谁在报（`active_sources`）；② 年龄（`<name>_age_s`、`stale_slots`）；
③ 跳变抑制是否被触发（`reacquire_s` 让门槛变成 800ms）。

## 8. 时延与长稳

```zsh
zsh -c 'source ~/.jg_env.zsh; source /opt/ros/jazzy/setup.zsh; \
        source build/radio/install/local_setup.zsh; \
        python3 tools/run_radio_vision.py --team red --radio-only --referee-tx \
        --referee-serial /dev/ttyUSB0 --referee-log out/long_run.jsonl --seconds 7200'
```

每 10 分钟记一次：进程 RSS、`referee_tx_final` 的关键计数、`max_frames_1s`。

**通过判据**：连续 2 小时不掉线；无 >0.5s 断流；RSS 无单调增长（有平台期）；
`rejected_radio_frames` 没有台阶式增长。

**失败先看**：① 内存增长先看是否某个队列无界；② 时间漂移看 `t` 列；③ `dmesg` 里的 USB/网卡事件。

## 8.5 一条命令出验收报告

走完 1~8 步、原始数据落盘之后，用一条命令出 `report.json`（结构见
[STAGE3_REPORT_SCHEMA.md](experiments/STAGE3_REPORT_SCHEMA.md)）：

```zsh
zsh -c 'source ~/.jg_env.zsh; python3 tools/stage3_accept.py \
    --radio-log out/live.jsonl --truth out/points.csv \
    --fusion-log out/fused.jsonl --switch-event B1=12.3 \
    --out out/stage3_accept'
```

- 四项判据写死在该工具里（`radio_accuracy` / `radio_uplink` / `vision_calibration` / `fusion_switch`），
  与 goal 完全一致；工具不会自己发明门槛。
- 每项三态 `pass | fail | blocked`，缺输入一律 `blocked`，**绝不判 pass**。
- 报告带 `commit`、`dirty_files`、`config_sha256`、`devices[]`、`artifacts[]`（含 sha256）和
  `caveats[]`；`evidence_grade` 取四项里最低，**工具最高只给 B**，A 级要人工签字。
- 设备没到位时跑它是合法的：输出全 `blocked` + `evidence_grade: D`，这份基线本身可以进台账。

真值文件格式（`--truth`）：CSV，列 `t0,t1,slot,x_cm,y_cm`，表示某车在 `[t0,t1)` 内的真值位置。
`--switch-event slot=秒` 是操作手制造换源刺激（拔网线/遮挡）的时刻，没有它换源延迟量不出来。

## 9. 症状 → 先查什么

| 症状 | 先查 |
| --- | --- |
| 一个业务包都解不出 | 解调等级是不是 3（信息波）；`radio_probe --mode bits` 的 sync 是否为 0 |
| sync 有、decoded 为 0 | CRC 失败 → `rejected` 计数；协议版本是否传对（2027/2026） |
| 能解出但对不上 | 时间戳对齐；点位真值；相机外参 |
| 小地图不动 | `referee_tx_error`；串口波特率；服务器是否收到 |
| 坐标来回跳 | `arbiter_switches`；两个源的年龄；跳变抑制是否生效 |
| 相机打不开 | 序列号、权限、MVS 是否占用 |

## 10. 本机已完成的准备性验证（**不等于**视觉/无线电验证）

这些是在设备到位前能做的、证明「管线没有断点」的检查。结论归属必须写清楚：

| 检查 | 命令 | 结果 | 归属 |
| --- | --- | --- | --- |
| 内置 UVC 取图 → Stage A（stub 检测）→ 跟踪 → 融合 → 发布 | 见下 | 60 帧跑完无异常；因 stub 无身份，槽位全 MISSING | 只证明管线不崩，**不证明检测/精度** |
| 视觉回放（带身份检测 + 真投影链） | 见下 | 96 帧：210 个槽位帧有坐标且全在场地内；`tier_share` 含 `vision_blind 0.87%` | 证明跟踪/投影/融合/发布通，**检测是造出来的** |
| 真实 ONNX 权重 CPU 推理（UVC 画面） | `vision_preview.py --config config/vision_check.yaml --config /tmp/uvc_onnx.yaml --video ...` | 模型就绪（CPU），8 帧 0 检测，954 ms/帧 | 证明模型能加载能推理，**不是精度**，且本机无 torch 时数字级不可用 |

复现命令：

```zsh
# 1) UVC 录 2 秒再喂进流水线
.venv/bin/python - <<'PY'
import cv2
cap = cv2.VideoCapture(0); out = cv2.VideoWriter('/tmp/uvc_30fps.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
for _ in range(60):
    ok, f = cap.read()
    if not ok: break
    out.write(f)
cap.release(); out.release()
PY
# 2) 视觉回放跑通跟踪/投影
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, '/home/meko/JGRadar')
from jg_radar.config import load_config
from jg_radar.pipeline.build import build_loop
import dataclasses
cfg = load_config('config/default.yaml')
cfg = dataclasses.replace(cfg, io=dataclasses.replace(
    cfg.io, frame_source='replay', replay_path='tests/fixtures/replay_seq_001'))
loop = build_loop(cfg)
outs = loop.run_to_completion()
print(len(outs), loop.stage_b.publisher.tier_share())
PY
```

**三个已知的本机缺口**（不是现场问题，是环境问题，先修再谈视觉精度）：

1. 数字分类模型需要 `torch`，本机未装 → `vision_check.yaml` 原样跑会报「数字模型未加载」。
2. onnxruntime 的 CUDA provider 缺 `libcudnn.so.9`，实际落在 CPU（954 ms/帧，达不到实时）。
3. 没有海康相机，`/dev/video0` 是笔记本内置 UVC，标定与机位都不对——**不能用它的数据验收视觉**。
