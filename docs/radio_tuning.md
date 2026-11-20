# 解调参数调优：离线回放与参数扫描

现场只有 PlutoSDR 实时源时，"调参"没法验证：同一段信号跑不第二遍，改一个
参数是变好了还是运气好，说不清。这一套工具把解调链搬到离线上：

```
合成或录制的 IQ ──▶ 真实流图（同一份 DSP）──▶ 比特流 ──▶ 接收包自己的解码器 ──▶ 三级计数
```

三个工具：

| 工具 | 作用 |
| --- | --- |
| `tools/iq_synth.py` | 按流图假定的波形合成空口 GFSK 信号，没有现场信号时当基线 |
| `tools/radio_replay.py` | 把签入流图派生回放变体（或录制真实 IQ），跑一遍出三级计数 |
| `tools/radio_sweep.py` | 在同一段 IQ 上扫参数，输出 CSV、曲线和报告 |

## 为什么不是复制一份流图

复制出来的第二条 DSP 链迟早和真实链漂移——真实链改了一个滤波器，回放链还
停在旧参数上，于是"离线测出来的最优"在场上不成立。

所以 `radio_replay.py` **读签入的 `.grc`，只在内存里替换源块和输出块**，把
变体写到临时目录再用 `grcc` 编译。DSP 链、AGC、限幅、SIC、判决器全部沿用实时
路径，签入文件一个字节都不改，所以实时行为天然不变；`tests/test_radio_replay.py`
里有一条测试专门盯这个哈希。

派生做了这些替换：

| 块 | 实时路径 | 回放 |
| --- | --- | --- |
| `nanosdr_source`（`iio_pluto_source`） | PlutoSDR | `blocks_file_source`（complex64，默认播一遍） |
| 两路比特流 `zeromq_pub_sink` | ZMQ 53103 / 53104 | `blocks_file_sink`（默认）或 ZMQ 53203 / 53204（`--live`） |
| 功率遥测 `zeromq_pub_sink` | 53105 | 53205 |
| `xmlrpc_server` | 53106 | 删掉（回放不需要实时调参） |

落盘模式是默认的：流图满速跑完文件、把比特写盘，再离线解码。这样不受
"Python 解码比流图慢"的约束，也没有流图退出时丢掉队列尾巴的问题——同一参数
点重跑三次，计数完全一致。

## 环境

```zsh
source ~/.jg_env.zsh    # 里面有 PYTHONNOUSERSITE=1，缺了它 GNU Radio 会被 ~/.local 的 numpy 拖崩
```

工具自己也会给子进程补上 `PYTHONNOUSERSITE=1`，但 pytest 只在 `.venv` 下跑，
所以基线命令是：

```zsh
PYTHONNOUSERSITE=1 .venv/bin/python -m pytest -q     # 680 passed, 1 skipped
```

直接 `.venv/bin/python -m pytest`（不带变量）会在 `test_video_validation.py`
和 `test_vision_preview.py` 上炸在 `cv2` 导入，那是 numpy 2.x 与系统 OpenCV
的 ABI 冲突，不是测试挂了。

## 1. 合成基线

```zsh
.venv/bin/python tools/iq_synth.py --out out/iq/synth_10s.c64 --seconds 10
```

- 符号率 = `samp_rate / sps` = 1e6 / 47 ≈ **21.28 kBd**
- 空口帧 = Access Code(8B) + Header(4B) + Payload(15B) = **27 字节 / 216 比特**
- 串口业务帧按字节流切开，每 15 字节塞一帧，跨帧重组交给 `SerialFrameDecoder`
- 业务包覆盖 0x0A01~0x0A05，数值按 2026 场地边界取

调制指数 `h=8`（频偏 ≈ 85 kHz）是**假设值**：规则文档里查不到官方波形参数，
取它是因为频偏落在流图的 270 kHz 中频滤波器内、判决器输出不贴 ±1.2 的限幅。
绝对门限随这个假设漂移，跨参数比较才可信。

## 2. 回放一段 IQ

```zsh
.venv/bin/python tools/radio_replay.py --iq out/iq/synth_10s.c64
.venv/bin/python tools/radio_replay.py --iq rec.c64 --set loop_bw=0.002 --interfere
.venv/bin/python tools/radio_replay.py --iq rec.c64 --live      # 边跑边看（会丢尾巴）
```

输出是两路的三级计数：

```
info     比特   200013 | 空口帧    913 | 序列帧    484 | 业务包    484 | 包/kbit 2.42 | 帧/同步 0.532
```

`--bits` 在 `--live` 下是停止条件；落盘模式按文件长度算，不看墙钟。

## 3. 录制真实空口

```zsh
.venv/bin/python tools/radio_replay.py --record out/iq/field_10s.c64 --seconds 10 \
    --sdr-uri ip:192.168.2.1
```

录制模式**保留 PlutoSDR 源**，在 `rx_raw_baseband_offset` 之前插一个 `blocks_head`
（收满 `seconds × samp_rate` 个样点就发 done，整张图随之收尾）和一个
`blocks_file_sink`，解调链照旧连着——所以录制的同时还能用
`tools/radio_probe.py` 看接收状态。

录制点就是回放时喂回去的那一点：录下来什么，回放就重放什么。

> SDR 一定要**直插笔记本本体的 USB 口**。经 GenesysLogic USB2.1 Hub 的 7-1.1
> 口一开流就 USB 复位。

## 4. 扫参数

```zsh
.venv/bin/python tools/radio_sweep.py --mode synth \
    --snr 0,5,8,10,12,14,16,18,20,25,30 \
    --loop-bw 0.002,0.005,0.01,0.02 \
    --bt 0.25,0.35,0.4,0.45,0.5 \
    --bits 200000 --out out/radio_sweep/refined_20261120
```

产物：`results.csv`（一行一个参数点）、`curves.svg`、`report.md`、`meta.json`。
报告写完后可以只重建、不重跑：

```zsh
.venv/bin/python tools/radio_sweep.py --rebuild out/radio_sweep/refined_20261120
```

有真实录制时用 `--mode file --iq rec.c64`，`--snr` 表示在录制上**附加**噪声。

### 指标口径

| 指标 | 定义 | 用来回答 |
| --- | --- | --- |
| `sync_rate` | 空口帧 / (比特 / 216) | 帧同步还在不在 |
| `packet_yield` | 业务包 / 期望业务包 | 真正解出来多少（只有合成模式有真值） |
| `packets_per_kbit` | 业务包 / 比特 × 1000 | 跨数据源可比的主指标 |

都按消耗的比特归一，不看墙钟，所以不同 CPU、不同参数点的结果可比。
单点只测一次，差异小于 1 个百分点的参数点不要当成真的不同。

### `rx_gain` 在回放里不可观测

硬件增益只决定**录制时的输入 SNR**：降 10 dB 增益，等价于降 10 dB SNR。
文件回放里没有硬件增益这回事（`--set rx_gain=...` 会被明确警告并忽略），
所以本报告的 SNR 轴就是增益轴平移；**真正的增益曲线必须在硬件上扫**。

## 基线结论（合成信号，2026-11-20）

220 个参数点、每点 200 kbit（约 926 个空口帧）。完整报告与 CSV 在
`out/radio_sweep/refined_20261120/`（`out/` 不进版本库，用上面的命令可复现）。

**敏感度排序**：SNR（相对极差 2.19）≫ BT（1.29）≫ 环路带宽（0.20）。
也就是说，在这一档信噪比范围里，先把信号弄干净比抠参数重要得多。

**解波门限**：SNR ≥ 20 dB 时全部参数组合都到平台；10 dB 以下全线归零；
拐点在 **14~18 dB** 之间（14 dB 出 0.72 包/kbit，18 dB 出 1.58，20 dB 出 1.75）。

**各轴最优区间**

| 参数 | 当前值 | 均值 包/kbit | 结论 |
| --- | --- | --- | --- |
| `loop_bw` | 0.005 | 0.85 | 偏慢。0.01 / 0.02 都是 0.89，比当前高约 5%；0.002 掉到 0.73 |
| `bt` | 0.35 | 1.07 | 已经接近最优。0.35~0.45 打平（1.07~1.09），0.5 略降，0.25 **完全打不开眼** |

结论：**`loop_bw` 从 0.005 提到 0.01，`bt` 保持 0.35**。前者是唯一有数据
支撑的改动，后者是已经站在平台上，不用动。`bt=0.25` 那种"窄一点少进噪声"
的直觉在这个波形上完全失效（同步率 <8%）。

## 局限（写报告时必须一起说）

- 合成分支是**模型信号**。符号率取自流图 `samp_rate / sps`，调制指数是按
  "频偏落在中频滤波器内"反推的假设值。绝对门限随假设漂移，**跨参数比较才可信**。
- 高 SNR 下仍有约 1.5% 的空口帧丢失（同步率 98.6%），来源是帧边界与符号
  同步恢复，不是噪声。所有参数点都带这个底，比较时相减即可。
- 只扫了解调链，**没有扫 SIC 的 `leak_beta`**，也没有合成功率更大的干扰波
  场景；这两项要等真实录制（`--mode file`）再补。
- 没有做硬件增益扫描，也没有验证 `stale_policy` 那类发送侧策略——那部分在
  `0x0305` 发送端里单独测。

## 端口速查

| 用途 | 实时链 | 回放 |
| --- | --- | --- |
| 信息波比特流 | 53103 | 53203 |
| 干扰波比特流 | 53104 | 53204 |
| 功率遥测 | 53105 | 53205 |
| SDR 调参 XMLRPC | 53106 | 不启用 |

回放端口刻意和实时链错开，所以回放可以和正在跑的接收节点同时用。
