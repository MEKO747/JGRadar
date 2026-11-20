# 2026 赛季雷达站兄弟队伍：视觉/无线电怎么选源（rev2，2026-11-21）

问题：JGRadar 的「优先级阶梯 + 驻留」要不要改？同行 2026 怎么写的？
方法：GitHub 拉源码 + 子模块，链接钉到 commit；否定性结论一律附检索式与时间（见 §2）。

**rev2 修订**：rev1 漏了江南大学 JNU-SHARK（`shark-radar-system` 及其 `radio`/`vision` 子模块）。漏检原因与检索日志见 §2，JNU 的拆解见 §4——它现在是本报告最重要的参照。

---

## 0. 一句话结论（rev2）

**结构不用改，阈值不用大改，改三处小地方。** 现在有多一家（JNU）加入统计：**五家双源队伍全部是「无线电优先 + 超时回退视觉」，全部没有驻留**。阈值 0.45~1.0 s。我们缺的不是结构，是**把「年龄/来源」说出来**，以及 0.3 s 驻留叠在 0.2 s 新鲜窗上零余量。

另外两条硬事实（这次从协议原文核到了）：
- `0x0305` **频率上限 5 Hz**，数据段 48 字节，雷达→服务器→己方所有选手端；
- 坐标 `x=0 且 y=0` **视为「未发送此机器人坐标」**——发零等于没发，会直接进 0.5 s 断流计数。

---

## 1. 学校结论表（修正）

| 学校 | 结论 | 证据 |
|---|---|---|
| **东北大学**（沈阳 T-DT） | ✅ 2026 开源 | `T-DT-Algorithm-2026/T-DT_Radar`（+ 无线电 `RM_sdr`） |
| **江南大学**（SHARK） | ✅ 2026 开源，**rev1 漏检** | `JNU-SHARK/shark-radar-system` + `radio`(6835547) / `vision`(a304fdb) 子模块 |
| **东南大学**（南京 3SE） | ⚠️ 只找到 2018 视觉开源；2026 雷达站未检索到 | 见 §2 检索式；元仓库「东南大学 3SE」节也只有 2018 仓库 |
| **复旦大学** | ❌ 未见 2026 雷达公开仓库 | 见 §2；元仓库无「复旦大学」节 |

---

## 2. 检索日志（为什么漏、怎么补）

时间取自本机（CST，分钟级）。「结果」只记与雷达相关的命中。

### 2.1 首次检索（rev1，2026-11-20 23:32–23:45）

| 检索式 / URL | 方式 | 结果 |
|---|---|---|
| `https://api.github.com/search/repositories?q=robomaster+radar&sort=stars` | REST | 命中 T-DT 2025、TCR、HUST、PnX 等；**无 JNU** |
| `q=robomaster+radar+2026` / `q=RMUC+radar+2026` | REST | 命中 XJTU Radar26、PnX 2026、TCR、YJsnz；**无 JNU**（其描述无 `2026`） |
| `gh search repos 雷达站` | gh | 40 条，命中 T-DT 2025、TCR、EGA、同济 sp_radar_26 等；**无 JNU**（描述是英文） |
| `gh search repos "复旦 雷达"` / `"江南大学 雷达"` / `"东北大学 雷达"` | gh | **均 0 条**（要求两个中文词同时出现在描述里，条件过严） |

### 2.2 补充检索（rev2，2026-11-20 23:47–23:55）

| 检索式 / URL | 方式 | 结果 |
|---|---|---|
| `gh search repos "SHARK radar"` | gh | **命中 JNU-SHARK 三仓**（system/radio/vision） |
| `gh search repos "shark-radar"` | gh | 同上 |
| `gh search repos "JNU radar"` | gh | 0 条（队名缩写 + radar 反而搜不到） |
| `gh search repos "江南大学"` | gh | 10 条，全是课程/论文模板类，无雷达 |
| `gh search code "江南大学 language:python"` | gh code | 8 条，无关 |
| `gh search repos "东南大学+RoboMaster"` | gh | 5 条：2017 项目、成贤学院 Solaris 主页、**元仓库 leaderboard**、2026 能量机关固件；**无 3SE 雷达** |
| `gh search code "东南大学 radar"` | gh code | 0 条 |
| `gh search repos "复旦大学+RoboMaster"` / `gh search code "复旦大学 radar"` | gh | **均 0 条** |
| 元仓库 `SEU-CX-RM-Solaris/rm-open-source-leaderboard` | 人工 | 有「东南大学 3SE」节（仅 2018）、「江南大学 SHARK」节（未收录 shark-radar-system）、**无复旦大学节** |

### 2.3 漏检根因（我的问题）

1. **只搜描述、不搜队名**：JNU 仓库描述是 `RoboMaster radar system orchestration and deployment`，既没有「雷达站」也没有「江南大学」，中文关键词必然漏。
2. **只用「学校名 + 雷达」这种交集词**：GitHub 搜索是描述文字的 AND 匹配，两个词缺一就 0 条，我把 0 条当成了「没有」。
3. **没查元仓库（awesome/leaderboard）**：这类目录是发现冷门仓库最稳的入口，我第一轮没做。
4. **code search 没先用已知仓库反查命名**：正确做法是先看一个已知雷达仓库的命名习惯（如 `radar_integration`、`0x0305`），再拿这些词去搜。

**以后的检索规则（自查用）**：队名（中英）+ 英文功能词 + 元仓库 + 已知仓库的 org 列表 + code search，四路至少走三路；否定结论必须贴检索式和条数。

---

## 3. 总表（rev2）

| 队伍 | 源 | 选源逻辑 | 滞回 | 阈值 | 超期处理 | 0x0305 | 冲突听谁 |
|---|---|---|---|---|---|---|---|
| **江南大学 SHARK** | 视觉 + 无线电 | 逐车：无线电 fresh → 视觉 fresh → `missing`（0,0） | **无** | 无线电 **1.0 s**、视觉 **0.5 s**、ghost **2.0 s**（仅展示） | 不标 stale 进坐标；全零则整帧不发；ghost 只做可见性 | ✅ **4.8 Hz**（<5Hz 上限），双源优先级在发送循环里 | 无线电 |
| **东北大学 T-DT 2026** | 视觉 + 无线电 | 无线电 valid 优先，否则雷达；每目标独立卡尔曼 | 无 | 两源均 **1.0 s** | `valid=false` + 滤波器 reset | ❌ 无发送端 | 无线电 |
| **厦大嘉庚 TCR** | 视觉 + 无线电 | 0.2 新鲜 → 视觉 → 0.5 兜底 → 最新并标 stale | 无 | 0.2 / 0.3 / 0.5 / **10 s** | 标 `INFO_STALE` 继续报 | ✅ 有 | 无线电 |
| **同济 sp_radar_26** | 视觉 + 信息波 | 信息波覆盖视觉；哨兵指令静默回退 | 无 | **0.5 s** | 照发（含零值） | ✅ **5 Hz** | 信息波 |
| **西交 Radar26** | 视觉 + 信息波 | 逐车覆盖 | 无 | **1.0 s** | 回退视觉，记每车来源 | ✅ **5 Hz** | 信息波 |
| **华南农 SCAU** | 视觉 + 无线电（基本弃用） | 视觉后写覆盖 | 无 | **1.0 s** TTL | TTL 清零 | ⚠️ 约 **15 Hz**（30Hz 隔一拍），**超 5Hz 上限** | 视觉 |
| **星云 EGA** | 相机 + 激光 + 信息波 | 开关控制信息波覆盖 | 无 | **0.25 s** | 立即回退 | ✅ **5 Hz** | 信息波（默认关） |
| **港科广 PnX** | 仅无线电 | — | — | — | — | 仅上传密钥 | — |
| **吉大 YJsnz / 北航 / 华北理工** | 仅视觉或仅激光 | — | — | — | — | 吉大有 0x0305 类；北航仅测试桩 | — |

---

## 4. JNU-SHARK 全量拆解（重点）

### 4.1 融合选源：`core/radar_integration.py`

三个阈值（`RadioFusionState.__init__`，`:83-84` 与节点 `:51-53` 一致）：

```python
vision_timeout_sec = 0.5   # 视觉新鲜窗
radio_timeout_sec  = 1.0   # 无线电新鲜窗
radio_ghost_sec    = 2.0   # 1.0~3.0s：只保留"可见"，不用于坐标
```

逐车判定（`:251-259`）：

```python
if not is_ally and radio['fresh']:      # 非己方 + 无线电 age<=1.0s 且 valid
    final = radio['position_cm']; source = 'radio'
elif visual['fresh']:                   # 视觉 age<=0.5s 且 valid
    final = visual['position_cm']; source = 'vision'
else:
    final = {'x':0,'y':0}; source = 'missing'
```

- **没有驻留、没有跳变门槛、没有按档位排序**：就是「无线电够新就用，否则视觉够新就用，都不够新就报 missing」。
- `radio['expired']`（age>1.0s）与 `radio['ghost_visible']`（1.0<age≤3.0）都会写进快照，**但 ghost 从不参与坐标**——相当于「记得它还活着，但不再用它」。

### 4.2 发送闸门：全零不发、阵营不符不发

- `build_0305_payload()`（`:59`）在 12 个坐标全为 0 时返回 `None`；`send_allowed = payload is not None and not mismatch`（`:273`），`block_reason` 两种：`configured_side_mismatch`、`all_positions_zero_or_stale`。
- 也就是说：**JNU 选择「没数据就不发帧」，而不是发旧值**。按协议「0,0 视为未发送」，发零和不发等价，所以他们干脆不发。

### 4.3 `0x0305` 发送循环：双源优先级 + 年龄闸门

`app_support/run_algorithm.py`：

- 调度频率默认 4.8 Hz（`radar_integration_node.py:49`），且节点把配置值强制夹到 **≤4.8 Hz**（`:153`），线程名 `rm-radio-0305-5hz`——**主动留出 0.2 Hz 余量，不贴着 5 Hz 上限打**。
- 两条候选源（`:313-340`）：
  1. 解码得到的 `0x0A01` 坐标，age ≤ **1.0 s**（`decoded_radar_client_max_age_sec`，`:129`）→ 优先；
  2. 视觉算法结果，age ≤ **0.45 s**（`radar_client_max_age_sec`，`:128`）→ 其次。
- 超龄直接 `continue` 并记 `0x0305 latest payload stale (…s old)`（`:335-340`）——**宁可这一拍不发，也不发旧数据**。

### 4.4 健康状态机（两层）

- **数据层**：每车 `source ∈ {radio, vision, missing}` + `age_sec` + `fresh`/`expired`/`ghost_visible` + `source_counts`，全部写进 `shark.radar.fusion.v1` 快照（`:298-310`）。
- **发送层**：dashboard 按「实际发送率」判 `ok/warn/bad`：低于配置一半记 warn、有请求无成功写记 bad、视觉在线但全无效记 bad（`app_support/match_dashboard_node.py:1113-1127`）。
- **视觉层**：telemetry 每车带 `state ∈ {measured, occlusion_hold, blind_prediction, missing}` + `valid`/`confidence`/`observed_at`（`vision/vision_telemetry.py:43-57`）——**这四个状态名和我们 `vision_measured / vision_occlusion_hold / vision_blind` 一一对应**。

### 4.5 IQ 归档与重放

- 录制：`nodes/match_recorder_node.py`（会话目录、压缩 JSONL、原子写、磁盘余量检查）。
- 重放：`core/iq_replay.py` + `nodes/iq_replay_tx_node.py`（GNU Radio 流图把 IQ 文件重新发射，用于台架复测无线电链路）。

### 4.6 协议原文（顺带核掉了 5 Hz 悬案）

JNU `radio/docs/` 里就带了官方协议 PDF（`RoboMaster 2026 高校系列赛通信协议 V2.0.0`）：

- `0x0305`，数据段 **48 字节**，说明「选手端小地图接收雷达数据，**频率上限为 5Hz**」，链路「雷达→服务器→己方所有选手端」；
- 表 1-36：12 组坐标（对方 6 + 己方 6），备注「当 x、y 均为 0 时，视为未发送此机器人坐标」；
- 修订日志里 `0x0101, 0x0305` 有改动。

→ 之前「5 Hz 是上限」从旁证升级为**原文确认**；SCAU 那个约 15 Hz 的发送可以算超发。论文里可以引用协议原文，不用再标待确认。

---

## 5. 对 JGRadar 的结论

### 5.1 不改的

| 项 | 理由 |
|---|---|
| 阶梯顺序（`schema.py:269`） | 5/5 双源队（JNU、T-DT、TCR、同济、西交）全是无线电优先；JNU 逐车选源、TCR 三段窗，本质同构 |
| `radio_stale_s = 10`（`schema.py:277`） | 与 TCR 同值；JNU 的「1.0 + ghost 2.0」比我们短，但 10 s 仍在同行区间上沿，先不动 |
| `vision_watchdog_s = 0.5`（`schema.py:278`） | JNU `vision_timeout_sec=0.5`、TCR 0.5，**三个项目同值**，没有理由改 |
| `switch_jump_cm = 150`（`schema.py:281`） | 同行没有等价物（T-DT 用 0.4 s 预测、JNU 直接切），我们无预测模型，保留 |

### 5.2 要改的（按优先级）

1. **`dwell_s` 0.3 → 0.2（`schema.py:279`）**：等效换源从 0.5 s 降到 0.4 s，留 0.1 s 余量。依据：JNU/T-DT/同济/西交全部当帧切换、零驻留；TCR 也没有驻留。
2. **0.5 s 标 stale（`arbiter.py:349` 放行处 + `SlotDecision` `arbiter.py:228`）**：加 `age_s` / `stale` / `source`。参照 JNU 的快照字段（`source` + `age_sec` + `fresh`/`expired`）和 TCR 的 `INFO_STALE`。
3. **输出层带 `source` + `age_s`**：给将来的 `0x0305` 发送端和论文统计用。JNU 的 `source_counts`、XJTU 的 `fused_sources`、EGA 的 `radar2client_enemy_source` 是同一思路。

**1.0 s 否决保持不动**：T-DT 和西交都是 1.0 s，JNU 也是 1.0 s（解码源），我们的 1.0 s 与同行一致。

### 5.3 一个新的分叉，需要你定（不是代码问题）

「过了新鲜窗之后，`0x0305` 到底发不发旧值」——同行给了两个相反答案：

- **TCR**：标 `INFO_STALE`，继续发（让下游/裁判看到「有值但旧」）；
- **JNU**：整帧不发（全零等于未发送），宁可这一拍空着；
- **同济**：每 200 ms 照发，含零值。

按协议「0,0 = 未发送」，发零 = 不发，都会进 0.5 s 断流计数；而发旧值会被逐次算距离误差。**哪边更划算取决于目标丢失时的实际距离分布**，这个应该用回放数据算，而不是拍脑袋。我的建议：仲裁器只负责把 `source/age/stale` 说清楚，发送端策略单独开一个开关（`stale_policy: drop | send_stale`），默认 `drop`（跟 JNU 一致），等实测数据再改默认值。

---

## 6. 证据强度 / 仍不确定

- JNU 的 4.8 Hz 是代码默认，实际比赛频率未实测；但「留余量」的设计意图明确。
- JNU 的 `0x0305` 两条源谁最终胜出，我读的是发送循环的 if 顺序（`decoded` 优先、`algorithm` 兜底），没有跑起来验证。
- 东南大学 3SE 2026 雷达是否开源：只证明「GitHub 描述/代码/元仓库里没有」，不能证明不存在（可能在 Gitee/私有）。
- 复旦：同理，且元仓库没有复旦节，旁证其开源密度低，不是「没有」。
- SCAU 的约 15 Hz 是代码推断（30 Hz 定时器 + `w_tick % 2`），未实测；协议原文已确认 5 Hz 上限，所以「超发」这一定性成立，但是否被裁判系统限流/丢弃未验证。

---

## 7. 链接（commit 固定）

- 江南大学 SHARK 聚合仓（`8748746`）：[README](https://github.com/JNU-SHARK/shark-radar-system/blob/8748746/README.md)
- SHARK radio（`6835547`）：[`radar_integration.py:83` 阈值](https://github.com/JNU-SHARK/shark-radar-radio/blob/6835547/src/rm_radio_ros/rm_radio_ros/core/radar_integration.py#L83)、[`:251` 逐车选源](https://github.com/JNU-SHARK/shark-radar-radio/blob/6835547/src/rm_radio_ros/rm_radio_ros/core/radar_integration.py#L251)、[`:273` 发送闸门](https://github.com/JNU-SHARK/shark-radar-radio/blob/6835547/src/rm_radio_ros/rm_radio_ros/core/radar_integration.py#L273)、[`radar_integration_node.py:49` 4.8 Hz](https://github.com/JNU-SHARK/shark-radar-radio/blob/6835547/src/rm_radio_ros/rm_radio_ros/nodes/radar_integration_node.py#L49)、[`run_algorithm.py:128` 0.45/1.0 s](https://github.com/JNU-SHARK/shark-radar-radio/blob/6835547/src/rm_radio_ros/rm_radio_ros/app_support/run_algorithm.py#L128)、[`run_algorithm.py:285` 发送循环](https://github.com/JNU-SHARK/shark-radar-radio/blob/6835547/src/rm_radio_ros/rm_radio_ros/app_support/run_algorithm.py#L285)
- SHARK vision（`a304fdb`）：[`vision_telemetry.py:43` 四种状态](https://github.com/JNU-SHARK/shark-radar-vision/blob/a304fdb/vision_telemetry.py#L43)
- 东北大学 T-DT 2026（`0bc3e37`）：[`kalman_filter.cpp:52` 超时](https://github.com/T-DT-Algorithm-2026/T-DT_Radar/blob/0bc3e37/src/fusion/kalman_filter/src/kalman_filter.cpp#L52)、[`:271` 选源](https://github.com/T-DT-Algorithm-2026/T-DT_Radar/blob/0bc3e37/src/fusion/kalman_filter/src/kalman_filter.cpp#L271)、[`kalman_filter.h:66` 阈值](https://github.com/T-DT-Algorithm-2026/T-DT_Radar/blob/0bc3e37/src/fusion/kalman_filter/include/kalman_filter.h#L66)
- 同济（`8ffe732`）：[`decision.cpp:26` 5 Hz](https://github.com/TongjiSuperPower/sp_radar_26/blob/8ffe732/src/main/decision/src/decision.cpp#L26)、[`:364` 覆盖与 0.5 s](https://github.com/TongjiSuperPower/sp_radar_26/blob/8ffe732/src/main/decision/src/decision.cpp#L364)
- 西交（`1ee1e10`）：[`main_event_loop.py:626` 融合](https://github.com/XJTU-RMV/Radar26/blob/1ee1e10/main_event_loop.py#L626)、[`params.yaml:38` 1.0 s](https://github.com/XJTU-RMV/Radar26/blob/1ee1e10/config/params.yaml#L38)
- TCR（`adcf312`）：[`position_fusion.py:87` 0.2/10/0.5](https://github.com/coach114514/TCR-RM2026-Radar-OpenSource/blob/adcf312/ground_vision/position_fusion.py#L87)
- 华南农（`8c2a225`）：[`serial_referee.cpp:1198` 隔一拍](https://github.com/SCAU-RM-NAV/SCAU_RADAR_26/blob/8c2a225/serial/src/serial_referee.cpp#L1198)
- 星云 EGA（`f7869c6`）：[`params.yaml:230` 0.25 s](https://github.com/Ryaxwn7/EGA_Radar_Algorithm/blob/f7869c6/Main_ws/config/params.yaml#L230)
- 元仓库：[`SEU-CX-RM-Solaris/rm-open-source-leaderboard`](https://github.com/SEU-CX-RM-Solaris/rm-open-source-leaderboard)
- JGRadar 现状：`jg_radar/config/schema.py:269/277/278/279/281`、`jg_radar/fuse/arbiter.py:228/349/476`
