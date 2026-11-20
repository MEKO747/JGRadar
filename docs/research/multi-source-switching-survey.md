# 多源定位切换：开源实现对照（2026-11-20）

调研问题：JGRadar 用「优先级阶梯 + 驻留」仲裁无线电与视觉，成熟项目怎么做同类决定？
范围：robot_localization、fuse、Autoware 定位栈、Apollo MSF、RoboMaster 开源雷达（TCR / HUST）。
方法：全部拉源码 grep 参数名与判定分支，链接钉到 commit，不引用二手博客。

---

## 0. 结论速览

1. **本轮查的五个项目里，没有一家用「优先级阶梯 + 时间驻留」做多源定位切换。** 主流只有两条路：
   - **不切换，融合**：robot_localization（协方差加权 + 马氏距离逐测量剔除）、fuse（因子图 + 鲁棒核降权）。
   - **切换的是「健康等级 / 谁在跑」，不是「信哪个值」**：Autoware（诊断等级 + 前置条件开关）、Apollo（GNSS 质量档 × LiDAR 状态 → 融合解 + 状态机）。
2. **最接近我们的同域实现是 TCR RM2026 雷达站**（地面视觉 + 无线电，仓库文档原话「新鲜信息波优先、地面视觉回退」）。它用的是**三段新鲜窗（0.3 / 0.2 / 0.5 s）+ 超期不丢、标 `stale` 位**，没有驻留、没有跳变门槛。
3. **过期数据的处理分歧才是关键**：TCR 与 Apollo 在数据超期后**继续给值并标注年龄/状态**，把取舍交给下游；我们则在 `max_position_error_cm` 处**直接否决**（`jg_radar/fuse/arbiter.py:349`）。这一步把「可用性」和「精度」压成了一条阈值——也正是 `radio_stale_s=10s` 名存实亡、以及 0.5 s 规则与 1 s 否决之间那 0.5 s 窗口的根因。
4. **滞回的等价物不是 dwell**：Autoware 用「warn/error 两级诊断计数」、HUST 用「帧数生命周期」（`life_span_init=20` 帧）、Apollo 用「多级时间阈值 + std 分级」。都不需要「连续 0.3 s 才换源」。
5. 延迟处理有成熟范式：**打戳（acquisition time）+ 固定偏移补偿 + 窗口内回填/超窗丢弃**，见 robot_localization 的 `smooth_lagged_data`、fuse 的 `transaction_timeout`、Autoware 的 `pose_additional_delay`、TCR 的 `received_at` 本机时钟。

---

## 1. 对照总表

| 项目 | 切换机制 | 触发条件 | 有没有滞回 | 阈值（默认值） | 怎么处理延迟 | 怎么处理两源冲突 |
|---|---|---|---|---|---|---|
| **robot_localization**（ROS 2 分支） | 不切换：所有输入进同一个 EKF/UKF，按协方差加权 | 无「失效判定」；某路不来就只是没有更新 | 无 | 剔除门限：`odom0_pose_rejection_threshold`=5.0、`odom0_twist_rejection_threshold`=1.0、`imu0_*_rejection_threshold`=0.8（默认 `max`＝不剔除）；`sensor_timeout`=1/30 s（ROS 2 里只作 `control_timeout` 默认值，不剔除传感器） | `transform_time_offset`（固定时延补偿，默认 0）；`smooth_lagged_data`+`history_length`：迟到测量回填历史、重算 | 靠协方差与马氏距离，异常测量**逐条**丢弃/降权；不投票、不驻留 |
| **fuse** | 不切换：因子图 / 固定滞后平滑器 | 无失效判定；某模型不发事务，图中就没有它的约束 | 无 | `lag_duration`=5.0 s（代码默认，教程示例 0.5 s）、`transaction_timeout`=0.1 s（代码默认，教程 0.01 s；太晚的事务直接丢）、`optimization_frequency`=20 Hz | 以测量时间戳建事务；早于窗口起点的直接忽略；窗口内迟到可重优化 | 鲁棒核（`loadLossConfig(nh,"loss")`，Huber/Cauchy/Tukey 等 13 种）**降权**而非丢弃 |
| **Autoware** `pose_estimator_arbiter` + `ekf_localizer` | 多个定位器并行，用 stop/resume 决定**谁在跑**；最终只融合在跑的那些 | 前置条件（地图区域、PCD 密度）；EKF 侧是逐测量门限 + 缺数据计数 | 无时间滞回 | `pose_gate_dist`=49.5（马氏，10⁻¹⁰ 显著性）；`pose_no_update_count_threshold_warn/error`=50/100 @50 Hz＝**1 s / 2 s**；`localization_error_monitor` warn/error 椭圆 1.2/1.5 m（横向 0.25/0.3 m） | `pose_additional_delay`（0）、`pose_smoothing_steps`=5、`max_pose_queue_size`=5；`pose_instability_detector` 跳变容差 0.11/0.11/0.5 m、0.0175 rad | **缺数据 ≠ 异常**：注释明说空 tick 通过门限、由 `no_update_count` 上报；两个都坏时降级诊断，不互换源 |
| **Apollo MSF** | 不切换源：GNSS 质量档 × LiDAR 状态 × 位置 std 三档 → 融合解 + `LocalizationIntegState` | GNSS/LiDAR 超时丢数、std 分层、IMU 缺失/延迟分级 | 无显式滞回；靠多级阈值与状态机 | `bestgnsspose_loss_time_threshold`=**2.0 s**、`lidar_loss_time_threshold`=**2.0 s**、`localization_std_x/y_threshold_1/2`=0.15/0.3 m、IMU missing 0.1/0.05/0.01 s、IMU delay 0.1/0.05/0.02 s | IMU 延迟三级阈值独立上报；融合结果带 `measurement_time` 与 `state_message` | 输出**组合状态** `MSF_SOL_LIDAR_GNSS / LIDAR_X / X_GNSS / …` + OK/WARNING/ERROR/CRITICAL/FATAL，把冲突原因写进 `LocalizationStatus.state_message` |
| **TCR RM2026 雷达站**（视觉+无线电，同域） | 四段式阶梯：主源新鲜 → 备源新鲜 → 主源兜底窗 → 任取最新并标 `stale` | 按 `now - received_at` 与三档窗口比较；仅 level 3（信息波）融合坐标 | 无驻留、无跳变门槛 | `MAIN_PRIMARY_S`=**0.3**、`AUX_PRIMARY_S`=**0.2**、`MAIN_FALLBACK_S`=**0.5**；`PUBLISH_HZ`=10；`MAX_RELAY_FUTURE_SKEW_S`=1.0 | `received_at` 用本机时钟；未来时间戳 >1 s 拒收；重连代次隔离 + 过期清理 | 超期**不丢**：继续给最新值并置 `stale`，打包 `source`+`age_ms`+`stale_mask` 交下游决定 |
| **HUST Radar 2024**（双相机+LiDAR） | 不切换：相机检测框 → 框内点云确认 → 赛场坐标；无跨源仲裁 | 框内无点、聚类为空 → 该帧丢弃 | 帧数生命周期：`life_span_init`=**20 帧**，连续未检测递减，<0 置 `trust=False` | 生命周期 20 帧（约 20 个检测帧） | 无专门处理 | 不解决冲突，只做「视觉框 + 点云确认」；不可信数据**仍发裁判系统、不发哨兵** |

---

## 2. 逐个拆解

### 2.1 robot_localization：用协方差投票，不用优先级

- 所有传感器进同一个滤波器，`*_config` 布尔向量决定每一路贡献哪些自由度（`odom0_config` 等），没有「当前生效源」这个状态。
- 异常测量靠**马氏距离**逐条剔除：参数按话题名生成，`<topic>_pose_rejection_threshold` / `<topic>_twist_rejection_threshold`，缺省 `numeric_limits::max()` 即不剔除；示例配置给的是 5.0/1.0（odom）、0.8（IMU）。
- 迟到的传感器靠 `smooth_lagged_data: true` + `history_length`：把新测量插回历史时间点重算，而不是硬等或丢掉。
- 注意一个易误读点：ROS 2 分支的 `sensor_timeout`（默认 1/30 s）只被读来当 `control_timeout` 的默认值，**没有**用来判断某路传感器是否超时——也就是说它连「超时」这个概念都不暴露给用户。

对我们的含义：它不做切换，所以没有「切换抖动」问题；代价是坏源不会被彻底排除，只能靠协方差和门限。我们如果继续用阶梯，就得自己补上它免费提供的东西：逐测量质量门限。

### 2.2 fuse：迟到的事务直接丢，异常值降权不丢

- 图优化框架，测量以「事务（transaction）」按时间戳入图；`transaction_timeout`（代码默认 0.1 s，教程示例 0.01 s）是**等待运动模型生成的上限**，超过就整笔事务忽略——宁可少一条约束，也不让晚到的数据搅乱图。
- 固定滞后窗口 `lag_duration`（代码默认 5.0 s，教程示例 0.5 s）：变量在窗口内可被重优化，滑出窗口才边缘化。
- 鲁棒核按 sensor model 配置（`loadLossConfig(nh, "loss")`），`fuse_loss` 提供 Huber/Cauchy/Tukey/Geman-McClure 等 13 种。
- 它没有「传感器失效 → 换源」逻辑（grep `fallback`/`failover` 无命中）；某路不发事务，图里就没有它，运动模型继续推。

对我们的含义：**「过期」和「异常」是两件事**，fuse 用两套机制分别处理（丢事务 / 降权）。我们现在用同一条年龄阈值同时承担两件事，容易两头不讨好。

### 2.3 Autoware：切「谁在跑」，不切「信谁」

- `pose_estimator_arbiter` 并行启多个定位器（NDT / YabLoc / eagleye / ArTag / landmark），按规则 stop/resume 以省算力并避免坏输出污染 EKF。README 明确：**当前只实现了「全部启用」这一条规则**，切换规则要做成插件；示例规则是 PCD 密度和地图区域，属于**前置条件**，不是运行时故障。
- `ekf_localizer` 里把两件事分开：逐测量门限 `pose_gate_dist`=49.5（马氏距离，显著性 10⁻¹⁰，对应位置/姿态偏差门限），以及**缺数据计数** `pose_no_update_count_threshold_warn/error`=50/100。源码注释写得很直白：*"A tick without new measurements passes the gates: missing input is reported by no_update_count."* 50 Hz 下 50/100 计数 = **1 s / 2 s**。
- 健康度还有独立的两级门限：`localization_error_monitor` 的 warn/error 椭圆 1.2/1.5 m（横向 0.25/0.3 m，`scale`=3.0），`pose_instability_detector` 的跳变容差（纵向/横向 0.11 m、垂向 0.5 m、角度 0.0175 rad，0.5 s 周期）。

对我们的含义：Autoware 的「1 s warn / 2 s error」正好说明**超时阈值应该分级并且给下游看**，而不是内部一刀切。我们的 `radio_stale_s=10s` 若只作「彻底 MISSING 线」、另设 0.5/1.0 s 两级对外口径，结构上就与它同构。

### 2.4 Apollo MSF：组合状态 + 原因文本

- 判定是一个二维矩阵：GNSS 质量档（`NARROW_INT`/`FLOATCONV`/`SINGLE`/…）× LiDAR 状态（正常/无输出），再按位置 std 三档（<0.15、<0.3、其余）落到 `MSF_SOL_LIDAR_GNSS` / `MSF_SOL_LIDAR_X` / `MSF_SOL_X_GNSS` / `MSF_SOL_X_X` 等组合，同时给 `OK`/`WARNING`/`ERROR`/`CRITICAL_ERROR`/`FATAL_ERROR` 和一句人话原因。
- 失联阈值都是**分级**的：GNSS 丢失 2.0 s、LiDAR 丢失 2.0 s、IMU 缺失 0.1/0.05/0.01 s、IMU 延迟 0.1/0.05/0.02 s。
- 它不切换到「另一个定位源」——GNSS 坏了只是组合从 `LIDAR_GNSS` 变成 `LIDAR_X`，状态降一级；两个都坏才 FATAL，并明确写「Current Localization Is NOT Available」。
- 补充一个容易误解的点：Apollo 的 MSF 与 RTK 是**两个独立模块、各自一份 DAG**（`dag_streaming_msf_localization.dag` / `dag_streaming_rtk_localization.dag`），部署期二选一，**运行期不互切**。所以它的「切换」是配置级，不是运行级。
- 结果通过 `LocalizationStatus`（`fusion_status` + `state_message`）持续发布。

对我们的含义：**「现在用的是哪几个源的组合」比「现在轮到哪一档」信息量大得多**。我们对外只报一个坐标，丢掉了「这条坐标是哪来的、多旧、可不可信」——这正是 0.5 s 规则冲突里下游无从判断的原因。

### 2.5 TCR RM2026：同域、几乎同构，但选了另一条路

官方变更日志原话：「坐标源采用『新鲜信息波优先、地面视觉回退』的策略；无线电断线或坐标过期时自动回到视觉结果。」判定只有四段：

1. 主源（无线电信息波）`age ≤ 0.3 s` → 用主源，`stale=False`
2. 备源（地面视觉）`age ≤ 0.2 s` → 用备源
3. 主源 `age ≤ 0.5 s` → 回到主源兜底（注意主源兜底窗比备源窗更长）
4. 都不满足 → 取三路里最新的一个，`stale=True`

关键细节：

- **超期不丢**，只标 `stale`；输出按车打包 `source`（2 bit）+ `age_ms` + `stale_mask`，10 Hz 发布。下游自己决定「stale 的能不能用」。
- 没有驻留、没有跳变门槛，也没看到「连续 N 秒才换」——换源是逐帧独立的。
- 时间用**接收时刻**（本机单调钟）算年龄，`MAX_RELAY_FUTURE_SKEW_S=1.0` 拒收未来时间戳，重连有代次隔离，防旧连接污染。
- 只有 level 3（信息波）才融合坐标，1/2 级直接 `return None`。

对我们的含义：这是「同题不同解」最有价值的样本——它把**不确定性外化**（age+stale 打包给下游），我们把不确定性**内化**（在仲裁器里一刀否决）。两种都能自洽，但前者天然不会和「0.5 s 记错」这类外部规则打架。

### 2.6 HUST Radar 2024：帧数生命周期 + 消费者分级

- 每辆车一个 `life_span`（初值 20 帧），本帧检测到就重置满，未检测到递减，<0 置 `trust=False`。这是纯帧数滞回，不需要时钟。
- 更有意思的是注释：不可信的车「**不发给哨兵，但是还是发给裁判系统**」——同一个数据对不同消费者有不同容忍度。
- 融合只是「检测框 + 框内点云确认」，没有跨源仲裁；框内无点直接丢，不做保持。

对我们的含义：「同一份状态、两套口径」在实战队里是常态。我们完全可以让上报链路（受 0.5 s 规则约束）和内部追踪（可容忍 1 s+）用不同阈值。

---

## 3. 我们 vs 他们

| 我们的机制 | 开源等价物 | 差异 / 风险 |
|---|---|---|
| 优先级阶梯 `radio_fresh > vision_measured > radio_stale > …`（`schema.py:269`） | TCR 三段新鲜窗 + 兜底窗；Apollo 组合状态 | 我们把「源」和「档」混在一个序里；TCR 只按年龄分窗，Apollo 只报组合 |
| `dwell_s=0.3` 换源驻留、`reacquire_s=0.8` 跳变加驻 | **没有对应物** | 开源里没有时间驻留；Autoware 干脆按前置条件预切换，fuse 用鲁棒核吸收抖动 |
| `switch_jump_cm=150` 跳变门槛 | Autoware `pose_instability_detector`（0.11 m 等容差） | 同量级，但它是**诊断报警**，不是禁止换源 |
| `max_position_error_cm=500` 硬否决（≈1.0 s 保质期） | Apollo `localization_std_*` 分级；TCR `stale` 标记 | 我们把「过期」变成「不可用」，且把 10 s 的 `radio_stale_s` 架空 |
| `vision_watchdog_s=0.5` | Autoware `no_update` warn/error；HUST 20 帧生命周期 | 我们只有 0.5 s 一个点，没有 warn/error 两级 |
| 只输出一个坐标 | TCR 输出 `source+age_ms+stale_mask`；Apollo 输出 `fusion_status+state_message` | 下游无法区分「刚算的」和「1 s 前的」 |

---

## 4. 三条能直接落地的建议

### 建议 1：过期不否决，改为「降级 + 打标」，把 0.5 s / 1.0 s 拆成两套口径

- 依据：TCR `stale_mask`+`age_ms`、Apollo `LocalizationStatus.state_message`、HUST「发裁判系统不发哨兵」。
- 改动：`jg_radar/fuse/arbiter.py:349` 的 `bound > max_position_error_cm` 不再 `return None`，改为返回 `(position, age_s, stale)`；`radio_stale` 档真正生效到 `radio_stale_s=10s`；`max_position_error_cm` 只作「彻底 MISSING 线」。
- 口径拆分：**0.5 s 起对外标 `stale`/记错误**（对齐 2027 规则），**1.0 s 起才不再作为换源候选**（内部仍可兜底）。这样那段 0.5 s 窗口不再是「算错还报旧值」，而是「报旧值且明确标了年龄」。
- 验收：构造「无线电 0.6 s 断流」回放，断言输出带 `age_s≥0.5` 且 `stale=True`，且上报层按 stale 计数；「1.2 s 断流」断言 `radio_stale` 不再参与换源但坐标仍可读。

### 建议 2：可用性（age）与质量（std/马氏）分成两条轴，阈值分级

- 依据：Autoware `pose_gate_dist=49.5` + no_update 50/100（1 s/2 s）+ 椭圆 1.2/1.5 m；Apollo std 0.15/0.3 m + 丢失 2.0 s；robot_localization 马氏阈值 5.0/1.0/0.8。
- 改动：`max_position_error_cm` 单阈值拆成 `warn=250 cm`（≈0.5 s）与 `error=500 cm`（≈1.0 s）；给视觉补一个逐测量门限（位置 std 或马氏距离，参考 Autoware 的 `pose_gate_dist` 语义），避免「视觉活着但离谱」时仍压过无线电；`max_robot_speed_cm_s=500` 是猜的，用实测最大速度替换。
- 验收：用现有回放 fixture 注入「视觉坐标跳 3 m」样本，断言仲裁不切到视觉或至少在 1 帧内回退；用真实录制测最大速度，回填 `max_robot_speed_cm_s` 并把两条阈值重算一遍。

### 建议 3：延迟用「打戳 + 固定偏移 + 窗口回填/超窗丢弃」，不要用驻留吸收

- 依据：robot_localization `transform_time_offset` + `smooth_lagged_data`/`history_length`；fuse `transaction_timeout`（0.1 s 默认）+ `lag_duration`（5.0 s 默认）；Autoware `pose_additional_delay`、`pose_smoothing_steps=5`；TCR 用 `received_at` 本机时钟 + 1 s 未来偏斜拒收。
- 改动：无线电报文同时保留 `measurement_time`（裁判系统时钟）与 `received_at`（本机单调钟）；仲裁年龄一律用 `received_at`，回放/对表用 `measurement_time`；加一个 `radio_time_offset` 固定时延补偿（学 `transform_time_offset`）；超过窗口的迟到帧直接丢，不参与预测。
- 验收：录制一段带已知处理时延的无线电数据，注入 ±200 ms 抖动，断言换源次数与时延补偿前对比下降；迟到 1 s 的报文断言被丢且不改变当前档位。

---

## 5. 不确定 / 没查的部分

- **RoboCup 队伍代码**本轮没拉源码（B-Human 等仓库体量大），所以不写进结论表。坊间说法是「多机观测进同一个滤波器」，但没有在本轮取证，不作为依据。
- Autoware `pose_estimator_arbiter` 的示例切换规则（PCD 密度 / 地图区域）只在 README 与 `example_rule` 里，主线未启用；不要把它当成「运行时故障切换」的现成实现。
- TCR 的 `stale` 位在**下游**怎么用（是否直接上报裁判系统）没有在本仓库闭环，只确认到打包发布为止。
- 各项目阈值都是其自身场景标定的（Apollo 是车规厘米级、Autoware 是城市道路米级），数值不能照抄，只能照「分级 + 打标」的结构。

---

## 6. 参考（commit 固定链接）

- robot_localization（`b1e3731`）：[`ros_filter.cpp:1503` 剔除门限](https://github.com/cra-ros-pkg/robot_localization/blob/b1e3731/src/ros_filter.cpp#L1503)、[`ros_filter.cpp:1223` `smooth_lagged_data`](https://github.com/cra-ros-pkg/robot_localization/blob/b1e3731/src/ros_filter.cpp#L1223)、[`params/ekf.yaml:127` 示例阈值](https://github.com/cra-ros-pkg/robot_localization/blob/b1e3731/params/ekf.yaml#L127)
- fuse（`8e3f0a1`）：[`fixed_lag_smoother.cpp:440` `transaction_timeout`](https://github.com/locusrobotics/fuse/blob/8e3f0a1/fuse_optimizers/src/fixed_lag_smoother.cpp#L440)、[`fixed_lag_smoother.cpp:538` 早于窗口起点忽略](https://github.com/locusrobotics/fuse/blob/8e3f0a1/fuse_optimizers/src/fixed_lag_smoother.cpp#L538)、[`pose_2d_params.h:94` 鲁棒核](https://github.com/locusrobotics/fuse/blob/8e3f0a1/fuse_models/include/fuse_models/parameters/pose_2d_params.h#L94)、[`fixed_lag_smoother_params.h:113` 代码默认值](https://github.com/locusrobotics/fuse/blob/8e3f0a1/fuse_optimizers/include/fuse_optimizers/fixed_lag_smoother_params.h#L113)、[`getting_started.rst:105` 参数说明](https://github.com/locusrobotics/fuse/blob/8e3f0a1/fuse_doc/doc/getting_started.rst#L105)
- Autoware Universe（`8361f0e`）：[`pose_estimator_arbiter/README.md`](https://github.com/autowarefoundation/autoware_universe/blob/8361f0e/localization/autoware_pose_estimator_arbiter/README.md)、[`example_rule/README.md` 切换规则](https://github.com/autowarefoundation/autoware_universe/blob/8361f0e/localization/autoware_pose_estimator_arbiter/example_rule/README.md)、[`localization_error_monitor.param.yaml`](https://github.com/autowarefoundation/autoware_universe/blob/8361f0e/localization/autoware_localization_error_monitor/config/localization_error_monitor.param.yaml)、[`pose_instability_detector/config`](https://github.com/autowarefoundation/autoware_universe/blob/8361f0e/localization/autoware_pose_instability_detector/config/pose_instability_detector.param.yaml)
- Autoware Core（`15408eb`）：[`ekf_localizer.param.yaml`](https://github.com/autowarefoundation/autoware_core/blob/15408eb/localization/autoware_ekf_localizer/config/ekf_localizer.param.yaml)、[`ekf_localizer.cpp:198` 缺数据注释](https://github.com/autowarefoundation/autoware_core/blob/15408eb/localization/autoware_ekf_localizer/src/ekf_localizer.cpp#L198)
- Apollo（`d53aa3d`）：[`online_localization_expert.cc:155` 丢失阈值](https://github.com/ApolloAuto/apollo/blob/d53aa3d/modules/localization/msf/local_integ/online_localization_expert.cc#L155)、[`online_localization_expert.cc:182` 组合状态矩阵](https://github.com/ApolloAuto/apollo/blob/d53aa3d/modules/localization/msf/local_integ/online_localization_expert.cc#L182)、[`localization_gflags.cc:149` 阈值默认值](https://github.com/ApolloAuto/apollo/blob/d53aa3d/modules/localization/common/localization_gflags.cc#L149)、[`dag_streaming_msf_localization.dag` 与 RTK 各自独立](https://github.com/ApolloAuto/apollo/blob/d53aa3d/modules/localization/dag/dag_streaming_msf_localization.dag)
- TCR RM2026（`adcf312`）：[`coordinate_fusion.py:52` 三段窗口](https://github.com/coach114514/TCR-RM2026-Radar-OpenSource/blob/adcf312/radar/CombatRadarSdr/coordinate_fusion.py#L52)、[`coordinate_fusion.py:271` 选择逻辑](https://github.com/coach114514/TCR-RM2026-Radar-OpenSource/blob/adcf312/radar/CombatRadarSdr/coordinate_fusion.py#L271)、[`CHANGES_FROM_HKUST.md:23` 官方描述](https://github.com/coach114514/TCR-RM2026-Radar-OpenSource/blob/adcf312/CHANGES_FROM_HKUST.md#L23)
- HUST Radar 2024（`f628641`）：[`Car/Car.py:10` 生命周期](https://github.com/HUSTLYRM/Hust-Radar-2024/blob/f628641/Car/Car.py#L10)、[`Car/Car.py:60` trust 衰减](https://github.com/HUSTLYRM/Hust-Radar-2024/blob/f628641/Car/Car.py#L60)
- JGRadar 现状：`jg_radar/config/schema.py:269`（优先级阶梯）、`jg_radar/config/schema.py:276`（`radio_fresh_s`）、`:277`（`radio_stale_s`）、`:293`（`max_position_error_cm`）、`jg_radar/fuse/arbiter.py:349`（硬否决）
