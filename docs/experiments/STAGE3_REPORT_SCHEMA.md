# 阶段 3 验收报告结构（`report.json`）↔ 真机记录模板 对照

> 研究员 · 2026-11-21。用途：`tools/stage3_accept.py` 产出的 `report.json` 必须能**原样落进**
> [REAL_RUN_TEMPLATE.md](REAL_RUN_TEMPLATE.md) 的八节，不允许出现「工具一套、模板一套」。
> 字段名可以商量，**语义和约束不能动**；要改先改本文件，再改工具。

## 0. 三条硬约束

1. **三态**：每项结果只能是 `pass | fail | blocked`。设备不在位、真值缺失、标定缺失 ⇒ `blocked`，
   **绝不许 `pass`**。`blocked` 必须带 `reason`。
2. **等级**：报告顶层 `evidence_grade ∈ {A,B,C,D}`，取所有已判定项里**最低**的那一级。
   没有真设备时上限是 **C**（合成输入）或 **D**（未跑）；`A` 只在「设备在位 + 真实信号 + 真值齐全」时出现。
3. **故障注入**：每项判据必须有「弄坏它」的用例，且工具在坏输入下判 `fail`（不是 `blocked`、更不是 `pass`）。
   过不了故障注入的验收器不算验收器。至少要覆盖：串口写失败、同帧 ≥2 块装甲、`camera.yaml` 主程序加载失败、主循环卡顿 0.2s。
4. **可回溯**：`pass` 必须同时给出 `value`、`threshold`、`artifacts[]`（路径 + sha256）与 `grade`。
   缺任何一个，该项按 `fail` 处理（不是 `blocked`，因为「拿不出证据」本身就是不通过）。

## 1. 顶层结构

```json
{
  "schema": "stage3-accept/1",
  "generated_at": "2026-11-21T01:55:00+08:00",
  "commit": "f6e0386",
  "dirty_files": 0,
  "config_sha256": { "default.yaml": "...", "camera.yaml": "..." },
  "evidence_grade": "C",
  "verdict": "blocked",
  "checks": [ ... ],
  "devices": { ... },
  "artifacts": [ ... ],
  "caveats": [ "..." ],
  "reproduce": { "commands": [ "..." ] }
}
```

`verdict` 聚合规则：任一 `fail` ⇒ `fail`；全部 `pass` ⇒ `pass`；否则 `blocked`。

## 2. 四项判据（与 goal 完全一致，不许自创）

| `checks[].id` | `criterion`（写进报告原文） | `threshold` | 需要的证据 |
|---|---|---|---|
| `radio_accuracy` | `0x0A01` 逐次判定 `d<0.8m` 比例 | `>= 0.90`；**样本 ≥30**；**每个真值段 ≥1 个采样点**，否则 `blocked` | 上报序列 + 真值轨迹 + 分母（有效上报次数） |
| `radio_uplink` | `0x0305` 实发频率与断流 | `>= 4.7 Hz`、任意 1s 窗口 `<= 5 帧`、无 `gap > 0.5s` | 发送日志（时间戳 + **每行 `write_ok`**）+ provider/端口信息；**缺 `write_ok` 按 fail** |
| `vision_calibration` | 标定四要件齐全且主程序可加载 | `camera.name`（型号，非 `unselected`）+ `camera.serial` + `camera.calibrated_at` + `transform.reproj_rms_px` 实测；**`RadarConfig.from_dict` 必须接受同一份 YAML** | 标定文件 + 棋盘图 + 相机序列号 + 加载成功的输出 |
| `fusion_switch` | 换源时间与跳变 | `<= 0.4 s`、跳变 `<= 150 cm` | 融合输出（含 `source`/`observed_at`）+ 真值 |

每项固定字段：

```json
{
  "id": "radio_accuracy",
  "status": "blocked",
  "reason": "无真值轨迹：未接 Pluto / 无已知点录制",
  "value": null,
  "threshold": ">= 0.90",
  "grade": "D",
  "artifacts": [],
  "note": "合成输入只能跑通逻辑，不能判 pass"
}
```

### 2.1 三项新增自证字段（2026-11-21 故障注入复盘后）

| 判据 | 必须出现在 `checks[].value` 里的字段 | 判据 |
|---|---|---|
| `radio_uplink` | `frames_sent`、`frames_written`、`write_ok_ratio` | 三者一致（`frames_sent == frames_written`、ratio 为 1）；日志行缺 `write_ok` 按 `fail` |
| `vision_calibration` | `config_loads: true`、`config_load_error`（为空） | 用运行时同一个 `load_config` 加载同一份 YAML 成功；否则 `fail` |
| `radio_accuracy` | `samples`、`truthed_segments`、`segments_covered` | `samples >= 30` 且 `segments_covered == truthed_segments`；不满足 `blocked` |

**标定序列号与日期放哪（2026-11-21 定稿）**：`camera.serial`、`camera.calibrated_at` 已加为
`CameraConfig` 的两个可选字段（默认空串），验收器与运行时**读同一份 schema**，不再用旁证文件，
也不再往 YAML 里塞 `calibration:` 节点（运行时会拒绝，`d06776c` 的验收器就是这么假通过的）。

**`--referee-log` 行格式必须自证**：每行至少包含 `t`、`slot`、`write_ok`（bool）、`write_error`、
`write_mode`；`delivered` 固定写 `null` 并注明「本地观测不到，不代表裁判系统收到」。
**缺 `write_ok` 的行按 `fail`**——验收器不能把「没写成功」当成「没这个字段所以不管」。

## 3. 八节映射（模板 → 报告字段）

| 模板节 | 报告字段 | 必填 | 说明 |
|---|---|---|---|
| §1 基本信息 | `generated_at`、`operator`、`reproduce.commands`、`verdict`、`caveats` | ✅ | **`operator` 必须非空**（谁跑的就是谁，不许留空）；目标一句话写在 `goal` |
| §2 设备与序列号 | `devices[]` | ✅ | 每台：`kind/model/serial/transport/present`；`present=false` 必须让相关项 `blocked` |
| §3 软件与配置指纹 | `commit`、`dirty_files`、`config_sha256`、标定四项 | ✅ | 标定四项可放顶层 `calibration` 或 `checks[vision_calibration].value`，但**四项必须齐全**；缺一项 ⇒ `blocked`，实测值超限 ⇒ `fail` |
| §4 原始数据落盘 | `artifacts[]` | ✅ | 每项 `kind/path/sha256/duration_s`；`pass` 必需的 artifact 缺 sha256 ⇒ 该项 `fail` |
| §5 执行步骤与结果 | `checks[]` | ✅ | 每项一条；`steps[]` 可另存命令输出，但**判定只认 `checks`** |
| §6 阶段 3 验收映射 | `checks[].id` 四项 + `verdict` | ✅ | 顺序固定：radio_accuracy → radio_uplink → vision_calibration → fusion_switch |
| §7 反例与风险 | `caveats[]` | ✅ | **非空**；至少写「本次不能证明什么」与「已知混杂因素」 |
| §8 复现包 | `reproduce.commands` + `artifacts` | ✅ | 命令必须**能原样复制运行**：只写实际给过的参数，不许留 `--radio-log  --truth` 这类空 flag |

## 4. 我会逐条核的检查表

- [ ] 设备不在位时，四项是否全为 `blocked`，且 `verdict` 不是 `pass`
- [ ] `blocked` 项是否都带 `reason`（空字符串不算）
- [ ] `pass` 项是否都有 `value`、`threshold`、`artifacts[]`（含 sha256）、`grade`
- [ ] 顶层 `evidence_grade` 是否等于各项等级里的**最低**级
- [ ] 没有真设备时 `evidence_grade` 是否 ≤ `C`
- [ ] `caveats` 是否非空
- [ ] `commit` / `config_sha256` 是否与 `git rev-parse HEAD`、`sha256sum config/*.yaml` 对得上
- [ ] 报告能否直接填满模板八节（不许留「见聊天记录」这种指代）

## 4.1 首次核对发现的两处缺口（2026-11-21，`d06776c`）

1. `operator` 是空字符串——模板 §1 要求写操作者，工具没填。
2. `reproduce.commands` 里是 `--radio-log  --truth  --fusion-log  --switch-event `（空 flag），
   照抄执行会立刻报错，不符合「能原样复制运行」。

其余六项通过。另有一处语义偏差已由本文件放开：标定四项实际放在
`checks[vision_calibration].value` 里，语义完整，不再要求搬去顶层。

**两处已于 `2d64058` 修复并复核**：`operator` 走 `--operator` → `git config user.name` → `"unset"` 回退链
（并记 `operator_source`，`git` 只代表仓库主人、不代表现场操作者）；复现命令只拼生效参数、
解释器写实际跑起来的 `.venv/bin/python`（本机没有 `python`）。修复后基线复跑 **14/14 通过**（见台账 R24）。

**标定 caveat 是条件性的**：只有当 `vision_calibration` 被判 `pass` 或 `fail` 时才要求出现
「配置合格 ≠ 真机标定做过；A 级签核补这一层」；`blocked` 时没有判定可收窄，硬加只是噪音。

**退出码**：工具在 `verdict != pass` 时返回 1（含 `blocked`），这是「未通过」而非崩溃；
CI/脚本请按非零=未通过处理，不要当成执行失败。

## 5. 现在就该能跑出来的东西（无设备基线）

设备不在位时，工具应当输出 `verdict: "blocked"`、四项全 `blocked`、`evidence_grade: "D"`（或合成输入跑通时为 `C`），
并且 `reason` 写明缺什么（Pluto 不在 USB / 无海康 / 无标定 / 无真值）。
这份基线本身可以进台账，作为「阶段 3 未开始」的正式证据——比一句「还没开始」有信息量。
