# 阶段 3 验收报告结构（`report.json`）↔ 真机记录模板 对照

> 研究员 · 2026-11-21。用途：`tools/stage3_accept.py` 产出的 `report.json` 必须能**原样落进**
> [REAL_RUN_TEMPLATE.md](REAL_RUN_TEMPLATE.md) 的八节，不允许出现「工具一套、模板一套」。
> 字段名可以商量，**语义和约束不能动**；要改先改本文件，再改工具。

## 0. 三条硬约束

1. **三态**：每项结果只能是 `pass | fail | blocked`。设备不在位、真值缺失、标定缺失 ⇒ `blocked`，
   **绝不许 `pass`**。`blocked` 必须带 `reason`。
2. **等级**：报告顶层 `evidence_grade ∈ {A,B,C,D}`，取所有已判定项里**最低**的那一级。
   没有真设备时上限是 **C**（合成输入）或 **D**（未跑）；`A` 只在「设备在位 + 真实信号 + 真值齐全」时出现。
3. **可回溯**：`pass` 必须同时给出 `value`、`threshold`、`artifacts[]`（路径 + sha256）与 `grade`。
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
| `radio_accuracy` | `0x0A01` 逐次判定 `d<0.8m` 比例 | `>= 0.90` | 上报序列 + 真值轨迹 + 分母（有效上报次数） |
| `radio_uplink` | `0x0305` 实发频率与断流 | `>= 4.7 Hz`、任意 1s 窗口 `<= 5 帧`、无 `gap > 0.5s` | 发送日志（时间戳）+ provider/端口信息 |
| `vision_calibration` | 标定四要件齐全 | 序列号 + 重标日期 + `reproj_rms_px` 实测 + 文件 sha256 | 标定文件 + 棋盘图 + 相机序列号 |
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

## 3. 八节映射（模板 → 报告字段）

| 模板节 | 报告字段 | 必填 | 说明 |
|---|---|---|---|
| §1 基本信息 | `generated_at`、`reproduce.commands`、`verdict`、`caveats` | ✅ | 操作者写在 `operator`；目标一句话写在 `goal` |
| §2 设备与序列号 | `devices[]` | ✅ | 每台：`kind/model/serial/transport/present`；`present=false` 必须让相关项 `blocked` |
| §3 软件与配置指纹 | `commit`、`dirty_files`、`config_sha256`、`calibration{...}` | ✅ | 标定四项进 `calibration` 子对象，缺一项 ⇒ `vision_calibration` 为 `blocked` |
| §4 原始数据落盘 | `artifacts[]` | ✅ | 每项 `kind/path/sha256/duration_s`；`pass` 必需的 artifact 缺 sha256 ⇒ 该项 `fail` |
| §5 执行步骤与结果 | `checks[]` | ✅ | 每项一条；`steps[]` 可另存命令输出，但**判定只认 `checks`** |
| §6 阶段 3 验收映射 | `checks[].id` 四项 + `verdict` | ✅ | 顺序固定：radio_accuracy → radio_uplink → vision_calibration → fusion_switch |
| §7 反例与风险 | `caveats[]` | ✅ | **非空**；至少写「本次不能证明什么」与「已知混杂因素」 |
| §8 复现包 | `reproduce.commands` + `artifacts` | ✅ | 别人拿到 commit + 配置哈希 + 标定 + 数据 + 命令能重跑 |

## 4. 我会逐条核的检查表

- [ ] 设备不在位时，四项是否全为 `blocked`，且 `verdict` 不是 `pass`
- [ ] `blocked` 项是否都带 `reason`（空字符串不算）
- [ ] `pass` 项是否都有 `value`、`threshold`、`artifacts[]`（含 sha256）、`grade`
- [ ] 顶层 `evidence_grade` 是否等于各项等级里的**最低**级
- [ ] 没有真设备时 `evidence_grade` 是否 ≤ `C`
- [ ] `caveats` 是否非空
- [ ] `commit` / `config_sha256` 是否与 `git rev-parse HEAD`、`sha256sum config/*.yaml` 对得上
- [ ] 报告能否直接填满模板八节（不许留「见聊天记录」这种指代）

## 5. 现在就该能跑出来的东西（无设备基线）

设备不在位时，工具应当输出 `verdict: "blocked"`、四项全 `blocked`、`evidence_grade: "D"`（或合成输入跑通时为 `C`），
并且 `reason` 写明缺什么（Pluto 不在 USB / 无海康 / 无标定 / 无真值）。
这份基线本身可以进台账，作为「阶段 3 未开始」的正式证据——比一句「还没开始」有信息量。
