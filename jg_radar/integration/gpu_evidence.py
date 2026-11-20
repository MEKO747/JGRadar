"""证据自证：ORT 到底把多少个计算节点分给了哪个 provider。

## 为什么光有 ``get_providers()`` 不够

``session.get_providers()`` 说的是"哪些执行器建起来了"，**不是"哪些真的在算"**。
CUDA 执行器能建起来、但一个节点都没分给它（比如被图优化整段融合掉、或者算子全都
不支持），它照样排在返回列表第一位。于是"JSON 里写着 CUDA = 真在 GPU 上跑"这个推理
是不成立的——三个月后翻证据的人没法自证。

这里的做法：开一次 ORT profiling（一次性开销，不计入计时），跑一遍，然后统计
profile 里 ``cat == "Node"`` 事件各自的 ``provider`` 字段。这才是"这个模型实际在哪儿算"。
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

__all__ = ["parse_profile", "provider_node_counts"]


def parse_profile(path: str | Path) -> dict[str, int]:
    """统计 ORT profile 里每个 provider 分到的节点数。

    ORT 写的是**一个 JSON 数组**（不是 JSONL），但不同版本的格式在变，所以两种都认：
    整文件能 ``json.load`` 就按数组走，否则按逐行 JSON 走。
    """
    text = Path(path).read_text(encoding="utf-8")
    events: list[dict] = []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    else:
        events = loaded if isinstance(loaded, list) else [loaded]

    counts: dict[str, int] = {}
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        provider = (event.get("args") or {}).get("provider") or "unknown"
        counts[provider] = counts.get(provider, 0) + 1
    return counts


def provider_node_counts(
    model_path: str | Path,
    *,
    input_shape: tuple[int, ...],
    providers: list[str] | None = None,
    ort=None,
) -> dict[str, int]:
    """建一个带 profiling 的会话跑一次，返回 ``{provider: 节点数}``。

    失败不抛：返回 ``{}``——调用方据此报"无法自证"，而不是让整个测量挂掉。
    """
    if ort is None:
        import onnxruntime as ort  # noqa: PLC0415 —— 可选依赖

    path = Path(model_path)
    if not path.is_file():
        return {}
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    with tempfile.TemporaryDirectory() as workdir:
        options.profile_file_prefix = str(Path(workdir) / "profile")
        try:
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=providers or None)
            name = session.get_inputs()[0].name
            tensor = np.zeros(input_shape, dtype=np.float32)
            session.run(None, {name: tensor})
            profile_path = session.end_profiling()
        except Exception:  # noqa: BLE001 —— 自证失败就是结论本身，不该把测量打断
            return {}
        return parse_profile(profile_path)
