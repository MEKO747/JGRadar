"""类别映射适配层。

用开源权重起步绕不开这件事：三家的类别定义完全不同——EGA 用 0–14 的颜色
编码，shark 直接用 ``B1``–``B5``/``R1``–``R5`` 这类槽位名，TCR 是 15 类。
网络的输出只是一个整数，整数是什么意思由训练时决定。

**这张表必须来自配置，不能写死在代码里。** 写死意味着换一份权重就要改代码、
就要重新 review 检测层——而换权重本来应该只是改一行配置。这里只提供
"把配置形式的表变成可查询对象"的机制，不内置任何一份具体表（内置的那份
一旦和权重不匹配，会安静地把敌方 ID 认成己方的）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

__all__ = ["ClassMap"]


@dataclass(frozen=True)
class ClassMap:
    """``class_id → 语义标签`` 的映射。

    ``label`` 只用于日志与调试；下游的判断一律基于 ``has``/``ids``，别用
    字符串比较去猜语义。
    """

    names: Mapping[int, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", {int(k): str(v) for k, v in self.names.items()})

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def identity(cls, num_classes: int) -> "ClassMap":
        """不做映射，标签就是 ``cls0``/``cls1``/…，用于后端类别未知时。"""
        return cls({i: f"cls{i}" for i in range(num_classes)})

    @classmethod
    def from_spec(cls, spec: Mapping | Sequence[str] | None, *, num_classes: int = 0) -> "ClassMap":
        """从配置里的三种写法之一构造。

        * ``None`` → 恒等映射（``num_classes`` 个）；
        * 列表 ``["R1","R2",...]`` → 下标即 class_id；
        * 字典 ``{0: "R1", 3: "B2"}`` → 显式指定（键写成字符串也行，YAML 的
          键经常被解析成字符串）。
        """
        if spec is None:
            return cls.identity(num_classes)
        if isinstance(spec, Mapping):
            return cls({int(k): str(v) for k, v in spec.items()})
        return cls({i: str(v) for i, v in enumerate(spec)})

    # -- 查询 -------------------------------------------------------------
    def label(self, class_id: int) -> str:
        return self.names.get(int(class_id), f"cls{class_id}")

    def has(self, class_id: int) -> bool:
        return int(class_id) in self.names

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, class_id: object) -> bool:
        try:
            return int(class_id) in self.names  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def ids(self, *labels: str) -> list[int]:
        """标签 → class_id 列表。找不到的标签被跳过（不报错：多份权重混用时
        某个标签不存在是常态，不该让整条流水线停下来）。"""
        want = {str(s) for s in labels}
        return [cid for cid, name in sorted(self.names.items()) if name in want]

    def subset(self, *labels: str) -> "ClassMap":
        """只要这几个标签，其余丢弃。用于"这一级只关心装甲板数字"这类过滤。"""
        want = {str(s) for s in labels}
        return ClassMap({cid: n for cid, n in self.names.items() if n in want})

    def describe(self) -> str:
        items = ", ".join(f"{cid}→{n}" for cid, n in sorted(self.names.items()))
        return f"ClassMap({len(self.names)} 类：{items})"
