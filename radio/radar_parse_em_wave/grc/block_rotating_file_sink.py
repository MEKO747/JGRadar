# -*- coding: utf-8 -*-
"""按固定时长自动切片的复数 IQ 文件录制块。

本文件需要放在 grc 目录下，便于 GNU Radio Companion 的 Python Block 直接导入：
_source_code: 'from block_rotating_file_sink import blk'
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import numpy as np
from gnuradio import gr


DEFAULT_RECORD_PREFIX = Path("records")
"""默认录制目录，相对于当前 ROS2 包根目录。"""


def _find_package_root() -> Path:
    """根据自定义块所在路径查找 ROS2 源码包根目录。

    源码运行时模块通常位于 <workspace>/src/<package>/grc；
    安装后运行时模块通常位于 <workspace>/install/.../share/<package>/grc。
    """
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if parent.name == "src":
            # 源码树中，src 后的第一级目录就是当前 ROS2 包。
            package_name = module_path.relative_to(parent).parts[0]
            return parent / package_name

        if parent.name == "install":
            # 安装树中，从 share/<package> 恢复包名，再映射回工作空间源码树。
            install_relative_parts = module_path.relative_to(parent).parts
            try:
                share_index = install_relative_parts.index("share")
                package_name = install_relative_parts[share_index + 1]
            except (ValueError, IndexError):
                package_name = module_path.parent.parent.name
            return parent.parent / "src" / package_name

    # 独立运行仓库时没有标准 src/install 层级，回退到 grc 的上一级工程目录。
    return module_path.parent.parent


def _resolve_record_prefix(record_prefix: str | os.PathLike[str]) -> Path:
    """解析录制目录；相对路径统一基于 ROS2 源码包根目录。"""
    configured_path = Path(record_prefix).expanduser()
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (_find_package_root() / configured_path).resolve()


class blk(gr.sync_block):
    """将输入复数 IQ 连续写入按时间编号的 cfile 文件。

    每个文件最多写入 sample_rate * segment_seconds 个样本；达到上限后关闭当前文件，
    并使用当前日期时间生成新文件名继续录制。
    """

    def __init__(
        self,
        record_prefix=str(DEFAULT_RECORD_PREFIX),
        sample_rate=1e6,
        segment_seconds=60,
        filename_prefix="rx_wideband",
        dtype="complex64",
    ):
        """初始化循环录制块。

        参数:
            record_prefix: 录制文件目录；相对路径以 ROS2 源码包根目录为基准。
            sample_rate: 输入 IQ 采样率，用于计算每段文件样本数。
            segment_seconds: 单个文件的录制时长，单位秒。
            filename_prefix: 文件名前缀，最终文件名会附加日期时间。
            dtype: 文件保存格式，当前默认 GNU Radio cfile 常用的 complex64。
        """
        gr.sync_block.__init__(
            self,
            name="rotating_file_sink",
            in_sig=[np.complex64],
            out_sig=None,
        )

        self.record_prefix = str(_resolve_record_prefix(record_prefix))
        self.sample_rate = max(1.0, float(sample_rate))
        self.segment_seconds = max(1.0, float(segment_seconds))
        self.filename_prefix = self._sanitize_filename_part(str(filename_prefix))
        self.dtype = np.dtype(dtype)

        # 每段文件的目标样本数，达到该数量后立即切换新文件。
        self.samples_per_segment = max(1, int(round(self.sample_rate * self.segment_seconds)))
        self.samples_in_current_file = 0
        self.file_index = 0
        self.current_file = None
        self.current_path = ""

        os.makedirs(self.record_prefix, exist_ok=True)

    def work(self, input_items, output_items):
        """写入一段输入样本，并在达到单文件长度后自动切到下一个文件。"""
        _ = output_items
        in0 = np.asarray(input_items[0], dtype=self.dtype)
        offset = 0

        while offset < len(in0):
            self._ensure_file_open()

            remaining = self.samples_per_segment - self.samples_in_current_file
            take = min(remaining, len(in0) - offset)
            if take <= 0:
                self._rotate_file()
                continue

            in0[offset : offset + take].tofile(self.current_file)
            self.samples_in_current_file += take
            offset += take

            if self.samples_in_current_file >= self.samples_per_segment:
                self._rotate_file()

        return len(in0)

    def stop(self):
        """流图停止时关闭当前文件，确保文件句柄被释放。"""
        self._close_file()
        return True

    def _ensure_file_open(self) -> None:
        """确保当前录制文件已经打开。"""
        if self.current_file is not None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{timestamp}_{self.file_index:04d}.cfile"
        self.current_path = os.path.join(self.record_prefix, filename)
        self.current_file = open(self.current_path, "wb")
        self.samples_in_current_file = 0
        self.file_index += 1

    def _rotate_file(self) -> None:
        """关闭当前文件，并让下一次写入自动创建新文件。"""
        self._close_file()

    def _close_file(self) -> None:
        """关闭并清空当前文件句柄。"""
        if self.current_file is None:
            return

        self.current_file.flush()
        os.fsync(self.current_file.fileno())
        self.current_file.close()
        self.current_file = None

    @staticmethod
    def _sanitize_filename_part(raw: str) -> str:
        """清理文件名前缀，避免路径分隔符进入文件名。"""
        cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw)
        return cleaned.strip("_") or "rx_wideband"
