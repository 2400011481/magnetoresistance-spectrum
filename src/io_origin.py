# -*- coding: utf-8 -*-
"""读取 Origin 导出的 .dat 数据。

文件格式（从 Origin 直接 Export ASCII 得到）：
    第 1 行：变量名，成对出现   B  R  B  R  ...
    第 2 行：单位               T  Ω  T  Ω  ...
    第 3 行：组标签             第 2i+1 列是第 i 组的名字，如 "17.71Hz 10Oe/s +"
    第 4 行起：数据，制表符分隔

编码是 GBK（含 Ω 字符），不是 UTF-8。
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    label: str          # 组标签，如 "17.71Hz 10Oe/s +"
    B: np.ndarray       # 磁场，单位 T
    R: np.ndarray       # 电阻，单位 Ω


def load_dat(path: str | Path) -> list[Dataset]:
    """解析 .dat，返回全部数据组（跳过空的尾列）。"""
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("gbk")
    except UnicodeDecodeError:
        text = raw.decode("utf-8")          # 兼容以后可能改成 UTF-8 的导出

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        raise ValueError(f"{path} 行数不足，不是预期的三行表头格式")
    lab_row = lines[2].split("\t")
    n_pair = len(lab_row) // 2

    out: list[Dataset] = []
    for i in range(n_pair):
        label = lab_row[2 * i + 1].strip()
        if not label:
            continue                        # 空的尾列
        B, R = [], []
        for ln in lines[3:]:
            cols = ln.split("\t")
            if len(cols) <= 2 * i + 1:
                continue
            bs, rs = cols[2 * i].strip(), cols[2 * i + 1].strip()
            if not bs or not rs:
                continue
            try:
                B.append(float(bs))
                R.append(float(rs))
            except ValueError:
                continue                    # 跳过非数值行
        if B:
            out.append(Dataset(label, np.array(B), np.array(R)))
    return out


def parse_label(label: str) -> dict:
    """从 "17.71Hz 10Oe/s +" 里拆出振动频率 / 扫场速率 / 场方向。"""
    import re
    m = re.match(r"([\d.]+)\s*Hz\s*([\d.]+)\s*Oe/s\s*([+-])", label)
    if not m:
        return {"freq_hz": np.nan, "rate_oe_s": np.nan, "sign": "?"}
    return {"freq_hz": float(m.group(1)),
            "rate_oe_s": float(m.group(2)),
            "sign": m.group(3)}
