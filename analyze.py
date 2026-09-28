# -*- coding: utf-8 -*-
"""主分析：检验「电阻随磁场波动」数据里的周期结构。

方法要点（与原始分析的三处不同，见 README）：
  1. 只取平台区（±0.5–1.05 T），避开超导转变的陡变；
  2. 用多项式去基线，不用窄窗平滑——窄窗平滑本身会滤掉低频成分，
     实测中会把 ~13 1/T 的真特征滤掉、只剩高频噪声；
  3. 报峰之前先跟置换检验的零分布比，并做正/负场平台独立对照。

跑法：  python analyze.py
输出：  out/results.txt + out/*.png
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.io_origin import load_dat, parse_label       # noqa: E402
from src import signal as sg                          # noqa: E402

DATA = HERE / "data" / "raw_data.dat"
OUT = HERE / "out"

PLATEAUS = {"正场平台": (0.50, 1.05), "负场平台": (-1.05, -0.50)}
BAND = (3.0, 40.0)      # 关注频段 1/T（周期 0.025–0.33 T）
DEG = 8                 # 去基线多项式阶数
N_PERM = 2000


def spectrum_plateau(B, R, lo, hi, deg=DEG):
    """平台区：取全部点 -> 排序去重 -> 重采样 -> 多项式去基线 -> 幅度谱。"""
    m = (B >= lo) & (B <= hi)
    b, r = B[m], R[m]
    o = np.argsort(b)
    b, r = b[o], r[o]
    bu, inv = np.unique(np.round(b, 9), return_inverse=True)
    ru = np.array([r[inv == j].mean() for j in range(len(bu))])
    bg = np.linspace(bu[0], bu[-1], len(bu))
    rg = np.interp(bg, bu, ru)

    coef = np.polyfit(bg - bg.mean(), rg, deg)
    pert = rg - np.polyval(coef, bg - bg.mean())
    dt = (bg[-1] - bg[0]) / (len(bg) - 1)
    freq, amp = sg.amplitude_spectrum(pert, dt)
    return dict(b=bu, r=ru, bg=bg, rg=rg, pert=pert, freq=freq, amp=amp,
                dt=dt, span=bg[-1] - bg[0])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = load_dat(DATA)
    dat_sets = [(d, parse_label(d.label)) for d in data]

    lines: list[str] = []
    add = lines.append
    add("电阻随磁场波动 —— 平台区周期结构的显著性检验")
    add("=" * 104)
    add(f"{len(data)} 组数据；平台窗口 正场 {PLATEAUS['正场平台']} / "
        f"负场 {PLATEAUS['负场平台']} T；多项式去基线 deg={DEG}；"
        f"关注频段 {BAND[0]}–{BAND[1]} 1/T；置换检验 {N_PERM} 次")
    add("")

    # ---------- 表 1：双平台各自的峰 + 置换检验 ----------
    add("【表 1】正/负场平台独立对照（两个平台是不同的场区间与扫描方向）")
    add(f"{'组标签':<20}{'平台':<10}{'点数':>6}{'跨度T':>7}{'分辨率':>8}"
        f"{'峰 1/T':>9}{'周期 T':>9}{'幅度Ω':>9}{'信噪比':>7}{'p 值':>8}")
    add("-" * 104)
    res = {}
    for d, meta in dat_sets:
        for name, (lo, hi) in PLATEAUS.items():
            s = spectrum_plateau(d.B, d.R, lo, hi)
            fp, ap, Tp = sg.peak_in_band(s["freq"], s["amp"], *BAND)
            noise = np.median(s["amp"][s["freq"] > 0.5])
            perm = sg.permutation_test(s["pert"], s["dt"], *BAND, n_perm=N_PERM)
            res[(d.label, name)] = dict(s=s, fp=fp, Tp=Tp, ap=ap,
                                        snr=ap / noise, p=perm["p"], perm=perm)
            add(f"{d.label:<20}{name:<10}{len(s['bg']):>6}{s['span']:>7.3f}"
                f"{1 / s['span']:>8.3f}{fp:>9.3f}{Tp:>9.4f}{ap:>9.5f}"
                f"{ap / noise:>7.1f}{perm['p']:>8.4f}")

    add("")
    add("【表 2】两平台峰位一致性（若周期内在，两者应吻合）")
    add(f"{'组标签':<20}{'正场峰 1/T':>12}{'负场峰 1/T':>12}{'差值':>9}"
        f"{'正场周期T':>11}{'负场周期T':>11}")
    add("-" * 104)
    diffs = []
    for d, _ in dat_sets:
        a = res[(d.label, "正场平台")]
        b = res[(d.label, "负场平台")]
        diffs.append(abs(a["fp"] - b["fp"]))
        add(f"{d.label:<20}{a['fp']:>12.3f}{b['fp']:>12.3f}"
            f"{abs(a['fp'] - b['fp']):>9.3f}{a['Tp']:>11.4f}{b['Tp']:>11.4f}")
    add(f"    两平台峰位差：中位 {np.median(diffs):.3f} 1/T，最大 {max(diffs):.3f} 1/T")

    # ---------- 表 3：Lomb-Scargle 独立复核（不做插值） ----------
    add("")
    add("【表 3】Lomb-Scargle 周期图复核（直接处理非均匀采样，不做插值）")
    from scipy.signal import lombscargle
    add(f"{'组标签':<20}{'平台':<10}{'LS 最强峰 1/T':>16}{'周期 T':>10}{'FFT 峰 1/T':>13}")
    add("-" * 104)
    for (lab, name), v in res.items():
        d = next(x for x in data if x.label == lab)
        lo, hi = PLATEAUS[name]
        m = (d.B >= lo) & (d.B <= hi)
        b, r = d.B[m], d.R[m]
        o = np.argsort(b)
        b, r = b[o], r[o]
        coef = np.polyfit(b - b.mean(), r, DEG)
        pert = r - np.polyval(coef, b - b.mean())
        f = np.linspace(*BAND, 4000)
        p = lombscargle(b - b.mean(), pert, 2 * np.pi * f, normalize=True)
        i = int(np.argmax(p))
        add(f"{lab:<20}{name:<10}{f[i]:>16.3f}{1 / f[i]:>10.4f}{v['fp']:>13.3f}")

    # ---------- 统计汇总 ----------
    add("")
    add("【汇总】")
    nsig = sum(1 for v in res.values() if v["p"] < 0.05)
    add(f"    置换检验 p < 0.05 的组合：{nsig} / {len(res)}")
    pers = [v["Tp"] for v in res.values()]
    add(f"    峰周期分布：中位 {np.median(pers):.4f} T，"
        f"范围 [{min(pers):.4f}, {max(pers):.4f}] T")
    amps = [v["ap"] for v in res.values()]
    add(f"    峰幅度：中位 {np.median(amps):.5f} Ω")

    (OUT / "results.txt").write_text("\n".join(lines), encoding="utf-8")

    import figures
    figures.make_all(data, dat_sets, res, OUT)
    print(f"[done] {OUT / 'results.txt'}")


if __name__ == "__main__":
    main()
