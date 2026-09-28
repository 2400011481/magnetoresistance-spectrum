# -*- coding: utf-8 -*-
"""出图。中文必须显式指定 SimSun —— 本机 matplotlib 的字体回退链不生效。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

plt.rcParams["font.family"] = "SimSun"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 120

C_MAIN = "#0f3d6e"
C_ALT = "#c0392b"
PLATEAUS = {"正场平台": (0.50, 1.05), "负场平台": (-1.05, -0.50)}


def _pick(dat_sets, rate=2.0, sign="+"):
    return next((d, m) for d, m in dat_sets
                if m["rate_oe_s"] == rate and m["sign"] == sign)


def fig1_overview(dat_sets, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for d, meta in dat_sets:
        if meta["rate_oe_s"] in (2.0, 15.0) and meta["sign"] == "+":
            axs[0].plot(d.B, d.R, ".", ms=1.5, lw=0, label=d.label)
    axs[0].set_xlabel("磁场 B (T)")
    axs[0].set_ylabel("电阻 R (Ω)")
    axs[0].set_title("全扫场范围：超导转变")
    axs[0].legend(fontsize=8, markerscale=6)
    for lo, hi in PLATEAUS.values():
        axs[0].axvspan(lo, hi, color=C_MAIN, alpha=0.12)
    axs[1].set_title("分析用的两个平台区（阴影）")

    (d, _) = _pick(dat_sets)
    m = (d.B >= 0.3) & (d.B <= 1.1)
    axs[1].plot(d.B[m], d.R[m], ".-", ms=2.5, lw=0.8, color="#444")
    axs[1].axvspan(*PLATEAUS["正场平台"], color=C_MAIN, alpha=0.12)
    axs[1].set_xlabel("磁场 B (T)")
    axs[1].set_ylabel("电阻 R (Ω)")
    axs[1].set_title(f"{d.label} 的平台区细节")
    fig.tight_layout()
    fig.savefig(out / "fig1_overview.png")
    plt.close(fig)


def fig2_perturbation(res, dat_sets, out: Path):
    """核心图：去掉基线后的振荡，正负场两个平台对照。"""
    (d, _) = _pick(dat_sets)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.4))
    for ax, name in zip(axs, PLATEAUS):
        v = res[(d.label, name)]
        ax.plot(v["s"]["b"], v["s"]["pert"], "-", lw=1.0, color=C_ALT)
        ax.set_ylabel("扰动 (Ω)")
        ax.set_title(f"{d.label}　{name}：峰周期 {v['Tp']:.4f} T，"
                     f"幅度 {v['ap']:.4f} Ω，p = {v['p']:.4f}", fontsize=9.5)
        lo, hi = PLATEAUS[name]
        step = 0.08
        k = np.ceil(min(lo, hi) / step) * step
        while k <= max(lo, hi):
            ax.axvline(k, color=C_MAIN, ls=":", lw=0.7, alpha=0.6)
            k += step
    axs[1].set_xlabel("磁场 B (T)")
    fig.suptitle("去基线后的扰动（蓝虚线间隔 0.08 T）—— 两个平台独立可见同一周期", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_perturbation.png")
    plt.close(fig)


def fig3_spectra(res, dat_sets, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, name in zip(axs, PLATEAUS):
        for d, meta in dat_sets:
            if meta["sign"] != "+":
                continue
            v = res[(d.label, name)]
            m = v["s"]["freq"] <= 40
            ax.plot(v["s"]["freq"][m], v["s"]["amp"][m], lw=0.9, label=d.label)
        ax.axvline(12.9, color=C_ALT, ls="--", lw=1.2)
        ax.set_title(f"{name}　幅度谱")
        ax.set_xlabel("频率 (1/T)")
    axs[0].set_ylabel("幅度 (Ω)")
    axs[0].legend(fontsize=7.5)
    fig.suptitle("各组正场数据的幅度谱（红虚线 12.9 1/T ≈ 周期 0.078 T）", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig3_spectra.png")
    plt.close(fig)


def fig4_consistency(res, dat_sets, out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    labels = [d.label for d, _ in dat_sets]
    y = np.arange(len(labels))
    fp_p = [res[(l, "正场平台")]["fp"] for l in labels]
    fp_n = [res[(l, "负场平台")]["fp"] for l in labels]
    axs[0].plot(fp_p, y, "o", ms=6, color=C_MAIN, label="正场平台")
    axs[0].plot(fp_n, y, "s", ms=6, color=C_ALT, label="负场平台")
    for i in range(len(labels)):
        axs[0].plot([fp_p[i], fp_n[i]], [i, i], "-", lw=0.8, color="#999")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labels, fontsize=8)
    axs[0].set_xlabel("最强峰频率 (1/T)")
    axs[0].set_title("两个平台的峰位几乎重合")
    axs[0].legend(fontsize=8)

    Tp = [res[(l, "正场平台")]["Tp"] for l in labels]
    Tn = [res[(l, "负场平台")]["Tp"] for l in labels]
    axs[1].plot(Tp, Tn, "o", ms=7, color=C_MAIN)
    lim = [min(Tp + Tn) * 0.97, max(Tp + Tn) * 1.03]
    axs[1].plot(lim, lim, "--", color="#999", lw=1)
    axs[1].set_xlabel("正场平台 峰周期 (T)")
    axs[1].set_ylabel("负场平台 峰周期 (T)")
    axs[1].set_title("峰周期对照（对角线 = 完全一致）")
    fig.tight_layout()
    fig.savefig(out / "fig4_consistency.png")
    plt.close(fig)


def fig5_permutation(res, dat_sets, out: Path):
    show = [(d, m) for d, m in dat_sets if m["sign"] == "+"][:8]
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.6))
    for ax, (d, meta) in zip(axs.ravel(), show):
        v = res[(d.label, "正场平台")]
        ax.hist(v["perm"]["null"], bins=30, color="#9bb4d0", edgecolor="none")
        ax.axvline(v["perm"]["obs"], color=C_ALT, lw=2)
        ax.set_title(f"{d.label}\np = {v['p']:.4f}", fontsize=8)
        ax.set_yticks([])
    fig.suptitle("置换检验：打乱数据后的最大谱峰（蓝）vs 实测（红）", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig5_permutation.png")
    plt.close(fig)


def make_all(data, dat_sets, res, out: Path):
    fig1_overview(dat_sets, out)
    fig2_perturbation(res, dat_sets, out)
    fig3_spectra(res, dat_sets, out)
    fig4_consistency(res, dat_sets, out)
    fig5_permutation(res, dat_sets, out)
