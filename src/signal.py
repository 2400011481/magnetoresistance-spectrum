# -*- coding: utf-8 -*-
"""信号处理：重采样、去基线、幅度谱、置换检验、自相关。

设计上的两个要点：
1. 数据在窗口内的采样不是均匀的（扫场速率会变），所以统一按 B 排序后
   重采样到等间距栅格，再做 FFT，避免频率轴被非均匀采样污染。
2. 报峰之前先跟「打乱后重算」的零分布比（置换检验），
   避免把噪声起伏当成周期性信号——这是原始分析里缺的一步。
"""
import numpy as np
from scipy.signal import savgol_filter


def resample_uniform(B, R, lo, hi, n=None):
    """取 [lo, hi] 内全部点，按 B 排序去重后重采样到等间距栅格。

    返回 (bg, rg, dt)：栅格磁场、重采样电阻、磁场步长。
    """
    m = (B >= lo) & (B <= hi)
    b, r = B[m], R[m]
    o = np.argsort(b)
    b, r = b[o], r[o]

    # 同一磁场值可能重复测量 -> 取平均
    bu, inv = np.unique(np.round(b, 9), return_inverse=True)
    ru = np.array([r[inv == j].mean() for j in range(len(bu))])

    n = len(bu) if n is None else n
    bg = np.linspace(bu[0], bu[-1], n)
    rg = np.interp(bg, bu, ru)
    return bg, rg, (bg[-1] - bg[0]) / (n - 1)


def detrend(r, frac=0.15):
    """减去大窗 savgol 平滑曲线，得到「扰动」分量（等价于差分+平滑）。"""
    n = len(r)
    win = max(11, (int(n * frac)) | 1)      # savgol 要求奇数窗
    win = min(win, n - 1 if (n - 1) % 2 else n - 2)
    return r - savgol_filter(r, win, 2), win


def amplitude_spectrum(rg, dt):
    """Hann 窗 FFT -> (freq[1/T], 幅度)。幅度已按正弦振幅归一化。"""
    n = len(rg)
    w = np.hanning(n)
    spec = np.fft.rfft(rg * w)
    amp = 2.0 * np.abs(spec) / w.sum()
    freq = np.fft.rfftfreq(n, d=dt)         # 单位 1/T
    return freq, amp


def peak_in_band(freq, amp, f_lo, f_hi):
    """在 [f_lo, f_hi] 频段内找最强峰，返回 (频率, 幅度, 周期)。"""
    m = (freq >= f_lo) & (freq <= f_hi)
    if not m.any():
        return np.nan, np.nan, np.nan
    f, a = freq[m], amp[m]
    i = int(np.argmax(a))
    return f[i], a[i], 1.0 / f[i]


def permutation_test(pert, dt, f_lo, f_hi, n_perm=2000, seed=0):
    """置换检验：打乱扰动序列后重算最大谱峰，得到零分布与 p 值。

    打乱会破坏一切时序结构，所以零分布代表「没有周期性时的最大谱峰有多大」。
    观测峰幅度超过零分布 95% 分位，才算显著。
    """
    rng = np.random.default_rng(seed)
    _, amp_obs = amplitude_spectrum(pert, dt)
    freq, _ = amplitude_spectrum(pert, dt)
    m = (freq >= f_lo) & (freq <= f_hi)
    obs = amp_obs[m].max()

    null = np.empty(n_perm)
    buf = pert.copy()
    for i in range(n_perm):
        rng.shuffle(buf)
        _, a = amplitude_spectrum(buf, dt)
        null[i] = a[m].max()

    p = (np.count_nonzero(null >= obs) + 1) / (n_perm + 1)
    return dict(obs=obs, null=null, p=p, q95=np.percentile(null, 95),
                q99=np.percentile(null, 99))


def autocorr(pert, dt, max_lag_T=None):
    """自相关（FFT 实现），返回 (滞后[T], 归一化自相关)。"""
    x = pert - pert.mean()
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[:n]
    ac /= ac[0]
    lags = np.arange(n) * dt
    if max_lag_T is not None:
        k = lags <= max_lag_T
        lags, ac = lags[k], ac[k]
    return lags, ac
