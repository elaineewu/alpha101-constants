"""Operators from Kakushadze (2016), "101 Formulaic Alphas".

Every input is a wide DataFrame (rows = dates, columns = tickers).
Window arguments may be non-integer, exactly as printed in the paper.
How a fractional window becomes an integer is controlled globally by
WINDOW_MODE, so the same alpha code can be run under different conventions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# "floor" is the paper's stated convention; "round" is the alternative we test.
WINDOW_MODE = "floor"


def set_window_mode(mode: str) -> None:
    global WINDOW_MODE
    assert mode in ("floor", "round")
    WINDOW_MODE = mode


def win(d: float, minimum: int = 1) -> int:
    n = int(np.floor(d)) if WINDOW_MODE == "floor" else int(np.floor(d + 0.5))
    return max(n, minimum)


# ---------- cross-sectional ----------

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank across tickers on each date."""
    return x.rank(axis=1, pct=True)


# ---------- time-series ----------

def delay(x, d):
    return x.shift(win(d))


def delta(x, d):
    return x - x.shift(win(d))


def ts_sum(x, d):
    return x.rolling(win(d)).sum()


def ts_min(x, d):
    return x.rolling(win(d)).min()


def ts_max(x, d):
    return x.rolling(win(d)).max()


def ts_rank(x, d):
    """Percentile rank of today's value within the trailing window."""
    n = win(d)
    if n == 1:
        return x.notna().astype(float).where(x.notna())
    return x.rolling(n).rank(pct=True)


def correlation(x, y, d):
    """Rolling Pearson correlation, column by column. Flat windows -> NaN."""
    n = win(d, minimum=2)
    c = x.rolling(n).corr(y)
    return c.replace([np.inf, -np.inf], np.nan).clip(-1, 1)


def decay_linear(x, d):
    """Linearly decaying weighted average, weights n, n-1, ..., 1 (today = n)."""
    n = win(d)
    total = n * (n + 1) / 2
    out = x * (n / total)
    for k in range(1, n):
        out = out + x.shift(k) * ((n - k) / total)
    return out


def log_product(x, d):
    """log(product(x, d)) computed as a rolling sum of logs (x > 0)."""
    return np.log(x).rolling(win(d)).sum()


def adv(dollar_volume: pd.DataFrame, d: int) -> pd.DataFrame:
    """Average daily dollar volume over the past d days (d is always an integer)."""
    return dollar_volume.rolling(int(d)).mean()
