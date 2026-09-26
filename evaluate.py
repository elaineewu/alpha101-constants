"""Signal evaluation: daily rank IC and a dollar-neutral rank-weighted portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd

IS_END = "2015-12-31"   # paper appeared on arXiv in Jan 2016
OOS_START = "2016-01-01"


def forward_returns(p) -> pd.DataFrame:
    """Signal known at close t, traded open t+1 -> open t+2 (no look-ahead).

    Moves larger than 35% (a handful of spin-offs / data errors) are masked.
    """
    r = p.open.shift(-2) / p.open.shift(-1) - 1
    return r.where(r.abs() < 0.35)


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    num = np.nansum(a * b, axis=1)
    den = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def daily_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    s = signal.replace([np.inf, -np.inf], np.nan)
    both = s.notna() & fwd.notna()
    sr = s.where(both).rank(axis=1)
    fr = fwd.where(both).rank(axis=1)
    ic = pd.Series(_rowwise_corr(sr.to_numpy(), fr.to_numpy()), index=s.index)
    ic[both.sum(axis=1) < 50] = np.nan
    return ic


def portfolio(signal: pd.DataFrame, fwd: pd.DataFrame):
    """Weights ∝ demeaned cross-sectional rank, gross exposure = 1."""
    s = signal.replace([np.inf, -np.inf], np.nan).where(fwd.notna())
    r = s.rank(axis=1, pct=True)
    w = r.sub(r.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1), axis=0).fillna(0.0)
    pnl = (w * fwd.fillna(0.0)).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1) / 2
    return pnl, turnover


def summarize(signal, fwd, start=None, end=None) -> dict:
    ic = daily_ic(signal, fwd).loc[start:end].dropna()
    pnl, to = portfolio(signal, fwd)
    pnl, to = pnl.loc[start:end], to.loc[start:end]
    pnl = pnl[ic.index.min():ic.index.max()] if len(ic) else pnl
    sharpe = np.sqrt(252) * pnl.mean() / pnl.std() if pnl.std() > 0 else np.nan
    return {
        "ic": float(ic.mean()) if len(ic) else np.nan,
        "ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if len(ic) > 1 else np.nan,
        "sharpe": float(sharpe),
        "turnover": float(to.mean()),
        "days": int(len(ic)),
    }


def is_oos(signal, fwd) -> dict:
    return {"is": summarize(signal, fwd, None, IS_END),
            "oos": summarize(signal, fwd, OOS_START, None)}
