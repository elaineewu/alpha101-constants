"""Twelve alphas from "101 Formulaic Alphas" with every constant exposed as a parameter.

Chosen because each has several fractional constants and none needs
IndNeutralize (industry classification data), so they can be computed from
daily OHLCV alone. Default values are copied verbatim from the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from operators import (adv, correlation, decay_linear, delta, log_product,
                       rank, ts_max, ts_min, ts_rank, ts_sum)


class Panel:
    """Wide OHLCV panel plus the paper's derived fields."""

    def __init__(self, df: pd.DataFrame):
        wide = {f: df.pivot(index="date", columns="Name", values=f)
                for f in ["open", "high", "low", "close", "volume"]}
        self.open, self.high, self.low = wide["open"], wide["high"], wide["low"]
        self.close, self.volume = wide["close"], wide["volume"]
        # Daily VWAP is not in free data; the typical price is a standard proxy.
        self.vwap = (self.high + self.low + self.close) / 3
        self.dollar_volume = self.vwap * self.volume
        self._adv = {}

    def adv(self, n: int) -> pd.DataFrame:
        if n not in self._adv:
            self._adv[n] = adv(self.dollar_volume, n)
        return self._adv[n]


def _ind(cond: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Boolean -> 1.0/0.0, keeping NaN wherever an input was missing."""
    return cond.astype(float).where(valid)


# ---------------------------------------------------------------------------
# Alpha definitions. Parameter names: w* = windows, a = blend weight.
# ---------------------------------------------------------------------------

def alpha61(p, w1=16.1219, w2=17.9282):
    A = rank(p.vwap - ts_min(p.vwap, w1))
    B = rank(correlation(p.vwap, p.adv(180), w2))
    return _ind(A < B, A.notna() & B.notna())


def alpha62(p, w1=22.4101, w2=9.91009):
    A = rank(correlation(p.vwap, ts_sum(p.adv(20), w1), w2))
    inner = (rank(p.open) + rank(p.open)) < (rank((p.high + p.low) / 2) + rank(p.high))
    B = rank(inner.astype(float))
    return -_ind(A < B, A.notna() & B.notna())


def alpha64(p, a=0.178404, w1=12.7054, w2=16.6208, w3=3.69741):
    A = rank(correlation(ts_sum(p.open * a + p.low * (1 - a), w1),
                         ts_sum(p.adv(120), w1), w2))
    B = rank(delta((p.high + p.low) / 2 * a + p.vwap * (1 - a), w3))
    return -_ind(A < B, A.notna() & B.notna())


def alpha65(p, a=0.00817205, w1=8.6911, w2=6.40374, w3=13.635):
    A = rank(correlation(p.open * a + p.vwap * (1 - a), ts_sum(p.adv(60), w1), w2))
    B = rank(p.open - ts_min(p.open, w3))
    return -_ind(A < B, A.notna() & B.notna())


def alpha68(p, a=0.518371, w1=8.91644, w2=13.9333, w3=1.06157):
    A = ts_rank(correlation(rank(p.high), rank(p.adv(15)), w1), w2)
    B = rank(delta(p.close * a + p.low * (1 - a), w3))
    return -_ind(A < B, A.notna() & B.notna())


def alpha71(p, w1=3.43976, w2=12.0647, w3=18.0175, w4=4.20501,
            w5=15.6948, w6=16.4662, w7=4.4388):
    A = ts_rank(decay_linear(correlation(ts_rank(p.close, w1),
                                         ts_rank(p.adv(180), w2), w3), w4), w5)
    B = ts_rank(decay_linear(rank((p.low + p.open) - (p.vwap + p.vwap)) ** 2, w6), w7)
    return np.maximum(A, B)


def alpha72(p, w1=8.93345, w2=10.1519, w3=3.72469, w4=18.5188,
            w5=6.86671, w6=2.95011):
    num = rank(decay_linear(correlation((p.high + p.low) / 2, p.adv(40), w1), w2))
    den = rank(decay_linear(correlation(ts_rank(p.vwap, w3),
                                        ts_rank(p.volume, w4), w5), w6))
    return num / den


def alpha73(p, a=0.147155, w1=4.72775, w2=2.91864, w3=2.03608,
            w4=3.33829, w5=16.7411):
    A = rank(decay_linear(delta(p.vwap, w1), w2))
    px = p.open * a + p.low * (1 - a)
    B = ts_rank(decay_linear(-(delta(px, w3) / px), w4), w5)
    return -np.maximum(A, B)


def alpha74(p, a=0.0261661, w1=37.4843, w2=15.1365, w3=11.4791):
    A = rank(correlation(p.close, ts_sum(p.adv(30), w1), w2))
    B = rank(correlation(rank(p.high * a + p.vwap * (1 - a)), rank(p.volume), w3))
    return -_ind(A < B, A.notna() & B.notna())


def alpha75(p, w1=4.24304, w2=12.4413):
    A = rank(correlation(p.vwap, p.volume, w1))
    B = rank(correlation(rank(p.low), rank(p.adv(50)), w2))
    return _ind(A < B, A.notna() & B.notna())


def alpha77(p, w1=20.0451, w2=3.1614, w3=5.64125):
    mid = (p.high + p.low) / 2
    A = rank(decay_linear((mid + p.high) - (p.vwap + p.high), w1))
    B = rank(decay_linear(correlation(mid, p.adv(40), w2), w3))
    return np.minimum(A, B)


def alpha78(p, a=0.352233, w1=19.7428, w2=6.83313, w3=5.77492):
    A = rank(correlation(ts_sum(p.low * a + p.vwap * (1 - a), w1),
                         ts_sum(p.adv(40), w1), w2))
    B = rank(correlation(rank(p.vwap), rank(p.volume), w3))
    return A ** B


def alpha81(p, w1=49.6054, w2=8.47743, w3=14.9655, w4=5.07914):
    inner = rank(rank(correlation(p.vwap, ts_sum(p.adv(10), w1), w2)) ** 4)
    A = rank(log_product(inner, w3))
    B = rank(correlation(rank(p.vwap), rank(p.volume), w4))
    return -_ind(A < B, A.notna() & B.notna())


@dataclass(frozen=True)
class AlphaSpec:
    fn: object
    formula: str  # the paper's formula, for display


ALPHAS = {
    "61": AlphaSpec(alpha61, "rank(vwap - ts_min(vwap, 16.1219)) < rank(correlation(vwap, adv180, 17.9282))"),
    "62": AlphaSpec(alpha62, "(rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1"),
    "64": AlphaSpec(alpha64, "(rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1"),
    "65": AlphaSpec(alpha65, "(rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1"),
    "68": AlphaSpec(alpha68, "(Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1"),
    "71": AlphaSpec(alpha71, "max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))"),
    "72": AlphaSpec(alpha72, "rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011))"),
    "73": AlphaSpec(alpha73, "max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1"),
    "74": AlphaSpec(alpha74, "(rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1"),
    "75": AlphaSpec(alpha75, "rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413))"),
    "77": AlphaSpec(alpha77, "min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))"),
    "78": AlphaSpec(alpha78, "rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492))"),
    "81": AlphaSpec(alpha81, "(rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1"),
}


def default_params(alpha_id: str) -> dict:
    import inspect
    sig = inspect.signature(ALPHAS[alpha_id].fn)
    return {k: v.default for k, v in sig.parameters.items() if k != "p"}
