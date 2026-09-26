"""Run every experiment (baseline, sweeps, floor vs. round, heatmaps, random null) and write results/results.json."""
import json, math, time, random
import numpy as np, pandas as pd
import operators
from alphas import Panel, ALPHAS, default_params
from evaluate import forward_returns, is_oos, daily_ic, IS_END

df = pd.read_csv("data/all_stocks_5yr.csv", parse_dates=["date"])
P = Panel(df); FWD = forward_returns(P)

def run(aid, **over):
    prm = default_params(aid); prm.update(over)
    return is_oos(ALPHAS[aid].fn(P, **prm), FWD)

def r4(x): return None if x is None or (isinstance(x,float) and math.isnan(x)) else round(float(x), 5)
def slim(res): return {k: {m: r4(v) for m, v in d.items()} for k, d in res.items()}

out = {"meta": {"start": str(P.close.index.min().date()), "end": str(P.close.index.max().date()),
                "tickers": int(P.close.shape[1]), "is_end": IS_END}, "alphas": {}}
t0 = time.time()
for aid, spec in ALPHAS.items():
    prm = default_params(aid)
    entry = {"formula": spec.formula, "params": prm}
    operators.set_window_mode("floor"); base = run(aid); entry["baseline"] = slim(base)
    # cumulative IC series (published params)
    ic = daily_ic(ALPHAS[aid].fn(P), FWD).fillna(0).cumsum()
    ic = ic.iloc[::5]
    entry["cum_ic"] = [[str(d.date()), r4(v)] for d, v in ic.items()]
    operators.set_window_mode("round"); entry["round"] = slim(run(aid)); operators.set_window_mode("floor")
    sweeps = {}
    for name, val in prm.items():
        pts = []
        if name == "a":
            grid = sorted(set([round(x, 2) for x in np.linspace(0, 1, 21)] + [val]))
            for g in grid:
                pts.append({"x": g, **{k: v for k, v in slim(run(aid, a=g)).items()}})
        else:
            f = int(math.floor(val)); lo = max(1 if name else 1, f - 5)
            for k in range(lo, f + 6):
                pts.append({"x": k, **slim(run(aid, **{name: k}))})
        sweeps[name] = {"published": val, "points": pts}
    entry["sweeps"] = sweeps
    out["alphas"][aid] = entry
    print(aid, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("results/results.json", "w"))

# 2D heatmaps
heat = {}
for aid, (p1, r1), (p2, r2) in [("75", ("w1", range(2, 13)), ("w2", range(5, 21))),
                                 ("61", ("w1", range(8, 25, 2)), ("w2", range(9, 28, 2)))]:
    cells = []
    for a in r1:
        for b in r2:
            res = slim(run(aid, **{p1: a, p2: b}))
            cells.append({"x": a, "y": b, "is": res["is"]["ic"], "oos": res["oos"]["ic"]})
    heat[aid] = {"xParam": p1, "yParam": p2, "cells": cells}
    print("heat", aid, f"{time.time()-t0:.0f}s", flush=True)
out["heatmaps"] = heat
json.dump(out, open("results/results.json", "w"))

# random-constant null
rng = random.Random(101)
null = {}
for aid in ALPHAS:
    prm = default_params(aid); draws = []
    for _ in range(60):
        over = {}
        for k, v in prm.items():
            if k == "a": over[k] = rng.random()
            else: over[k] = rng.randint(max(2, round(0.5 * v)), max(3, round(2 * v)))
        res = slim(run(aid, **over)); draws.append([res["is"]["ic"], res["oos"]["ic"]])
    null[aid] = draws
    print("null", aid, f"{time.time()-t0:.0f}s", flush=True)
out["null"] = null
json.dump(out, open("results/results.json", "w"))
print("done")
