"""Render the README figures from results/results.json."""
import json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

R = json.load(open("results/results.json"))
IS_C, OOS_C, RAND_C = "#7c3aed", "#0f9488", "#94a3b8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
ids = sorted(R["alphas"], key=int)


def clean(points, name, key):
    return [p for p in points if p["is"]["ic"] is not None and p["oos"]["ic"] is not None
            and not (name != "a" and p["x"] == 1 and key >= 2)]


# 1. Neighborhood percentile of every published constant
pis, poos = [], []
for aid in ids:
    for name, s in R["alphas"][aid]["sweeps"].items():
        key = s["published"] if name == "a" else math.floor(s["published"])
        pts = clean(s["points"], name, key)
        xs = [p["x"] for p in pts]
        if key not in xs:
            continue
        i = xs.index(key)
        for arr, per in ((pis, "is"), (poos, "oos")):
            v = [p[per]["ic"] for p in pts]
            arr.append(sum(x < v[i] for x in v) / (len(v) - 1))
fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.linspace(0, 1.0001, 6)
ax.hist([pis, poos], bins=bins, color=[IS_C, OOS_C], label=["In-sample (2013–15)", "Out-of-sample (2016–18)"])
ax.axhline(len(pis) / 5, color="k", ls="--", lw=1, label="Uniform (no information)")
ax.set_xlabel("Percentile of published constant among its integer neighbors")
ax.set_ylabel("Constants")
ax.set_title(f"{len(pis)} published constants vs. their neighbors")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/neighborhood_percentiles.png"); plt.close(fig)

# 2. One-at-a-time sweep example: Alpha #61, window 1
s = R["alphas"]["61"]["sweeps"]["w1"]
pts = clean(s["points"], "w1", 16)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot([p["x"] for p in pts], [p["is"]["ic"] * 1e4 for p in pts], "-o", color=IS_C, ms=4, label="In-sample")
ax.plot([p["x"] for p in pts], [p["oos"]["ic"] * 1e4 for p in pts], "-o", color=OOS_C, ms=4, label="Out-of-sample")
ax.axvline(16, color="k", ls="--", lw=1)
ax.text(16.1, ax.get_ylim()[1] * 0.95, "published 16.1219 → 16", fontsize=8, va="top")
ax.set_xlabel("ts_min window (days)"); ax.set_ylabel("Mean daily rank IC × 10⁴")
ax.set_title("Alpha #61: best in-sample at the published window, 8th of 11 out-of-sample")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("figures/sweep_alpha61.png"); plt.close(fig)

# 3. Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for row, aid in enumerate(["61", "75"]):
    h = R["heatmaps"][aid]
    xs = sorted({c["x"] for c in h["cells"]}); ys = sorted({c["y"] for c in h["cells"]})
    vmax = max(abs(c[k]) for c in h["cells"] for k in ("is", "oos") if c[k] is not None) * 1e4
    prm = R["alphas"][aid]["params"]
    px, py = math.floor(prm[h["xParam"]]), math.floor(prm[h["yParam"]])
    for col, per in enumerate(["is", "oos"]):
        Z = np.full((len(ys), len(xs)), np.nan)
        for c in h["cells"]:
            if c[per] is not None:
                Z[ys.index(c["y"]), xs.index(c["x"])] = c[per] * 1e4
        ax = axes[row, col]; ax.grid(False)
        im = ax.imshow(Z, origin="lower", cmap="RdBu", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        ax.set_xticks(range(len(xs)), xs, fontsize=7); ax.set_yticks(range(len(ys)), ys, fontsize=7)
        ax.scatter([xs.index(px)], [ys.index(py)], s=120, facecolors="none", edgecolors="k", lw=1.5)
        ax.set_xlabel(f"{h['xParam']} (days)"); ax.set_ylabel(f"{h['yParam']} (days)")
        ax.set_title(f"Alpha #{aid}, {'in-sample' if per == 'is' else 'out-of-sample'}")
        fig.colorbar(im, ax=ax, label="IC × 10⁴", shrink=0.8)
fig.suptitle("Spike vs. plateau (circle = published windows)")
fig.tight_layout(); fig.savefig("figures/heatmaps.png"); plt.close(fig)

# 4. Random-constant null, small multiples
fig, axes = plt.subplots(3, 5, figsize=(13, 7.5))
for ax, aid in zip(axes.flat, ids):
    d = np.array([x for x in R["null"][aid] if None not in x]) * 1e4
    b = R["alphas"][aid]["baseline"]
    ax.scatter(d[:, 0], d[:, 1], s=10, color=RAND_C)
    ax.scatter([b["is"]["ic"] * 1e4], [b["oos"]["ic"] * 1e4], s=60, color=OOS_C, edgecolors="k", zorder=3)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"#{aid}", fontsize=9); ax.tick_params(labelsize=7)
for ax in list(axes.flat)[len(ids):]:
    ax.axis("off")
fig.supxlabel("In-sample IC × 10⁴"); fig.supylabel("Out-of-sample IC × 10⁴")
fig.suptitle("Published constants (teal) vs. 60 random constant sets (gray) per alpha")
fig.tight_layout(); fig.savefig("figures/random_null.png"); plt.close(fig)
print("figures written")
