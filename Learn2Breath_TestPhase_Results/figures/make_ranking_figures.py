"""Figures for the public results summary: how the Learn2Breath ranking comes about.

Reads only ../ranking_table.csv, so it can be rerun from this repository:
    python3 make_ranking_figures.py
"""
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
rows = sorted(csv.DictReader(open(OUT.parent / "ranking_table.csv")), key=lambda r: (int(r["rank"]), r["team"]))
teams = [r["team"] for r in rows]; R = {r["team"]: {k: (v if k == "team" else float(v)) for k, v in r.items()} for r in rows}; K = len(teams)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"; INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
METRICS = [("DSC_lobe", "Lobe DSC"), ("Fissure_ASD", "Fissure\nASD"), ("Fissure_HD95", "Fissure\nHD95"), ("NDV", "%NDV"), ("Costophrenic_ASD", "Costo.\nASD"), ("Costophrenic_HD95", "Costo.\nHD95")]
ylabels = [f"{int(R[t]['rank'])}. {t}" for t in teams]

def strip(ax, left=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if not left: ax.spines["left"].set_visible(False)

# 1. per-metric significance scores -------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 6.4))
grid = np.array([[R[t]["sig_" + m] for m, _ in METRICS] for t in teams])
im = ax.imshow(grid, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
for i in range(K):
    for j in range(len(METRICS)):
        v = grid[i, j]; ax.text(j, i, f"{int(round(v * K)) - 1}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([n for _, n in METRICS]); ax.xaxis.tick_top()
ax.set_yticks(range(K)); ax.set_yticklabels(ylabels)
ax.tick_params(length=0); strip(ax); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
for x in (0.5, 2.5, 3.5): ax.axvline(x, color="white", lw=3)
for x, name in ((0, "Lobar"), (1.5, "Fissure"), (3, "Folding"), (4.5, "Costophrenic")): ax.text(x, -2.0, name, ha="center", color=INK2, fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label(f"significance score = (methods beaten + 1) / {K}", color=INK2); cb.outline.set_visible(False)
ax.set_title(f"Cell value: number of the other {K - 1} methods significantly outperformed on that metric\n(one-sided Wilcoxon on 200 subjects, Holm-adjusted); rows in final-rank order.", fontsize=9, color=INK2, loc="left", pad=46)
fig.tight_layout(); fig.savefig(OUT / "fig_metric_scores.png", dpi=200, facecolor="#fcfcfb"); plt.close(fig)

# 2. components -> final -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 6.6))
y = np.arange(K)[::-1]; h = 0.21
comps = [("comp_Lobar", "Lobar", BLUE), ("comp_Fissure", "Fissure", VIOLET), ("comp_Folding", "Folding", AQUA), ("comp_Costophrenic", "Costophrenic", ORANGE)]
for k, (key, name, col) in enumerate(comps):
    ax.barh(y + (1.5 - k) * h, [R[t][key] for t in teams], height=h * 0.92, color=col, label=name, edgecolor="#fcfcfb", linewidth=0.6)
for i, t in enumerate(teams):
    ax.text(1.02, y[i], f"{R[t]['final_score']:.1f}", va="center", fontsize=8.5, color=INK, fontweight="bold")
ax.text(1.02, K - 0.3, "Final", fontsize=8.5, color=INK2, va="bottom")
ax.set_yticks(y); ax.set_yticklabels(ylabels); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1.0); ax.set_xlabel("component score (mean significance score of its metrics)"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
strip(ax, left=False); ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.set_title("Final = 100 × (Lobar × Fissure × Folding × Costophrenic)^0.25", loc="left", fontsize=9.5, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig_components.png", dpi=200, facecolor="#fcfcfb"); plt.close(fig)

# 3. rank stability ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.0))
for i, t in enumerate(teams):
    lo, hi, med = R[t]["rank_ci95_lower"], R[t]["rank_ci95_upper"], R[t]["bootstrap_median_rank"]
    ax.plot([lo, hi], [y[i], y[i]], color=BLUE, lw=2, solid_capstyle="round", alpha=0.45)
    ax.plot(med, y[i], "o", color="#fcfcfb", mec=BLUE, mew=1.6, ms=7)
    ax.plot(R[t]["rank"], y[i], "D", color=ORANGE, ms=6)
ax.set_yticks(y); ax.set_yticklabels(teams); ax.tick_params(axis="y", length=0)
ax.set_xticks(range(1, K + 1)); ax.set_xlim(0.5, K + 0.5); ax.set_xlabel("rank"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); strip(ax, left=False)
ax.legend(handles=[Line2D([], [], marker="D", color=ORANGE, ls="", ms=6, label="official rank (full cohort)"), Line2D([], [], marker="o", color="#fcfcfb", mec=BLUE, mew=1.6, ls="", ms=7, label="median bootstrap rank"), Line2D([], [], color=BLUE, lw=2, alpha=0.45, label="95% interval over 1,000 subject bootstraps")], loc="upper right", frameon=False, fontsize=8.5)
ax.set_title("Ranking stability: the complete scoring was repeated on 1,000 resamples of the 200 subjects", loc="left", fontsize=9.5, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig_rank_stability.png", dpi=200, facecolor="#fcfcfb"); plt.close(fig)

# 4. lobe overlap vs costophrenic boundary: why both are scored --------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.0))
xs = [R[t]["mean_DSC_lobe"] for t in teams]; ys = [R[t]["mean_Costophrenic_HD95"] for t in teams]
ax.scatter(xs, ys, s=50, color=BLUE, edgecolor="#fcfcfb", linewidth=1, zorder=3)
OFFSETS = {"dslee": (-25, -14), "AI4Viaevum": (-30, 12), "hemoyang1999": (-12, 30), "UzL": (4, 32), "hengjieliu96": (28, 20), "jbgeb_VROC": (-6, -28), "vjaouen_LaTIM": (18, -16), "aryapmenon": (5, -6), "ali_raza": (5, -7), "zeybekoglu": (5, -3)}
CLUSTER = {"dslee", "AI4Viaevum", "hemoyang1999", "UzL", "hengjieliu96", "jbgeb_VROC", "vjaouen_LaTIM"}
for t, x, v in zip(teams, xs, ys):
    dx, dy = OFFSETS.get(t, (5, 3))
    ax.annotate(t, (x, v), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK2, ha="right" if dx < 0 else "left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=3) if t in CLUSTER else None)
ax.set_xlim(0.66, 1.0); ax.set_ylim(-4, 58)
ax.set_xlabel("lobe Dice (cohort mean) — higher is better"); ax.set_ylabel("costophrenic HD95, mm (cohort mean) — lower is better")
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); strip(ax)
ax.set_title("Lobe overlap and lung-base alignment mostly agree, but not always: songtaowei (Dice 0.89) misses the\ncostophrenic recess by 31 mm on average, while zeybekoglu (Dice 0.83) stays within 5 mm.", loc="left", fontsize=9, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig_lobar_vs_costophrenic.png", dpi=200, facecolor="#fcfcfb"); plt.close(fig)
print("figures written to", OUT)
