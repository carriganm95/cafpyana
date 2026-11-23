"""Draw docs/processing_chain.png: the nueNp0Pi processing chain, CAF files -> plots.

One horizontal band per stage. In each band a processor block (solid, rounded: code that
runs) feeds an output block (white, folded corner: what gets saved).

    python analysis_village/nueNp0Pi/docs/make_processing_chain.py
"""
from os import path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# (stage name, where it runs, band colour, processor colour,
#  processor title, processor command, output title, output detail)
STAGES = [
    ("1  File lists", "interactive", "#eef2f7", "#64748b",
     "List the CAF files", "create_list.py / make_filelist.py",
     "File list", "one xrootd URL per CAF file"),
    ("2  Dataframes", "grid", "#fff4e5", "#d97706",
     "Flatten CAFs into tables", "run_df_maker.py",
     "Dataframes (.df)", "HDF5: slices, particles, POT, truth"),
    ("3  Skim", "grid, then merge", "#ecfdf5", "#059669",
     "Pre-select slices", "run_event_selection_map.py -skim",
     "Skim (Parquet)", "FV, flash-matched, contained slices"),
    ("4  Selection", "interactive", "#eff6ff", "#2563eb",
     "Apply cuts, fill histograms", "run_event_selection_map.py -from_skim",
     "Histograms (.pkl)", "signal and π⁰ sideband"),
    ("5  Plots", "interactive", "#f5f3ff", "#7c3aed",
     "Combine, scale to POT, draw", "event_selection_aggregate",
     "Plots and tables", "PNG"),
]

W, BAND_H = 12.0, 1.55
TOP_H = 1.0                      # space above stage 1 for the CAF input
H = TOP_H + BAND_H * len(STAGES) + 0.1
LABEL_X = 0.25                   # stage label column
P_X, P_W = 2.9, 4.3              # processor block
O_X, O_W = 8.1, 3.6              # output block
BOX_H = 0.95

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
plt.rcParams["font.family"] = "DejaVu Sans"


def processor(x, y, w, h, colour, title, cmd):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={min(0.18, 0.3 * h)}",
                                facecolor=colour, edgecolor="none", zorder=3))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.28, cmd, ha="center", va="center", color="white",
            fontsize=9, family="DejaVu Sans Mono", alpha=0.92, zorder=4)


def output(x, y, w, h, colour, title, detail):
    fold = min(0.22, 0.3 * h)
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - fold), (x + w - fold, y + h),
                          (x, y + h)], closed=True, facecolor="white", edgecolor=colour,
                         linewidth=1.8, zorder=3))
    ax.add_patch(Polygon([(x + w - fold, y + h), (x + w - fold, y + h - fold),
                          (x + w, y + h - fold)], closed=True, facecolor=colour,
                         edgecolor=colour, linewidth=1.2, alpha=0.35, zorder=4))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", color="#1f2937",
            fontsize=11.5, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center", color="#4b5563",
            fontsize=9, zorder=4)


def arrow(p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 color="#374151", linewidth=1.6,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


# CAF input at the top
caf_y = H - TOP_H + 0.12
output(P_X + P_W / 2 - 1.5, caf_y, 3.0, 0.7, "#64748b", "CAF files", "flat CAF .root on dCache")
prev_bottom = (P_X + P_W / 2, caf_y)

for i, (name, where, band, colour, p_title, p_cmd, o_title, o_detail) in enumerate(STAGES):
    y0 = H - TOP_H - BAND_H * (i + 1)
    ax.add_patch(plt.Rectangle((0, y0), W, BAND_H, facecolor=band, edgecolor="none", zorder=0))
    ax.plot([0, W], [y0 + BAND_H, y0 + BAND_H], color="#9ca3af", linewidth=1.0, zorder=1)
    ax.text(LABEL_X, y0 + BAND_H * 0.58, name, fontsize=13, fontweight="bold",
            color=colour, va="center")
    ax.text(LABEL_X + 0.32, y0 + BAND_H * 0.32, where, fontsize=9.5, color="#4b5563",
            va="center", style="italic")

    by = y0 + (BAND_H - BOX_H) / 2
    processor(P_X, by, P_W, BOX_H, colour, p_title, p_cmd)
    output(O_X, by, O_W, BOX_H, colour, o_title, o_detail)
    arrow(prev_bottom, (P_X + P_W / 2, by + BOX_H))                 # previous output -> processor
    arrow((P_X + P_W, by + BOX_H / 2), (O_X, by + BOX_H / 2))       # processor -> output
    prev_bottom = (O_X + O_W / 2, by)
    if i < len(STAGES) - 1:
        # next arrow leaves the bottom of this output
        prev_bottom = (O_X + O_W / 2, by)

ax.plot([0, W], [y0, y0], color="#9ca3af", linewidth=1.0, zorder=1)

# legend: small, top right
for k, (kind, text) in enumerate((("processor", "processor: code that runs"),
                                  ("output", "output: saved files"))):
    ly = H - 0.42 - 0.38 * k
    if kind == "processor":
        processor(W - 3.05, ly, 0.75, 0.26, "#6b7280", "", "")
    else:
        output(W - 3.05, ly, 0.75, 0.26, "#6b7280", "", "")
    ax.text(W - 2.18, ly + 0.13, text, fontsize=8.5, va="center", color="#374151")

out = path.join(path.dirname(path.abspath(__file__)), "processing_chain.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", out)
