import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from pyanalib.variable_config import is_integrated_var_config

# =========================
# Okabe-Ito palette
# =========================
OKABE_ITO_COLORS = {
    "orange": (0.90, 0.60, 0.0),
    "sky_blue": (0.35, 0.70, 0.90),
    "blue_green": (0.0, 0.60, 0.50),
    "yellow": (0.95, 0.90, 0.25),
    "blue": (0.0, 0.45, 0.70),
    "vermilion": (0.80, 0.30, 0.0),
    "red_purple": (0.80, 0.60, 0.70),
}

COLOR_CYCLES = {
    "okabe_ito": [
        "black",
        OKABE_ITO_COLORS["vermilion"],
        OKABE_ITO_COLORS["sky_blue"],
        OKABE_ITO_COLORS["orange"],
        OKABE_ITO_COLORS["blue_green"],
        OKABE_ITO_COLORS["red_purple"],
        OKABE_ITO_COLORS["blue"],
        OKABE_ITO_COLORS["yellow"]
    ],
    "sbnd_logo": [
        OKABE_ITO_COLORS["vermilion"],
        OKABE_ITO_COLORS["blue_green"],
        OKABE_ITO_COLORS["blue"],
        OKABE_ITO_COLORS["sky_blue"]
    ]
}

# =========================
# Core style functions
# =========================
def apply_color_cycle(cycle_name="okabe_ito"):
    colors = COLOR_CYCLES.get(cycle_name, COLOR_CYCLES["okabe_ito"])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def apply_cvd_palette():
    plt.set_cmap("cividis")

def apply_sea_palette():
    colors = [(1, 1, 1), (0.0, 0.45, 0.70)]
    if "SeaPalette" not in plt.colormaps():
        cmap = LinearSegmentedColormap.from_list("SeaPalette", colors)
        plt.register_cmap("SeaPalette", cmap)
    plt.set_cmap("SeaPalette")


def apply_symmetric_palette():
    colors = [
        (0.0, 0.45, 0.70),
        (1.0, 1.0, 1.0),
        (0.8, 0.3, 0.0)
    ]
    if "SymmetricPalette" not in plt.colormaps():
        cmap = LinearSegmentedColormap.from_list("SymmetricPalette", colors)
        plt.register_cmap(name="SymmetricPalette", cmap=cmap)
    plt.set_cmap("SymmetricPalette")

_color_counter = {c: 0 for c in COLOR_CYCLES.keys()}

def next_color(cycle="okabe_ito", start=None):
    """Return next color in a given SBND color cycle."""
    colors = COLOR_CYCLES[cycle]
    if start is not None:
        _color_counter[cycle] = start % len(colors)
    idx = _color_counter[cycle]
    _color_counter[cycle] = (idx + 1) % len(colors)
    return colors[idx]

# =========================
# SBND text helpers
# =========================

def sbnd_watermark():
    return r"$\bf{SBND}$"

def sbnd_wip(ax, x, y, fontsize=30):
    ax.text(x, y, f"{sbnd_watermark()} Work in Progress", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize,  **{'fontname':'Helvetica'})

def sbnd_preliminary(ax, x, y, ha='left', va='top', fontsize=30):
    ax.text(x, y, f"{sbnd_watermark()} Preliminary", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize,  **{'fontname':'Helvetica'})

def sbnd_data(ax, x, y, ha='left', va='top', fontsize=30):
    ax.text(x, y, f"{sbnd_watermark()} Data", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize,  **{'fontname':'Helvetica'})

def sbnd_official(ax, x, y, fontsize=30):
    ax.text(x, y, f"{sbnd_watermark()}", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize,  **{'fontname':'Helvetica'})

# =========================
# Layout helpers
# =========================
def split_canvas(fig, ysplit=0.23):
    gs = fig.add_gridspec(2, 1, height_ratios=[1 - ysplit, ysplit], hspace=0.)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    return ax1, ax2

def center_titles(ax):
    ax.title.set_ha('center')
    ax.set_title(ax.get_title(), ha='center')
    ax.xaxis.label.set_ha('center')
    ax.yaxis.label.set_va('center')
    ax.yaxis.labelpad = 22

# =========================
# Plot annotation helpers (approval watermark / POT / chi2 / GENIE version)
# Moved from analysis_village/nueNp0Pi/utils.py -- analysis-agnostic, operate
# on the current figure's first axes (``plt.gcf().axes[0]``). Not yet merged
# with the sbnd_wip/sbnd_preliminary/sbnd_data/sbnd_official helpers above
# (those take an explicit ``ax``, these use the implicit current-figure axes,
# and add_approval_text has an "internal" mode those don't) -- reconciling
# the two conventions is a follow-up, not done here.
# =========================
def get_textloc_x(values, bins, textloc=[0.05, 0.55]):
    """Pick a text x-position + h-alignment that avoids the fuller half of a histogram."""
    textloc_x, _ = textloc
    n_firsthalf = np.sum(values[:len(bins)//2])
    n_secondhalf = np.sum(values[len(bins)//2:])
    if n_firsthalf < n_secondhalf:
        textloc_x, textloc_ha = textloc_x, 'left'
    else:
        textloc_x, textloc_ha = 1 - textloc_x, 'right'
    return textloc_x, textloc_ha


def add_approval_text(approval, textloc_x, textloc_y, textloc_ha, fontsize=20):
    """Add an 'SBND Internal'/'SBND Preliminary'/'SBND Work in Progress' watermark to the
    current figure's first axes.

    ``"wip"`` was added alongside the original ``"internal"``/``"preliminary"`` modes so that
    plots wanting a "Work in Progress" tag (e.g.
    ``pyanalib.response_matrix_plotting.response_matrix_from_histdata``) render it through this
    same function -- same position/fontsize/mathtext-bold-SBND styling as every other plot in
    the pipeline calls this with -- instead of a separately-styled one-off (see
    ``sbnd_wip`` above, which has its own different position/fontsize/font-family defaults and
    is not used by the render pipeline).
    """
    if approval == "internal":
        approval_text = r"$\mathbf{SBND}$ Internal"
        textcolor = 'rosybrown'
    elif approval == "preliminary":
        approval_text = r"$\mathbf{SBND}$ Preliminary"
        textcolor = 'gray'
    elif approval == "wip":
        approval_text = r"$\mathbf{SBND}$ Work in Progress"
        textcolor = 'black'
    else:
        return  # don't add anything

    ax = plt.gcf().axes[0]
    ax.text(
        textloc_x, textloc_y,
        approval_text,
        transform=ax.transAxes,
        ha=textloc_ha, va='top',
        fontsize=fontsize, color=textcolor
    )


def add_pot_text(pot_text, textloc_x, textloc_y, textloc_ha, fontsize=20):
    textcolor = 'black'
    ax = plt.gcf().axes[0]
    ax.text(
        textloc_x, textloc_y,
        pot_text,
        transform=ax.transAxes,
        ha=textloc_ha, va='top',
        fontsize=fontsize, color=textcolor
    )


def add_chi2_text(
    chi2_val,
    p_val,
    ndof,
    textloc_x,
    textloc_y,
    textloc_ha,
    label="",
    chi2_shape=None,
    p_val_shape=None,
    ndof_shape=None,
):
    ax = plt.gcf().axes[0]
    prefix = f"{label} " if label else ""
    lines = [
        f"{prefix}$\\chi^2$/ndof = {chi2_val:.1f}/{int(ndof)} (p-value = {p_val:.2f})"
    ]
    if chi2_shape is not None and p_val_shape is not None:
        ndof_s = int(ndof_shape) if ndof_shape is not None else max(int(ndof) - 1, 1)
        lines.append(
            f"{prefix}$\\chi^2_{{\\mathrm{{shape}}}}$/ndof = "
            f"{chi2_shape:.1f}/{ndof_s} (p-value = {p_val_shape:.2f})"
        )
    ax.text(
        textloc_x,
        textloc_y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=textloc_ha,
        va="top",
        fontsize=12,
        color="black",
        linespacing=1.35,
    )


def add_genie_version_text(textloc_x, textloc_y, textloc_ha, version="GENIE v3.4.0 AR23_00i_00_000"):
    """Annotate the current figure with a GENIE tune/version string.

    ``version`` defaults to the tag nueNp0Pi was using when this moved out of
    ``utils.py``; pass your analysis's actual tag explicitly rather than relying on
    the default staying correct as productions move on.
    """
    ax = plt.gcf().axes[0]
    ax.text(textloc_x, textloc_y,
            version,
            transform=ax.transAxes,
            ha=textloc_ha, va='top',
            fontsize=12, color='gray')


def format_singlebin_plot():
    ax = plt.gcf().axes[0]
    ax.set_xticks([])


# =========================
# 2D / heatmap plotting helpers
# Moved from analysis_village/nueNp0Pi/utils.py. ``dpi``/``fig_ext`` are now
# explicit kwargs instead of nueNp0Pi/utils.py module globals.
# =========================
def get_text_color(value, cmap=cm.viridis, norm=None):
    """Pick black/white text color with enough contrast against ``cmap(norm(value))``."""
    if norm is None:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    rgba = cmap(norm(value))
    # Compute luminance (perceived brightness)
    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "black" if luminance > 0.5 else "white"


def bin_range_labels(edges):
    return [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges)-1)]


def plot_heatmap(matrix,
                 bins,
                 plot_labels=["", "", ""],
                 approval="internal",
                 verbose=False,
                 plot=True,
                 cmap="bwr",
                 save_fig=False,
                 save_name=None,
                 leave_open=False,
                 dpi=300,
                 fig_ext=".png"):

    nbins = len(bins)
    assert nbins-1 == matrix.shape[0] == matrix.shape[1]
    unif_bin = np.linspace(0., float(nbins - 1), nbins)
    extent = [unif_bin[0], unif_bin[-1], unif_bin[0], unif_bin[-1]]

    x_edges, y_edges = np.array(bins), np.array(bins)
    x_tick_positions, y_tick_positions = (unif_bin[:-1] + unif_bin[1:]) / 2, (unif_bin[:-1] + unif_bin[1:]) / 2
    x_labels, y_labels = bin_range_labels(x_edges), bin_range_labels(y_edges)

    fig, ax = plt.subplots(figsize=(12, 12))
    if cmap == "bwr":
        plt.imshow(matrix, extent=extent, origin="lower", vmin=-1, vmax=1, cmap=cmap)
    else:
        plt.imshow(matrix, extent=extent, origin="lower", cmap=cmap)

    # Find the power-of-10 exponent from one of the (non-NaN) values
    exponent = 0
    flat_matrix = matrix[~np.isnan(matrix)]
    if flat_matrix.size > 0 and np.any(flat_matrix != 0):
        example_value = flat_matrix[0]
        exponent = np.floor(np.log10(abs(example_value)))
        # if exponent is infinite (matrix contains only zeros), set to 0
        if np.isinf(exponent):
            exponent = 0
        exponent = int(exponent)

        formatter = mpl.ticker.FuncFormatter(lambda x, _: f"{x/10**exponent:.2f}")
        cbar = plt.colorbar(shrink=0.7)
        if exponent != 0 and exponent != -1:
            cbar.set_label(plot_labels[2] + f" [10$^{{{exponent}}}$]", fontsize=16)
        else:
            cbar.set_label(plot_labels[2], fontsize=16)
        cbar.ax.yaxis.set_major_formatter(formatter)

    for i in range(nbins-1):      # rows (y)
        for j in range(nbins-1):  # columns (x)
            value = matrix[i, j]
            if not np.isnan(value):  # skip NaNs
                if exponent != -1:
                    significand = value / 10**exponent
                else:
                    significand = value
                plt.text(
                    j + 0.5, i + 0.5,
                    f"{significand:.2f}",
                    ha="center", va="center",
                    color=get_text_color(value),
                    fontsize=10
                )

    plt.xticks(x_tick_positions, x_labels, rotation=45, ha="right")
    plt.yticks(y_tick_positions, y_labels)
    plt.xlabel(plot_labels[0], fontsize=20)
    plt.ylabel(plot_labels[1], fontsize=20)
    if len(plot_labels) > 3:
        plt.title(plot_labels[3], fontsize=20)
    else:
        plt.title(plot_labels[2], fontsize=20)

    if verbose:
        n_diag = np.sum(np.diag(matrix))
        diagonal_ratio = n_diag / np.sum(matrix)
        print(f"Diagonal ratio: {diagonal_ratio:.2f}")
        print(f"True ratio: {np.diag(matrix) / np.sum(matrix, axis=0)}")

    # ===== plot additions =====
    add_approval_text(approval, 0.95, 1.05, "right")

    if save_fig:
        plt.savefig(save_name+fig_ext, bbox_inches='tight', dpi=dpi)

    if plot:
        plt.show()
    elif not leave_open:
        plt.close()


def plot_frac_unc(frac_unc_list,
                  var_config,
                  plot_labels=["", "", ""],
                  legends=None,
                  textloc=[0.05, 0.55],
                  approval="internal",
                  plot=True,
                  save_fig=False,
                  save_name=None,
                  dpi=300,
                  fig_ext=".png"):

    for fidx, frac_unc in enumerate(frac_unc_list):
        color = "C{}".format(fidx)
        if len(frac_unc_list) == 1:
            color = "black"
        plt.hist(var_config.bin_centers, bins=var_config.bins, weights=frac_unc, histtype="step", color=color)

    plt.xlim(var_config.bins[0], var_config.bins[-1])
    plt.xlabel(var_config.var_labels[0])
    plt.ylabel("Fractional Uncertainty")
    plt.title(plot_labels[2])
    plt.grid(True)

    if legends is not None:
        plt.legend(legends)

    textloc_x, textloc_ha = get_textloc_x(frac_unc, var_config.bins, textloc)
    textloc_y = textloc[1]
    add_approval_text(approval, textloc_x, textloc_y, textloc_ha)

    if is_integrated_var_config(var_config):
        format_singlebin_plot()

    if save_fig:
        plt.savefig(save_name+fig_ext, bbox_inches='tight', dpi=dpi)

    if plot == True:
        plt.show()
    else:
        plt.close()


# =========================
# DEFAULT STYLE (auto-apply)
# =========================
def _apply_default_style():
    apply_color_cycle("okabe_ito")

    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "errorbar.capsize": 3,
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "font.family": "sans-serif",
        "font.size": 14,
        #"axes.labelsize": 25,
        #"axes.titlesize": 25,
        "axes.linewidth": 1,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "xtick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.top": False,
        "ytick.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 20,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.subplot.left": 0.15,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.92

    })

_apply_default_style()
