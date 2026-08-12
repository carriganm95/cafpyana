"""Generic systematics-covariance loading: the syst-disk tree (see
:mod:`pyanalib.syst_disk_layout`), the category-summary export, and a
GENIE-signal/background covariance pickle.

No hardcoded analysis-specific paths or values live here -- analysis packages
(e.g. ``analysis_village.nueNp0Pi.utils``) supply their own default roots and,
for the two loaders with an analysis-specific data source
(:func:`get_syst_unc`'s ``cosmics`` component, :func:`load_overlay_syst_cov_frac`'s
category-summary export), an explicit loader **hook** callable. Mirrors the
``pyanalib.dataset_paths`` / ``pyanalib.event_selection_pipeline`` split: this
module is the reusable engine, the analysis package is the wiring.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np

from pyanalib.logging_utils import get_logger
from pyanalib.syst_disk_layout import (
    FILE_GENIE,
    SUB_GENIE,
    SYST_DISK_ENV,
    category_out_dir,
    category_summary_npz_path,
    syst_disk_paths,
)

logger = get_logger(__name__)


def _fail_syst_disk(msg: str) -> None:
    """Emit a high-visibility error and abort (no silent fallbacks).

    Logged at ERROR level, which is shown regardless of CAFPYANA_LOG_LEVEL
    (default WARNING) -- see ``pyanalib.logging_utils``.
    """
    logger.error("[systematics disk] %s", msg)
    raise FileNotFoundError(msg)


def resolve_syst_disk_root(explicit, syst_disk_env: str = SYST_DISK_ENV):
    """Return normalized syst disk root from argument or the ``syst_disk_env`` env var."""
    root = explicit or os.environ.get(syst_disk_env)
    if not root:
        _fail_syst_disk(
            "No syst disk root. Set environment variable %s or pass syst_disk_root= to "
            "get_syst_unc(). Expected layout: <root>/MCstat/mcstat_syst_dict.npz, "
            "<root>/Flux/flux_syst_dict.npz, … — see pyanalib.syst_disk_layout."
            % syst_disk_env
        )
    resolved = syst_disk_paths(root)["root"]
    logger.debug("resolved syst disk root: %s", resolved)
    return resolved


# Keys accepted by ``get_syst_unc(..., syst_components=...)`` (case-insensitive strings).
SYST_UNC_DISK_KEYS = ("mcstat", "flux", "g4", "genie", "cosmics", "detector")
SYST_UNC_FLAT_KEYS = ("pot", "ntargets")
SYST_UNC_ALL_KEYS = SYST_UNC_DISK_KEYS + SYST_UNC_FLAT_KEYS
_SYST_UNC_DISK_LABELS = {
    "mcstat": "MC stat.",
    "genie": "GENIE",
    "flux": "Flux",
    "g4": "G4",
    "cosmics": "Cosmics",
    "detector": "Detector",
}


def get_syst_unc(
    var_config,
    plot=False,
    save_fig=False,
    save_name=None,
    syst_disk_root=None,
    syst_components=None,
    genie_cov_frac_key: str = "genie",
    skip_missing_vars: bool = False,
    *,
    syst_disk_env: str = SYST_DISK_ENV,
    cosmics_flat_uncorrelated_cov_frac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    pot_frac_unc: float = 0.02,
    ntargets_frac_unc: float = 0.01,
    dpi: int = 300,
    fig_ext: str = ".png",
):
    """Load fractional covariance blocks from the syst-disk tree and combine into total covariance.

    All inputs live under a single root directory (see ``pyanalib.syst_disk_layout``): ``MCstat/``,
    ``Flux/``, ``G4/``, ``GENIE/``, ``Cosmics/``, ``Detector/``. If ``syst_disk_root`` is omitted,
    the ``syst_disk_env`` env var must be set. **Missing files abort with a loud error** — there
    are no alternate search paths or dated campaign fallbacks.

    Parameters
    ----------
    syst_components
        Optional subset of uncertainty sources to include. Each entry is a string, case-insensitive,
        chosen from disk-backed keys ``mcstat``, ``flux``, ``g4``, ``genie``, ``cosmics``,
        ``detector`` and flat correlated terms ``pot``, ``ntargets``. If ``None`` (default), all
        of the above are included (original behavior). Only files needed for the selected disk
        keys are required on disk.
    genie_cov_frac_key
        Which matrix to read from ``GENIE/cov_mat_dict.pkl`` for the ``genie`` disk component:
        ``"genie"`` (response / **xsec** path) or ``"genie_rate"`` (**rate** reweight path).
    skip_missing_vars
        If ``True``, omit disk-backed components whose files lack ``var_config.var_save_name``,
        whose covariance shape does not match ``var_config`` bins, or that otherwise fail to
        combine for this variable, instead of raising. Useful for overlay plots when only a
        subset of variables has been produced on the syst disk.
    cosmics_flat_uncorrelated_cov_frac
        Optional ``cov_frac -> cov_frac`` callable applied to the loaded ``cosmics`` fractional
        covariance before use. If ``cosmics`` is requested (via ``syst_components``) and this
        hook is omitted, the caller's analysis-specific cosmics-flattening step is skipped
        entirely rather than guessed at here.
    """
    if syst_components is None:
        active = frozenset(SYST_UNC_ALL_KEYS)
    else:
        active = frozenset(str(x).lower() for x in syst_components)
        unknown = active - frozenset(SYST_UNC_ALL_KEYS)
        if unknown:
            raise ValueError(
                "Invalid syst_components keys: %s. Allowed: %s"
                % (", ".join(sorted(unknown)), ", ".join(SYST_UNC_ALL_KEYS))
            )

    logger.debug(
        "get_syst_unc(var=%r, components=%s, skip_missing_vars=%s)",
        var_config.var_save_name, sorted(active), skip_missing_vars,
    )

    need_disk = any(k in active for k in SYST_UNC_DISK_KEYS)
    root = None
    paths = None
    if need_disk:
        root = resolve_syst_disk_root(syst_disk_root, syst_disk_env=syst_disk_env)
        paths = syst_disk_paths(root)
        missing = [
            (k, paths[k])
            for k in SYST_UNC_DISK_KEYS
            if k in active and not os.path.isfile(paths[k])
        ]
        if missing:
            detail = "\n".join("  [%s] %s" % (role, pth) for role, pth in missing)
            _fail_syst_disk(
                "Missing systematic covariance file(s). Run the producer pipelines into the "
                "expected locations, then retry:\n%s" % detail
            )

    def _load_disk_frac_cov(key: str) -> np.ndarray:
        assert paths is not None
        if key == "mcstat":
            blob = np.load(paths["mcstat"], allow_pickle=True)
            return dict(blob)[var_config.var_save_name].item()["MCstat"]["cov_frac"]
        if key == "flux":
            blob = np.load(paths["flux"], allow_pickle=True)
            return dict(blob)[var_config.var_save_name].item()["flux"]["cov_frac"]
        if key == "g4":
            blob = np.load(paths["g4"], allow_pickle=True)
            return dict(blob)[var_config.var_save_name].item()["G4"]["cov_frac"]
        if key == "genie":
            if genie_cov_frac_key not in ("genie", "genie_rate"):
                raise ValueError(
                    "genie_cov_frac_key must be 'genie' or 'genie_rate', got %r" % (genie_cov_frac_key,)
                )
            with open(paths["genie"], "rb") as gf:
                genie_blob = pickle.load(gf)
            row = genie_blob[var_config.var_save_name]
            if genie_cov_frac_key not in row:
                raise KeyError(
                    "GENIE pickle for %r has no %r (keys: %s)"
                    % (var_config.var_save_name, genie_cov_frac_key, sorted(row.keys()))
                )
            return row[genie_cov_frac_key]
        if key == "cosmics":
            blob = np.load(paths["cosmics"], allow_pickle=True)
            return dict(blob)[var_config.var_save_name].item()["Cosmics"]["cov_frac"]
        if key == "detector":
            blob = np.load(paths["detector"], allow_pickle=True)
            return dict(blob)["detector"].item()[var_config.var_save_name]["cov_frac"]
        raise KeyError(key)

    import matplotlib.pyplot as plt  # local import: only needed when plot=True

    frac_uncert_total = np.zeros(len(var_config.bin_centers))
    frac_cov_matrix_total = np.zeros((len(var_config.bin_centers), len(var_config.bin_centers)))

    n_bins = len(var_config.bin_centers)
    for key in SYST_UNC_DISK_KEYS:
        if key not in active:
            continue
        syst_name = _SYST_UNC_DISK_LABELS[key]
        try:
            logger.debug("loading %s fractional covariance from %s", syst_name, paths[key])
            syst = _load_disk_frac_cov(key)
            if key == "cosmics" and cosmics_flat_uncorrelated_cov_frac is not None:
                syst = cosmics_flat_uncorrelated_cov_frac(syst)
            syst_uncert = np.sqrt(np.diag(syst))
            if key == "cosmics" and cosmics_flat_uncorrelated_cov_frac is not None:
                flat_val = float(np.max(syst_uncert)) if len(syst_uncert) else 0.0
                syst_uncert = flat_val * np.ones(n_bins)
            if syst.shape != (n_bins, n_bins):
                raise ValueError(
                    "cov_frac shape %s does not match %d bins for %r"
                    % (syst.shape, n_bins, var_config.var_save_name)
                )
            frac_uncert_total += syst_uncert ** 2
            frac_cov_matrix_total += syst
            if plot:
                plt.hist(
                    var_config.bin_centers,
                    bins=var_config.bins,
                    weights=syst_uncert,
                    histtype="step",
                    linewidth=2,
                    label=syst_name,
                )
        except (KeyError, ValueError) as ex:
            if not skip_missing_vars:
                raise
            logger.warning(
                "[get_syst_unc] skip %s for %r: %s",
                syst_name, var_config.var_save_name, ex,
            )
            continue

    if "pot" in active:
        syst_name = "POT"
        syst_uncert = pot_frac_unc * np.ones(len(var_config.bin_centers))
        frac_uncert_total += syst_uncert ** 2
        frac_cov_matrix_total += np.diag(syst_uncert ** 2)
        if plot:
            plt.hist(var_config.bin_centers, bins=var_config.bins, weights=syst_uncert, histtype="step", linewidth=2, label=syst_name)
    if "ntargets" in active:
        syst_name = "Ntargets"
        syst_uncert = ntargets_frac_unc * np.ones(len(var_config.bin_centers))
        frac_uncert_total += syst_uncert ** 2
        frac_cov_matrix_total += np.diag(syst_uncert ** 2)
        if plot:
            plt.hist(var_config.bin_centers, bins=var_config.bins, weights=syst_uncert, histtype="step", linewidth=2, label=syst_name)

    frac_uncert_total = np.sqrt(frac_uncert_total)
    syst = frac_uncert_total
    logger.info(
        "get_syst_unc(%r): mean frac. unc. = %.4f over %d bins (components=%s)",
        var_config.var_save_name, float(np.mean(frac_uncert_total)), n_bins, sorted(active),
    )

    if plot:
        plt.hist(var_config.bin_centers, bins=var_config.bins, weights=frac_uncert_total, histtype="step", linewidth=2, color="k", label="Total")

        plt.xlim(var_config.bins[0], var_config.bins[-1])
        plt.ylim(0, max(frac_uncert_total) * 1.4)

        plt.xlabel(var_config.var_labels[1])
        plt.ylabel("Uncertainty [%]")
        plt.legend(fontsize=11, ncol=3, loc="upper center")

        plt.grid(which='major', linestyle='-', linewidth=0.7, alpha=0.7)
        plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.minorticks_on()

        if save_fig:
            plt.savefig(save_name + fig_ext, bbox_inches='tight', dpi=dpi)

        if not plot:
            plt.close()
        else:
            plt.show()

    return syst, frac_cov_matrix_total


# ===========================================================================
# Category-summary & GENIE signal/background covariance loaders.
# ===========================================================================
_CATEGORY_SYST_SUMMARY_CACHE: dict = {}
_GENIE_SB_COV_MAT_CACHE: dict = {}

GENIE_SB_BKGD_RATE_KEY = "genie_bkgd_rate"


def resolve_category_syst_summary_path(
    category_syst_summary_path=None,
    syst_disk_root=None,
    *,
    default_root: str,
    syst_disk_env: str = SYST_DISK_ENV,
):
    """Path to ``CategorySummary/category_syst_summary.npz`` from an export cell.

    ``default_root`` (required) is the analysis's own fallback syst-disk root, used only
    when neither ``syst_disk_root`` nor the ``syst_disk_env`` env var is set.
    """
    if category_syst_summary_path:
        return os.path.abspath(os.path.expanduser(category_syst_summary_path))
    root = syst_disk_root or os.environ.get(syst_disk_env)
    if not root:
        logger.warning(
            "resolve_category_syst_summary_path: no syst_disk_root/%s set, "
            "falling back to hardcoded default root %s",
            syst_disk_env, default_root,
        )
        root = default_root
    return category_summary_npz_path(root)


def load_overlay_syst_cov_frac(
    var_config,
    *,
    syst_kind="rate",
    syst_disk_root=None,
    category_syst_summary_path=None,
    default_root: str,
    syst_category_summary_loader: Callable[[], Tuple[Callable, Callable]],
):
    """Fractional covariance for overlay bands (default: summed category summary).

    ``syst_category_summary_loader`` (required) returns
    ``(load_category_syst_summary, total_cov_frac)`` -- the analysis-specific pair of
    functions that read and combine the category-summary export. There is no generic
    default for this: the export format/location is analysis-specific.
    """
    load_category_syst_summary, total_cov_frac = syst_category_summary_loader()

    path = resolve_category_syst_summary_path(
        category_syst_summary_path, syst_disk_root, default_root=default_root,
    )
    vsn = (
        getattr(var_config, "category_syst_var_save_name", None)
        or var_config.var_save_name
    )
    cache_key = (path, vsn, syst_kind)
    if cache_key in _CATEGORY_SYST_SUMMARY_CACHE:
        logger.debug("category syst summary cache hit: %s", cache_key)
        return _CATEGORY_SYST_SUMMARY_CACHE[cache_key]
    logger.debug("category syst summary cache miss, loading from %s", path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Category syst summary not found: %s (run systematics-summary export cell)"
            % path
        )
    summary = load_category_syst_summary(path)
    cov = total_cov_frac(summary, vsn, kind=syst_kind)
    _CATEGORY_SYST_SUMMARY_CACHE[cache_key] = cov
    return cov


def get_category_summary_syst_unc(
    var_config,
    *,
    syst_kind="rate",
    syst_disk_root=None,
    category_syst_summary_path=None,
    default_root: str,
    syst_category_summary_loader: Callable[[], Tuple[Callable, Callable]],
):
    """Fractional diagonal uncertainty and covariance from ``category_syst_summary.npz``.

    ``syst_kind`` ``"rate"`` uses ``total_rate`` (GENIE rate); ``"xsec"`` uses ``total_xsec``.
    """
    cov = load_overlay_syst_cov_frac(
        var_config,
        syst_kind=syst_kind,
        syst_disk_root=syst_disk_root,
        category_syst_summary_path=category_syst_summary_path,
        default_root=default_root,
        syst_category_summary_loader=syst_category_summary_loader,
    )
    unc = np.sqrt(np.diag(cov))
    return unc, cov


def resolve_genie_sb_cov_mat_pkl(genie_sb_cov_mat_pkl=None, *, default_root_base: str):
    """Path to a ``systematics-genie-SB`` ``GENIE/cov_mat_dict.pkl`` (``genie_bkgd_rate``).

    ``default_root_base`` (required) is the analysis's own plots/output base directory;
    the default sub-path (``systematics-notebook-genie-SB-integrated``) is a shared
    production convention, not something each analysis needs to override.
    """
    if genie_sb_cov_mat_pkl:
        return os.path.abspath(os.path.expanduser(genie_sb_cov_mat_pkl))
    sb_root = os.path.join(default_root_base, "systematics-notebook-genie-SB-integrated")
    return os.path.join(category_out_dir(sb_root, SUB_GENIE), FILE_GENIE)


def load_genie_sb_bkgd_rate_cov_frac(
    var_config, genie_sb_cov_mat_pkl=None, *, default_root_base: str,
):
    """Fractional covariance on background topology rate from a GENIE-SB systematics notebook."""
    pkl_path = resolve_genie_sb_cov_mat_pkl(genie_sb_cov_mat_pkl, default_root_base=default_root_base)
    vsn = var_config.var_save_name
    cache_key = (pkl_path, vsn, GENIE_SB_BKGD_RATE_KEY)
    if cache_key in _GENIE_SB_COV_MAT_CACHE:
        logger.debug("GENIE-SB cov_mat cache hit: %s", cache_key)
        return _GENIE_SB_COV_MAT_CACHE[cache_key]
    logger.debug("GENIE-SB cov_mat cache miss, loading from %s", pkl_path)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            "GENIE SB cov_mat_dict not found: %s (run systematics-genie-SB.ipynb)" % pkl_path
        )
    with open(pkl_path, "rb") as f:
        cov_mat_dict = pickle.load(f)
    if vsn not in cov_mat_dict:
        raise KeyError(
            "Variable %r not in %s (keys sample: %s)"
            % (vsn, pkl_path, ", ".join(sorted(cov_mat_dict.keys())[:8]))
        )
    row = cov_mat_dict[vsn]
    if GENIE_SB_BKGD_RATE_KEY not in row:
        raise KeyError(
            "%r missing in %s for %r (have: %s)"
            % (
                GENIE_SB_BKGD_RATE_KEY,
                pkl_path,
                vsn,
                ", ".join(sorted(row.keys())[:12]),
            )
        )
    cov = np.asarray(row[GENIE_SB_BKGD_RATE_KEY], dtype=np.float64)
    _GENIE_SB_COV_MAT_CACHE[cache_key] = cov
    return cov
