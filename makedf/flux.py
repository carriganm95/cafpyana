"""Flux integration, detector active-volume, and cross-section unit normalization.

Shared across any SBND analysis that needs to convert selected/generated event
counts into a cross section: integrate a flux histogram over POT, look up
detector active-volume constants for a named fiducial-volume convention, and
combine them into a per-bin cross-section normalization unit.

No input-file paths are hardcoded here -- callers pass their own ``fluxfile``
path explicitly (e.g. from a per-analysis paths config such as
``analysis_village/nueNp0Pi/config/datasets.py``) rather than this module
defaulting to one physicist's personal directory.
"""

import numpy as np
import uproot
import matplotlib.pyplot as plt

from makedf.constants import RHO, N_A, M_AR

# Named SBND fiducial-volume conventions (cm^3).
_ACTIVE_VOLUMES_CM3 = {
    "SBND": 380 * 380 * 440,
    "SBND_nohighyz": 380 * 380 * 440 - 380 * (190 - 100) * (450 - 250),
    "SBND_face": 380 * 380 * 50,
    "SBND_face_yzcut": 380 * 380 * 50 - 380 * (190 - 100) * 50,
    "SBND_end": 380 * 380 * 50,
}


def get_integrated_flux(fluxfile, plot=False):
    """Integrate the SBND numu flux histogram in ``fluxfile`` over its full energy range.

    ``fluxfile``: path to a ROOT file containing a ``flux_sbnd_numu`` TH1
    (units: /m^2/1e6 POT, 50 MeV bins).
    """
    flux = uproot.open(fluxfile)
    numu_flux = flux["flux_sbnd_numu"].to_numpy()
    bin_edges = numu_flux[1]
    flux_vals = numu_flux[0]

    if plot:
        fig, ax = plt.subplots()
        plt.hist(bin_edges[:-1], bins=bin_edges, weights=flux_vals, histtype="step", linewidth=2, color="C0")
        plt.xlim(0, 3)
        plt.xlabel("Neutrino Energy [GeV]")
        plt.ylabel("Flux [/m$^{2}$/10$^{6}$ POT]")
        plt.title("SBND $\\nu_\\mu$ Flux")
        plt.savefig("sbnd-flux.pdf", bbox_inches='tight')

    integrated_flux = flux_vals.sum() / (1e4 * 1e6)  # to cm2 # to POT
    print("Integrated flux: %.3e" % integrated_flux)
    return integrated_flux


def get_active_volume(detector="SBND"):
    """Active volume (cm^3) for a named SBND fiducial-volume convention.

    Raises ``ValueError`` for an unrecognized ``detector`` string (previously this
    silently fell through to an ``UnboundLocalError`` -- fixed here).
    """
    try:
        return _ACTIVE_VOLUMES_CM3[detector]
    except KeyError:
        raise ValueError(
            "Unrecognized detector=%r; expected one of %s"
            % (detector, sorted(_ACTIVE_VOLUMES_CM3))
        )


def print_sbnd_octant_vertex_ranges(x0, y0, z0):
    """Print octant labels vs reco vertex (x, y, z) in cm.

    Matches the convention in ``selected_events`` octant labeling: split planes at
    ``x0``, ``y0``, ``z0``; E/W from ``x``, N/S from ``z``, Top/Bottom from ``y``.
    SBND: **East** is ``x < x0``; **West** is ``x >= x0``.
    **South** is ``z < z0``; **North** is ``z > z0`` (``z >= z0`` at the split plane).
    **Top** is ``y >= y0``; **Bottom** is ``y < y0``.
    """
    xf, yf, zf = float(x0), float(y0), float(z0)
    xs, ys, zs = "{:.6g}".format(xf), "{:.6g}".format(yf), "{:.6g}".format(zf)
    print("\n=== SBND octants vs reco vertex [cm]; planes x={}, y={}, z={} ===".format(xs, ys, zs))
    print("  E/W (TPC sides):  E if x < {} (negative x),    W if x >= {}".format(xs, xs))
    print("  N/S:              S if z < {} (lower z),    N if z >= {}".format(zs, zs))
    print("  Top / Bottom:     Bottom if y < {},    Top if y >= {}".format(ys, ys))
    rows = [
        ("W-S-Bottom", "[{}, +inf)".format(xs), "(-inf, {})".format(zs), "(-inf, {})".format(ys)),
        ("W-S-Top", "[{}, +inf)".format(xs), "(-inf, {})".format(zs), "[{}, +inf)".format(ys)),
        ("W-N-Bottom", "[{}, +inf)".format(xs), "[{}, +inf)".format(zs), "(-inf, {})".format(ys)),
        ("W-N-Top", "[{}, +inf)".format(xs), "[{}, +inf)".format(zs), "[{}, +inf)".format(ys)),
        ("E-S-Bottom", "(-inf, {})".format(xs), "(-inf, {})".format(zs), "(-inf, {})".format(ys)),
        ("E-S-Top", "(-inf, {})".format(xs), "(-inf, {})".format(zs), "[{}, +inf)".format(ys)),
        ("E-N-Bottom", "(-inf, {})".format(xs), "[{}, +inf)".format(zs), "(-inf, {})".format(ys)),
        ("E-N-Top", "(-inf, {})".format(xs), "[{}, +inf)".format(zs), "[{}, +inf)".format(ys)),
    ]
    hdr = "{:14}  {:^26}  {:^26}  {:^26}".format("octant", "x range", "z range", "y range")
    print("\n" + hdr)
    print(" " + "-" * (len(hdr) + 2))
    for name, xr, zr, yr in rows:
        print("{:14}  {:^26}  {:^26}  {:^26}".format(name, xr, zr, yr))
    print(
        "\nBoundary vertices: x=x0 uses >= toward West; z=z0 uses >= toward North; y=y0 uses >= toward Top.\n"
    )


def get_xsec_unit(tot_pot, fluxfile, detector="SBND", volume=None):
    """Per-bin cross-section normalization unit: 1 / (integrated flux x N targets).

    ``fluxfile``: required -- no hardcoded default. Pass your analysis's flux-file
    path explicitly (e.g. from a per-analysis paths config).
    """
    tot_flux = get_integrated_flux(fluxfile, plot=False)
    tot_flux *= tot_pot
    print("integrated flux: ", tot_flux)

    V_SBND = get_active_volume(detector)
    if volume is not None:
        print("using custom volume: ", volume)
        V_SBND = volume

    NTARGETS = RHO * V_SBND * (N_A / M_AR)
    print("# of targets: ", NTARGETS)

    xsec_unit = 1 / (tot_flux * NTARGETS)
    print("xsec unit: ", xsec_unit)
    return xsec_unit
