"""
Data-driven bin-edge determination for cross-section variables.

Goal
----
Given the true and reco values of a variable for the reco-selected signal
sample (i.e. `var_signal_df` / `true_var_signal_sel_df` as used elsewhere in
`analysis_village/unfolding/`), choose N bin edges (shared by both the true
and reco axes, as required for a square response matrix) such that:

  1. (baseline) each bin contains roughly the same number of reco-selected
     events -- this keeps the relative statistical uncertainty flat across
     bins.
  2. (hybrid constraint) each bin's diagonal purity and stability stay above
     a chosen floor -- this keeps the true-vs-reco response matrix well
     conditioned for unfolding. A bin narrower than the local detector
     resolution will leak events into its neighbors on the 2D true-vs-reco
     histogram; equal statistics alone does not catch this.

Strategy
--------
Start from an equal-population split of the seed distribution (N quantile
bins). Compute the true-vs-reco 2D histogram on those edges and, for every
bin, the purity (diagonal / reco-marginal) and stability
(diagonal / true-marginal). If any bin fails the purity/stability floor, the
bin count has to shrink -- but rather than only ever merging the one bad bin
into a neighbor (which permanently freezes every other bin's edges at their
original, now-stale quantile split), the resolution loop first looks for the
largest bin count below the current one at which a FRESH equal-population
redistribution of the whole range clears the floor in every bin. If one
exists, it's adopted -- the whole binning gets rebalanced, not just the
region that failed. Only when no fresh redistribution at any bin count down
to `min_bins` can clear the floor (e.g. a narrow, badly-resolved region that
would force redistribution to sacrifice granularity everywhere just to fix
one spot) does the loop fall back to a single targeted merge of just the
worst bin into whichever neighbor gives the better result, then resumes
searching for a redistribution at the new, smaller count. Merging and
redistributing can each necessarily reduce the bin count below the requested
N when the requested N asks for finer binning than the detector resolution
supports; the final N actually achieved -- and how much came from each
mechanism -- is always reported rather than silently forcing the request.

Finally, edges are rounded to `sig_figs` significant figures for readability,
with a monotonicity fix-up in case rounding collapses two close edges.

This module deliberately only takes plain numpy arrays (not dataframes / CAF
column tuples) so it has no dependency on the rest of the analysis_village
tree and can be unit-tested with synthetic data. `determine_bins()` is the
one function most callers need; everything else is a building block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# 1. Rounding to N significant figures, with a monotonicity guarantee.
# ---------------------------------------------------------------------------

def round_to_sigfigs(x: float, sig_figs: int = 2) -> float:
    """Round a single scalar to `sig_figs` significant figures. x == 0 -> 0.0."""
    if x == 0 or not math.isfinite(x):
        return 0.0
    digits = sig_figs - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, digits)


def _step_at(x: float, sig_figs: int) -> float:
    """Size of the smallest step at `x`'s own sig-fig precision."""
    if x == 0:
        return 10 ** (-sig_figs)
    digits = sig_figs - int(math.floor(math.log10(abs(x)))) - 1
    return 10 ** (-digits)


def _nudge_up(x: float, sig_figs: int) -> float:
    """Bump `x` up by one step at its own sig-fig precision, used to break
    ties after rounding collapses two edges onto the same value."""
    return x + _step_at(x, sig_figs)


def _nudge_down(x: float, sig_figs: int) -> float:
    """Bump `x` down by one step at its own sig-fig precision."""
    return x - _step_at(x, sig_figs)


def round_edges_sigfigs(edges: np.ndarray, sig_figs: int = 2,
                         keep_exact: tuple[int, ...] = ()) -> np.ndarray:
    """Round every edge to `sig_figs` sig figs and guarantee the result is
    strictly increasing.

    Parameters
    ----------
    edges : sorted, strictly increasing bin edges (length n_bins + 1).
    sig_figs : significant figures to round to.
    keep_exact : indices (can be negative, e.g. (0, -1)) that should be left
        exactly as given rather than rounded -- use this for physical bounds
        such as a hard 0 threshold or an analysis-cut edge that must stay
        exact rather than drift under rounding.
    """
    edges = np.asarray(edges, dtype=float)
    n = len(edges)
    keep_idx = {i % n for i in keep_exact}

    rounded = np.array([
        edges[i] if i in keep_idx else round_to_sigfigs(edges[i], sig_figs)
        for i in range(n)
    ])

    # Monotonicity fix-up: if rounding (or an already-close pair of quantile
    # edges) collapsed edge i into edge i-1, separate them by one step at the
    # movable edge's own precision. Prefer nudging the later (non-exact) edge
    # up; if that edge is pinned exact instead, nudge the earlier edge down
    # and cascade backward in case that creates a new collision. This is a
    # rare edge case (very narrow bins, or an edge landing right next to a
    # pinned vmin/vmax) but silently returning non-monotonic edges would be
    # worse.
    for i in range(1, n):
        if rounded[i] <= rounded[i - 1]:
            if i not in keep_idx:
                rounded[i] = _nudge_up(rounded[i - 1], sig_figs)
            elif i - 1 not in keep_idx:
                k = i - 1
                rounded[k] = _nudge_down(rounded[i], sig_figs)
                while k > 0 and rounded[k] <= rounded[k - 1]:
                    if k - 1 in keep_idx:
                        raise ValueError(
                            f"Edge {k - 1} is pinned exact via keep_exact but "
                            f"rounding collapsed it against edge {k} "
                            f"({rounded[k]}); widen the bin or drop it from "
                            f"keep_exact."
                        )
                    rounded[k - 1] = _nudge_down(rounded[k], sig_figs)
                    k -= 1
            else:
                raise ValueError(
                    f"Edges {i - 1} and {i} are both pinned exact via "
                    f"keep_exact but rounding collapsed them onto the same "
                    f"value ({rounded[i]}); the requested bin is narrower "
                    f"than {sig_figs} sig figs can represent here."
                )

    return rounded


# ---------------------------------------------------------------------------
# 2. Equal-population initial split.
# ---------------------------------------------------------------------------

def equal_population_edges(values: np.ndarray, n_bins: int,
                            vmin: float | None = None,
                            vmax: float | None = None) -> np.ndarray:
    """Quantile-based bin edges giving (as close to) equal counts of
    `values` per bin. `vmin`/`vmax` override the outer edges with an exact
    physical bound (e.g. an analysis threshold) instead of the sample min/max.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite values to bin.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) < n_bins + 1:
        raise ValueError(
            f"Only {len(edges) - 1} distinct bins possible from {len(values)} "
            f"values for n_bins={n_bins} (too many repeated values / too few "
            f"events). Reduce n_bins."
        )

    if vmin is not None:
        edges[0] = vmin
    if vmax is not None:
        edges[-1] = vmax
    return edges


# ---------------------------------------------------------------------------
# 3. Diagonal purity / stability from the true-vs-reco 2D histogram.
# ---------------------------------------------------------------------------

def diagonal_metrics(true_vals: np.ndarray, reco_vals: np.ndarray,
                      edges: np.ndarray, weights: np.ndarray | None = None):
    """2D-histogram `true_vals` vs `reco_vals` on shared `edges` and return
    (counts2d, purity, stability, bin_totals_true, bin_totals_reco).

    counts2d[i, j] = # events with true in bin i AND reco in bin j
    (matches the (true, reco) argument order used by
    `unfolding.unfolding_inputs.get_smear_matrix`).

    purity[i]    = counts2d[i, i] / sum_k counts2d[k, i]   (reco-bin-i purity:
                   of events reconstructed into reco bin i, what fraction are
                   truly from bin i)
    stability[i] = counts2d[i, i] / sum_k counts2d[i, k]   (true-bin-i
                   stability: of events truly in bin i, what fraction are
                   reconstructed back into reco bin i)
    """
    counts2d, _, _ = np.histogram2d(true_vals, reco_vals, bins=[edges, edges],
                                     weights=weights)
    diag = np.diag(counts2d)
    true_marginal = counts2d.sum(axis=1)   # sum over reco, per true bin
    reco_marginal = counts2d.sum(axis=0)   # sum over true, per reco bin

    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.divide(diag, reco_marginal,
                            out=np.zeros_like(diag), where=reco_marginal != 0)
        stability = np.divide(diag, true_marginal,
                               out=np.zeros_like(diag), where=true_marginal != 0)

    return counts2d, purity, stability, true_marginal, reco_marginal


# ---------------------------------------------------------------------------
# 4. Purity-floor resolution loop: prefer a fresh redistribution, fall back
#    to a targeted merge only where redistribution can't clear the floor.
# ---------------------------------------------------------------------------

def _score(edges, true_vals, reco_vals, weights):
    _, purity, stability, *_ = diagonal_metrics(true_vals, reco_vals, edges, weights)
    return purity, stability


def _passes(edges, true_vals, reco_vals, weights, threshold):
    p, s = _score(edges, true_vals, reco_vals, weights)
    return np.minimum(p, s).min() >= threshold


def _merge_worst_bin_once(edges: np.ndarray, true_vals: np.ndarray,
                           reco_vals: np.ndarray,
                           weights: np.ndarray | None) -> np.ndarray:
    """Merge the single worst-scoring bin into whichever neighbor gives the
    better resulting diagonal. One step only -- the caller loops."""
    purity, stability = _score(edges, true_vals, reco_vals, weights)
    worst_metric = np.minimum(purity, stability)
    i = int(np.argmin(worst_metric))
    n_bins = len(edges) - 1

    candidates = []
    if i > 0:
        candidates.append(np.delete(edges, i))       # merge bin i into bin i-1
    if i < n_bins - 1:
        candidates.append(np.delete(edges, i + 1))   # merge bin i into bin i+1
    if not candidates:
        return edges  # single bin left, nothing to merge with

    best_trial, best_floor = None, -np.inf
    for trial in candidates:
        p, s = _score(trial, true_vals, reco_vals, weights)
        floor = np.minimum(p, s).min()
        if floor > best_floor:
            best_floor, best_trial = floor, trial
    return best_trial


def enforce_purity_floor(edges: np.ndarray, true_vals: np.ndarray,
                          reco_vals: np.ndarray, seed_vals: np.ndarray,
                          threshold: float = 0.6,
                          weights: np.ndarray | None = None,
                          min_bins: int = 2,
                          vmin: float | None = None,
                          vmax: float | None = None,
                          prefer_redistribution: bool = True,
                          ) -> tuple[np.ndarray, int, int, bool]:
    """Shrink the bin count until every bin clears `min(purity, stability) >=
    threshold`, preferring a full rebalance over a local merge at each step.

    At each iteration where the current edges have a failing bin: if
    `prefer_redistribution`, search bin counts from (current - 1) down to
    `min_bins` for the LARGEST one at which a fresh equal-population split of
    `seed_vals` (via `equal_population_edges`) clears the floor in every bin;
    adopt it if found. Only when no such redistribution exists at any bin
    count in that range does this fall back to `_merge_worst_bin_once`, which
    only touches the offending bin and its chosen neighbor, leaving everyone
    else's edges alone. Either way the loop then re-checks and repeats.

    `min_bins` is a hard floor: if it's reached while a bin still fails the
    threshold (e.g. several spatially-separated bad regions that no single
    global redistribution can fix at once, combined with a `min_bins` too
    tight for enough merges), the loop stops anyway rather than going lower --
    `satisfied=False` in the return signals this rather than silently
    pretending the floor was cleared.

    Returns (final_edges, n_redistributions, n_merges, satisfied).
    """
    edges = np.array(edges, dtype=float)
    n_redistributions = 0
    n_merges = 0
    satisfied = _passes(edges, true_vals, reco_vals, weights, threshold)

    while len(edges) - 1 > min_bins and not satisfied:
        n_current = len(edges) - 1
        redistributed = False
        if prefer_redistribution:
            for candidate_n in range(n_current - 1, min_bins - 1, -1):
                candidate_edges = equal_population_edges(
                    seed_vals, candidate_n, vmin=vmin, vmax=vmax
                )
                if _passes(candidate_edges, true_vals, reco_vals, weights, threshold):
                    edges = candidate_edges
                    n_redistributions += 1
                    redistributed = True
                    satisfied = True
                    break

        if not redistributed:
            new_edges = _merge_worst_bin_once(edges, true_vals, reco_vals, weights)
            if len(new_edges) == len(edges):
                break  # couldn't merge further (single bin), give up
            edges = new_edges
            n_merges += 1
            satisfied = _passes(edges, true_vals, reco_vals, weights, threshold)

    return edges, n_redistributions, n_merges, satisfied


# ---------------------------------------------------------------------------
# 5. Top-level API.
# ---------------------------------------------------------------------------

@dataclass
class BinningResult:
    edges: np.ndarray                 # final, rounded, shared true/reco edges
    n_bins_requested: int
    n_bins_final: int
    counts_true: np.ndarray           # per-bin true-marginal count (selected sample)
    counts_reco: np.ndarray           # per-bin reco-marginal count
    purity: np.ndarray
    stability: np.ndarray
    counts2d: np.ndarray
    redistributions_applied: int      # times the whole binning was rebalanced at a lower N
    merges_applied: int               # times a single bin was merged into a neighbor as a fallback
    floor_satisfied: bool             # False if min_bins was hit before every bin cleared the floor
    diagnostics: str = field(repr=False, default="")

    def print_table(self):
        print(self.diagnostics)


def _diagnostics_table(edges, counts_true, counts_reco, purity, stability) -> str:
    lines = [
        f"{'bin':>4} {'[lo, hi)':>22} {'N_true':>8} {'N_reco':>8} "
        f"{'purity':>7} {'stability':>9}"
    ]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lines.append(
            f"{i:>4} {f'[{lo:g}, {hi:g})':>22} {counts_true[i]:>8.0f} "
            f"{counts_reco[i]:>8.0f} {purity[i]:>7.2f} {stability[i]:>9.2f}"
        )
    return "\n".join(lines)


def determine_bins(true_vals: np.ndarray, reco_vals: np.ndarray, n_bins: int,
                    population: str = "reco",
                    purity_threshold: float = 0.6,
                    stability_threshold: float | None = None,
                    enforce_purity: bool = True,
                    prefer_redistribution: bool = True,
                    sig_figs: int = 2,
                    vmin: float | None = None,
                    vmax: float | None = None,
                    keep_exact: tuple[int, ...] = (0, -1),
                    min_bins: int = 2,
                    weights: np.ndarray | None = None) -> BinningResult:
    """Determine `n_bins` shared true/reco bin edges for a cross-section
    variable from the reco-selected signal sample.

    Parameters
    ----------
    true_vals, reco_vals : true and reco values for the SAME reco-selected
        signal events (same length, index-aligned) -- e.g.
        `true_var_signal_sel_df`, `var_signal_df` as passed to
        `unfolding.unfolding_inputs.get_smear_matrix`.
    n_bins : requested/target number of bins.
    population : "reco" (default) or "true" -- which array's quantiles seed
        the initial equal-population split. "reco" matches what actually
        limits your measured statistical uncertainty per bin.
    purity_threshold, stability_threshold : minimum acceptable per-bin
        purity/stability (diagonal fraction of the true-vs-reco histogram).
        stability_threshold defaults to purity_threshold if not given.
    enforce_purity : if True (the hybrid strategy), bins failing the floor
        force the bin count down (via redistribution and/or merging -- see
        `prefer_redistribution`), so the final bin count can be < n_bins.
        If False, returns the raw equal-population split at exactly n_bins
        with diagnostics only (no adjustment) -- use this to see what plain
        equal-statistics binning would look like.
    prefer_redistribution : if True (default), each time the bin count has to
        shrink to fix a purity/stability violation, first search for the
        largest bin count at which a FRESH equal-population redistribution of
        the whole range clears the floor everywhere, and adopt it if found --
        this rebalances all bins, not just the one that failed. Only when no
        redistribution at any bin count (down to min_bins) clears the floor
        does a single targeted merge of the worst bin get applied as a
        fallback, after which redistribution is tried again at the new,
        smaller count. Set to False to skip straight to merge-only behavior
        (each violation merges just that bin into a neighbor, leaving every
        other bin's edges untouched -- cheaper, but bin counts can end up
        very uneven since nothing ever rebalances after a merge).
    sig_figs : round final edges to this many significant figures.
    vmin, vmax : pin the outer edges to exact physical bounds instead of the
        sample min/max (e.g. an analysis threshold or a hard 0).
    keep_exact : edge indices left unrounded (default: first and last, since
        those are usually the physical range bounds set by vmin/vmax).
    min_bins : the bin count never drops below this, from either mechanism.
    weights : optional per-event weights for the 2D histogram.
    """
    if stability_threshold is None:
        stability_threshold = purity_threshold
    # the resolution loop uses a single threshold; take the stricter of the
    # two so both floors are honored (equal in the common case).
    merge_threshold = max(purity_threshold, stability_threshold)

    seed_vals = reco_vals if population == "reco" else true_vals
    edges = equal_population_edges(seed_vals, n_bins, vmin=vmin, vmax=vmax)

    n_redistributions = 0
    n_merges = 0
    if enforce_purity:
        edges, n_redistributions, n_merges, _ = enforce_purity_floor(
            edges, true_vals, reco_vals, seed_vals, threshold=merge_threshold,
            weights=weights, min_bins=min_bins, vmin=vmin, vmax=vmax,
            prefer_redistribution=prefer_redistribution,
        )

    edges = round_edges_sigfigs(edges, sig_figs=sig_figs, keep_exact=keep_exact)

    # Recompute on the FINAL, rounded edges -- rounding can shift purity/stability
    # slightly from whatever `enforce_purity_floor` last saw pre-rounding, so this
    # is the authoritative check, and it's computed the same way whether or not
    # `enforce_purity` even ran (an unenforced equal-population split might
    # legitimately already satisfy the floor, or might not -- either way this
    # reports the truth rather than assuming).
    counts2d, purity, stability, counts_true, counts_reco = diagonal_metrics(
        true_vals, reco_vals, edges, weights=weights
    )
    floor_satisfied = bool(np.minimum(purity, stability).min() >= merge_threshold)

    diagnostics = _diagnostics_table(edges, counts_true, counts_reco, purity, stability)
    if n_redistributions or n_merges:
        verb = "applied to satisfy" if floor_satisfied else "applied but did NOT fully clear"
        diagnostics = (
            f"requested n_bins={n_bins}, final n_bins={len(edges) - 1} "
            f"({n_redistributions} redistribution(s), {n_merges} merge(s) "
            f"{verb} purity/stability >= {merge_threshold:.2f})\n" + diagnostics
        )
    else:
        diagnostics = f"n_bins={len(edges) - 1} (no adjustment needed)\n" + diagnostics
    if not floor_satisfied:
        if enforce_purity:
            warning = (
                f"WARNING: min_bins={min_bins} was reached before every bin "
                f"cleared the purity/stability floor -- see rows below the "
                f"threshold. Lower min_bins, lower purity_threshold, or accept "
                f"the remaining migration."
            )
        else:
            warning = (
                f"WARNING: enforce_purity=False, so no adjustment was attempted "
                f"-- this plain equal-population split does not clear "
                f"purity/stability >= {merge_threshold:.2f} in every bin. See "
                f"rows below the threshold."
            )
        diagnostics = warning + "\n" + diagnostics

    return BinningResult(
        edges=edges,
        n_bins_requested=n_bins,
        n_bins_final=len(edges) - 1,
        counts_true=counts_true,
        counts_reco=counts_reco,
        purity=purity,
        stability=stability,
        counts2d=counts2d,
        redistributions_applied=n_redistributions,
        merges_applied=n_merges,
        floor_satisfied=floor_satisfied,
        diagnostics=diagnostics,
    )
