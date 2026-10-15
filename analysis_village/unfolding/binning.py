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

Edge-of-range handling: stability is defined as the diagonal of the
row-normalised smearing matrix -- i.e. counts2d[i,i] / counts2d[i,:].sum(),
where the denominator counts only events with BOTH true in bin i AND reco
inside `[vmin, vmax]`.  Events whose reco migrates entirely outside the
measured range contribute to efficiency loss (true_marginal vs row sums of
counts2d) but do not count against stability; stability therefore measures
smearing-matrix conditioning and matches what the 2D migration-matrix plot
shows.  Purity uses an independent 1D reco-marginal histogram (not the 2D
column sums) so that events whose true value falls outside `edges` still
count against reco-bin purity.

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

    # Restrict the population used for the quantile split to [vmin, vmax]
    # BEFORE computing quantiles, not just after. `values` can legitimately
    # have mass outside that range (e.g. containment loss -- reco values
    # spilling past the stated measurement bounds; see `diagonal_metrics`'s
    # docstring for why callers are expected to pass such values unclipped
    # rather than folding them into an overflow edge bin). Without this, a
    # natural quantile edge can land on the wrong side of vmin/vmax, and then
    # overriding edges[0]/edges[-1] below would break monotonicity -- which
    # crashes `np.histogram2d` downstream rather than failing informatively.
    # Clamping first guarantees every quantile edge already lies within
    # [vmin, vmax], so the override can only ever widen the outer bins toward
    # the stated bound, never invalidate ordering.
    quantile_values = values
    if vmin is not None or vmax is not None:
        lo = vmin if vmin is not None else values.min()
        hi = vmax if vmax is not None else values.max()
        quantile_values = np.clip(values, lo, hi)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(quantile_values, quantiles)
    edges = np.unique(edges)
    if len(edges) < n_bins + 1:
        raise ValueError(
            f"Only {len(edges) - 1} distinct bins possible from {len(values)} "
            f"values for n_bins={n_bins} (too many repeated values -- e.g. a "
            f"large fraction sitting at or beyond vmin/vmax -- or too few "
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
    (counts2d, purity, stability, true_marginal, reco_marginal).

    counts2d[i, j] = # events with true in bin i AND reco in bin j
    (matches the (true, reco) argument order used by
    `unfolding.unfolding_inputs.get_smear_matrix`).

    purity[i]    = counts2d[i, i] / reco_marginal[i]   (reco-bin-i purity: of
                   events reconstructed into reco bin i, what fraction are
                   truly from bin i)
    stability[i] = counts2d[i, i] / counts2d[i, :].sum()   (true-bin-i
                   stability: of events with true in bin i that reconstructed
                   into ANY signal bin, what fraction landed in the correct
                   reco bin i; i.e. the diagonal of the row-normalised
                   smearing matrix)

    `true_marginal` (4th return) is an INDEPENDENT 1D histogram of the full
    `true_vals` array on `edges`, counting all true-bin events regardless of
    where reco falls.  It is returned for efficiency and cross-section
    calculations where containment loss is relevant, but is NOT used as the
    stability denominator.  Stability uses `counts2d.sum(axis=1)` (events
    with both true and reco inside `edges`) so that it measures smearing
    matrix conditioning and agrees with what the 2D plot's row sums show.
    Events whose reco migrates outside `edges` entirely are accounted for as
    efficiency loss (true_marginal vs counts2d row sums) rather than being
    charged against stability.
    """
    counts2d, _, _ = np.histogram2d(true_vals, reco_vals, bins=[edges, edges],
                                     weights=weights)
    diag = np.diag(counts2d)
    true_marginal, _ = np.histogram(true_vals, bins=edges, weights=weights)
    reco_marginal, _ = np.histogram(reco_vals, bins=edges, weights=weights)

    row_sums = counts2d.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        purity = np.divide(diag, reco_marginal,
                            out=np.zeros_like(diag), where=reco_marginal != 0)
        stability = np.divide(diag, row_sums,
                               out=np.zeros_like(diag), where=row_sums != 0)

    return counts2d, purity, stability, true_marginal, reco_marginal


# ---------------------------------------------------------------------------
# 4a. Greedy constraint-based edge finder.
# ---------------------------------------------------------------------------

def _greedy_constrained_edges(
    true_vals: np.ndarray,
    reco_vals: np.ndarray,
    vmin: float,
    vmax: float,
    min_width: float = 0.0,
    min_events: float = 0.0,
    stability_fraction: float = 0.0,
    weights: np.ndarray | None = None,
    n_bins_max: int | None = None,
    n_candidates: int = 500,
    sig_figs: int = 2,
) -> np.ndarray:
    """Greedy forward scan producing bin edges that satisfy hard constraints.

    Starting at vmin, each bin grows rightward until all active constraints
    are met:
      - width:       hi - lo >= min_width
      - events:      diagonal weighted count in [lo, hi) >= min_events
      - stability:   diagonal / true_marginal >= stability_fraction

    Stability = diagonal / true_marginal (fraction of true-bin events reco'd
    back into this bin). Uses independent 1-D marginals on the full arrays,
    consistent with `diagonal_metrics`.

    Candidate right edges are taken from quantiles of the in-range true
    distribution (n_candidates points), so the scan is O(n_bins * n_candidates)
    array operations rather than O(n_events^2). Uses sorted cumulative weight
    arrays for O(log n) per candidate lookup.

    Returns edges always spanning [vmin, vmax]. If the final partial bin cannot
    satisfy all constraints, it is closed at vmax anyway; the caller can detect
    this from BinningResult.floor_satisfied.
    """
    w = (np.ones(len(true_vals)) if weights is None
         else np.asarray(weights, dtype=float))
    true_vals = np.asarray(true_vals, dtype=float)
    reco_vals = np.asarray(reco_vals, dtype=float)

    in_range = (true_vals >= vmin) & (true_vals <= vmax)
    if not in_range.any():
        return np.array([vmin, vmax])

    # Fine candidate right-edge positions drawn from the true distribution.
    q = np.linspace(0.0, 1.0, n_candidates)
    cands = np.unique(np.concatenate([np.quantile(true_vals[in_range], q), [vmax]]))
    cands = np.sort(cands[(cands > vmin) & (cands <= vmax)])

    # Mask for events whose reco is inside the signal range [vmin, vmax].
    # Used as the stability denominator (smearing matrix row sums), so that
    # events with reco outside the signal region count as efficiency loss
    # rather than being charged against stability.
    reco_in_signal = (reco_vals >= vmin) & (reco_vals <= vmax)

    edges: list[float] = [float(vmin)]
    lo = float(vmin)

    while lo < vmax:
        if n_bins_max is not None and len(edges) - 1 >= n_bins_max:
            break

        # Build sorted cumulative-weight arrays for this lo value so each
        # candidate hi costs only O(log n) via searchsorted.

        # True marginal for stability: events with true in [lo, hi) AND reco
        # inside the signal region [vmin, vmax].  Matches the smearing matrix
        # row-sum denominator used by diagonal_metrics.
        reco_ge = reco_vals >= lo
        true_ge = true_vals >= lo
        ts_mask = true_ge & reco_in_signal
        ts_sub = true_vals[ts_mask]
        ts_w = w[ts_mask]
        ts_ord = np.argsort(ts_sub)
        ts_sorted = ts_sub[ts_ord]
        ts_cumw = np.concatenate([[0.0], np.cumsum(ts_w[ts_ord])])

        # Diagonal counts: events with BOTH true and reco in [lo, hi).
        # Key = max(true, reco) — an event is in the diagonal iff its max is < hi.
        both = true_ge & reco_ge
        d_max = np.maximum(true_vals[both], reco_vals[both])
        d_w = w[both]
        d_ord = np.argsort(d_max)
        d_sorted = d_max[d_ord]
        d_cumw = np.concatenate([[0.0], np.cumsum(d_w[d_ord])])

        found_hi: float | None = None
        seen_hi_r: set[float] = set()
        for hi in cands:
            if hi <= lo:
                continue
            # Round the candidate as it will appear in the final edge list so
            # that the constraint check is done on the same value that
            # diagonal_metrics will see. vmax is kept exact (never rounded).
            hi_r = hi if hi == vmax else float(round_to_sigfigs(hi, sig_figs))
            if hi_r in seen_hi_r:
                continue
            seen_hi_r.add(hi_r)
            if hi_r <= lo:
                continue
            if hi_r - lo < min_width:
                continue

            true_marg = ts_cumw[int(np.searchsorted(ts_sorted, hi_r, side="left"))]
            diag = d_cumw[int(np.searchsorted(d_sorted, hi_r, side="left"))]
            if diag < min_events:
                continue

            stability = diag / true_marg if true_marg > 0.0 else 0.0
            if stability < stability_fraction:
                continue

            found_hi = hi_r
            break

        if found_hi is None:
            break

        edges.append(found_hi)
        lo = found_hi

    residual_appended = False
    if edges[-1] != vmax:
        # If the cap stopped us before reaching vmax, extend the last bin
        # to cover the remaining range rather than opening a new bin past the cap.
        if n_bins_max is not None and len(edges) - 1 >= n_bins_max:
            edges[-1] = float(vmax)
        else:
            # No valid edge was found from `lo` — the greedy scan could not
            # satisfy all constraints before reaching vmax.  The remaining
            # events are forced into a residual bin that may violate the
            # stability floor.  Signal this to the caller so it can decide
            # whether to fall back to a different strategy.
            edges.append(float(vmax))
            residual_appended = True

    return np.array(edges), residual_appended


# ---------------------------------------------------------------------------
# 4b. Purity-floor resolution loop: prefer a fresh redistribution, fall back
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
                          sig_figs: int = 2,
                          keep_exact: tuple[int, ...] = (0, -1),
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

    Every candidate is rounded to `sig_figs` (via `round_edges_sigfigs`,
    pinning `keep_exact` indices) BEFORE it's scored against `threshold` or
    adopted -- not just once at the very end in `determine_bins`. Validating
    the pre-rounding version and rounding afterward would let a binning that
    only marginally clears the floor pass the search, then silently fail once
    rounding nudges an edge enough to meaningfully narrow a bin (confirmed in
    practice: one case had pre-rounding worst-case purity/stability at 0.616
    against a 0.6 threshold, then dropped to 0.508 after 2-sig-fig rounding
    moved a single edge from 1.454 to 1.5). Rounding every candidate up front
    means what the search accepts is exactly what ships, so the final round
    in `determine_bins` becomes a no-op re-round for whatever path this
    function took (merging only ever deletes an edge, which can't un-round
    the edges left behind).

    `min_bins` is a hard floor: if it's reached while a bin still fails the
    threshold (e.g. several spatially-separated bad regions that no single
    global redistribution can fix at once, combined with a `min_bins` too
    tight for enough merges), the loop stops anyway rather than going lower --
    `satisfied=False` in the return signals this rather than silently
    pretending the floor was cleared.

    Returns (final_edges, n_redistributions, n_merges, satisfied).
    """
    def _round(e):
        return round_edges_sigfigs(e, sig_figs=sig_figs, keep_exact=keep_exact)

    edges = _round(np.array(edges, dtype=float))
    n_redistributions = 0
    n_merges = 0
    satisfied = _passes(edges, true_vals, reco_vals, weights, threshold)

    while len(edges) - 1 > min_bins and not satisfied:
        n_current = len(edges) - 1
        redistributed = False
        if prefer_redistribution:
            for candidate_n in range(n_current - 1, min_bins - 1, -1):
                candidate_edges = _round(equal_population_edges(
                    seed_vals, candidate_n, vmin=vmin, vmax=vmax
                ))
                if _passes(candidate_edges, true_vals, reco_vals, weights, threshold):
                    edges = candidate_edges
                    n_redistributions += 1
                    redistributed = True
                    satisfied = True
                    break

        if not redistributed:
            # `edges` is already rounded, and merging only deletes an edge
            # (never changes the value of a surviving one), so the result is
            # automatically still rounded -- no extra `_round` call needed.
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


def _diagnostics_table(edges, counts_true, counts_reco, counts_diag, purity, stability) -> str:
    lines = [
        f"{'bin':>4} {'[lo, hi)':>22} {'N_true':>8} {'N_reco':>8} {'N_diag':>8}"
        f"{'purity':>7} {'stability':>9}"
    ]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lines.append(
            f"{i:>4} {f'[{lo:g}, {hi:g})':>22} {counts_true[i]:>8.0f} "
            f"{counts_reco[i]:>8.0f} {counts_diag[i]:>8.0f} {purity[i]:>7.2f} {stability[i]:>9.2f}"
        )
    return "\n".join(lines)


def determine_bins(true_vals: np.ndarray, reco_vals: np.ndarray, n_bins: int | None = None,
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
                    weights: np.ndarray | None = None,
                    min_width: float | None = None,
                    min_events: float | None = None,
                    n_candidates: int = 500,
                    greedy_fallback: bool = False,
                    ) -> BinningResult:
    """Determine shared true/reco bin edges for a cross-section variable.

    Two modes are available, selected automatically based on which parameters
    are supplied.

    Constraint-based mode (new)
    ---------------------------
    Activated when ``min_width`` or ``min_events`` is provided.  A greedy
    forward scan grows each bin from its left edge until all active constraints
    are *simultaneously* satisfied:

      - width:       hi - lo >= min_width
      - events:      diagonal weighted count in [lo, hi) >= min_events
      - stability:   diagonal / true_marginal >= purity_threshold  (per-bin)

    This replaces the equal-population seed with a geometry/statistics-driven
    partition, which avoids very narrow bins where detector resolution drives
    off-diagonal migration.  ``n_bins`` is treated as a soft upper cap on the
    number of bins; omit it (or pass None) to impose no cap.

    Equal-population mode (original)
    ---------------------------------
    Used when neither ``min_width`` nor ``min_events`` is set.  Seeds from
    ``n_bins`` equal-population quantile bins and then shrinks the bin count
    via redistribution and/or merging until ``purity_threshold`` is cleared
    everywhere (if ``enforce_purity=True``).  ``n_bins`` is required in this
    mode.

    Parameters
    ----------
    true_vals, reco_vals : true and reco values for the SAME reco-selected
        signal events (same length, index-aligned).
    n_bins : target/maximum number of bins.  Required for equal-population
        mode; optional (cap) in constraint-based mode.
    population : "reco" (default) or "true" -- which array's quantiles seed
        the equal-population split (equal-population mode only).
    purity_threshold, stability_threshold : minimum acceptable per-bin
        purity/stability (equal-population mode).  In constraint-based mode
        ``purity_threshold`` serves as the stability floor.
        stability_threshold defaults to purity_threshold if not given.
    enforce_purity : equal-population mode only -- if True, merge/redistribute
        to clear the purity floor; if False, return the raw equal-population
        split with diagnostics only.
    prefer_redistribution : equal-population mode only -- prefer a full
        rebalance over a local merge at each step (see module docstring).
    sig_figs : round final edges to this many significant figures.
    vmin, vmax : pin outer edges to exact physical bounds.
    keep_exact : edge indices left unrounded (default: first and last).
    min_bins : equal-population mode only -- bin count floor.
    weights : optional per-event weights.
    min_width : constraint-based mode -- minimum allowed bin width.  Setting
        this (or min_events) activates constraint-based mode.
    min_events : constraint-based mode -- minimum diagonal weighted event count
        per bin (events with both true and reco inside the bin).
    n_candidates : constraint-based mode -- number of quantile points used as
        candidate right edges in the greedy scan (default 500).
    greedy_fallback : constraint-based mode only -- if True and the greedy
        scan leaves a residual last bin that fails the stability floor (because
        no valid right edge was found before vmax), fall back to equal-
        population redistribution with progressively fewer bins until stability
        is cleared.  Stability only is checked (not purity) for consistency
        with the greedy scan's own constraint.  Default False preserves the
        original behaviour of keeping the greedy result and emitting a WARNING.
    """
    if stability_threshold is None:
        stability_threshold = purity_threshold
    merge_threshold = max(purity_threshold, stability_threshold)

    constrained_mode = (min_width is not None) or (min_events is not None)

    if constrained_mode:
        _vmin = float(vmin) if vmin is not None else float(np.nanmin(true_vals))
        _vmax = float(vmax) if vmax is not None else float(np.nanmax(true_vals))
        _min_width = float(min_width) if min_width is not None else 0.0
        _min_events = float(min_events) if min_events is not None else 0.0

        raw_edges, residual_appended = _greedy_constrained_edges(
            true_vals, reco_vals,
            vmin=_vmin, vmax=_vmax,
            min_width=_min_width,
            min_events=_min_events,
            stability_fraction=merge_threshold,
            weights=weights,
            n_bins_max=n_bins,
            n_candidates=n_candidates,
            sig_figs=sig_figs,
        )
        edges = round_edges_sigfigs(raw_edges, sig_figs=sig_figs, keep_exact=keep_exact)

        counts2d, purity, stability, _, counts_reco = diagonal_metrics(
            true_vals, reco_vals, edges, weights=weights
        )
        counts_true = counts2d.sum(axis=1)   # smearing matrix row sums = stability denominator
        floor_satisfied = bool(stability.min() >= merge_threshold)
        n_bins_out = len(edges) - 1
        n_redistributions = 0

        if not floor_satisfied and residual_appended and greedy_fallback:
            # The greedy scan could not find a valid right edge from the final
            # lo position and was forced to close the last bin at vmax with
            # whatever events remained (residual bin).  Fall back to equal-
            # population redistribution, trying progressively fewer bins from
            # n_bins_out-1 down to min_bins and adopting the first split whose
            # stability clears the floor everywhere.  Purity is deliberately
            # not checked here: the greedy scan enforces only stability, and
            # purity can be structurally low near the low edge of the range
            # (many background events reco'd into the first bin) regardless of
            # how bins are divided.
            seed_vals_fb = true_vals if population == "true" else reco_vals
            for fallback_n in range(n_bins_out - 1, min_bins - 1, -1):
                try:
                    fb_edges = round_edges_sigfigs(
                        equal_population_edges(seed_vals_fb, fallback_n,
                                               vmin=_vmin, vmax=_vmax),
                        sig_figs=sig_figs, keep_exact=keep_exact,
                    )
                except ValueError:
                    continue
                _, s = _score(fb_edges, true_vals, reco_vals, weights)
                if s.min() >= merge_threshold:
                    edges = fb_edges
                    counts2d, purity, stability, _, counts_reco = diagonal_metrics(
                        true_vals, reco_vals, edges, weights=weights
                    )
                    counts_true = counts2d.sum(axis=1)
                    floor_satisfied = True
                    n_redistributions = n_bins_out - fallback_n
                    n_bins_out = fallback_n
                    break

        diagnostics = _diagnostics_table(
            edges, counts_true, counts_reco, counts2d.diagonal(), purity, stability
        )
        cap_str = f", cap={n_bins}" if n_bins is not None else ""
        header = (
            f"constraint-based greedy scan: n_bins={n_bins_out}{cap_str}, "
            f"min_width={_min_width:g}, min_events={_min_events:g}, "
            f"stability>={merge_threshold:.2f}"
        )
        if n_redistributions:
            header += (
                f" → equal-pop fallback ({n_redistributions} fewer bin(s) "
                f"than greedy)"
            )
        diagnostics = header + "\n" + diagnostics
        if not floor_satisfied:
            if residual_appended:
                warn = (
                    "WARNING: greedy scan left a residual last bin and the "
                    "equal-pop fallback also failed to clear the stability "
                    "threshold down to min_bins. Lower the threshold, widen "
                    "the range, or reduce min_bins.\n"
                )
            else:
                warn = (
                    "WARNING: last bin was extended to vmax after hitting the "
                    "n_bins cap, but its stability is below the threshold. "
                    "Increase n_bins or lower the threshold.\n"
                )
            diagnostics = warn + diagnostics

        return BinningResult(
            edges=edges,
            n_bins_requested=n_bins if n_bins is not None else n_bins_out,
            n_bins_final=n_bins_out,
            counts_true=counts_true,
            counts_reco=counts_reco,
            purity=purity,
            stability=stability,
            counts2d=counts2d,
            redistributions_applied=n_redistributions,
            merges_applied=0,
            floor_satisfied=floor_satisfied,
            diagnostics=diagnostics,
        )

    # --- Equal-population mode (original) ---
    if n_bins is None:
        raise ValueError(
            "n_bins is required in equal-population mode. Either pass n_bins "
            "or activate constraint-based mode by setting min_width or min_events."
        )

    seed_vals = reco_vals if population == "reco" else true_vals
    edges = equal_population_edges(seed_vals, n_bins, vmin=vmin, vmax=vmax)

    n_redistributions = 0
    n_merges = 0
    if enforce_purity:
        edges, n_redistributions, n_merges, _ = enforce_purity_floor(
            edges, true_vals, reco_vals, seed_vals, threshold=merge_threshold,
            weights=weights, min_bins=min_bins, vmin=vmin, vmax=vmax,
            prefer_redistribution=prefer_redistribution,
            sig_figs=sig_figs, keep_exact=keep_exact,
        )

    edges = round_edges_sigfigs(edges, sig_figs=sig_figs, keep_exact=keep_exact)

    counts2d, purity, stability, _, counts_reco = diagonal_metrics(
        true_vals, reco_vals, edges, weights=weights
    )
    counts_true = counts2d.sum(axis=1)   # smearing matrix row sums = stability denominator
    floor_satisfied = bool(np.minimum(purity, stability).min() >= merge_threshold)

    diagnostics = _diagnostics_table(edges, counts_true, counts_reco, counts2d.diagonal(), purity, stability)
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


# ---------------------------------------------------------------------------
# 6. Plotting helpers: visualize under/overflow WITHOUT feeding clipped
#    values into determine_bins/diagonal_metrics (see module docstring's
#    "Edge-of-range handling" note -- clipping before analysis is exactly the
#    mistake this section exists to avoid; it belongs only at the very last
#    step, for matplotlib's benefit, on a throwaway copy of the array).
# ---------------------------------------------------------------------------

def add_overflow_edges(edges: np.ndarray, low_pad_frac: float = 0.25,
                        high_pad_frac: float = 0.25,
                        mode: str = 'multiplicative') -> np.ndarray:
    """Extend `edges` with one extra bin below `edges[0]` and one above
    `edges[-1]`, for VISUALIZING under/overflow only.

    mode='multiplicative' (default): padding is a fraction of the edge value
    itself -- stays strictly positive and is safe for log-scale axes.  Raises
    if ``edges[0] <= 0``.

    mode='additive': padding is a fraction of the total range
    ``edges[-1] - edges[0]`` -- safe for variables that reach 0 or go negative
    (TKI variables, angles, cosines, etc.).

    Never pass values clipped with `clip_to_plot_edges` (below) into
    `determine_bins` or `diagonal_metrics` -- those need the true unclipped
    values to correctly count migration into/out of the signal region (see
    `diagonal_metrics`'s docstring). This pair of functions is for the plot
    only.
    """
    edges = np.asarray(edges, dtype=float)
    if mode == 'multiplicative':
        if edges[0] <= 0:
            raise ValueError(
                "add_overflow_edges uses multiplicative padding and requires "
                f"edges[0] > 0; got {edges[0]}. Use additive padding instead for "
                "a variable whose range can reach 0 or go negative."
            )
        lo_edge = edges[0] / (1.0 + low_pad_frac)
        hi_edge = edges[-1] * (1.0 + high_pad_frac)
    elif mode == 'additive':
        span = edges[-1] - edges[0]
        lo_edge = edges[0] - low_pad_frac * span
        hi_edge = edges[-1] + high_pad_frac * span
    else:
        raise ValueError(f"mode must be 'multiplicative' or 'additive', got {mode!r}")
    return np.concatenate([[lo_edge], edges, [hi_edge]])


def clip_to_plot_edges(values: np.ndarray, plot_edges: np.ndarray) -> np.ndarray:
    """Clip `values` into `[plot_edges[0], plot_edges[-1])` so under/overflow
    events land in the dedicated outer bins added by `add_overflow_edges`
    instead of being silently dropped by matplotlib's `hist`/`hist2d` (which
    omit points outside the given bin range entirely, rather than showing
    them anywhere). For plotting only -- see `add_overflow_edges`'s docstring.
    """
    plot_edges = np.asarray(plot_edges, dtype=float)
    span = plot_edges[-1] - plot_edges[0]
    eps = max(span * 1e-9, np.finfo(float).eps)
    return np.clip(np.asarray(values, dtype=float),
                    plot_edges[0], plot_edges[-1] - eps)
