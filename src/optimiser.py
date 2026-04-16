"""
src/optimiser.py
----------------
Block adjacency graph, connectivity check, and the geofence optimiser.

The optimiser supports two modes via a single unified loop:

  Greedy  — always adds the highest-revenue external neighbour (deterministic).
  SA      — samples neighbours by softmax(delta / T); temperature decays each
            iteration so the algorithm transitions from exploration to exploitation.

Public API
----------
  build_adjacency      — build block adjacency graph from a GeoDataFrame
  is_connected         — BFS connectivity check on a block subgraph
  run_algorithm        — single-seed optimiser (greedy or SA)
  run_multi_start      — run from each seed, return the best result

Private helpers
---------------
  _get_external  — all blocks adjacent to the zone but not yet inside it
  _expand        — select the next block to add (greedy argmax or SA softmax)
  _swap          — deterministic escape: try all (remove b, add c) pairs
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from src.revenue import marginal_revenue


# ── Adjacency graph ────────────────────────────────────────────────────────────

def build_adjacency(blocks_gdf: gpd.GeoDataFrame) -> dict[int, set]:
    """
    Build block adjacency: two blocks are neighbours only if they share a
    boundary edge (intersection length > 0).

    Corner-touching blocks are excluded — they cause pinch-point MultiPolygon
    artefacts in the greedy output.
    """
    adjacency = {i: set() for i in blocks_gdf["block_id"]}
    sindex    = blocks_gdf.sindex

    for idx, block in tqdm(blocks_gdf.iterrows(), total=len(blocks_gdf),
                           desc="Building adjacency"):
        for cand_idx in sindex.intersection(block.geometry.bounds):
            if cand_idx == idx:
                continue
            shared = block.geometry.intersection(
                blocks_gdf.loc[cand_idx, "geometry"]
            )
            if not shared.is_empty and shared.length > 0:
                bid  = block["block_id"]
                cbid = blocks_gdf.loc[cand_idx, "block_id"]
                adjacency[bid].add(cbid)
                adjacency[cbid].add(bid)

    return adjacency


# ── Connectivity check ─────────────────────────────────────────────────────────

def is_connected(selected_ids: set, adjacency: dict) -> bool:
    """Return True if the selected blocks form a connected subgraph (BFS)."""
    if len(selected_ids) <= 1:
        return True
    s       = set(selected_ids)
    visited = set()
    queue   = [next(iter(s))]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in adjacency.get(node, []):
            if nb in s and nb not in visited:
                queue.append(nb)
    return visited == s


# ── Shared sub-steps ───────────────────────────────────────────────────────────

def _get_external(selected: set, adjacency: dict) -> set:
    """All blocks adjacent to the current zone but not yet inside it."""
    external = set()
    for b in selected:
        external.update(adjacency.get(b, set()))
    return external - selected


def _expand(
    selected: set,
    adjacency: dict,
    block_lookup: dict,
    block_area_m2: dict,
    current_area_m2: float,
    max_area_sqkm: float,
    temperature: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int | None, float]:
    """
    Select one external neighbour block to add to the zone.

    Greedy mode (``temperature=None``)
        Returns the block with the highest *positive* marginal revenue.
        Returns ``(None, 0.0)`` when no block improves revenue.

    SA mode (``temperature=float``)
        Samples a block from a softmax distribution weighted by
        ``exp(delta / temperature)``.  Returns a candidate whenever any
        area-feasible external block exists, regardless of sign of delta.
        High temperature → near-uniform; low temperature → near-argmax.

    """
    external   = _get_external(selected, adjacency)
    candidates = []
    deltas     = []

    for block_id in external:
        if (current_area_m2 + block_area_m2.get(block_id, 0)) / 1e6 > max_area_sqkm:
            continue
        delta = marginal_revenue(block_id, selected, block_lookup)
        candidates.append(block_id)
        deltas.append(delta)

    if not candidates:
        return None, 0.0

    if temperature is None:
        # ── Greedy: argmax, require strictly positive delta ────────────────
        best_idx = int(np.argmax(deltas))
        if deltas[best_idx] <= 0.0:
            return None, 0.0
        return candidates[best_idx], deltas[best_idx]

    else:
        # ── SA: softmax sampling; shift by max for numerical stability ─────
        d       = np.array(deltas, dtype=float)
        weights = np.exp((d - d.max()) / max(temperature, 1e-9))
        weights /= weights.sum()
        idx     = rng.choice(len(candidates), p=weights)
        return candidates[idx], deltas[idx]


def _swap(
    selected: set,
    adjacency: dict,
    block_lookup: dict,
    block_area_m2: dict,
    current_area_m2: float,
    max_area_sqkm: float,
) -> tuple[int | None, int | None, float]:
    """
    Deterministic escape: try every (remove b, add c) pair and return the
    best improving swap.

    Only considers swaps that:
    - keep the remaining zone connected after removing *b*
    - keep *c* adjacent to the remaining zone
    - stay within the area cap

    """
    external   = _get_external(selected, adjacency)
    best_delta = 0.0
    best_swap  = None

    for b in list(selected):
        rest = selected - {b}
        if not rest or not is_connected(rest, adjacency):
            continue
        rev_lost = marginal_revenue(b, rest, block_lookup)

        for c in external:
            if not adjacency.get(c, set()) & rest:
                continue
            area_after = (current_area_m2
                          - block_area_m2.get(b, 0)
                          + block_area_m2.get(c, 0))
            if area_after / 1e6 > max_area_sqkm:
                continue
            delta = marginal_revenue(c, rest, block_lookup) - rev_lost
            if delta > best_delta:
                best_delta, best_swap = delta, (b, c)

    if best_swap is None:
        return None, None, 0.0
    return best_swap[0], best_swap[1], best_delta


# ── Unified single-seed optimiser ─────────────────────────────────────────────

def run_algorithm(
    seed_block_id: int,
    adjacency: dict,
    block_lookup: dict,
    block_area_m2: dict,
    mode: str = "greedy",
    T_start: float = 300.0,
    T_end: float = 1.0,
    cooling_rate: float = 0.995,
    max_area_sqkm: float = 15.0,
    max_iters: int = 300,
    random_seed: int = 42,
) -> tuple[set, float, pd.DataFrame]:
    """
    Run the expand-swap optimiser from a single seed block.

    Parameters
    ----------
    mode : {"greedy", "sa"}
        ``"greedy"`` — deterministic argmax expansion (SA params ignored).
        ``"sa"``     — softmax expansion with geometric temperature decay.
    T_start, T_end, cooling_rate : float
        SA schedule: temperature starts at *T_start* and multiplies by
        *cooling_rate* each iteration until it drops below *T_end*.
        Ignored when ``mode="greedy"``.
    random_seed : int
        Seed for the RNG used in SA sampling.  Ignored in greedy mode.

    Returns
    -------
    (best_selected, best_revenue, history_df)
        *best_selected* and *best_revenue* always reflect the highest-revenue
        state ever visited.  In greedy mode these equal the final state (revenue
        is monotonically non-decreasing).  In SA mode they may differ because
        the algorithm can wander downhill after the peak.

    History columns
    ---------------
    iteration, action, block, revenue, best_revenue, area_sqkm, n_blocks,
    temperature  (None in every greedy row)
    """
    if mode not in ("greedy", "sa"):
        raise ValueError(f"mode must be 'greedy' or 'sa', got {mode!r}")

    rng         = np.random.default_rng(random_seed)
    temperature = T_start if mode == "sa" else None

    selected        = {seed_block_id}
    current_revenue = marginal_revenue(seed_block_id, set(), block_lookup)
    current_area_m2 = block_area_m2.get(seed_block_id, 0.0)

    # Best-ever snapshot — identical to current in greedy (revenue never drops)
    best_selected = set(selected)
    best_revenue  = current_revenue
    best_area_m2  = current_area_m2

    history = [{
        "iteration":    0,
        "action":       "seed",
        "block":        seed_block_id,
        "revenue":      current_revenue,
        "best_revenue": best_revenue,
        "area_sqkm":    current_area_m2 / 1e6,
        "n_blocks":     1,
        "temperature":  temperature,
    }]

    mode_info = (f"T_start={T_start}  cooling={cooling_rate}"
                 if mode == "sa" else "greedy")
    print(f"Seed {seed_block_id}  rev=£{current_revenue:.2f}  {mode_info}")

    for iteration in range(1, max_iters + 1):

        # SA-only stop condition
        if mode == "sa" and temperature < T_end:
            print(f"Temperature below T_end at iteration {iteration}.")
            break

        improved = False

        # ── Expand ────────────────────────────────────────────────────────
        block, delta = _expand(
            selected, adjacency, block_lookup,
            block_area_m2, current_area_m2, max_area_sqkm,
            temperature=temperature,
            rng=rng if mode == "sa" else None,
        )
        if block is not None:
            selected.add(block)
            current_revenue += delta
            current_area_m2 += block_area_m2.get(block, 0.0)
            improved = True
            if current_revenue > best_revenue:
                best_revenue  = current_revenue
                best_selected = set(selected)
                best_area_m2  = current_area_m2
            history.append({
                "iteration":    iteration,
                "action":       "expand",
                "block":        block,
                "revenue":      current_revenue,
                "best_revenue": best_revenue,
                "area_sqkm":    current_area_m2 / 1e6,
                "n_blocks":     len(selected),
                "temperature":  temperature,
            })
            if current_area_m2 / 1e6 >= max_area_sqkm:
                print(f"Area cap reached at iteration {iteration}.")
                break

        # ── Swap (deterministic in both modes) ────────────────────────────
        rb, ac, swap_delta = _swap(
            selected, adjacency, block_lookup,
            block_area_m2, current_area_m2, max_area_sqkm,
        )
        if rb is not None:
            selected.discard(rb)
            selected.add(ac)
            current_revenue += swap_delta
            current_area_m2 += block_area_m2.get(ac, 0) - block_area_m2.get(rb, 0)
            improved = True
            if current_revenue > best_revenue:
                best_revenue  = current_revenue
                best_selected = set(selected)
                best_area_m2  = current_area_m2
            history.append({
                "iteration":    iteration,
                "action":       "swap",
                "block":        ac,
                "revenue":      current_revenue,
                "best_revenue": best_revenue,
                "area_sqkm":    current_area_m2 / 1e6,
                "n_blocks":     len(selected),
                "temperature":  temperature,
            })

        if not improved:
            print(f"Converged at iteration {iteration}.")
            break

        if mode == "sa":
            temperature *= cooling_rate

    print(f"RESULT  seed={seed_block_id}  blocks={len(best_selected)}  "
          f"best_rev=£{best_revenue:,.2f}  final_rev=£{current_revenue:,.2f}  "
          f"area={best_area_m2 / 1e6:.2f} sqkm")
    return best_selected, best_revenue, pd.DataFrame(history)


# ── Multi-start ────────────────────────────────────────────────────────────────

def run_multi_start(
    seeds: list[int],
    adjacency: dict,
    block_lookup: dict,
    block_area_m2: dict,
    mode: str = "greedy",
    T_start: float = 300.0,
    T_end: float = 1.0,
    cooling_rate: float = 0.995,
    max_area_sqkm: float = 15.0,
    max_iters: int = 300,
    random_seed: int = 42,
) -> tuple[set, float, pd.DataFrame, int, pd.DataFrame]:
    """
    Run :func:`run_algorithm` from each seed and return the best result.

    Each seed receives a different random seed (``random_seed + i``) so SA runs
    are independent but still reproducible.

    """
    best_revenue  = -1.0
    best_selected = None
    best_history  = None
    best_seed     = None
    seed_results  = []

    for i, seed in enumerate(seeds):
        print(f'\n{"─" * 50}')
        print(f"Run {i + 1}/{len(seeds)}  seed={seed}  mode={mode}")
        print("─" * 50)
        sel, rev, hist = run_algorithm(
            seed, adjacency, block_lookup, block_area_m2,
            mode=mode,
            T_start=T_start, T_end=T_end, cooling_rate=cooling_rate,
            max_area_sqkm=max_area_sqkm, max_iters=max_iters,
            random_seed=random_seed + i,
        )
        seed_results.append({"seed": seed, "revenue": rev, "n_blocks": len(sel)})
        if rev > best_revenue:
            best_revenue  = rev
            best_selected = sel
            best_history  = hist
            best_seed     = seed

    print(f'\n{"=" * 50}')
    print(f"BEST SEED: {best_seed}   revenue=£{best_revenue:,.2f}  mode={mode}")
    print("=" * 50)

    return best_selected, best_revenue, best_history, best_seed, pd.DataFrame(seed_results)

