"""
src/revenue.py
--------------
Revenue lookup tables, computation helpers, and seed selection for the greedy
geofence optimiser.
"""

import geopandas as gpd
import pandas as pd
from collections import defaultdict


def build_revenue_lookup(
    trips_df: pd.DataFrame,
) -> dict[tuple[int, int], float]:
    """
    Aggregate sampled revenue into a ``{(pickup_block, dropoff_block): revenue}``
    dict for O(1) pair lookups.
    """
    grouped = (
        trips_df
        .groupby(["pickup_block", "dropoff_block"])["sampled_revenue"]
        .sum()
    )
    return {(int(i), int(j)): float(rev) for (i, j), rev in grouped.items()}


def build_block_lookup(
    revenue_lookup: dict[tuple[int, int], float],
) -> dict[int, dict[int, float]]:
    """
    Build a per-block index for fast marginal-revenue queries.

    ``block_lookup[b][other]`` is the total revenue from all trips where one
    endpoint is *b* and the other is *other*.  Directional pairs are merged so
    each appears once under each block.
    """
    blk: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for (i, j), rev in revenue_lookup.items():
        blk[i][j] += rev
        if i != j:
            blk[j][i] += rev
    return {k: dict(v) for k, v in blk.items()}


def compute_revenue(
    selected_ids: set | list,
    revenue_lookup: dict[tuple[int, int], float],
) -> float:
    """
    Sum all revenue where *both* pickup and dropoff blocks are in *selected_ids*.

    O(|revenue_lookup|) full scan — use for verification or seed evaluation.
    Use :func:`marginal_revenue` for the greedy inner loop.
    """
    s = set(selected_ids)
    return sum(rev for (i, j), rev in revenue_lookup.items()
               if i in s and j in s)


def compute_pickup_revenue(
    selected_ids: set | list,
    revenue_lookup: dict[tuple[int, int], float],
) -> float:
    """
    Sum all revenue where the pickup block is in *selected_ids*.

    Dropoff block is irrelevant — once a trip is accepted the revenue is
    committed regardless of where the car delivers the passenger or whether
    the dropoff falls inside a later geofence boundary.

    Use this for per-hour evaluation of a breathing geofence: trips are
    attributed to the hour they were quoted (via quote_creation_timestamp),
    and revenue is earned if the car was available to accept the trip
    (pickup_block in zone), not gated on the dropoff location.
    """
    s = set(selected_ids)
    return sum(rev for (i, j), rev in revenue_lookup.items() if i in s)


def marginal_revenue(
    block_id: int,
    selected_set: set,
    block_lookup: dict[int, dict[int, float]],
) -> float:
    """
    Revenue gained by adding *block_id* to *selected_set*.

    Counts trips where one endpoint is *block_id* and the other is already in
    *selected_set*, plus same-block trips.

    *selected_set* must NOT already contain *block_id*.
    """
    nbrs       = block_lookup.get(block_id, {})
    same_block = nbrs.get(block_id, 0.0)
    cross      = sum(rev for other, rev in nbrs.items()
                     if other in selected_set and other != block_id)
    return same_block + cross


def find_local_maxima_seeds(
    blocks_gdf: gpd.GeoDataFrame,
    adjacency: dict[int, set],
    block_lookup: dict[int, dict[int, float]],
    k: int = 5,
    center_bbox: list[float] | None = None,
) -> tuple[list[int], dict[int, float]]:
    """
    Identify *k* well-separated local revenue maxima as greedy starting seeds.

    A block is a local maximum if its unilateral revenue (total revenue from
    all trips touching that block) is ≥ every direct neighbour's.  Seeds are
    pruned so that no two seeds are within 2 graph hops of each other.

    Parameters
    ----------
    center_bbox : [min_lon, min_lat, max_lon, max_lat], optional
        If provided, seed candidates are restricted to blocks whose centroid
        falls within this box.  Use to avoid picking seeds in sparse outer
        areas where trip density is too low to grow a meaningful geofence.

    Returns
    -------
    seeds : list[int]
        Block IDs of candidate seeds, ranked by unilateral revenue.
    unilateral_rev : dict[int, float]
        Unilateral revenue for every block (use for heatmap visualisation).
    """
    unilateral_rev: dict[int, float] = {}
    for block_id in blocks_gdf["block_id"]:
        nbrs = block_lookup.get(int(block_id), {})
        unilateral_rev[int(block_id)] = sum(nbrs.values())

    # Restrict seed candidates to center_bbox if provided
    if center_bbox is not None:
        min_lon, min_lat, max_lon, max_lat = center_bbox
        centroids = blocks_gdf.set_index("block_id").geometry.centroid.to_crs("EPSG:4326")
        center_ids = set(
            centroids.index[
                centroids.x.between(min_lon, max_lon)
                & centroids.y.between(min_lat, max_lat)
            ].astype(int)
        )
        print(f"Seed candidates restricted to center_bbox: {len(center_ids):,} blocks")
    else:
        center_ids = set(int(b) for b in blocks_gdf["block_id"])

    local_maxima: list[tuple[int, float]] = []
    for block_id, rev in unilateral_rev.items():
        if block_id not in center_ids:
            continue
        neighbors = adjacency.get(block_id, set())
        if all(rev >= unilateral_rev.get(n, 0.0) for n in neighbors):
            local_maxima.append((block_id, rev))

    local_maxima.sort(key=lambda x: x[1], reverse=True)
    print(f"Local revenue maxima found: {len(local_maxima):,}")

    seeds: list[int] = []
    excluded: set[int] = set()

    for block_id, rev in local_maxima:
        if block_id in excluded:
            continue
        seeds.append(block_id)
        print(f"  Seed {len(seeds)}: block_id={block_id}, "
              f"unilateral_rev=£{rev:,.2f}")
        if len(seeds) >= k:
            break
        n1 = adjacency.get(block_id, set())
        n2: set[int] = set()
        for n in n1:
            n2.update(adjacency.get(n, set()))
        excluded.update(n1 | n2 | {block_id})

    return seeds, unilateral_rev
