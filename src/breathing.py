"""
src/breathing.py
----------------
Dynamic "breathing" geofence — hourly transition functions.

The breathing geofence adapts from one hour to the next via four stages:

  1. Border contract — peel zero-revenue blocks from the zone's outer edge;
                       processes tip-blocks first (fewest internal neighbours)
                       to avoid connectivity refusals on appendix shapes.
                       Only border blocks (those with at least one external
                       neighbour) are candidates; interior blocks are untouched.
  2. Swap            — run the deterministic swap-escape loop to convergence so
                       the zone migrates toward better territory under the new
                       demand signal.  Reuses optimiser._swap unchanged.
  3. Expand          — greedily grow into adjacent blocks that add positive
                       revenue until the area cap is reached or no improving
                       neighbour exists.  Reuses optimiser._expand unchanged.
  4. Border continuity (Option C) — every block dropped from the previous zone
                       must be adjacent to the new zone so a car there can drive
                       one step into the new geofence.  BFS bridging repairs any
                       gap left after stages 1–3.

Revenue semantics
-----------------
  Evaluation uses compute_pickup_revenue (pickup block in zone is sufficient;
  dropoff location does not gate revenue — the trip is committed once accepted).
  The optimisation signal (block_lookup) remains symmetric so that popular
  dropoff areas are valued: cars landing there are already positioned for the
  next pickup within the same zone.

Public API
----------
  load_dated_trips          — join trips with quote date and hour
  build_hourly_lookups      — {hour: revenue_lookup} for a date and list of hours
  build_cumulative_lookup   — single revenue_lookup aggregated across date+hours
  run_border_contract       — peel zero-revenue border blocks, tip-first
  run_transition            — one full hourly transition + border continuity
  enforce_border_continuity — post-transition Option-C repair via BFS bridging
  diff_geofences            — (added_blocks, removed_blocks) between two zones
  project_bbox              — reproject CENTER_BBOX lon/lat into TARGET_CRS
  plot_transition           — single-panel map with added/removed blocks highlighted

Private helpers
---------------
  _bfs_to_zone — BFS shortest path from a block to the nearest zone block
"""

from collections import deque

import pandas as pd
import geopandas as gpd
from shapely.geometry import box as shapely_box

from src.revenue import build_revenue_lookup, marginal_revenue
from src.optimiser import is_connected, _expand, _swap


# ── Data preparation ───────────────────────────────────────────────────────────

def load_dated_trips(
    trips: pd.DataFrame,
    quotes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join trips with quote creation dates and hours.

    Parameters
    ----------
    trips  : trips_assigned DataFrame
             columns: quote_id, sampled_revenue, pickup_block, dropoff_block
    quotes : quotes_canonical DataFrame
             must contain: quote_id, quote_creation_timestamp (tz-aware UTC)

    Returns
    -------
    trips with added 'date' (datetime.date) and 'hour' (int 0-23) columns,
    both in Europe/London timezone.
    """
    dated = trips.merge(
        quotes[["quote_id", "quote_creation_timestamp"]], on="quote_id", how="left"
    )
    ts = pd.to_datetime(dated["quote_creation_timestamp"]).dt.tz_convert("Europe/London")
    dated["date"] = ts.dt.date
    dated["hour"] = ts.dt.hour
    return dated.drop(columns=["quote_creation_timestamp"])


def build_hourly_lookups(
    dated_trips: pd.DataFrame,
    date,
    hours: list,
) -> dict:
    """
    Build one revenue_lookup per hour for a given date.

    Parameters
    ----------
    dated_trips : output of load_dated_trips (must have 'date' and 'hour' columns)
    date        : datetime.date — the day to filter to
    hours       : list of int hour values (e.g. [8, 9, 10, 11, 12, 13])

    Returns
    -------
    {hour: revenue_lookup}
    revenue_lookup = {(pickup_block, dropoff_block): total_revenue}
    """
    day_trips = dated_trips[dated_trips["date"] == date]
    return {
        h: build_revenue_lookup(day_trips[day_trips["hour"] == h])
        for h in hours
    }


def build_cumulative_lookup(
    dated_trips: pd.DataFrame,
    date,
    hours: list,
) -> dict:
    """
    Revenue lookup built from all trips across the given date and hours combined.

    Returns
    -------
    {(pickup_block, dropoff_block): total_revenue}
    """
    mask = (dated_trips["date"] == date) & (dated_trips["hour"].isin(hours))
    return build_revenue_lookup(dated_trips[mask])


# ── Border continuity (Option C) ───────────────────────────────────────────────

def _bfs_to_zone(start: int, curr: set, adjacency: dict) -> list:
    """
    Find the shortest path from *start* to the nearest block in *curr* via BFS.

    Returns
    -------
    Path as a list of block IDs from *start* up to (but not including) the
    first block found in *curr*.  Returns [] if *start* is already adjacent
    to *curr* (no bridging needed).

    Adding this path in reverse order (zone-outward → start) keeps the zone
    connected throughout, since each added block touches the previous one.
    """
    if adjacency.get(start, set()) & curr:
        return []  # already adjacent — no bridge needed

    queue   = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()
        for nb in adjacency.get(node, set()):
            if nb in curr:
                return path  # path to zone found; nb (in curr) is excluded
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))

    return []  # no path found (shouldn't occur in a connected block graph)


def enforce_border_continuity(
    prev: set,
    curr: set,
    adjacency: dict,
    block_area_m2: dict,
    max_area_sqkm: float,
) -> tuple[set, float, int, int]:
    """
    Ensure every block dropped from *prev* is adjacent to *curr*.

    For each removed block with no neighbour in the new zone, the shortest
    path of blocks connecting it to the zone is added in reverse order
    (zone outward toward the removed block).  This preserves connectivity
    because each step attaches to the existing zone before the next one.

    If adding the bridge path would push the zone above the area cap, the
    repair is skipped entirely for that block — the violation is left
    unresolved rather than breaching the cap.

    Motivation
    ----------
    Individual swap steps guarantee the removed block is adjacent to the zone
    *at the time of removal*, but subsequent swaps may move the zone away so
    the block's neighbour is itself gone by the end of the transition.
    This post-processing step repairs those cases where the cap allows it.

    Parameters
    ----------
    prev          : previous hour's block set
    curr          : new zone after contract → swap → expand
    max_area_sqkm : hard area cap — bridges that would breach it are skipped

    Returns
    -------
    (new_curr, new_area_m2, n_bridge_blocks, n_skipped)
    n_bridge_blocks : blocks added as bridges for continuity
    n_skipped       : violations left unresolved because bridge would breach cap
    """
    curr            = set(curr)
    current_area_m2 = sum(block_area_m2.get(b, 0) for b in curr)
    n_bridge        = 0
    n_skipped       = 0

    changed = True
    while changed:
        changed = False
        removed = prev - curr
        for b in removed:
            if not (adjacency.get(b, set()) & curr):
                path = _bfs_to_zone(b, curr, adjacency)

                # Check if bridge fits within the area cap before adding anything
                bridge_area_m2 = sum(
                    block_area_m2.get(br, 0) for br in path if br not in curr
                )
                if (current_area_m2 + bridge_area_m2) / 1e6 > max_area_sqkm:
                    # Bridge would breach cap — skip this violation
                    n_skipped += 1
                    continue

                # Add in reverse: zone-adjacent block first, then outward
                for bridge in reversed(path):
                    if bridge not in curr:
                        curr.add(bridge)
                        current_area_m2 += block_area_m2.get(bridge, 0)
                        n_bridge += 1
                changed = True
                break  # restart with updated curr

    if n_skipped > 0:
        print(
            f"  Border continuity: {n_skipped} violation(s) skipped "
            f"(bridge would breach {max_area_sqkm} sq km cap)"
        )

    return curr, current_area_m2, n_bridge, n_skipped


# ── Transition mechanics ───────────────────────────────────────────────────────

MIN_BLOCKS = 20  # contract will not shrink the zone below this block count


def run_border_contract(
    selected: set,
    adjacency: dict,
    block_lookup: dict,
    block_area_m2: dict,
) -> tuple[set, float]:
    """
    Peel zero-revenue blocks from the zone's outer edge, processing tips first.

    Only *border* blocks are candidates — blocks with at least one neighbour
    outside the current zone.  Interior blocks are never touched directly.

    Ordering: sort border blocks by internal degree ascending (fewest in-zone
    neighbours first) so tip blocks (dead-end appendages) are evaluated before
    bridge blocks that connect larger sub-regions.  Restarts after each removal
    so connectivity checks always reflect the current zone.

    A block is removed when ALL four conditions hold:
      1. It is a border block (has at least one external neighbour).
      2. marginal_revenue(b, rest) <= 0  — zero or negative contribution.
      3. Removing it leaves the zone connected.
      4. The zone would not shrink below MIN_BLOCKS.

    Parameters
    ----------
    block_lookup : this hour's {block_id: {other_block_id: revenue}} lookup

    Returns
    -------
    (new_selected, new_area_m2)
    """
    selected        = set(selected)
    current_area_m2 = sum(block_area_m2.get(b, 0) for b in selected)

    changed = True
    while changed:
        changed = False
        if len(selected) <= MIN_BLOCKS:
            break  # floor reached — stop contracting

        # Border blocks only: at least one neighbour is outside the zone
        # Sorted by internal degree ascending — tips (degree 1) before bridges
        border_blocks = sorted(
            (b for b in selected if adjacency.get(b, set()) - selected),
            key=lambda b: len(adjacency.get(b, set()) & selected),
        )

        for b in border_blocks:
            rest = selected - {b}
            if len(rest) < MIN_BLOCKS:
                continue  # removal would breach the floor
            if not rest or not is_connected(rest, adjacency):
                continue  # removal would disconnect the zone
            if marginal_revenue(b, rest, block_lookup) <= 0:
                selected         = rest
                current_area_m2 -= block_area_m2.get(b, 0)
                changed          = True
                break  # restart with updated selected

    return selected, current_area_m2


BRIDGE_BUFFER_SQKM = 0.5  # sq km reserved for border continuity bridges


def run_transition(
    selected: set,
    adjacency: dict,
    block_lookup: dict,
    block_area_m2: dict,
    max_area_sqkm: float,
) -> tuple[set, float, dict]:
    """
    Adapt a geofence to a new revenue signal:
    border_contract → swap → expand → continuity.

    Swap and expand reuse optimiser._swap and optimiser._expand unchanged and
    use a working cap of (max_area_sqkm - BRIDGE_BUFFER_SQKM) to reserve
    headroom for border continuity bridge blocks.  The full max_area_sqkm cap
    is applied only in the continuity stage — bridges that would still breach
    it are skipped and logged as n_continuity_skipped.

    Parameters
    ----------
    selected     : previous hour's block set
    block_lookup : this hour's {block_id: {other_block_id: revenue}} lookup
    block_area_m2: {block_id: area_m²}
    max_area_sqkm: hard area cap (bridges may use up to BRIDGE_BUFFER_SQKM of it)

    Returns
    -------
    (new_selected, new_area_m2, stats)

    stats
    -----
    n_removed            : blocks dropped by border contract
    n_swapped            : swap iterations applied
    n_added              : net new blocks vs. start of transition
    n_bridge_blocks      : blocks added as continuity bridges
    n_continuity_skipped : violations skipped because bridge would breach cap
    n_blocks             : total blocks in final zone
    area_sqkm            : final zone area in sq km
    """
    prev        = set(selected)
    working_cap = max_area_sqkm - BRIDGE_BUFFER_SQKM  # 14.5 sq km

    # 1. Border contract — peel zero-revenue border blocks tip-first
    selected, current_area_m2 = run_border_contract(
        selected, adjacency, block_lookup, block_area_m2
    )

    # 2. Swap until convergence — migrate toward better territory
    #    Uses working_cap so swap cannot consume the bridge buffer
    n_swapped = 0
    while True:
        rb, ac, delta = _swap(
            selected, adjacency, block_lookup,
            block_area_m2, current_area_m2, working_cap,
        )
        if rb is None:
            break
        selected.discard(rb)
        selected.add(ac)
        current_area_m2 += block_area_m2.get(ac, 0) - block_area_m2.get(rb, 0)
        n_swapped += 1

    # 3. Expand greedily — grow into new territory
    #    Uses working_cap so expand cannot consume the bridge buffer
    while True:
        block, delta = _expand(
            selected, adjacency, block_lookup,
            block_area_m2, current_area_m2, working_cap,
        )
        if block is None:
            break
        selected.add(block)
        current_area_m2 += block_area_m2.get(block, 0)

    # 4. Border continuity (Option C) — ensure all dropped blocks are adjacent
    #    Uses full max_area_sqkm so bridges can use the 0.5 sq km buffer
    #    Bridges that would still breach the hard cap are skipped and logged
    selected, current_area_m2, n_bridge, n_skipped = enforce_border_continuity(
        prev, selected, adjacency, block_area_m2, max_area_sqkm
    )

    added, removed = diff_geofences(prev, selected)
    stats = {
        "n_removed":            len(removed),
        "n_added":              len(added),
        "n_swapped":            n_swapped,
        "n_bridge_blocks":      n_bridge,
        "n_continuity_skipped": n_skipped,
        "n_blocks":             len(selected),
        "area_sqkm":            current_area_m2 / 1e6,
    }
    return selected, current_area_m2, stats


# ── Utilities ─────────────────────────────────────────────────────────────────

def diff_geofences(prev: set, curr: set) -> tuple[set, set]:
    """Return (added_blocks, removed_blocks) between two consecutive geofences."""
    return curr - prev, prev - curr


def project_bbox(center_bbox: list, target_crs: str) -> tuple:
    """
    Reproject CENTER_BBOX [min_lon, min_lat, max_lon, max_lat] into target_crs.

    Returns
    -------
    (xmin, ymin, xmax, ymax) in target_crs units (metres for EPSG:27700)
    """
    geom = (
        gpd.GeoSeries(
            [shapely_box(
                center_bbox[0], center_bbox[1],
                center_bbox[2], center_bbox[3],
            )],
            crs="EPSG:4326",
        )
        .to_crs(target_crs)
        .iloc[0]
    )
    return geom.bounds  # (xmin, ymin, xmax, ymax)


def plot_transition(
    ax,
    blocks_gdf: gpd.GeoDataFrame,
    roads_gdf: gpd.GeoDataFrame,
    prev: set,
    curr: set,
    bbox_bounds: tuple,
    title: str,
) -> None:
    """
    Plot one hour's geofence within bbox, highlighting changes vs. previous hour.

    Colours
    -------
    Light grey  — outside zone (unchanged)
    Steel blue  — inside zone, unchanged from previous hour
    Lime green  — added blocks (new this hour)
    Crimson     — removed blocks (dropped this hour)

    Parameters
    ----------
    prev        : previous hour's block set (pass curr itself for hour 1)
    curr        : this hour's block set
    bbox_bounds : (xmin, ymin, xmax, ymax) in the GeoDataFrame's CRS
    """
    added, removed = diff_geofences(prev, curr)
    unchanged_in   = prev & curr

    xmin, ymin, xmax, ymax = bbox_bounds

    roads_gdf.plot(ax=ax, linewidth=0.2, color="grey", alpha=0.3)

    # Outside zone
    blocks_gdf[~blocks_gdf["block_id"].isin(curr | removed)].plot(
        ax=ax, color="#f5f5f5", edgecolor="#dddddd", linewidth=0.15,
    )
    # Unchanged interior
    blocks_gdf[blocks_gdf["block_id"].isin(unchanged_in)].plot(
        ax=ax, color="steelblue", alpha=0.4, edgecolor="white", linewidth=0.2,
    )
    # Added this hour
    blocks_gdf[blocks_gdf["block_id"].isin(added)].plot(
        ax=ax, color="limegreen", alpha=0.7, edgecolor="white", linewidth=0.2,
    )
    # Removed this hour
    blocks_gdf[blocks_gdf["block_id"].isin(removed)].plot(
        ax=ax, color="crimson", alpha=0.7, edgecolor="white", linewidth=0.2,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
