"""
src/trips.py
------------
Trip dataset construction, GPS snapping, and point-to-block assignment.
"""

import numpy as np
import pandas as pd
import geopandas as gpd


# ── Call table ─────────────────────────────────────────────────────────────────

def build_call_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse raw quote rows into one row per pricing call.

    Groups by ``quote_id``, sorts by timestamp, then collapses rows with the
    same timestamp into one call record with a ``prices`` list (A/B arms).
    The entire quote_id is one session — no sub-session splitting.

    Returns
    -------
    pd.DataFrame
        One row per pricing call with columns:
        quote_id, call_id, call_ts, delta_ms_prev,
        pickup_lat, pickup_lon, dropoff_lat, dropoff_lon,
        time_travel_estimated, prices, price_gaps, n_prices,
        is_repeat_call, quote_converted.
    """
    records = []

    for qid, grp in df.groupby("quote_id", sort=False):
        grp = grp.sort_values("quote_creation_timestamp")

        by_ts: dict = {}
        for _, row in grp.iterrows():
            ts = row["quote_creation_timestamp"]
            by_ts.setdefault(ts, []).append(row)

        sorted_calls = sorted(by_ts.items())
        prev_prices  = None
        prev_ts      = None
        converted    = grp.iloc[0]["quote_converted"]

        for call_id, (ts, call_rows) in enumerate(sorted_calls):
            prices = sorted({r["quoted_price"] for r in call_rows})
            gaps   = [round(prices[i + 1] - prices[i], 2)
                      for i in range(len(prices) - 1)]
            delta  = ((ts - prev_ts).total_seconds() * 1000
                      if prev_ts is not None else float("nan"))
            is_rep = (prev_prices is not None
                      and frozenset(prices) == frozenset(prev_prices))

            records.append({
                "quote_id":              qid,
                "call_id":               call_id,
                "call_ts":               ts,
                "delta_ms_prev":         delta,
                "pickup_lat":            call_rows[0]["pickup_latitude"],
                "pickup_lon":            call_rows[0]["pickup_longitude"],
                "dropoff_lat":           call_rows[0]["dropoff_latitude"],
                "dropoff_lon":           call_rows[0]["dropoff_longitude"],
                "time_travel_estimated": call_rows[0].get("estimated_travel_time"),
                "prices":                prices,
                "price_gaps":            gaps,
                "n_prices":              len(prices),
                "is_repeat_call":        is_rep,
                "quote_converted":       converted,
            })
            prev_prices = prices
            prev_ts     = ts

    return pd.DataFrame(records)


# ── Trip dataset from cleaned quotes 

def build_trip_dataset(df_clean: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Produce one row per converted ``quote_id`` from the cleaned quotes DataFrame.

    Steps
    -----
    1. Keep only converted sessions (``quote_converted == True``).
    2. Keep only rows from the last pricing call per ``quote_id``
       (``max(quote_creation_timestamp)``).
    3. Randomly sample one row — one A/B price arm — per ``quote_id``.
    4. Rename coordinate columns to the trips-pipeline convention.

    Coordinates are already canonical after :func:`~src.quotes_cleaning.assign_canonical_location`:
    pickup = mean across calls, dropoff = last-call location.

    Parameters
    """
    converted = df_clean[df_clean["quote_converted"]].copy()

    last_ts   = converted.groupby("quote_id")["quote_creation_timestamp"].transform("max")
    last_call = converted[converted["quote_creation_timestamp"] == last_ts]

    trips = (
        last_call.groupby("quote_id", sort=False)
        .apply(lambda g: g.sample(n=1, random_state=int(rng.integers(0, 2**31))))
        .reset_index(drop=True)
    )

    return (
        trips[[
            "quote_id", "pickup_latitude", "pickup_longitude",
            "dropoff_latitude", "dropoff_longitude",
            "quoted_price", "estimated_travel_time",
        ]]
        .rename(columns={
            "pickup_latitude":  "pickup_lat",
            "pickup_longitude": "pickup_lon",
            "dropoff_latitude": "dropoff_lat",
            "dropoff_longitude":"dropoff_lon",
            "quoted_price":     "sampled_revenue",
        })
        .reset_index(drop=True)
    )


# ── GPS snapping ───────────────────────────────────────────────────────────────

def snap_to_road(
    points_gdf: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    max_distance_m: float = 100.0,
) -> gpd.GeoDataFrame:
    """
    Project each point onto the nearest road segment within *max_distance_m* metres.

    """
    roads_r = roads[["geometry"]].reset_index(drop=True)
    pts     = points_gdf[["geometry"]].reset_index(drop=True)

    joined = gpd.sjoin_nearest(
        pts,
        roads_r,
        how="left",
        max_distance=max_distance_m,
        distance_col="snap_dist",
    )
    # sjoin_nearest can duplicate rows on exact distance ties — keep first
    joined = joined.loc[~joined.index.duplicated(keep="first")]

    snapped_geoms = []
    for i in range(len(pts)):
        match = joined.iloc[i]
        if pd.isna(match["snap_dist"]):
            snapped_geoms.append(pts.geometry.iloc[i])
        else:
            road_geom = roads_r.geometry.iloc[int(match["index_right"])]
            pt        = pts.geometry.iloc[i]
            snapped_geoms.append(road_geom.interpolate(road_geom.project(pt)))

    result             = points_gdf.copy().reset_index(drop=True)
    result["geometry"] = snapped_geoms
    return gpd.GeoDataFrame(result, geometry="geometry", crs=points_gdf.crs)


# ── Point-to-block assignment ──────────────────────────────────────────────────

def assign_to_blocks(
    trips_df: pd.DataFrame,
    blocks: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    main_hull: gpd.GeoDataFrame,
    snap_distance_m: float = 100.0,
) -> pd.DataFrame:
    """
    Assign pickup and dropoff coordinates to block IDs.

    **Step 0** — Filter: keep only trips where *both* pickup and dropoff fall
    within the main connected component hull.

    **Step 1** — Snap to nearest road: assign the block whose centroid is
    closest among all blocks touching the nearest road segment.

    **Step 2** — Fallback: if no road is found within *snap_distance_m*,
    assign to the block with the nearest centroid.

    All trips that survive the hull filter are guaranteed to be assigned
    (no NaN in ``pickup_block`` or ``dropoff_block``).

    """
    target_crs = blocks.crs

    def _make_gdf(lats, lons):
        return gpd.GeoDataFrame(
            {"_idx": range(len(lats))},
            geometry=gpd.points_from_xy(lons, lats),
            crs="EPSG:4326",
        ).to_crs(target_crs)

    pickup_gdf  = _make_gdf(trips_df["pickup_lat"],  trips_df["pickup_lon"])
    dropoff_gdf = _make_gdf(trips_df["dropoff_lat"], trips_df["dropoff_lon"])

    # Step 0: hull filter
    hull    = main_hull.geometry.iloc[0]
    both_in = pickup_gdf.geometry.within(hull) & dropoff_gdf.geometry.within(hull)

    n_before    = len(trips_df)
    trips_filt  = trips_df.loc[both_in.values].reset_index(drop=True)
    pickup_pts  = pickup_gdf.loc[both_in].reset_index(drop=True)
    dropoff_pts = dropoff_gdf.loc[both_in].reset_index(drop=True)

    print(f"Trips within main component: {len(trips_filt):,} / {n_before:,} "
          f"({100 * len(trips_filt) / max(n_before, 1):.1f} %)")

    # Precompute road-block adjacency (reused for pickup + dropoff)
    print("Building road-block adjacency table...")
    roads_r = roads[["geometry"]].reset_index(drop=True)
    roads_r["_road_idx"] = range(len(roads_r))

    rb = gpd.sjoin(
        roads_r,
        blocks[["block_id", "geometry"]],
        predicate="intersects",
        how="inner",
    )[["_road_idx", "block_id"]].drop_duplicates()

    centroids = blocks.set_index("block_id")["geometry"].centroid
    rb = rb.copy()
    rb["cx"] = rb["block_id"].map(centroids.x).values
    rb["cy"] = rb["block_id"].map(centroids.y).values
    print(f"  Road-block pairs: {len(rb):,}")

    block_centroids_gdf = gpd.GeoDataFrame(
        blocks[["block_id"]].copy(),
        geometry=centroids.loc[blocks["block_id"]].values,
        crs=target_crs,
    ).reset_index(drop=True)

    def _assign(pts_gdf: gpd.GeoDataFrame, label: str) -> np.ndarray:
        n   = len(pts_gdf)
        pts = pts_gdf.copy()
        pts["pt_idx"] = range(n)
        pts["pt_x"]   = pts.geometry.x
        pts["pt_y"]   = pts.geometry.y

        snapped = gpd.sjoin_nearest(
            pts[["pt_idx", "pt_x", "pt_y", "geometry"]],
            roads_r,
            how="left",
            max_distance=snap_distance_m,
            distance_col="snap_dist",
        )

        matched   = snapped[snapped["snap_dist"].notna()].copy()
        unmatched = set(snapped.loc[snapped["snap_dist"].isna(), "pt_idx"].astype(int))

        result: dict[int, int] = {}

        if len(matched) > 0:
            merged = matched[["pt_idx", "pt_x", "pt_y", "_road_idx"]].merge(
                rb, on="_road_idx", how="left"
            )
            merged = merged.dropna(subset=["block_id"])
            merged["block_id"] = merged["block_id"].astype(int)

            if len(merged) > 0:
                merged["dist"] = np.sqrt(
                    (merged["pt_x"] - merged["cx"]) ** 2
                    + (merged["pt_y"] - merged["cy"]) ** 2
                )
                best = merged.loc[merged.groupby("pt_idx")["dist"].idxmin()]
                for row in best.itertuples(index=False):
                    result[int(row.pt_idx)] = int(row.block_id)

            road_assigned = set(result.keys())
            matched_idxs  = set(matched["pt_idx"].astype(int))
            unmatched.update(matched_idxs - road_assigned)

        n_road = len(result)

        if unmatched:
            fallback_pts = pts.loc[list(unmatched)].copy()
            nearest = gpd.sjoin_nearest(
                gpd.GeoDataFrame(
                    fallback_pts[["pt_idx", "geometry"]], crs=target_crs
                ),
                block_centroids_gdf,
                how="left",
            )
            for row in nearest.itertuples(index=False):
                result[int(row.pt_idx)] = int(row.block_id)

        print(f"  {label}: road-snapped={n_road:,}  "
              f"fallback={len(unmatched):,}  total={n:,}")
        return np.array([result.get(i, -1) for i in range(n)], dtype=np.int64)

    print("Assigning pickup points...")
    pickup_block_ids  = _assign(pickup_pts, "pickup")

    print("Assigning dropoff points...")
    dropoff_block_ids = _assign(dropoff_pts, "dropoff")

    trips_filt = trips_filt.copy()
    trips_filt["pickup_block"]  = pickup_block_ids
    trips_filt["dropoff_block"] = dropoff_block_ids

    n_full = int(((pickup_block_ids >= 0) & (dropoff_block_ids >= 0)).sum())
    print(f"\nFully assigned trips: {n_full:,} / {len(trips_filt):,}")
    assert n_full == len(trips_filt), (
        f"BUG: {len(trips_filt) - n_full} trips inside the main component "
        "were not assigned to a block."
    )

    return trips_filt
