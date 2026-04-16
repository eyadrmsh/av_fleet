"""
src/quotes_cleaning.py
----------------------
Cleaning pipeline for the raw quotes DataFrame.

Each public function takes the raw-rows DataFrame (one row per A/B arm per
pricing call) and returns a filtered copy.  All thresholds are keyword
arguments with documented defaults so callers can override them.

Typical usage
-------------
    from src.quotes_cleaning import (
        run_pipeline,
        compute_midpoint,
        check_quote_consistency,
        haversine_m,
        max_spread_m,
    )

    df_clean = run_pipeline(df)
    df_clean = compute_midpoint(df_clean)
    report   = check_quote_consistency(df_clean)
"""

from __future__ import annotations

from itertools import combinations
from math import atan2, cos, radians, sin, sqrt
from typing import Any

import numpy as np
import pandas as pd

from src.config import BBOX, MAX_SPREAD_M, MAX_ETT_DIFF_S


# ── Geo helpers (public — importable for use in notebooks) ─────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1.0 - a))


def max_spread_m(lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Return the maximum pairwise haversine distance (metres) among a set of
    (lat, lon) pairs.  Returns 0.0 when fewer than two distinct points exist.
    """
    coords = np.unique(np.column_stack([lats, lons]), axis=0)
    if len(coords) < 2:
        return 0.0
    return max(
        haversine_m(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
        for i, j in combinations(range(len(coords)), 2)
    )


# ── Filter 1 — inconsistent quote_converted ────────────────────────────────────

def filter_inconsistent_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop quote_ids where ``quote_converted`` takes more than one distinct value
    across all rows of that quote session.

    Expected to remove ~147 quote_ids from the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw quotes DataFrame with columns ``quote_id`` and ``quote_converted``.

    Returns
    -------
    pd.DataFrame
        Filtered copy with consistent-conversion quote_ids only.
    """
    n_before = df["quote_id"].nunique()

    inconsistent = (
        df.groupby("quote_id")["quote_converted"]
        .nunique()
        .gt(1)
    )
    bad_ids = inconsistent[inconsistent].index
    df_out  = df[~df["quote_id"].isin(bad_ids)].copy()

    n_after   = df_out["quote_id"].nunique()
    n_dropped = n_before - n_after
    print(f"[filter_inconsistent_conversion]  dropped {n_dropped:,} quote_ids "
          f"({n_before:,} → {n_after:,})")
    return df_out


# ── Filter 2 — points outside London ──────────────────────────────────────────

def filter_outside_london(
    df: pd.DataFrame,
    bbox: list[float] | None = None,
) -> pd.DataFrame:
    """
    Drop quote_ids where **any** pickup or dropoff coordinate falls outside the
    London bounding box.

    Parameters
    ----------
    df : pd.DataFrame
        Raw quotes DataFrame with columns ``pickup_longitude``,
        ``pickup_latitude``, ``dropoff_longitude``, ``dropoff_latitude``.
    bbox : [min_lon, min_lat, max_lon, max_lat], optional
        Defaults to ``src.config.BBOX``.

    Returns
    -------
    pd.DataFrame
        Filtered copy with only London-bound quote_ids.
    """
    if bbox is None:
        bbox = BBOX
    min_lon, min_lat, max_lon, max_lat = bbox

    n_before = df["quote_id"].nunique()

    in_london = (
        df["pickup_longitude"].between(min_lon, max_lon)
        & df["pickup_latitude"].between(min_lat, max_lat)
        & df["dropoff_longitude"].between(min_lon, max_lon)
        & df["dropoff_latitude"].between(min_lat, max_lat)
    )
    # A quote_id is kept only if every one of its rows is inside London
    bad_ids = df.loc[~in_london, "quote_id"].unique()
    df_out  = df[~df["quote_id"].isin(bad_ids)].copy()

    n_after   = df_out["quote_id"].nunique()
    n_dropped = n_before - n_after
    print(f"[filter_outside_london]           dropped {n_dropped:,} quote_ids "
          f"({n_before:,} → {n_after:,})")
    return df_out


# ── Filter 3 — location spread > 500 m ────────────────────────────────────────

def filter_location_spread(
    df: pd.DataFrame,
    max_spread_m_threshold: float = MAX_SPREAD_M,
) -> pd.DataFrame:
    """
    Drop quote_ids where the maximum pairwise distance between any two **pickup**
    points exceeds *max_spread_m_threshold* metres.

    Dropoff spread is intentionally ignored — dropoff changes across calls are
    deliberate destination edits by the customer, not GPS noise.

    Parameters
    ----------
    df : pd.DataFrame
        Raw quotes DataFrame.
    max_spread_m_threshold : float
        Pickup spread threshold in metres.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    n_before = df["quote_id"].nunique()

    pu_spread = (
        df.groupby("quote_id")
        .apply(
            lambda g: max_spread_m(
                g["pickup_latitude"].to_numpy(), g["pickup_longitude"].to_numpy()
            )
        )
    )

    bad_ids = pu_spread.index[pu_spread > max_spread_m_threshold]
    df_out  = df[~df["quote_id"].isin(bad_ids)].copy()

    n_after   = df_out["quote_id"].nunique()
    n_dropped = n_before - n_after
    print(f"[filter_location_spread]          dropped {n_dropped:,} quote_ids "
          f"(pickup spread > {max_spread_m_threshold:.0f} m)  ({n_before:,} → {n_after:,})")
    return df_out


# ── Filter 4 — ETT variation > 250 s ──────────────────────────────────────────

def filter_ett_variation(
    df: pd.DataFrame,
    max_ett_diff_s: float = MAX_ETT_DIFF_S,
) -> pd.DataFrame:
    """
    Drop quote_ids where the range (max − min) of ``estimated_travel_time``
    across **distinct pricing calls** exceeds *max_ett_diff_s* seconds.

    Pricing calls are identified by unique ``quote_creation_timestamp`` within
    each ``quote_id``; only the first ETT value per call is used (ETT is
    constant within a call).

    Parameters
    ----------
    df : pd.DataFrame
        Raw quotes DataFrame with columns ``estimated_travel_time`` and
        ``quote_creation_timestamp``.
    max_ett_diff_s : float
        ETT range threshold in seconds.  Default 250 s.

    Returns
    -------
    pd.DataFrame
        Filtered copy.
    """
    n_before = df["quote_id"].nunique()

    # One ETT value per pricing call (first row per timestamp group)
    ett_per_call = (
        df.groupby(["quote_id", "quote_creation_timestamp"])["estimated_travel_time"]
        .first()
        .reset_index()
    )
    ett_range = (
        ett_per_call.groupby("quote_id")["estimated_travel_time"]
        .agg(lambda s: s.max() - s.min())
    )
    bad_ids = ett_range.index[ett_range > max_ett_diff_s]
    df_out  = df[~df["quote_id"].isin(bad_ids)].copy()

    n_after   = df_out["quote_id"].nunique()
    n_dropped = n_before - n_after
    print(f"[filter_ett_variation]            dropped {n_dropped:,} quote_ids "
          f"(ETT range > {max_ett_diff_s:.0f} s)  ({n_before:,} → {n_after:,})")
    return df_out


# ── Canonical location assignment ─────────────────────────────────────────────

def assign_canonical_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace per-row pickup **and** dropoff coordinates with the values from the
    **final pricing call** for that ``quote_id``.

    The final call (``max(quote_creation_timestamp)``) is used for both
    endpoints — it reflects the most up-to-date location the customer had
    confirmed before converting or abandoning.

    After this step every row in a ``quote_id`` shares the same
    ``pickup_latitude``, ``pickup_longitude``, ``dropoff_latitude``, and
    ``dropoff_longitude``.

    """
    df_out = df.copy()

    last_call_idx = df_out.groupby("quote_id")["quote_creation_timestamp"].idxmax()
    last_call_locs = (
        df_out.loc[
            last_call_idx,
            ["quote_id", "pickup_latitude", "pickup_longitude",
             "dropoff_latitude", "dropoff_longitude"],
        ]
        .set_index("quote_id")
        .add_suffix("_c")
    )

    df_out = df_out.join(last_call_locs, on="quote_id")
    for col in ["pickup_latitude", "pickup_longitude",
                "dropoff_latitude", "dropoff_longitude"]:
        df_out[col] = df_out[f"{col}_c"]
        df_out.drop(columns=[f"{col}_c"], inplace=True)

    n_affected = (
        df.groupby("quote_id")[
            ["pickup_latitude", "pickup_longitude",
             "dropoff_latitude", "dropoff_longitude"]
        ]
        .nunique()
        .gt(1)
        .any(axis=1)
        .sum()
    )
    print(f"[assign_canonical_location]       canonicalised coordinates for "
          f"{n_affected:,} quote_ids with location drift")
    return df_out


# ── Canonical one-row-per-quote dataset ───────────────────────────────────────

def build_canonical_quotes(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Collapse cleaned quotes to **one row per** ``quote_id``.

    For each quote session:

    1. Keep only rows from the **last pricing call**
       (``max(quote_creation_timestamp)``).
    2. If that call has two A/B-price rows, **randomly sample one**.

    Pickup coordinates, dropoff coordinates, and ``estimated_travel_time``
    are all taken from that last call.

    """
    last_ts   = df.groupby("quote_id")["quote_creation_timestamp"].transform("max")
    last_call = df[df["quote_creation_timestamp"] == last_ts]

    canonical = (
        last_call
        .groupby("quote_id", sort=False)
        .apply(lambda g: g.sample(n=1, random_state=int(rng.integers(0, 2**31))))
        .reset_index(drop=True)
    )

    cols = [
        "quote_id", "quote_creation_timestamp",
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude",
        "quoted_price", "estimated_travel_time", "quote_converted",
    ]
    print(f"[build_canonical_quotes]          {len(canonical):,} canonical quotes "
          f"({canonical['quote_converted'].sum():,.0f} converted)")
    return canonical[cols].reset_index(drop=True)


# ── Midpoint computation ───────────────────────────────────────────────────────

# ── Consistency check ──────────────────────────────────────────────────────────

def check_quote_consistency(df: pd.DataFrame) -> dict[str, Any]:
    """
    Verify that after cleaning every ``quote_id`` satisfies:

    1. Exactly one distinct ``quote_converted`` value.
    2. Exactly one distinct pickup location  (lat, lon pair).
    3. Exactly one distinct dropoff location (lat, lon pair).


    """
    n_quotes = df["quote_id"].nunique()

    def _n_mixed(val_cols: list[str]) -> int:
        return int(
            df.groupby("quote_id")[val_cols]
            .nunique()
            .gt(1)
            .any(axis=1)
            .sum()
        )

    n_mixed_conv    = _n_mixed(["quote_converted"])
    n_mixed_pickup  = _n_mixed(["pickup_latitude", "pickup_longitude"])
    n_mixed_dropoff = _n_mixed(["dropoff_latitude", "dropoff_longitude"])

    report: dict[str, Any] = {
        "n_quotes":             n_quotes,
        "conversion_ok":        n_mixed_conv    == 0,
        "n_mixed_conversion":   n_mixed_conv,
        "pickup_location_ok":   n_mixed_pickup  == 0,
        "n_mixed_pickup":       n_mixed_pickup,
        "dropoff_location_ok":  n_mixed_dropoff == 0,
        "n_mixed_dropoff":      n_mixed_dropoff,
        "all_ok":               (n_mixed_conv == 0
                                 and n_mixed_pickup == 0
                                 and n_mixed_dropoff == 0),
    }

    status = "PASS" if report["all_ok"] else "FAIL"
    print(f"\n[check_quote_consistency]  {status}  ({n_quotes:,} quote_ids)")
    print(f"  quote_converted consistent:   {report['conversion_ok']}"
          f"  (violations: {n_mixed_conv})")
    print(f"  pickup  location consistent:  {report['pickup_location_ok']}"
          f"  (violations: {n_mixed_pickup})")
    print(f"  dropoff location consistent:  {report['dropoff_location_ok']}"
          f"  (violations: {n_mixed_dropoff})")
    return report


# ── Pipeline wrapper ───────────────────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    *,
    bbox: list[float] | None = None,
    max_spread_m_threshold: float = MAX_SPREAD_M,
    max_ett_diff_s: float = MAX_ETT_DIFF_S,
) -> pd.DataFrame:
    """
    Apply all four cleaning filters in sequence and return the cleaned DataFrame.

    Order
    -----
    1. :func:`filter_inconsistent_conversion`   — ~147 quote_ids
    2. :func:`filter_outside_london`            — out-of-bbox coordinates
    3. :func:`filter_location_spread`           — GPS drift > *max_spread_m_threshold*
    4. :func:`filter_ett_variation`             — ETT range > *max_ett_diff_s*

    """
    n_rows_start   = len(df)
    n_quotes_start = df["quote_id"].nunique()
    print(f"=== Quotes cleaning pipeline ===")
    print(f"Input:  {n_rows_start:,} rows  |  {n_quotes_start:,} quote_ids\n")

    df = filter_inconsistent_conversion(df)
    df = filter_outside_london(df, bbox=bbox)
    df = filter_location_spread(df, max_spread_m_threshold=max_spread_m_threshold)
    df = filter_ett_variation(df, max_ett_diff_s=max_ett_diff_s)

    n_rows_end   = len(df)
    n_quotes_end = df["quote_id"].nunique()
    print(f"\nOutput: {n_rows_end:,} rows  |  {n_quotes_end:,} quote_ids  "
          f"({n_quotes_start - n_quotes_end:,} removed in total)")
    return df


# ── Save canonical dataset ─────────────────────────────────────────────────────

def build_and_save_canonical(
    cleaned_csv_path: str,
    output_path: str,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load ``quotes_cleaned.csv``, collapse to one row per ``quote_id`` via
    :func:`build_canonical_quotes`, and save to *output_path* (parquet).
    """
    rng = np.random.default_rng(random_seed)

    df = pd.read_csv(cleaned_csv_path)
    df["quote_creation_timestamp"] = pd.to_datetime(
        df["quote_creation_timestamp"], format="ISO8601", utc=True
    )
    canonical = build_canonical_quotes(df, rng)
    canonical.to_parquet(output_path, index=False)
    print(f"Saved → {output_path}  ({len(canonical):,} rows)")
    return canonical
