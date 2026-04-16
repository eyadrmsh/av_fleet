"""
src/forecasting_vis_helpers.py
------------------------------
Reusable visualisation helpers for the demand forecasting notebook.

Design notes
------------
Functions here cover patterns that appear more than once across notebook cells.
One-off complex charts (the 6-model forecast comparison, the April line plots)
are left inline in the notebook because parameterising them would add more
complexity than the single call site warrants.

Public API
----------
  make_zone_day_pivot    — zone × day MAPE pivot from day_mape_df
  zone_day_heatmap       — plot one zone × day heatmap on a given Axes
  build_metric_pivots    — build mape + coverage pivots for a tuned refit
  plot_heatmap_pair      — side-by-side default vs tuned MAPE heatmaps
  print_mape_cov_summary — coverage table + 3-line MAPE/coverage summary
  h3_ids_to_geodataframe — convert H3 cell IDs to GeoDataFrame with polygons
  plot_h3_demand_maps    — three-panel H3 hexagon demand map
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import h3
from shapely.geometry import Polygon


# ── Heatmap pivot helpers ──────────────────────────────────────────────────────

def make_zone_day_pivot(
    mape_df: pd.DataFrame,
    model: str,
    h_days: int,
    col_labels: list[str],
) -> pd.DataFrame:
    """
    Build a zone × eval-day MAPE pivot from the day-level MAPE DataFrame.

    Parameters
    ----------
    mape_df    : day-level MAPE DataFrame with columns
                 zone, model, h_days, day_i (1–7), mape.
    model      : model key to filter on (e.g. "prophet_lags").
    h_days     : horizon to filter on (14 or 21).
    col_labels : 8-element list — 7 day labels + "Avg" — used to rename
                 the pivot columns (e.g. ["Sun\\n22 Mar", …, "Avg"]).

    Returns
    -------
    DataFrame indexed by zone, columns = col_labels, sorted by Avg ascending
    (best-performing zone first).
    """
    sub = mape_df[
        (mape_df["model"]  == model) &
        (mape_df["h_days"] == h_days)
    ]
    p = sub.pivot(index="zone", columns="day_i", values="mape")
    p = p[sorted(p.columns)]
    p["Avg"] = p.mean(axis=1)
    p = p.sort_values("Avg")
    p.columns = col_labels
    return p


def build_metric_pivots(
    tuned_df: pd.DataFrame,
    zone_order,
    col_labels: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Build zone × day pivots for both mape and coverage from a tuned-refit
    DataFrame, aligned to a reference zone ordering.

    Parameters
    ----------
    tuned_df   : DataFrame with columns zone, day_i, mape, coverage
                 (one row per zone × day, as produced by the tuned refit loops).
    zone_order : index-like giving the desired zone ordering (e.g. the index
                 of the corresponding default pivot so both share the same y-axis).
    col_labels : 8-element list — 7 day labels + "Avg".

    Returns
    -------
    {"mape": pd.DataFrame, "coverage": pd.DataFrame}
    """
    pivots: dict[str, pd.DataFrame] = {}
    for col in ["mape", "coverage"]:
        p = tuned_df.pivot(index="zone", columns="day_i", values=col)
        p = p[sorted(p.columns)]
        p["Avg"] = p.mean(axis=1)
        p = p.reindex(zone_order)
        p.columns = col_labels
        pivots[col] = p
    return pivots


# ── Heatmap plot helpers ───────────────────────────────────────────────────────

def zone_day_heatmap(
    pivot: pd.DataFrame,
    title: str,
    vmax: float,
    ax,
    cmap: str = "YlOrRd",
    fmt: str = ".0f",
    vmin: float = 0,
):
    """
    Plot a zone × day heatmap with annotated cell values on a given Axes.

    A dashed vertical line separates daily columns from the "Avg" column.
    Cell text colour flips to white when the normalised value exceeds 0.70,
    and the "Avg" column is always rendered bold.

    Parameters
    ----------
    pivot : zone × day DataFrame (output of make_zone_day_pivot or
            build_metric_pivots).  Last column is expected to be "Avg".
    title : axes title string.
    vmax  : colour scale upper bound.
    ax    : matplotlib Axes to draw on.
    cmap  : matplotlib colormap name (default "YlOrRd").
    fmt   : format spec for cell annotations (default ".0f").
    vmin  : colour scale lower bound (default 0).

    Returns
    -------
    The AxesImage returned by imshow — pass to plt.colorbar() if needed.
    """
    data = pivot.values
    im   = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([z[:12] for z in pivot.index], fontsize=7)
    ax.axvline(data.shape[1] - 1.5, color="black", linewidth=1.5, linestyle="--")
    span = vmax - vmin if vmax > vmin else 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val      = data[i, j]
            norm_val = (val - vmin) / span
            txt_col  = "white" if norm_val > 0.70 else "black"
            weight   = "bold" if pivot.columns[j] == "Avg" else "normal"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=7, color=txt_col, fontweight=weight)
    ax.set_title(title, fontsize=9)
    return im


def plot_heatmap_pair(
    pivot_def: pd.DataFrame,
    pivot_tuned: pd.DataFrame,
    vmax: float,
    suptitle: str,
) -> tuple:
    """
    Side-by-side MAPE heatmaps: default params (left) vs tuned params (right).

    Both panels share the same colour scale and y-axis so zones align visually.

    Parameters
    ----------
    pivot_def   : default-params zone × day pivot.
    pivot_tuned : tuned-params zone × day pivot (same zone order as pivot_def).
    vmax        : shared colour scale upper bound (typically the 95th percentile
                  of pivot_def values).
    suptitle    : figure-level title string.

    Returns
    -------
    (fig, axes) — the matplotlib Figure and the two-element Axes array.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 5.5), sharey=True)
    zone_day_heatmap(pivot_def,   "Default params",                      vmax, axes[0])
    zone_day_heatmap(pivot_tuned, "Tuned params (per-zone grid search)", vmax, axes[1])
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig, axes


# ── Summary helpers ────────────────────────────────────────────────────────────

def print_mape_cov_summary(
    pivot_def: pd.DataFrame,
    pivots_tuned: dict[str, pd.DataFrame],
    model_label: str,
) -> None:
    """
    Print a CI-coverage table (per zone, sorted ascending) and a 3-line
    summary comparing default vs tuned MAPE and tuned CI coverage.

    Parameters
    ----------
    pivot_def    : default-params MAPE pivot (zone × day, with "Avg" column).
    pivots_tuned : output of build_metric_pivots — must have "mape" and
                   "coverage" keys.
    model_label  : model name for the table header string
                   (e.g. "prophet_lags" or "prophet_full_temp_vis").
    """
    cov_table = (
        pivots_tuned["coverage"][["Avg"]]
        .rename(columns={"Avg": "coverage (%)"})
        .round(1)
        .sort_values("coverage (%)")
    )
    print(f"95% CI coverage per zone — {model_label} (tuned), H21 eval (Mar 22–28):\n")
    print(cov_table.to_string())
    print(f"\nMedian Avg MAPE     — default : {pivot_def['Avg'].median():.1f}%")
    print(f"Median Avg MAPE     — tuned   : {pivots_tuned['mape']['Avg'].median():.1f}%")
    print(f"Median Avg Coverage — tuned   : {pivots_tuned['coverage']['Avg'].median():.1f}%  (target 95%)")


# ── H3 map helpers ─────────────────────────────────────────────────────────────

def _h3_to_polygon(h3_id: str) -> Polygon:
    """Convert one H3 cell ID string to a Shapely Polygon (lng, lat order)."""
    try:
        boundary = h3.cell_to_boundary(h3_id)      # h3-py >= 4
    except AttributeError:
        boundary = h3.h3_to_geo_boundary(h3_id)    # h3-py 3
    return Polygon([(lng, lat) for lat, lng in boundary])


def h3_ids_to_geodataframe(
    df: pd.DataFrame,
    zone_col: str = "zone",
) -> gpd.GeoDataFrame:
    """
    Attach H3 hexagon polygons to a DataFrame and return a GeoDataFrame.

    Parameters
    ----------
    df       : DataFrame that includes a column of H3 cell identifier strings.
               All other columns are preserved in the output.
    zone_col : name of the column containing H3 cell IDs (default "zone").

    Returns
    -------
    GeoDataFrame with a "geometry" column of Shapely Polygons, CRS EPSG:4326.
    """
    out = df.copy()
    out["geometry"] = out[zone_col].apply(_h3_to_polygon)
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def plot_h3_demand_maps(
    gdf: gpd.GeoDataFrame,
    scenario_cols: list[str],
    scenario_labels: list[str],
    suptitle: str,
    cmap: str = "YlOrRd",
    colorbar_label: str = "Estimated trips",
    vmax: float | None = None,
) -> plt.Figure:
    """
    Three-panel H3 hexagon map with annotated cell values and a shared
    horizontal colorbar below.

    Typical use: lower-bound, forecast, upper-bound trip estimates across zones.

    Parameters
    ----------
    gdf             : GeoDataFrame (output of h3_ids_to_geodataframe) containing
                      the columns named in scenario_cols.
    scenario_cols   : exactly 3 column names to plot (one per panel).
    scenario_labels : exactly 3 panel title strings matched to scenario_cols.
    suptitle        : figure-level title; supports two-line strings with \\n.
    cmap            : matplotlib colormap name (default "YlOrRd").
    colorbar_label  : label for the shared colorbar (default "Estimated trips").
    vmax            : colour scale upper bound; defaults to the maximum value
                      across all three scenario columns.

    Returns
    -------
    matplotlib Figure.
    """
    if vmax is None:
        vmax = float(gdf[scenario_cols].max().max())

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)

    for ax, col, label in zip(axes, scenario_cols, scenario_labels):
        gdf.plot(column=col, ax=ax, cmap=cmap,
                 vmin=0, vmax=vmax, edgecolor="white", linewidth=0.8)
        for _, row in gdf.iterrows():
            cx = row["geometry"].centroid.x
            cy = row["geometry"].centroid.y
            ax.text(cx, cy, f"{row[col]:.1f}",
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color="white" if row[col] / max(vmax, 1e-6) > 0.7 else "black")
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="horizontal",
                 pad=0.02, shrink=0.45, label=colorbar_label)
    fig.suptitle(suptitle, fontsize=11)
    return fig
