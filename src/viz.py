"""
src/viz.py
----------
Plotting helpers for block coverage and geofence visualisation.
All functions display inline — no file saving.
"""

import geopandas as gpd
import matplotlib.pyplot as plt


def plot_fraction_choropleth(
    blocks: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    column: str,
    label: str,
    title: str,
    main_gdf: gpd.GeoDataFrame = None,
) -> None:
    """
    Choropleth map shading blocks by a fraction column.

    Parameters
    ----------
    blocks : GeoDataFrame  Must contain *column*.
    roads : GeoDataFrame
    column : str  Column to visualise (e.g. ``"park_fraction"``).
    label : str  Colourbar label.
    title : str  Plot title.
    main_gdf : GeoDataFrame or None  Optional hull outline.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    blocks.plot(
        ax=ax, column=column, cmap="RdYlGn_r",
        legend=True, legend_kwds={"label": label},
        edgecolor="none", alpha=0.7,
    )
    roads.plot(ax=ax, linewidth=0.3, color="gray", alpha=0.4)
    if main_gdf is not None:
        main_gdf.plot(ax=ax, alpha=0, edgecolor="navy", linewidth=2)
    ax.set_title(title, fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_before_after_filter(
    blocks_before: gpd.GeoDataFrame,
    blocks_after: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    main_gdf: gpd.GeoDataFrame = None,
) -> None:
    """Side-by-side: blocks before filtering (left) and after park/water removal (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    pairs = [
        (axes[0], blocks_before, f"Before filter: {len(blocks_before):,}"),
        (axes[1], blocks_after,  f"After filter:  {len(blocks_after):,} blocks"),
    ]
    for ax, gdf, title in pairs:
        gdf = gdf.copy()
        gdf["_idx"] = range(len(gdf))
        gdf.plot(ax=ax, column="_idx", cmap="tab20",
                 alpha=0.5, edgecolor="white", linewidth=0.2)
        roads.plot(ax=ax, linewidth=0.3, color="gray", alpha=0.4)
        if main_gdf is not None:
            main_gdf.plot(ax=ax, alpha=0, edgecolor="darkgreen", linewidth=2)
        ax.set_title(title, fontsize=13)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_final_component(blocks_main: gpd.GeoDataFrame) -> None:
    """Simple map of the final connected component after all filtering."""
    fig, ax = plt.subplots(figsize=(14, 10))
    blocks_main.plot(
        ax=ax, color="lightblue", edgecolor="gray", linewidth=0.3, alpha=0.7
    )
    ax.set_title(
        f"Main connected component — {len(blocks_main):,} blocks", fontsize=13
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
