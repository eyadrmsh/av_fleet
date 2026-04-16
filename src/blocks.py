"""
src/blocks.py
-------------
OSM road-block construction and coverage-fraction computation.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pyrosm
from shapely.ops import unary_union, polygonize


def load_osm_data(
    pbf_file: str,
    bbox: list,
    road_types: list,
    target_crs: str,
) -> tuple:
    """
    Create an OSM reader and load the driving road network.

    Returns
    -------
    (osm, roads) : (pyrosm.OSM, GeoDataFrame)
    """
    osm   = pyrosm.OSM(pbf_file, bounding_box=bbox)
    roads = osm.get_network(network_type="driving")
    roads = roads[roads["highway"].isin(road_types)].copy()
    roads = roads.reset_index(drop=True).to_crs(target_crs)

    print(f"Roads loaded:  {len(roads):,}")
    print(f"Road types:    {sorted(roads['highway'].unique())}")
    print(f"CRS:           {roads.crs}")
    return osm, roads


def make_road_blocks(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Polygonize road centre-lines into enclosed blocks.

    Merges all road geometries into one unary union, then extracts every
    closed ring as a polygon. Each block gets a ``block_id`` and ``area_m2``.
    """
    all_roads_union = unary_union(roads.geometry)
    blocks = gpd.GeoDataFrame(
        geometry=list(polygonize(all_roads_union)), crs=roads.crs
    ).reset_index(drop=True)
    blocks["block_id"] = blocks.index
    blocks["area_m2"]  = blocks.geometry.area

    print(f"Blocks after polygonisation: {len(blocks):,}")
    return blocks


def find_main_component(blocks: gpd.GeoDataFrame) -> tuple:
    """
    Keep only blocks whose centroid falls inside the largest connected polygon.

    Returns
    -------
    (blocks_main, main_hull) : (GeoDataFrame, GeoDataFrame)
    """
    dissolved = unary_union(blocks.geometry)
    parts = (
        list(dissolved.geoms)
        if dissolved.geom_type == "MultiPolygon"
        else [dissolved]
    )
    main_component = max(parts, key=lambda x: x.area)
    print(f"Connected components found: {len(parts)}")

    blocks_main = (
        blocks[blocks.geometry.centroid.within(main_component)]
        .copy()
        .reset_index(drop=True)
    )
    blocks_main["block_id"] = blocks_main.index
    main_hull = gpd.GeoDataFrame(geometry=[main_component], crs=blocks.crs)

    print(f"Blocks in main component:   {len(blocks_main):,}")
    return blocks_main, main_hull


def load_osm_polygons(
    osm: pyrosm.OSM,
    target_crs: str,
    landuse_tags: list = None,
    natural_tags: list = None,
    leisure_tags: list = None,
    roads: gpd.GeoDataFrame = None,
    road_buffer_m: float = 20.0,
) -> gpd.GeoDataFrame:
    """
    Fetch OSM polygons filtered by any combination of landuse, natural, and
    leisure tags — covering both parks and water bodies with one function.

    Pass ``roads`` only for water (bridge subtraction); omit for parks.
    """
    landuse = osm.get_landuse()
    natural = osm.get_natural()

    frames = []
    if landuse_tags and "landuse" in landuse.columns:
        frames.append(landuse[landuse["landuse"].isin(landuse_tags)].copy())
    if natural_tags and "natural" in natural.columns:
        frames.append(natural[natural["natural"].isin(natural_tags)].copy())
    if leisure_tags and "leisure" in natural.columns:
        frames.append(natural[natural["leisure"].isin(leisure_tags)].copy())

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    polygons = (
        gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
        .to_crs(target_crs)
    )
    polygons = (
        polygons[polygons.geom_type.isin(["Polygon", "MultiPolygon"])]
        .copy()
        .reset_index(drop=True)
    )
    polygons["geometry"] = polygons.geometry.make_valid()
    # make_valid() can produce GeometryCollection — explode and re-filter so
    # overlay() never sees mixed geometry types
    polygons = polygons.explode(index_parts=False).reset_index(drop=True)
    polygons = (
        polygons[polygons.geom_type.isin(["Polygon", "MultiPolygon"])]
        .copy()
        .reset_index(drop=True)
    )
    polygons = polygons[~polygons.geometry.is_empty].copy()
    print(f"OSM polygons loaded: {len(polygons):,}")

    if roads is not None and len(polygons) > 0:
        poly_union    = unary_union(polygons.geometry)
        roads_on_poly = roads[roads.geometry.intersects(poly_union)].copy()
        if len(roads_on_poly) > 0:
            road_mask            = unary_union(roads_on_poly.geometry.buffer(road_buffer_m))
            polygons["geometry"] = polygons.geometry.difference(road_mask)
            polygons             = polygons[~polygons.geometry.is_empty].copy()
            print(f"After road masking ({road_buffer_m} m buffer): {len(polygons):,}")

    return polygons


def compute_coverage_fraction(
    blocks: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    col_prefix: str,
) -> gpd.GeoDataFrame:
    """
    Compute what fraction of each block's area is covered by *polygons*.

    Adds ``{col_prefix}_area_m2`` and ``{col_prefix}_fraction`` columns.
    ``block_area_m2`` is added on first call and reused on subsequent ones.
    """
    intersection = blocks.overlay(polygons[["geometry"]], how="intersection")
    intersection[f"{col_prefix}_area_m2"] = intersection.geometry.area

    area_per_block = (
        intersection.groupby("block_id")[f"{col_prefix}_area_m2"]
        .sum()
        .reset_index()
    )

    blocks = blocks.copy()
    if "block_area_m2" not in blocks.columns:
        blocks["block_area_m2"] = blocks.geometry.area

    blocks = blocks.merge(area_per_block, on="block_id", how="left")
    blocks[f"{col_prefix}_area_m2"] = blocks[f"{col_prefix}_area_m2"].fillna(0)
    blocks[f"{col_prefix}_area_m2"] = np.minimum(
        blocks[f"{col_prefix}_area_m2"], blocks["block_area_m2"]
    )
    blocks[f"{col_prefix}_fraction"] = (
        blocks[f"{col_prefix}_area_m2"] / blocks["block_area_m2"]
    ).fillna(0)
    return blocks


def filter_blocks(
    blocks: gpd.GeoDataFrame,
    park_threshold: float = 0.1,
    water_threshold: float = 0.1,
) -> gpd.GeoDataFrame:
    """
    Remove blocks that are predominantly park or water.
    """
    before   = len(blocks)
    filtered = blocks[
        (blocks["park_fraction"]  < park_threshold) &
        (blocks["water_fraction"] < water_threshold)
    ].copy()
    print(
        f"Blocks removed (park≥{park_threshold} or water≥{water_threshold}): "
        f"{before - len(filtered):,}  →  {len(filtered):,} remain"
    )
    return filtered
