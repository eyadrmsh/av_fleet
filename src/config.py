import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── File paths ─────────────────────────────────────────────────────────────────
PBF_FILE   = str(_ROOT / "data" / "raw"       / "greater-london-260411.osm.pbf")
TRIPS_FILE = str(_ROOT / "data" / "raw"       / "THA_Dataset_MarketplaceDS.csv")

FILTERED_BLOCKS_FILE   = str(_ROOT / "data" / "processed" / "filtered_main_blocks.geojson")
TRIPS_ASSIGNED_FILE    = str(_ROOT / "data" / "processed" / "trips_assigned.csv")
QUOTES_CLEANED_FILE    = str(_ROOT / "data" / "processed" / "quotes_cleaned.csv")
QUOTES_CANONICAL_FILE  = str(_ROOT / "data" / "processed" / "quotes_canonical.parquet")

# ── Spatial extent (minx, miny, maxx, maxy) ────────────────────────────────────
BBOX = [-0.681, 51.260, 0.423, 51.764]

# ── Coordinate reference system ────────────────────────────────────────────────
# EPSG:27700 = British National Grid, appropriate for London distance/area calcs
TARGET_CRS = "EPSG:27700"

# ── Road network ───────────────────────────────────────────────────────────────
ROAD_TYPES = [
    "motorway",    "motorway_link",
    "trunk",       "trunk_link",
    "primary",     "primary_link",
    "secondary",   "secondary_link",
    "tertiary",    "tertiary_link",
    "residential", "living_street",
]

# ── Park filtering ─────────────────────────────────────────────────────────────
PARK_LANDUSE = [
    "park", "recreation_ground", "grass",
    "meadow", "forest", "nature_reserve", "cemetery",
]
PARK_LEISURE = [
    "park", "garden", "nature_reserve",
    "golf_course", "pitch", "common",
]
BLOCK_PARK_THRESHOLD  = 0.1

# ── Water filtering ────────────────────────────────────────────────────────────
WATER_NATURAL = ["water", "wetland", "bay", "strait"]
WATER_LANDUSE = ["reservoir", "basin"]
ROAD_BUFFER_M = 20
WATER_SPURIOUS_THRESHOLD = 0.95
BLOCK_WATER_THRESHOLD = 0.1

# ── Geofence seed search area ──────────────────────────────────────────────────
# Seeds are restricted to this tighter box where trip density is high.
# Wider BBOX is used for block-building and spatial filtering only.
CENTER_BBOX = [-0.267, 51.449, 0.009, 51.575]

# ── Quotes cleaning thresholds ─────────────────────────────────────────────────
MAX_SPREAD_M   = 1000.0   # drop quote_ids whose pickup or dropoff GPS drift exceeds this
MAX_ETT_DIFF_S = 1000.0   # drop quote_ids whose ETT range across calls exceeds this

# ── Trip revenue pipeline ──────────────────────────────────────────────────────
RANDOM_SEED     = 42
SNAP_DISTANCE_M = 100

# ── Greedy optimiser ───────────────────────────────────────────────────────────
N_SEEDS       = 5
MAX_AREA_SQKM = 15
MAX_ITERS     = 300
