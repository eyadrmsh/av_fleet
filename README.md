# AV Fleet — London Geofence & Demand Forecasting

End-to-end pipeline for optimising an autonomous vehicle fleet's operating
geofence in London, from raw quote data through block-level road network
construction, revenue-maximising geofence optimisation, and hourly demand
forecasting.

---

## Repository layout

```
data/
  raw/
    THA_Dataset_MarketplaceDS.csv   # raw quotes with pricing calls
    greater-london-260411.osm.pbf   # OpenStreetMap road network
  processed/
    quotes_canonical.parquet        # one deduplicated record per quote
    quotes_cleaned.csv              # cleaned quotes (pre-canonical)
    filtered_main_blocks.geojson    # road-network blocks (main connected component)
    trips_assigned.csv              # trips with pickup/dropoff block IDs
    quotes_hourly_h3.parquet        # quotes aggregated to H3 zones by hour
    quotes_weather_h3.parquet       # quotes joined with weather features
    demand_forecast_apr2026.parquet # April 2026 demand forecast output
    weather_forecast_apr2026.parquet
    breathing_results/              # breathing geofence SA checkpoints

notebooks/                          # run in the order listed below
src/                                # shared Python modules
```

---

## Notebooks — run in order

### 0. `0_quotes_eda.ipynb` — Quotes EDA & Cleaning

Explores the raw quotes CSV (`THA_Dataset_MarketplaceDS.csv`) and produces the
canonical quote table used by all downstream notebooks.

- **Session structure** — each `quote_id` consists of repeated pricing calls;
  the notebook maps how many calls appear per session and why.
- **A/B price gap** — the data contains a £2 price experiment; the notebook
  characterises the split and its effect on conversion.
- **GPS & ETT behaviour** — examines coordinate drift and estimated travel time
  variation across repeated calls within a session.
- **Cleaning rules** — deduplication logic that collapses each session to a
  single canonical record; outputs `quotes_canonical.parquet`.

---

### Task 1 — Geofence

#### 1.1 `1.1_geofence_roads_closed.ipynb` — Road Network Blocks

Builds the block graph that all geofence algorithms operate on.

- Parses the OSM PBF file and closes road segments into discrete city blocks.
- Identifies the main connected component; excludes disconnected islands.
- Masks out parks and water bodies so blocks reflect driveable area only.
- Outputs `filtered_main_blocks.geojson`.

#### 1.2 `1.2_trip_points.ipynb` — Trip-to-Block Assignment

Projects raw trip pickup/dropoff coordinates onto the block graph.

- Snaps GPS coordinates to the nearest road within the main connected
  component, correcting for GPS drift.
- Filters out trips that fall outside the main component.
- Assigns each trip a `pickup_block` and `dropoff_block` from the block graph.
- Outputs `trips_assigned.csv`.

#### 1.3 `1.3_geofence_revenue_optimiser.ipynb` — Revenue Optimiser

Finds the revenue-maximising connected geofence zone within the block graph.

- Introduces the greedy expand algorithm and the simulated annealing (SA)
  variant; implementation details in `src/optimiser.py`.
- Compares both approaches on cumulative trip revenue; SA consistently
  outperforms greedy by escaping local optima.
- Produces the final London geofence (`london_geofence.geojson`).

---

### Task 2 — Demand Forecasting

#### 2.1 `2.1_h3_zone_assignment.ipynb` — H3 Zone Assignment

Maps quotes to Uber H3 hexagonal zones for spatial demand modelling.

- Assigns every cleaned quote to an H3 zone; filters zones with insufficient
  history to model reliably.
- Analyses demand and conversion patterns by hour of day and day of week.
- Identifies conversion-stable regions to prioritise during forecasting.
- Outputs `quotes_hourly_h3.parquet`.

#### 2.2 `2.2_weather_features.ipynb` — Weather Features

Fetches and analyses London weather data as a demand driver.

- Retrieves hourly weather observations for the study period.
- Examines the relationship between weather variables (rain, temperature, wind)
  and quote volume.
- Outputs `quotes_weather_h3.parquet` and `weather_forecast_apr2026.parquet`.

#### 2.3 `2.3_demand_forecast.ipynb` — Demand Forecast

Produces an hourly, zone-level demand forecast for April 2026.

**Approach:**
1. Predict the raw number of quotes appearing per H3 zone per hour using
   Prophet models (with optional lag features and weather regressors) and an
   OLS baseline.
2. Multiply predicted quote volume by the zone's historical average conversion
   rate for that hour-of-day × day-of-week combination.
3. Report three scenarios — lower bound, point estimate, and upper bound of the
   credible interval — so fleet planning can consider conservative, expected,
   and optimistic demand.

Outputs `demand_forecast_apr2026.parquet`.

---

### Experimental — `geofence_breathing.ipynb` — Breathing Geofence

> **Status: work in progress.** The breathing geofence currently produces lower
> revenue than the static geofence due to a revenue attribution error in the
> evaluation: `compute_revenue` requires both pickup **and** dropoff blocks to
> be inside the zone. Under this criterion the hourly geofences are unfairly
> penalised because trips accepted within the zone often drop off outside it.
> The fix (`compute_pickup_revenue`) is implemented but the comparison needs to
> be re-run.

Explores whether a geofence that adapts hour-by-hour to shifting demand can
capture more revenue than a zone fixed for the whole day.

- **Static baseline** — SA-optimised on 6-hour cumulative revenue, held fixed.
- **Hour 08:00** — fresh SA on that hour's trips alone; result saved to
  `data/processed/breathing_results/checkpoint_h8.pkl` to avoid re-running.
- **Hours 09:00–13:00** — each hour runs `run_transition` from `src/breathing.py`:
  1. *Border contract* — peels zero-revenue blocks from the zone's outer edge,
     tip-first; interior blocks are untouched.
  2. *Swap* — migrates toward better territory (reuses `optimiser._swap`).
  3. *Expand* — grows into adjacent positive-revenue blocks (reuses
     `optimiser._expand`).
  4. *Border continuity* — ensures every dropped block remains adjacent to the
     new zone so cars can physically reach it.
- Revenue evaluation uses **pickup-committed revenue**: a trip is credited to
  the hour's zone that accepted it (pickup block in zone), regardless of where
  the dropoff occurs or whether the trip spans the hour boundary.

---

## Source modules (`src/`)

| Module | Purpose |
|---|---|
| `config.py` | Shared paths, CRS, area cap, seed count |
| `blocks.py` | OSM parsing, block polygon construction |
| `optimiser.py` | `build_adjacency`, `is_connected`, `_expand`, `_swap`, `run_algorithm`, `run_multi_start` |
| `revenue.py` | `build_revenue_lookup`, `build_block_lookup`, `build_pickup_block_lookup`, `compute_revenue`, `compute_pickup_revenue`, `marginal_revenue`, `find_local_maxima_seeds` |
| `breathing.py` | `run_border_contract`, `run_transition`, `enforce_border_continuity`, hourly data helpers, plotting |
| `trips.py` | Trip loading and GPS snapping |
| `quotes_cleaning.py` | Deduplication and canonical quote construction |
| `forecasting.py` | Prophet & OLS model builders, zone forecast runner |
| `forecasting_vis_helpers.py` | Forecast visualisation utilities |
| `viz.py` | Shared map and block plotting helpers |

---

## Data inputs required

| File | Source |
|---|---|
| `data/raw/THA_Dataset_MarketplaceDS.csv` | Provided dataset |
| `data/raw/greater-london-260411.osm.pbf` | OpenStreetMap via Geofabrik |
