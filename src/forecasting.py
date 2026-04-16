"""
src/forecasting.py
------------------
Demand forecasting helpers for H3-zone hourly quote counts.

Supports six model variants:
  "base"          — Prophet with daily + weekly seasonality + UK holidays
  "lags"          — same, plus lag_1w and lag_2w as external regressors
  "full_temp"     — lags + temperature_2m
  "full_temp_vis" — lags + temperature_2m + visibility
  "full"          — lags + all four weather regressors

Lag construction note
---------------------
Lag regressors reference the same hour N weeks ago.  Where that reference
hour predates the start of the training window (i.e. the first 7/14 days of
data lack lag_1w/lag_2w history) those values are filled with 0 — a
conservative assumption of "no observed demand before the window."  This keeps
all rows usable rather than shrinking the effective training set.

Public API
----------
  prepare_zone_df              — reindex one zone to a full hourly grid and add lags
  build_prophet_model          — configure a Prophet instance (any variant)
  fit_and_forecast             — fit a Prophet model and return forecast DataFrame
  build_baseline_ols           — fit OLS with lags + hour/weekday dummies
  compute_mape                 — MAPE ignoring zero-actual rows
  run_zone_forecast            — orchestrate all variants for one zone × one split
  run_zone_grid_search         — grid search over changepoint/seasonality priors
  compute_historical_conversion — mean conversion rate per (hour × dow) up to a cutoff
  apply_conversion_to_forecast  — join conversion rates + compute trips columns
  save_demand_forecast          — save standardised demand forecast to parquet
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
import holidays as hols
import statsmodels.formula.api as smf
from prophet import Prophet

# Suppress verbose Stan/cmdstanpy output when the module is imported
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ── Constants ─────────────────────────────────────────────────────────────────

LAG_WEEKS = [1, 2]   # lag regressors: same hour N weeks ago

WEATHER_REGRESSORS = [
    "temperature_2m",
    "visibility",
    "precipitation_probability",
    "wind_speed_10m",
]

# Weather regressor subsets for progressive model variants
_VARIANT_WEATHER: dict = {
    "full_temp":     ["temperature_2m"],
    "full_temp_vis": ["temperature_2m", "visibility"],
    "full":          WEATHER_REGRESSORS,
}

# UK public holiday frame for Prophet (covers training + forecast window)
_uk_dates   = hols.country_holidays("GB", years=range(2025, 2028))
UK_HOLIDAYS = pd.DataFrame([
    {"holiday": name, "ds": pd.Timestamp(date)}
    for date, name in sorted(_uk_dates.items())
])

ProphetVariant = Literal["base", "lags", "full_temp", "full_temp_vis", "full"]


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_zone_df(
    df: pd.DataFrame,
    h3_zone: str,
    full_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Filter to one H3 zone, reindex to a complete hourly grid, and add lags.

    Missing hours (zone absent in that slot) are filled with quote_count = 0.
    Lag values that reference hours before the start of full_index are also
    filled with 0 (see module-level note).

    Parameters
    ----------
    df         : full quotes_weather_h3 DataFrame (all zones, all hours)
    h3_zone    : H3 cell identifier string
    full_index : complete tz-aware hourly DatetimeIndex for the desired period
                 (e.g. March 1 00:00 → March 31 23:00 UTC)

    Returns
    -------
    DataFrame with columns:
        hour_bucket  — original UTC timestamp (tz-aware)
        ds           — tz-naive datetime (Prophet requirement)
        y            — quote_count (int, 0-filled for missing hours)
        lag_1w       — quote_count same hour 7 days ago  (0-filled for gaps)
        lag_2w       — quote_count same hour 14 days ago (0-filled for gaps)
        <weather>    — weather regressors (ffill-filled, then 0 for any gaps)
    """
    weather_cols = [c for c in WEATHER_REGRESSORS if c in df.columns]
    zone_df = (
        df[df["h3_zone"] == h3_zone]
        .set_index("hour_bucket")[["quote_count"] + weather_cols]
        .reindex(full_index)
    )
    zone_df.index.name = "hour_bucket"
    zone_df["quote_count"] = zone_df["quote_count"].fillna(0).astype(int)

    # Weather is identical across zones — forward-fill any gap hours, then 0
    for col in weather_cols:
        zone_df[col] = zone_df[col].ffill().fillna(0)

    for w in LAG_WEEKS:
        zone_df[f"lag_{w}w"] = (
            zone_df["quote_count"].shift(w * 7 * 24).fillna(0)
        )

    zone_df = zone_df.reset_index()                  # hour_bucket becomes column
    zone_df["ds"] = zone_df["hour_bucket"].dt.tz_localize(None)  # Prophet: tz-naive
    zone_df["y"]  = zone_df["quote_count"]

    return_cols = ["hour_bucket", "ds", "y", "lag_1w", "lag_2w"] + weather_cols
    return zone_df[return_cols]


# ── Prophet model factory ─────────────────────────────────────────────────────

def build_prophet_model(
    variant: ProphetVariant,
    changepoint_prior_scale: float | None = None,
    seasonality_prior_scale: float | None = None,
) -> Prophet:
    """
    Configure a Prophet instance with daily + weekly seasonality and UK holidays.

    Daily  seasonality — period = 1 day  (24 hourly data points per cycle)
    Weekly seasonality — period = 7 days (168 hourly data points per cycle)

    Parameters
    ----------
    variant      : one of "base", "lags", "full_temp", "full_temp_vis", "full"
        "base"          — seasonality + holidays, no extra regressors
        "lags"          — same + lag_1w and lag_2w
        "full_temp"     — lags + temperature_2m
        "full_temp_vis" — lags + temperature_2m + visibility
        "full"          — lags + all four weather regressors
    changepoint_prior_scale : override Prophet's default (0.05); useful for grid search.
    seasonality_prior_scale : override Prophet's default (10.0); useful for grid search.

    Returns
    -------
    Unfitted Prophet instance.
    """
    if variant not in ("base", "lags", "full_temp", "full_temp_vis", "full"):
        raise ValueError(
            f"variant must be one of 'base', 'lags', 'full_temp', 'full_temp_vis', "
            f"or 'full', got {variant!r}"
        )

    prophet_kwargs = dict(
        holidays=UK_HOLIDAYS,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.95,
        uncertainty_samples=1000,
    )

    # Allow caller to override either prior (e.g. for grid search)
    if changepoint_prior_scale is not None:
        prophet_kwargs["changepoint_prior_scale"] = changepoint_prior_scale
    if seasonality_prior_scale is not None:
        prophet_kwargs["seasonality_prior_scale"] = seasonality_prior_scale

    m = Prophet(**prophet_kwargs)
    m.add_seasonality(name="daily",  period=1, fourier_order=8)
    m.add_seasonality(name="weekly", period=7, fourier_order=3)

    if variant in ("lags", "full_temp", "full_temp_vis", "full"):
        m.add_regressor("lag_1w")
        m.add_regressor("lag_2w")

    for col in _VARIANT_WEATHER.get(variant, []):
        m.add_regressor(col)

    return m


# ── Fit and forecast ──────────────────────────────────────────────────────────

def fit_and_forecast(
    model: Prophet,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fit a Prophet model and return predictions for future_df rows.

    Parameters
    ----------
    train_df  : must have ds, y; also regressors if model uses them
    future_df : must have ds; also regressors if model uses them

    Returns
    -------
    DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.
    yhat is clipped to ≥ 0 (demand cannot be negative).
    """
    model.fit(train_df)
    forecast         = model.predict(future_df)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ── OLS baseline ──────────────────────────────────────────────────────────────

def build_baseline_ols(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    OLS regression: y ~ lag_1w + lag_2w + C(hour) + C(dow).

    Hour-of-day and day-of-week dummies capture the diurnal and weekly
    patterns that Prophet handles through its seasonality components.

    Parameters
    ----------
    train_df  : columns ds, y, lag_1w, lag_2w
    future_df : columns ds, lag_1w, lag_2w

    Returns
    -------
    DataFrame with columns: ds, yhat  (clipped to ≥ 0).
    """
    def _add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
        out       = frame.copy()
        ts        = pd.to_datetime(out["ds"])
        out["hour"] = ts.dt.hour.astype("category")
        out["dow"]  = ts.dt.dayofweek.astype("category")
        return out

    train  = _add_time_features(train_df)
    future = _add_time_features(future_df)

    result = smf.ols("y ~ lag_1w + lag_2w + C(hour) + C(dow)", data=train).fit()
    yhat   = result.predict(future).clip(lower=0)

    return pd.DataFrame({"ds": future["ds"].values, "yhat": yhat.values})


# ── Evaluation metric ─────────────────────────────────────────────────────────

def compute_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Mean Absolute Percentage Error, skipping rows where actual == 0.

    Skipping zeros avoids infinite errors from hours with no demand.
    Returns NaN if no nonzero-actual rows remain.
    """
    mask = actual > 0
    if mask.sum() == 0:
        return float("nan")
    pct_errors = (actual[mask] - predicted[mask]).abs() / actual[mask] * 100
    return float(pct_errors.mean())


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_zone_forecast(
    zone_df: pd.DataFrame,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
) -> dict:
    """
    Fit all five variants for one zone × one train/eval split.

    All boundary timestamps must be tz-naive (matched to zone_df["ds"]).

    Parameters
    ----------
    zone_df    : output of prepare_zone_df for this zone
    train_end  : last hour of training window (inclusive, tz-naive)
    eval_start : first hour of evaluation window (tz-naive)
    eval_end   : last hour of evaluation window (inclusive, tz-naive)

    Returns
    -------
    {
      "ols":                  {"mape": float, "forecast": pd.DataFrame},
      "prophet_base":         {"mape": float, "forecast": pd.DataFrame},
      "prophet_lags":         {"mape": float, "forecast": pd.DataFrame},
      "prophet_full_temp":    {"mape": float, "forecast": pd.DataFrame},
      "prophet_full_temp_vis":{"mape": float, "forecast": pd.DataFrame},
      "prophet_full":         {"mape": float, "forecast": pd.DataFrame},
    }
    Each forecast DataFrame has columns: ds, yhat[, yhat_lower, yhat_upper].
    MAPE is computed over the eval window, aligned on ds.
    """
    train = zone_df[zone_df["ds"] <= train_end]
    eval_ = zone_df[
        (zone_df["ds"] >= eval_start) & (zone_df["ds"] <= eval_end)
    ].copy()

    def _mape_from_fc(fc: pd.DataFrame) -> float:
        merged = eval_[["ds", "y"]].merge(fc[["ds", "yhat"]], on="ds", how="left")
        return compute_mape(merged["y"].reset_index(drop=True),
                            merged["yhat"].reset_index(drop=True))

    results = {}

    # ── OLS baseline ──────────────────────────────────────────────────────
    ols_fc = build_baseline_ols(train, eval_[["ds", "lag_1w", "lag_2w"]])
    results["ols"] = {"mape": _mape_from_fc(ols_fc), "forecast": ols_fc}

    # ── Prophet base ──────────────────────────────────────────────────────
    m_base  = build_prophet_model("base")
    base_fc = fit_and_forecast(m_base, train, eval_[["ds"]])
    results["prophet_base"] = {"mape": _mape_from_fc(base_fc), "forecast": base_fc}

    # ── Prophet + lags ────────────────────────────────────────────────────
    m_lags  = build_prophet_model("lags")
    lags_fc = fit_and_forecast(m_lags, train, eval_[["ds", "lag_1w", "lag_2w"]])
    results["prophet_lags"] = {"mape": _mape_from_fc(lags_fc), "forecast": lags_fc}

    # ── Progressive weather variants ──────────────────────────────────────
    avail = set(eval_.columns)
    for variant, result_key in [
        ("full_temp",     "prophet_full_temp"),
        ("full_temp_vis", "prophet_full_temp_vis"),
        ("full",          "prophet_full"),
    ]:
        weather_cols = [c for c in _VARIANT_WEATHER[variant] if c in avail]
        future       = eval_[["ds", "lag_1w", "lag_2w"] + weather_cols]
        m            = build_prophet_model(variant)
        fc           = fit_and_forecast(m, train, future)
        results[result_key] = {"mape": _mape_from_fc(fc), "forecast": fc}

    return results


# ── Per-zone hyperparameter grid search ───────────────────────────────────────

def run_zone_grid_search(
    zone_df: pd.DataFrame,
    param_grid: dict,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    variant: ProphetVariant = "full",
) -> pd.DataFrame:
    """
    Grid search over changepoint_prior_scale × seasonality_prior_scale for
    any weather-bearing Prophet variant, evaluated on one train/eval split.

    Parameters
    ----------
    zone_df    : output of prepare_zone_df for this zone
    param_grid : {
                   "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.5],
                   "seasonality_prior_scale": [1.0, 5.0, 10.0, 20.0],
                 }
    train_end  : last hour of training window (inclusive, tz-naive)
    eval_start : first hour of evaluation window (tz-naive)
    eval_end   : last hour of evaluation window (inclusive, tz-naive)
    variant    : Prophet variant to tune (must be one of the weather variants:
                 "full_temp", "full_temp_vis", or "full"). Default "full".

    Returns
    -------
    DataFrame with columns: changepoint_prior_scale, seasonality_prior_scale, mape
    sorted by mape ascending (best first).
    """
    train = zone_df[zone_df["ds"] <= train_end]
    eval_ = zone_df[
        (zone_df["ds"] >= eval_start) & (zone_df["ds"] <= eval_end)
    ].copy()

    weather_cols = [c for c in _VARIANT_WEATHER.get(variant, []) if c in eval_.columns]
    future = eval_[["ds", "lag_1w", "lag_2w"] + weather_cols]

    rows = []
    for cps_val in param_grid["changepoint_prior_scale"]:
        for sps_val in param_grid["seasonality_prior_scale"]:
            m  = build_prophet_model(
                variant,
                changepoint_prior_scale=cps_val,
                seasonality_prior_scale=sps_val,
            )
            fc = fit_and_forecast(m, train, future)
            merged = eval_[["ds", "y"]].merge(fc[["ds", "yhat"]], on="ds", how="left")
            mape = compute_mape(
                merged["y"].reset_index(drop=True),
                merged["yhat"].reset_index(drop=True),
            )
            rows.append({
                "changepoint_prior_scale": cps_val,
                "seasonality_prior_scale": sps_val,
                "mape": mape,
            })

    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


# ── Conversion rate helpers ───────────────────────────────────────────────────

#: Standard column order for the saved demand forecast parquet.
DEMAND_FORECAST_COLS: list[str] = [
    "ds", "zone",
    "yhat_lower", "yhat", "yhat_upper",
    "conv_rate",
    "trips_lower", "trips_pred", "trips_upper",
]


def compute_historical_conversion(
    df: pd.DataFrame,
    h3_zone: str,
    cutoff_ts: pd.Timestamp,
) -> pd.DataFrame:
    """
    Mean conversion rate per (hour_of_day × day_of_week) for one H3 zone,
    restricted to records with hour_bucket strictly before `cutoff_ts`.

    Using a cutoff prevents leakage: only data that would have been available
    at prediction time is used to estimate conversion rates.

    Parameters
    ----------
    df        : quotes_weather_h3 DataFrame with columns
                hour_bucket (tz-aware), h3_zone, conversion_rate.
    h3_zone   : H3 cell identifier.
    cutoff_ts : tz-aware Timestamp.  Only rows where hour_bucket < cutoff_ts
                are included.  Typical values:
                  - eval Mar 22–28  → pd.Timestamp("2026-03-22", tz="UTC")
                  - April forecast  → pd.Timestamp("2026-04-01", tz="UTC")

    Returns
    -------
    DataFrame with columns:
        hour_of_day (int 0–23), dow (int 0=Mon … 6=Sun), conv_rate (float)
    Up to 168 rows (24 × 7); fewer if some (hour, dow) combos lack history.
    """
    mask = (df["h3_zone"] == h3_zone) & (df["hour_bucket"] < cutoff_ts)
    sub  = df.loc[mask].copy()
    sub["hour_of_day"] = sub["hour_bucket"].dt.hour
    sub["dow"]         = sub["hour_bucket"].dt.dayofweek
    return (
        sub.groupby(["hour_of_day", "dow"])["conversion_rate"]
        .mean()
        .reset_index()
        .rename(columns={"conversion_rate": "conv_rate"})
    )


def apply_conversion_to_forecast(
    fc: pd.DataFrame,
    conv_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich a forecast DataFrame with historical conversion rates and
    derived trip-count estimates.

    Parameters
    ----------
    fc      : forecast DataFrame with columns ds (tz-naive or tz-aware),
              yhat, yhat_lower, yhat_upper.
    conv_df : output of compute_historical_conversion
              (columns: hour_of_day, dow, conv_rate).

    Returns
    -------
    Copy of fc with added columns:
        hour_of_day  — extracted from ds
        dow          — extracted from ds (0 = Monday)
        conv_rate    — historical average conversion for that (hour, dow)
        trips_lower  — max(yhat_lower, 0) × conv_rate, rounded to 2 d.p.
        trips_pred   — yhat × conv_rate, rounded to 2 d.p.
        trips_upper  — yhat_upper × conv_rate, rounded to 2 d.p.
    """
    out = fc.copy()
    ts  = pd.to_datetime(out["ds"])
    out["hour_of_day"] = ts.dt.hour
    out["dow"]         = ts.dt.dayofweek
    out = out.merge(conv_df, on=["hour_of_day", "dow"], how="left")
    out["trips_lower"] = (out["yhat_lower"].clip(lower=0) * out["conv_rate"]).round(2)
    out["trips_pred"]  = (out["yhat"]                      * out["conv_rate"]).round(2)
    out["trips_upper"] = (out["yhat_upper"]                * out["conv_rate"]).round(2)
    return out


def save_demand_forecast(
    forecast_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """
    Save a demand forecast DataFrame to parquet using the standard column schema.

    """
    cols = [c for c in DEMAND_FORECAST_COLS if c in forecast_df.columns]
    out  = forecast_df[cols].sort_values(["zone", "ds"]).reset_index(drop=True)
    out.to_parquet(output_path, index=False)
    return out
