#!/usr/bin/env python3
"""
Ginger farmgate price forecast - every month of 2027
====================================================

Algorithm
---------
Regression with seasonal-ARIMA errors (SARIMAX) fitted on LOG price, one model per
province, with (log) ginger production volume as an exogenous regressor.

Why this algorithm (evidence from the two files)
  * ~198 monthly observations per province           -> too few for GBM / LSTM.
  * Price ACF(1) ~ 0.88 with slow decay, multi-year  -> persistent, non-stationary
    cycles (2014-15, 2019-20 spikes)                    -> ARIMA-style differencing.
  * Seasonality is real but modest (strength ~0.2,   -> seasonal AR/MA terms, NOT
    Apr-Jul peak, Sep-Dec trough)                       forced seasonal differencing.
  * Production volume is negatively correlated with  -> exogenous regressor.
    price (about -0.3 quarterly, -0.27 de-seasonalised)
  * Davao de Oro is missing 57 of 198 months (29 %)  -> state-space (Kalman) form
    once '..' and hidden 0.00 placeholders are          handles gaps natively, no
    treated as missing                                  imputation needed.
  * Forecast horizon is 18 months beyond the last    -> need honest prediction
    observation (Jun 2026 -> Dec 2027)                  intervals.

Model selection is data driven and honest:
  1. d (regular differencing) from ADF + KPSS, D (seasonal differencing) from
     seasonal strength (Hyndman's 0.64 rule).
  2. Grid over (p, q, P, Q) x {with volume, without volume}; AIC shortlist
     (AIC is only compared between models that share the same d and D).
  3. Shortlist is scored with rolling-origin cross-validation (18-month horizon,
     scored on horizons 7-18 = the 2027 months) against naive baselines.
     Future volume inside the CV is *forecast*, never peeked at.
  4. Champion is refit on all data and forecasts Jan-Dec 2027 with 80 % / 95 %
     intervals. A volume-weighted composite of the provinces is also produced.

Requirements:  pip install pandas numpy matplotlib statsmodels
Inputs      :  prices.csv, volume_production.csv  (same folder, or edit CONFIG)
Outputs     :  ./output/  (clean data, CV scores, model summaries, forecast CSV, plots)
"""

from __future__ import annotations

import itertools
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; plots are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =============================================================================
# CONFIG
# =============================================================================
PRICE_FILE = Path("prices.csv")
VOLUME_FILE = Path("volume_production.csv")
OUT_DIR = Path("output")

TARGET_YEAR = 2027
PERIOD = 12                       # monthly data -> yearly seasonality

# Cleaning
OUTLIER_ACTION = "flag"           # "flag" = report only | "drop" = set flagged points to NaN
HAMPEL_WINDOW, HAMPEL_K = 13, 3.5

# Order-search grid (d and D are chosen from the data, see choose_d / choose_D)
P_GRID, Q_GRID = [0, 1, 2], [0, 1, 2]
SP_GRID, SQ_GRID = [0, 1], [0, 1]
TOP_K_PER_GROUP = 3               # AIC shortlist size (with-volume / without-volume)

# Rolling-origin cross-validation
CV_HORIZON = 18                   # Jun-2026 -> Dec-2027 is 18 steps
CV_MIN_TRAIN = 84                 # first origin after 7 years of data
CV_STEP = 6                       # months between origins
SELECT_HORIZONS = list(range(7, CV_HORIZON + 1))  # = the 2027 months (h = 7..18)

VOLUME_SEASONAL_YEARS = 3         # future volume = median of same quarter, last 3 years
PI_LEVELS = (0.80, 0.95)


# =============================================================================
# 1. LOAD + CLEAN
# =============================================================================
_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS)}


def _header_row(path: Path) -> int:
    """Exports start with a title line + blank line; the header is the first ';' line."""
    with open(path, encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh):
            if line.count(";") >= 3:
                return i
    raise ValueError(f"No ';'-delimited header row found in {path}")


def _read_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=_header_row(path), dtype=str,
                     encoding="utf-8-sig", keep_default_na=False)
    df.columns = df.columns.str.strip()
    return df


def _clean_geo(s: pd.Series) -> pd.Series:
    """'....Davao de Oro' -> 'Davao de Oro'"""
    return s.str.replace(r"^[.\s]+", "", regex=True).str.strip()


def _parse_month_col(col: str):
    """'2012 January' -> Timestamp('2012-01-01'); '2012 Annual' -> NaT"""
    m = re.fullmatch(r"(\d{4})\s+([A-Za-z]+)", col.strip())
    if m and m.group(2) in _MONTH_NUM:
        return pd.Timestamp(int(m.group(1)), _MONTH_NUM[m.group(2)], 1)
    return pd.NaT


def _parse_quarter_col(col: str):
    """'2012 Quarter3' -> Timestamp('2012-07-01'); Semester / Annual columns -> NaT"""
    m = re.fullmatch(r"(\d{4})\s+Quarter\s*([1-4])", col.strip())
    if m:
        return pd.Timestamp(int(m.group(1)), 3 * (int(m.group(2)) - 1) + 1, 1)
    return pd.NaT


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_prices(path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Returns monthly ginger farmgate price (index = month start, columns = province).

    Cleaning rules
      * drop title/blank lines; ';' delimiter; strip '....' prefix from province names
      * keep ginger rows only
      * keep the 12 monthly columns per year; DROP the '<year> Annual' columns
        (they disagree with the monthly values in several years, e.g. 2024-25)
      * '..' (not available)  -> NaN
      * 0.00 -> NaN : a farmgate price of zero is a missing-data code, not a price
        (all of 2021-22 for Davao de Oro, plus scattered months in 2024-25)
      * drop trailing months with no data in any province (Jul-Dec 2026 placeholders)
      * reindex to a gap-free monthly calendar
    """
    raw = _read_export(path)
    raw["Geolocation"] = _clean_geo(raw["Geolocation"])
    raw = raw[raw["Commodity"].str.contains("ginger", case=False, na=False)]

    month_cols = {c: _parse_month_col(c) for c in raw.columns[2:]}
    month_cols = {c: d for c, d in month_cols.items() if pd.notna(d)}

    long = raw.melt(id_vars="Geolocation", value_vars=list(month_cols),
                    var_name="col", value_name="raw")
    long["date"] = long["col"].map(month_cols)
    long["price"] = pd.to_numeric(long["raw"], errors="coerce")

    report = {
        "dot_tokens": int((long["raw"].str.strip() == "..").sum()),
        "zero_prices_by_province": long.loc[long["price"] == 0].groupby("Geolocation").size().to_dict(),
    }
    long.loc[long["price"] <= 0, "price"] = np.nan

    wide = long.groupby(["date", "Geolocation"])["price"].mean().unstack("Geolocation").sort_index()
    last_obs = wide.dropna(how="all").index.max()
    wide = wide.loc[:last_obs].asfreq("MS")
    wide.columns.name = None

    report["months"] = len(wide)
    report["first_month"], report["last_month"] = wide.index[0], wide.index[-1]
    report["missing_after_cleaning"] = wide.isna().sum().to_dict()
    return wide, report


def load_volume(path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Returns quarterly ginger production volume (index = quarter start, columns = province).

    Cleaning rules
      * keep ginger rows only (file also contains Turmeric rows - irrelevant, all ~0)
      * keep 'YYYY QuarterN' columns only. 'Semester' and 'Annual' columns are
        redundant sums of the quarters (verified below) and would double count.
      * '..' -> NaN, drop trailing empty quarters (2026 Q3-Q4)
    """
    raw = _read_export(path)
    raw["Geolocation"] = _clean_geo(raw["Geolocation"])
    raw = raw[raw["Crop"].str.contains("ginger", case=False, na=False)]

    # sanity check: Annual == sum of the four quarters (for complete years)
    years = sorted({int(c[:4]) for c in raw.columns[2:] if c[:4].isdigit()})
    max_gap = 0.0
    for _, row in raw.iterrows():
        for y in years:
            q = [_num(row.get(f"{y} Quarter{i}")) for i in range(1, 5)]
            a = _num(row.get(f"{y} Annual"))
            if not np.isnan(q).any() and not np.isnan(a):
                max_gap = max(max_gap, abs(sum(q) - a))

    q_cols = {c: _parse_quarter_col(c) for c in raw.columns[2:]}
    q_cols = {c: d for c, d in q_cols.items() if pd.notna(d)}

    long = raw.melt(id_vars="Geolocation", value_vars=list(q_cols),
                    var_name="col", value_name="raw")
    long["date"] = long["col"].map(q_cols)
    long["volume"] = pd.to_numeric(long["raw"], errors="coerce")
    long.loc[long["volume"] <= 0, "volume"] = np.nan

    vol = long.groupby(["date", "Geolocation"])["volume"].mean().unstack("Geolocation").sort_index()
    vol = vol.loc[:vol.dropna(how="all").index.max()]
    vol.columns.name = None

    report = {"max_abs_gap_annual_vs_quarters": round(max_gap, 4),
              "first_quarter": vol.index[0], "last_quarter": vol.index[-1],
              "missing_after_cleaning": vol.isna().sum().to_dict()}
    return vol, report


def flag_outliers(prices: pd.DataFrame) -> pd.DataFrame:
    """Hampel filter on log price (rolling median +/- k * scaled MAD). Diagnostic by default."""
    flags = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    for col in prices:
        s = np.log(prices[col].dropna())
        med = s.rolling(HAMPEL_WINDOW, center=True, min_periods=5).median()
        mad = (s - med).abs().rolling(HAMPEL_WINDOW, center=True, min_periods=5).median()
        mad = mad.clip(lower=0.03)  # floor so flat stretches do not over-flag
        flags.loc[s.index, col] = ((s - med).abs() > HAMPEL_K * 1.4826 * mad).values
    return flags


# =============================================================================
# 2. FEATURES: monthly exogenous volume (actual where known, forecast afterwards)
# =============================================================================
def monthly_volume(vq: pd.Series, months: pd.DatetimeIndex, known_until: pd.Timestamp) -> pd.Series:
    """
    Average monthly volume (= quarter total / 3) for every month in `months`.

    * quarters that are COMPLETE at `known_until` use the reported value
    * later quarters use the median of the same quarter number over the last
      VOLUME_SEASONAL_YEARS completed years (robust to one-off spikes such as
      Davao de Oro Q3-2023). Volume is very regular by quarter, so this is a
      sound, leakage-free forecast for a regressor.
    """
    vq = vq.dropna()
    known = vq[(vq.index + pd.DateOffset(months=2)) <= known_until]
    out = []
    for m in months:
        qs = m.to_period("Q").start_time
        if qs in known.index:
            v = known.loc[qs]
        else:
            same_q = known[known.index.quarter == qs.quarter].tail(VOLUME_SEASONAL_YEARS)
            v = same_q.median()
        out.append(v / 3.0)
    return pd.Series(out, index=months, name="volume")


def build_exog(vq: pd.Series, months: pd.DatetimeIndex, known_until: pd.Timestamp,
               mu: float, sd: float) -> pd.DataFrame:
    """Standardised log monthly volume, the single exogenous regressor."""
    v = monthly_volume(vq, months, known_until)
    return ((np.log(v) - mu) / sd).to_frame("log_volume_z")


# =============================================================================
# 3. DIAGNOSTICS -> choose d and D
# =============================================================================
def choose_d(y_log: pd.Series, max_d: int = 2) -> int:
    """Smallest d for which ADF rejects a unit root AND KPSS does not reject stationarity."""
    x = y_log.interpolate(limit_area="inside").dropna().to_numpy()  # diagnostic use only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in range(max_d + 1):
            z = np.diff(x, n=d) if d else x
            adf_p = adfuller(z, autolag="AIC")[1]
            kpss_p = kpss(z, regression="c", nlags="auto")[1]
            if adf_p < 0.05 and kpss_p > 0.05:
                return d
    return 1  # persistent series with no clean verdict -> difference once


def seasonal_strength(y_log: pd.Series, period: int = PERIOD) -> float:
    """Hyndman's seasonal strength Fs = max(0, 1 - Var(R) / Var(S + R)), classical decomposition."""
    s = y_log.interpolate(limit_area="inside")
    trend = s.rolling(period, center=True).mean().rolling(2, center=True).mean().shift(-1)
    detr = (s - trend).dropna()
    seas = detr.groupby(detr.index.month).transform("mean")
    resid = detr - seas
    return float(max(0.0, 1.0 - resid.var() / (resid + seas).var()))


def choose_D(y_log: pd.Series) -> tuple[int, float]:
    strength = seasonal_strength(y_log)
    return (1 if strength > 0.64 else 0), strength


# =============================================================================
# 4. MODEL SEARCH + ROLLING-ORIGIN CROSS-VALIDATION
# =============================================================================
@dataclass(frozen=True)
class Spec:
    order: tuple
    seasonal: tuple
    trend: str
    use_exog: bool

    @property
    def label(self) -> str:
        return f"SARIMAX{self.order}x{self.seasonal}{'+volume' if self.use_exog else ''}"


def fit_spec(y_log: pd.Series, exog: pd.DataFrame | None, spec: Spec):
    """Fit one SARIMAX. NaNs in y_log are handled by the Kalman filter (no imputation)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(y_log,
                        exog=exog if spec.use_exog else None,
                        order=spec.order,
                        seasonal_order=spec.seasonal,
                        trend=spec.trend)
        return model.fit(disp=False, maxiter=200)


def aic_shortlist(y_log: pd.Series, exog: pd.DataFrame, d: int, D: int) -> list[Spec]:
    """Grid search on AIC (valid because every candidate shares the same d and D)."""
    trend = "c" if (d + D) == 0 else "n"   # a constant is unidentified once differenced
    rows = []
    for use_exog in (False, True):
        for p, q, P, Q in itertools.product(P_GRID, Q_GRID, SP_GRID, SQ_GRID):
            spec = Spec((p, d, q), (P, D, Q, PERIOD), trend, use_exog)
            try:
                aic = float(fit_spec(y_log, exog, spec).aic)
            except Exception:
                continue
            if np.isfinite(aic):
                rows.append((use_exog, aic, spec))
    shortlist = []
    for use_exog in (False, True):
        grp = sorted([r for r in rows if r[0] == use_exog], key=lambda r: r[1])
        shortlist += [r[2] for r in grp[:TOP_K_PER_GROUP]]
    return shortlist


def baseline_forecasts(y_train: pd.Series, horizon: int) -> dict[str, np.ndarray]:
    """Benchmarks every real model has to beat (price scale)."""
    ff = y_train.ffill()
    n = len(ff)
    seasonal_naive = np.array([ff.iloc[n - 1 + h - PERIOD * int(np.ceil(h / PERIOD))]
                               for h in range(1, horizon + 1)])
    return {
        "Baseline: naive (last value)": np.repeat(ff.iloc[-1], horizon),
        "Baseline: seasonal naive": seasonal_naive,
        "Baseline: mean of last 12 months": np.repeat(ff.iloc[-PERIOD:].mean(), horizon),
    }


def rolling_origin_cv(y_log: pd.Series, vq: pd.Series, specs: list[Spec],
                      mu: float, sd: float) -> pd.DataFrame:
    """Expanding-window CV. Future volume is forecast from information available at the origin."""
    y_raw = np.exp(y_log)
    first = y_log.index[CV_MIN_TRAIN - 1]
    last = y_log.index[-1] - pd.DateOffset(months=CV_HORIZON)
    origins = pd.date_range(first, last, freq=f"{CV_STEP}MS")
    print(f"  {len(origins)} forecast origins x {len(specs)} candidate models "
          f"(+ baselines), horizon {CV_HORIZON} months")

    recs = []
    for origin in origins:
        train = y_log.loc[:origin]
        fut = pd.date_range(origin, periods=CV_HORIZON + 1, freq="MS")[1:]
        actual = y_raw.reindex(fut)
        if actual.notna().sum() == 0:
            continue
        exog_all = build_exog(vq, train.index.append(fut), origin, mu, sd)
        exog_train, exog_fut = exog_all.loc[train.index], exog_all.loc[fut]

        preds = {}
        for spec in specs:
            try:
                res = fit_spec(train, exog_train, spec)
                fc = res.get_forecast(steps=CV_HORIZON, exog=exog_fut if spec.use_exog else None)
                preds[spec.label] = np.exp(np.asarray(fc.predicted_mean))
            except Exception:
                continue
        preds.update(baseline_forecasts(np.exp(train), CV_HORIZON))

        for model, arr in preds.items():
            for h, (dt, p) in enumerate(zip(fut, arr), start=1):
                recs.append({"model": model, "origin": origin, "h": h, "date": dt,
                             "actual": actual.loc[dt], "pred": float(p)})
    return pd.DataFrame(recs)


def summarize_cv(rec: pd.DataFrame) -> pd.DataFrame:
    rec = rec.dropna(subset=["actual"]).copy()
    rec["err"] = rec["pred"] - rec["actual"]
    rec["abs_err"] = rec["err"].abs()
    rec["ape"] = rec["abs_err"] / rec["actual"]
    sel = rec[rec["h"].isin(SELECT_HORIZONS)]
    g = sel.groupby("model")
    out = pd.DataFrame({
        "MAE_h7_18": g["abs_err"].mean(),
        "RMSE_h7_18": g["err"].apply(lambda e: float(np.sqrt((e ** 2).mean()))),
        "MAPE_pct_h7_18": g["ape"].mean() * 100,
        "n_points": g.size(),
    })
    out["MAE_all_h"] = rec.groupby("model")["abs_err"].mean()
    return out.sort_values("MAE_h7_18")


# =============================================================================
# 5. FINAL FIT + 2027 FORECAST
# =============================================================================
def forecast_target_year(y_log: pd.Series, vq: pd.Series, spec: Spec,
                         mu: float, sd: float):
    """Refit the champion on all data and forecast through December of TARGET_YEAR."""
    last_obs = y_log.index[-1]
    future = pd.date_range(last_obs, f"{TARGET_YEAR}-12-01", freq="MS")[1:]
    steps = len(future)

    exog_all = build_exog(vq, y_log.index.append(future), last_obs, mu, sd)
    res = fit_spec(y_log, exog_all.loc[y_log.index], spec)
    fc = res.get_forecast(steps=steps, exog=exog_all.loc[future] if spec.use_exog else None)

    out = pd.DataFrame({"forecast": np.exp(np.asarray(fc.predicted_mean))}, index=future)
    for lvl in PI_LEVELS:
        ci = fc.conf_int(alpha=1 - lvl)
        tag = int(round(lvl * 100))
        out[f"lo{tag}"] = np.exp(np.asarray(ci.iloc[:, 0]))
        out[f"hi{tag}"] = np.exp(np.asarray(ci.iloc[:, 1]))
    return out.loc[str(TARGET_YEAR)], res


def composite_forecast(fc_by_prov: dict[str, pd.DataFrame], vol_by_prov: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-weighted average of the province forecasts (weights = forecast monthly
    production share). Interval bounds are weighted averages of the province
    bounds, i.e. they assume perfectly correlated errors -> conservative (wide).
    """
    w = vol_by_prov.div(vol_by_prov.sum(axis=1), axis=0)
    cols = next(iter(fc_by_prov.values())).columns
    comp = sum(fc_by_prov[p][cols].mul(w[p], axis=0) for p in fc_by_prov)
    return comp


# =============================================================================
# 6. PLOTS
# =============================================================================
def plot_forecast(history: pd.Series, fc: pd.DataFrame, title: str, path: Path, years_back: int = 6):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    h = history.dropna()
    hist = h[h.index >= history.index[-1] - pd.DateOffset(years=years_back)]
    ax.plot(hist.index, hist.values, color="#444", lw=1.6, label="Actual (cleaned)")
    ax.plot(fc.index, fc["forecast"], color="#c0392b", lw=2, marker="o", ms=3.5, label="Forecast")
    ax.fill_between(fc.index, fc["lo95"], fc["hi95"], color="#c0392b", alpha=0.12, label="95% interval")
    ax.fill_between(fc.index, fc["lo80"], fc["hi80"], color="#c0392b", alpha=0.25, label="80% interval")
    ax.set_title(title)
    ax.set_ylabel("Farmgate price (as reported in source)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. load + clean ----------------------------------------------------
    prices, p_rep = load_prices(PRICE_FILE)
    vol_q, v_rep = load_volume(VOLUME_FILE)
    print("PRICE  cleaning report :", p_rep)
    print("VOLUME cleaning report :", v_rep)

    flags = flag_outliers(prices)
    for col in prices:
        hits = prices.loc[flags[col], col]
        print(f"Hampel outlier candidates - {col}: {len(hits)}")
        if len(hits):
            print(hits.round(2).to_string())
    if OUTLIER_ACTION == "drop":
        prices = prices.mask(flags)

    prices.to_csv(OUT_DIR / "clean_prices_monthly.csv")
    vol_q.to_csv(OUT_DIR / "clean_volume_quarterly.csv")

    provinces = [c for c in prices.columns if c in vol_q.columns]
    assert provinces, "No province appears in both files - check the Geolocation names."

    # ---- 2-5. per-province model search, CV, final forecast -----------------
    fc_by_prov, vol_fc, champions, cv_tables = {}, {}, {}, []
    for prov in provinces:
        print("\n" + "=" * 78 + f"\n{prov}\n" + "=" * 78)
        y_log = np.log(prices[prov])
        vq = vol_q[prov]
        last_obs = y_log.index[-1]

        hist_vol = np.log(monthly_volume(vq, y_log.index, last_obs))
        mu, sd = float(hist_vol.mean()), float(hist_vol.std())     # scaling only (no target leakage)
        exog_hist = build_exog(vq, y_log.index, last_obs, mu, sd)

        d = choose_d(y_log)
        D, strength = choose_D(y_log)
        print(f"d = {d} (ADF/KPSS) | seasonal strength = {strength:.2f} -> D = {D}")

        print("AIC grid search ...")
        shortlist = aic_shortlist(y_log, exog_hist, d, D)
        print("Shortlist:", [s.label for s in shortlist])

        print("Rolling-origin cross-validation ...")
        cv = rolling_origin_cv(y_log, vq, shortlist, mu, sd)
        cv_summary = summarize_cv(cv) if not cv.empty else pd.DataFrame()
        sarimax_rows = cv_summary[cv_summary.index.str.startswith("SARIMAX")] if len(cv_summary) else cv_summary
        base_rows = cv_summary[cv_summary.index.str.startswith("Baseline")] if len(cv_summary) else cv_summary

        if len(sarimax_rows):
            champion = next(s for s in shortlist if s.label == sarimax_rows.index[0])
            best_base = base_rows["MAE_h7_18"].min()
            skill = 1 - sarimax_rows["MAE_h7_18"].iloc[0] / best_base
            print(cv_summary.round(2).to_string())
            print(f"\nChampion: {champion.label} | skill vs best baseline (MAE, h=7-18): {skill:+.1%}")
            if skill <= 0:
                print("WARNING: champion does not beat the best naive baseline - treat the forecast "
                      "as a level anchor with wide uncertainty, not as a precise prediction.")
        else:  # CV failed for every candidate -> fall back to the best AIC model
            champion = shortlist[0]
            print(f"CV produced no results; falling back to lowest-AIC model: {champion.label}")

        cv_summary.assign(province=prov).to_csv(OUT_DIR / f"cv_scores_{prov.replace(' ', '_')}.csv")
        cv_tables.append(cv_summary.assign(province=prov))
        champions[prov] = champion

        fc, res = forecast_target_year(y_log, vq, champion, mu, sd)
        fc_by_prov[prov] = fc

        months = fc.index
        vol_fc[prov] = monthly_volume(vq, months, last_obs)

        (OUT_DIR / f"model_summary_{prov.replace(' ', '_')}.txt").write_text(str(res.summary()))
        try:
            res.plot_diagnostics(figsize=(12, 8)).savefig(
                OUT_DIR / f"residual_diagnostics_{prov.replace(' ', '_')}.png", dpi=120)
            plt.close("all")
        except Exception as exc:  # diagnostics are a nice-to-have
            print("Residual diagnostics plot skipped:", exc)

        plot_forecast(prices[prov], fc, f"Ginger farmgate price - {prov} - forecast {TARGET_YEAR}",
                      OUT_DIR / f"forecast_{prov.replace(' ', '_')}.png")

    # ---- 6. composite + outputs --------------------------------------------
    comp = composite_forecast(fc_by_prov, pd.DataFrame(vol_fc))

    # history of the composite (for the plot only): volume-weighted over the provinces
    # that have a price in that month
    w_hist = pd.DataFrame({p: monthly_volume(vol_q[p], prices.index, prices.index[-1]) for p in provinces})
    avail = prices[provinces].notna()
    comp_hist = (prices[provinces].fillna(0) * w_hist).sum(axis=1) / w_hist.where(avail).sum(axis=1)
    plot_forecast(comp_hist, comp, f"Ginger farmgate price - volume-weighted composite - forecast {TARGET_YEAR}",
                  OUT_DIR / "forecast_composite.png")

    tables = {**fc_by_prov, "Composite (volume-weighted)": comp}
    final = pd.concat({k: v.round(2) for k, v in tables.items()}, names=["series", "month"]).reset_index()
    final.to_csv(OUT_DIR / f"ginger_price_forecast_{TARGET_YEAR}.csv", index=False)
    if cv_tables:
        pd.concat(cv_tables).to_csv(OUT_DIR / "cv_scores_all.csv")

    print("\n" + "=" * 78 + f"\nFORECAST {TARGET_YEAR}\n" + "=" * 78)
    for name, tbl in tables.items():
        print(f"\n{name}" + (f"   [{champions[name].label}]" if name in champions else ""))
        print(tbl.round(2).to_string())
    print(f"\nAll outputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()