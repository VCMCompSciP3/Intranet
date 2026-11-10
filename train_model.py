"""Step 2: linear regression on the local database.

Predicts the forward return over the next HORIZON bars (6 bars = 30 minutes)
from recent returns, volatility, volume, and time of day.

Install: pip install pandas numpy scikit-learn joblib
"""
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOCAL_DB = "my_stocks_5m.sqlite"
HORIZON = 6            # predict return over the next 6 five-minute bars
TEST_FRACTION = 0.2    # most recent 20% of days held out for testing
LAGS = (1, 3, 6, 12, 36)

FEATURES = [f"ret_{k}" for k in LAGS] + [
    "vol_12", "volume_ratio", "range_pct", "mins_since_open",
]


def load():
    conn = sqlite3.connect(LOCAL_DB)
    df = pd.read_sql("SELECT * FROM bars_5m ORDER BY symbol, ts_utc", conn)
    conn.close()

    ts = df["ts_utc"]
    if pd.api.types.is_numeric_dtype(ts):
        # Epoch seconds (use unit="ms" if values are 13 digits)
        df["ts"] = pd.to_datetime(ts, unit="s", utc=True)
    else:
        df["ts"] = pd.to_datetime(ts, utc=True)
    df["ts"] = df["ts"].dt.tz_convert("America/New_York")

    # Keep regular market hours only (9:30 to 16:00 ET)
    mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[(mins >= 9 * 60 + 30) & (mins < 16 * 60)].copy()
    df["mins_since_open"] = mins[df.index] - (9 * 60 + 30)
    df["date"] = df["ts"].dt.date
    return df.reset_index(drop=True)


def add_features(df):
    keys = ["symbol", "date"]  # never let features cross overnight gaps
    df["logc"] = np.log(df["close"])
    g = df.groupby(keys)["logc"]

    for k in LAGS:
        df[f"ret_{k}"] = df["logc"] - g.shift(k)
    df["target"] = g.shift(-HORIZON) - df["logc"]

    df["ret_1_tmp"] = df["logc"] - g.shift(1)
    df["vol_12"] = df.groupby(keys)["ret_1_tmp"].transform(
        lambda s: s.rolling(12).std())
    df["volume_ratio"] = df["volume"] / df.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(78, min_periods=20).mean())
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=FEATURES + ["target"])


def main():
    df = add_features(load())
    print(f"{len(df)} usable rows, {df['symbol'].nunique()} symbols")

    # Chronological split by day: train on the past, test on the future
    days = np.sort(df["date"].unique())
    cutoff = days[int(len(days) * (1 - TEST_FRACTION))]
    train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff]
    print(f"Train: {len(train)} rows before {cutoff} | Test: {len(test)} rows")

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(train[FEATURES], train["target"])
    pred = model.predict(test[FEATURES])
    y = test["target"].to_numpy()

    print("\n=== Test results (data the model never saw) ===")
    print(f"R^2:                 {r2_score(y, pred):.5f}")
    print(f"MAE:                 {mean_absolute_error(y, pred):.6f}")
    print(f"MAE predicting 0:    {mean_absolute_error(y, np.zeros_like(y)):.6f}")
    print(f"Direction accuracy:  {(np.sign(pred) == np.sign(y)).mean():.3%}")
    print(f"Always-up baseline:  {(y > 0).mean():.3%}")

    print("\n=== Coefficients (standardized features) ===")
    coefs = model.named_steps["linearregression"].coef_
    for name, c in sorted(zip(FEATURES, coefs), key=lambda t: -abs(t[1])):
        print(f"{name:16s} {c:+.6f}")

    print("\n=== Direction accuracy by symbol (test period) ===")
    test = test.assign(pred=pred)
    for sym, grp in test.groupby("symbol"):
        acc = (np.sign(grp["pred"]) == np.sign(grp["target"])).mean()
        print(f"{sym:6s} {acc:.3%}  (n={len(grp)})")

    joblib.dump({"model": model, "features": FEATURES, "horizon": HORIZON},
                "linreg_model.joblib")
    print("\nSaved linreg_model.joblib")
    print("Reminder: R^2 near 0 is normal for 5-minute returns. Real trading "
          "also pays spreads and fees, which can wipe out a tiny edge.")


if __name__ == "__main__":
    main()