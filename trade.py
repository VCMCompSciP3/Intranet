"""Step 3: use the trained model to place PAPER trades on Alpaca."""
import os
import time
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv
load_dotenv()

SYMBOLS = ["NVDA", "AMD", "TSM", "MSFT", "GOOGL", "PLTR", "TSLA", "ENPH",
           "FSLR", "ISRG", "CRSP", "ILMN", "TDOC", "RKLB", "IONQ"]
DOLLARS_PER_TRADE = 100          # size of each buy
THRESHOLD = 0.0005               # buy only if predicted 30-min return > 0.05%
MAX_POSITIONS = 5
MAX_RUN_MINUTES = int(os.getenv("MAX_RUN_MINUTES", "60"))
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"   # set DRY_RUN=0 to actually send orders
LAGS = (1, 3, 6, 12, 36)

KEY = os.environ["ALPACA_API_KEY"]
SECRET = os.environ["ALPACA_SECRET_KEY"]
trading = TradingClient(KEY, SECRET, paper=True)   # paper=True: fake money
data = StockHistoricalDataClient(KEY, SECRET)

artifact = joblib.load("linreg_model.joblib")
model, FEATURES = artifact["model"], artifact["features"]


def get_bars():
    now = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=SYMBOLS,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=now - timedelta(days=7),
        feed=DataFeed.IEX,   # free plan feed
    )
    df = data.get_stock_bars(req).df.reset_index()
    df = df.rename(columns={"timestamp": "ts"})
    # drop the bar still in progress
    df = df[df["ts"] + timedelta(minutes=5) <= now]
    return df


def build_features(df):
    """Same formulas as train_model.py, minus the target."""
    df = df.sort_values(["symbol", "ts"]).copy()
    df["ts"] = df["ts"].dt.tz_convert("America/New_York")
    mins = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[(mins >= 570) & (mins < 960)].copy()
    df["mins_since_open"] = mins.loc[df.index] - 570
    df["date"] = df["ts"].dt.date
    df = df.reset_index(drop=True)

    keys = ["symbol", "date"]
    df["logc"] = np.log(df["close"])
    g = df.groupby(keys)["logc"]
    for k in LAGS:
        df[f"ret_{k}"] = df["logc"] - g.shift(k)
    df["ret_1_tmp"] = df["logc"] - g.shift(1)
    df["vol_12"] = df.groupby(keys)["ret_1_tmp"].transform(
        lambda s: s.rolling(12).std())
    df["volume_ratio"] = df["volume"] / df.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(78, min_periods=20).mean())
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    return df.replace([np.inf, -np.inf], np.nan)


def run_once():
    df = build_features(get_bars())
    latest = df.groupby("symbol").tail(1).dropna(subset=FEATURES).copy()
    # skip symbols whose newest bar is stale (IEX has gaps in thin stocks)
    cutoff = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(minutes=15)
    latest = latest[latest["ts"] >= cutoff]
    if latest.empty:
        print("No fresh features yet (the model needs ~3 hours of the day's bars).")
        return
    latest["pred"] = model.predict(latest[FEATURES])

    held = {p.symbol for p in trading.get_all_positions()}
    for _, r in latest.sort_values("pred", ascending=False).iterrows():
        sym, pred = r["symbol"], r["pred"]
        print(f"{sym:6s} pred={pred:+.5f} held={sym in held}")
        if sym in held and pred <= 0:
            print(f"  -> SELL {sym}")
            if not DRY_RUN:
                trading.close_position(sym)
        elif sym not in held and pred > THRESHOLD and len(held) < MAX_POSITIONS:
            print(f"  -> BUY ${DOLLARS_PER_TRADE} {sym}")
            if not DRY_RUN:
                trading.submit_order(MarketOrderRequest(
                    symbol=sym, notional=DOLLARS_PER_TRADE,
                    side=OrderSide.BUY, time_in_force=TimeInForce.DAY))
            held.add(sym)


def main():
    print(f"DRY_RUN={DRY_RUN} (paper account)")
    deadline = time.time() + MAX_RUN_MINUTES * 60
    while time.time() < deadline:
        clock = trading.get_clock()
        if not clock.is_open:
            print("Market is closed. Exiting.")
            return
        # flatten 15 minutes before the close
        if clock.next_close - clock.timestamp <= timedelta(minutes=15):
            print("Near close: closing all positions.")
            if not DRY_RUN:
                trading.close_all_positions(cancel_orders=True)
            return
        run_once()
        time.sleep(300)   # one pass per 5-minute bar


if __name__ == "__main__":
    main()