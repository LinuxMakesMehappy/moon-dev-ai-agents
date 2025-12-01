import os
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone


def write_csv(symbol: str, timeframe: str, df: pd.DataFrame):
    out_dir = Path("src/data/session_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{symbol}-{timeframe}.csv"
    df.reset_index().to_csv(p, index=False)
    return p


def make_synthetic_series(start: datetime, periods: int, freq: str = "5T", base: float = 100.0):
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    close = pd.Series(base, index=idx)
    df = pd.DataFrame({
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1.0
    }, index=idx)
    return df


def test_smoke_eval_runs():
    start = datetime(2025, 1, 5, tzinfo=timezone.utc)
    # Create 2 days of 5m bars
    periods = 2 * 24 * 12

    # BTC with a London sweep day: simulate slightly higher high during LDN and close at Asia high
    btc = make_synthetic_series(start, periods)
    # Day 1 Asia: 00:00-06:59, set highs/lows; Day 1 LDN: 07:00-12:59
    d1 = start
    asia_mask = (btc.index >= d1) & (btc.index < d1 + timedelta(hours=7))
    ldn_mask = (btc.index >= d1 + timedelta(hours=7)) & (btc.index < d1 + timedelta(hours=13))
    btc.loc[asia_mask, "high"] = 101.0
    btc.loc[asia_mask, "low"] = 99.0
    btc.loc[asia_mask, "close"] = 100.5
    btc.loc[ldn_mask, "high"] = 102.0
    btc.loc[ldn_mask, "low"] = 98.0
    btc.loc[ldn_mask, "close"] = 101.0  # close back inside Asia high
    write_csv("BTC-USD", "5m", btc)

    # EURUSD target with mild correlation: make returns proportional to BTC close changes
    eur = make_synthetic_series(start, periods)
    # Create small positive drift aligned with BTC LDN event
    eur.loc[ldn_mask, "close"] = eur.loc[ldn_mask, "close"] * 0.999  # small change
    write_csv("EURUSD", "5m", eur)

    # Run eval with small corr window and short horizons to avoid heavy computation
    cmd = [
        "python",
        "src/evals/btc_macro_eval.py",
        "--symbols", "EURUSD",
        "--btc", "BTC-USD",
        "--timeframe", "5m",
        "--corr_days", "0.05",
        "--corr_min", "0.0",
        "--asia_q", "1.0",
        "--horiz_bars", "1"
    ]
    res = subprocess.run(cmd, text=True, capture_output=True)
    # Allow success even if few events; we just want it to run without crashing
    assert res.returncode == 0, f"stderr: {res.stderr}"


