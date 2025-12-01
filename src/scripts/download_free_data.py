import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf


TICKER_MAP = {
    # output_symbol: yfinance_ticker
    "BTC-USD": "BTC-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "XAUUSD=X",
}


def fetch_yf(symbol: str, interval: str = "5m", period: str = "60d") -> pd.DataFrame:
    y_tkr = TICKER_MAP.get(symbol, symbol)
    hist = yf.Ticker(y_tkr).history(interval=interval, period=period, auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"No data returned for {symbol} ({y_tkr})")
    # Standardize columns and index name
    df = hist.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[["open", "high", "low", "close", "volume"]]
    df.index.name = "time"
    # Ensure UTC tz-awareness
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def save_csv(df: pd.DataFrame, out_symbol: str, timeframe: str):
    out_dir = Path("src/data/session_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_symbol}-{timeframe}.csv"
    df.reset_index().to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Download free 5m data via yfinance for BTC/FX/XAU")
    ap.add_argument("--symbols", default="BTC-USD,EURUSD,GBPUSD,USDJPY,XAUUSD")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--period", default="60d", help="yfinance period (max 60d for 5m)")
    args = ap.parse_args()

    tf = args.timeframe
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]

    for sym in syms:
        try:
            df = fetch_yf(sym, interval=tf, period=args.period)
            out = save_csv(df, sym, tf)
            print(f"✅ Saved {sym} -> {out}")
        except Exception as e:
            print(f"❌ {sym}: {e}")


if __name__ == "__main__":
    main()


