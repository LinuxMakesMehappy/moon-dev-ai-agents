import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone


TICKER_MAP = {
    # output_symbol: yfinance_ticker
    "BTC-USD": "BTC-USD",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "XAUUSD=X",
}


def fetch_yf(symbol: str, interval: str = "5m", period: str = "60d", start: str = None, end: str = None) -> pd.DataFrame:
    y_tkr = TICKER_MAP.get(symbol, symbol)
    tkr = yf.Ticker(y_tkr)
    if start and end:
        hist = tkr.history(interval=interval, start=start, end=end, auto_adjust=False)
    else:
        hist = tkr.history(interval=interval, period=period, auto_adjust=False)
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
    ap.add_argument("--timeframe", default="5m", help="yfinance intervals: 1m(7d max),2m,5m,15m,30m,60m,1h,...")
    ap.add_argument("--period", default="60d", help="yfinance period (e.g., 7d for 1m; 60d for 5m)")
    ap.add_argument("--start", default=None, help="UTC start YYYY-MM-DD (for 1m pagination)")
    ap.add_argument("--end", default=None, help="UTC end YYYY-MM-DD (exclusive)")
    args = ap.parse_args()

    tf = args.timeframe
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]

    for sym in syms:
        try:
            if tf == "1m" and args.start and args.end:
                # Paginate in 7-day windows due to Yahoo limits for 1m
                start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
                end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
                cur = start_dt
                frames = []
                while cur < end_dt:
                    win_end = min(cur + timedelta(days=7), end_dt)
                    df_part = fetch_yf(sym, interval=tf, start=cur.date().isoformat(), end=win_end.date().isoformat())
                    if not df_part.empty:
                        frames.append(df_part)
                    cur = win_end
                if not frames:
                    raise RuntimeError("No data returned in paginated 1m fetch")
                df = pd.concat(frames).sort_index().~drop_duplicates~(keep="last")
            else:
                df = fetch_yf(sym, interval=tf, period=args.period)
            out = save_csv(df, sym, tf)
            print(f"✅ Saved {sym} -> {out}")
        except Exception as e:
            print(f"❌ {sym}: {e}")


if __name__ == "__main__":
    main()


