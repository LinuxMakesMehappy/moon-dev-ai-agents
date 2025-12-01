import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd

BINANCE_SPOT = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"  # Closest to BTC-USD; saved as BTC-USD in output
MAX_LIMIT = 1000


def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(start_ms: int, end_ms: int, interval: str = "5m"):
    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "limit": MAX_LIMIT,
        "startTime": start_ms,
        "endTime": end_ms
    }
    r = requests.get(BINANCE_SPOT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def paginate(start: datetime, end: datetime, interval: str = "5m") -> pd.DataFrame:
    """
    Page through Binance klines without API key.
    """
    out = []
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    last_ts = start_ms
    while last_ts < end_ms:
        data = fetch_klines(last_ts, end_ms, interval)
        if not data:
            break
        out.extend(data)
        # Next page starts after last candle open time + one interval
        last_open = data[-1][0]
        # add 1 ms to avoid repeat
        last_ts = last_open + 1
        # avoid rate limits
        time.sleep(0.2)
    if not out:
        return pd.DataFrame()
    cols = ["open_time","open","high","low","close","volume","close_time","qvol","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(out, columns=cols)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("time")
    df = df[["open","high","low","close","volume"]].astype(float).sort_index()
    return df


def save_csv(df: pd.DataFrame, timeframe: str):
    out_dir = Path("src/data/session_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"BTC-USD-{timeframe}.csv"
    df.reset_index().to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Download BTC 5m multi-month data from Binance (no API key)")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--start", required=True, help="UTC start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="UTC end date YYYY-MM-DD (exclusive)")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    df = paginate(start, end, interval=args.timeframe)
    if df.empty:
        print("No data returned. Check dates.")
        return
    out = save_csv(df, args.timeframe)
    print(f"✅ Saved {len(df)} candles to {out}")


if __name__ == "__main__":
    main()


