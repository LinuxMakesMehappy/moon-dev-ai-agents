import argparse
from pathlib import Path
from datetime import datetime, timezone
import sys

"""
Best-effort Dukascopy FX/XAU 1m downloader wrapper.
This tries multiple popular Python packages. If none are installed,
it prints clear install instructions without crashing the repo.

Supported symbols (examples): EURUSD, GBPUSD, USDJPY, XAUUSD
Output CSVs are saved to src/data/session_eval/{SYMBOL}-1m.csv
"""


def parse_args():
    ap = argparse.ArgumentParser(description="Download FX/XAU 1m candles from Dukascopy")
    ap.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,XAUUSD")
    ap.add_argument("--start", required=True, help="UTC start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="UTC end date YYYY-MM-DD (exclusive)")
    ap.add_argument("--outdir", default="src/data/session_eval")
    return ap.parse_args()


def try_duka_cli(symbols, start, end, outdir) -> bool:
    """
    Try using 'duka' package via its Python API if available, otherwise return False.
    The duka project primarily offers a CLI, so this path may not be available.
    """
    try:
        from duka.core import TimeFrame, get_data
        # duka API typically expects timezone-naive datetimes, but UTC date strings are fine
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        for sym in symbols:
            out_path = Path(outdir) / f"{sym}-1m.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # duka writes to directory; get_data may require symbol formatting like "EURUSD"
            get_data(sym, start_dt, end_dt, TimeFrame.MIN_1, outdir=str(out_path.parent), filename=out_path.name)
            print(f"✅ duka saved {sym} -> {out_path}")
        return True
    except Exception:
        return False


def try_dukascopy_pkg(symbols, start, end, outdir) -> bool:
    """
    Try using a 'dukascopy' style package.
    Multiple community packages exist; we attempt a couple of import conventions.
    """
    # Attempt 1: dukascopy (Downloader-like)
    try:
        from dukascopy import Downloader, Timeframe
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        for sym in symbols:
            out_path = Path(outdir) / f"{sym}-1m.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            dl = Downloader(sym, timeframe=Timeframe.M1, start=start_dt, end=end_dt)
            df = dl.download()  # Expect DataFrame with ohlc/volume
            if df is None or df.empty:
                print(f"⚠️ dukascopy: no data for {sym}")
                continue
            df = df.rename(columns={c: c.lower() for c in df.columns})
            # Ensure required columns
            needed = ["open","high","low","close","volume"]
            if not all(c in df.columns for c in needed):
                # Try common alternatives
                alt = {
                    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
                    "BidOpen": "open", "BidHigh": "high", "BidLow": "low", "BidClose": "close", "TickVolume": "volume",
                }
                df = df.rename(columns=alt)
            df = df[["open","high","low","close","volume"]]
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.index.name = "time"
            df.reset_index().to_csv(out_path, index=False)
            print(f"✅ dukascopy saved {sym} -> {out_path}")
        return True
    except Exception:
        pass

    # Attempt 2: dukascopy_data style
    try:
        from dukascopy_data import instruments as d_instruments
        from dukascopy_data import candles as d_candles
        start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
        for sym in symbols:
            out_path = Path(outdir) / f"{sym}-1m.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Some libs require instrument codes; assume sym string works
            df = d_candles.get(sym, start_dt, end_dt, timeframe="m1")
            if df is None or df.empty:
                print(f"⚠️ dukascopy_data: no data for {sym}")
                continue
            df = df.rename(columns={c: c.lower() for c in df.columns})
            needed = ["open","high","low","close","volume"]
            if not all(c in df.columns for c in needed):
                alt = {
                    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
                    "BidOpen": "open", "BidHigh": "high", "BidLow": "low", "BidClose": "close", "TickVolume": "volume",
                }
                df = df.rename(columns=alt)
            df = df[["open","high","low","close","volume"]]
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            df.index.name = "time"
            df.reset_index().to_csv(out_path, index=False)
            print(f"✅ dukascopy_data saved {sym} -> {out_path}")
        return True
    except Exception:
        return False


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    outdir = args.outdir

    # Try known options
    ok = try_dukascopy_pkg(symbols, args.start, args.end, outdir)
    if not ok:
        ok = try_duka_cli(symbols, args.start, args.end, outdir)

    if not ok:
        print("\n❌ No Dukascopy downloader package found.")
        print("Install one of the following and re-run:")
        print("  pip install dukascopy")
        print("  pip install dukascopy-data")
        print("  pip install duka")
        sys.exit(1)


if __name__ == "__main__":
    main()


