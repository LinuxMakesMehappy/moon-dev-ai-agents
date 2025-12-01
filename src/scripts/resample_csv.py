import argparse
from pathlib import Path
import pandas as pd


def resample_1m_to_5m(in_path: Path, out_path: Path):
    df = pd.read_csv(in_path, parse_dates=['time'])
    df = df.set_index('time')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    o = df['open'].resample('5T').first()
    h = df['high'].resample('5T').max()
    l = df['low'].resample('5T').min()
    c = df['close'].resample('5T').last()
    v = df['volume'].resample('5T').sum()
    out = pd.concat([o.rename('open'), h.rename('high'), l.rename('low'), c.rename('close'), v.rename('volume')], axis=1)
    out = out.dropna(how='any')
    out.index.name = 'time'
    out.reset_index().to_csv(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Resample 1m OHLCV CSV to 5m")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--indir", default="src/data/session_eval")
    ap.add_argument("--outdir", default="src/data/session_eval")
    args = ap.parse_args()

    in_path = Path(args.indir) / f"{args.symbol}-1m.csv"
    out_path = Path(args.outdir) / f"{args.symbol}-5m.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = resample_1m_to_5m(in_path, out_path)
    print(f"✅ Resampled -> {p}")


if __name__ == "__main__":
    main()


