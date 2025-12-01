import argparse
import subprocess
from pathlib import Path


def run(cmd: list):
    print(f"→ {' '.join(cmd)}")
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def ensure_btc(btc_tf: str, start: str, end: str):
    out = Path("src/data/session_eval") / f"BTC-USD-{btc_tf}.csv"
    if out.exists():
        print(f"BTC exists: {out}")
        return
    run(["python", "src/scripts/download_binance_btc.py", "--timeframe", btc_tf, "--start", start, "--end", end])


def ensure_fx_yf(symbols: list, tf: str, period: str):
    missing = []
    for s in symbols:
        p = Path("src/data/session_eval") / f"{s}-{tf}.csv"
        if not p.exists():
            missing.append(s)
    if not missing:
        print("FX/XAU found for all symbols (yfinance).")
        return
    run(["python", "src/scripts/download_free_data.py", "--symbols", ",".join(missing), "--timeframe", tf, "--period", period])


def ensure_fx_dukascopy(symbols: list, start: str, end: str):
    # Download 1m then resample to 5m
    need_dl = []
    for s in symbols:
        p5 = Path("src/data/session_eval") / f"{s}-5m.csv"
        if not p5.exists():
            need_dl.append(s)
    if not need_dl:
        print("FX/XAU 5m present (from prior runs).")
        return

    run(["python", "src/scripts/download_dukascopy_fx.py", "--symbols", ",".join(need_dl), "--start", start, "--end", end])
    for s in need_dl:
        run(["python", "src/scripts/resample_csv.py", "--symbol", s])


def run_eval(symbols: list, btc_symbol: str, timeframe: str, corr_days: float, corr_min: float, asia_q: float, horiz: str):
    cmd = [
        "python", "src/evals/btc_macro_eval.py",
        "--symbols", ",".join(symbols),
        "--btc", btc_symbol,
        "--timeframe", timeframe,
        "--corr_days", str(corr_days),
        "--corr_min", str(corr_min),
        "--asia_q", str(asia_q),
        "--horiz_bars", horiz
    ]
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description="End-to-end pipeline: download BTC/FX data and run BTC macro-gated eval")
    ap.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,XAUUSD")
    ap.add_argument("--use_dukascopy", action="store_true", help="Fetch FX/XAU from Dukascopy 1m and resample to 5m")
    ap.add_argument("--btc_start", required=True, help="BTC start YYYY-MM-DD (UTC)")
    ap.add_argument("--btc_end", required=True, help="BTC end YYYY-MM-DD (UTC)")
    ap.add_argument("--fx_start", default=None, help="FX start YYYY-MM-DD (UTC) for Dukascopy")
    ap.add_argument("--fx_end", default=None, help="FX end YYYY-MM-DD (UTC) for Dukascopy")
    ap.add_argument("--yf_period", default="60d", help="yfinance period (5m max 60d)")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--corr_days", type=float, default=3.0)
    ap.add_argument("--corr_min", type=float, default=0.30)
    ap.add_argument("--asia_q", type=float, default=0.40)
    ap.add_argument("--horiz_bars", default="12,48")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # 1) Ensure BTC
    ensure_btc(args.timeframe, args.btc_start, args.btc_end)

    # 2) Ensure FX/XAU
    if args.use_dukascopy:
        if not args.fx_start or not args.fx_end:
            raise SystemExit("fx_start and fx_end are required when --use_dukascopy is set")
        ensure_fx_dukascopy(symbols, args.fx_start, args.fx_end)
    else:
        ensure_fx_yf(symbols, args.timeframe, args.yf_period)

    # 3) Run evaluation
    run_eval(symbols, "BTC-USD", args.timeframe, args.corr_days, args.corr_min, args.asia_q, args.horiz_bars)


if __name__ == "__main__":
    main()


