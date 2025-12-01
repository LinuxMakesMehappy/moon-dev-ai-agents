import argparse
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

from src.skills.session_features import session_ranges, london_sweeps_asia
from src.skills.cross_asset import rolling_corr_beta
from src.skills.btc_events import label_btc_events


def load_csv(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Expected CSV at: src/data/session_eval/{symbol}-{timeframe}.csv
    Columns: time,open,high,low,close,volume (UTC or tz-aware)
    """
    p = Path("src/data/session_eval") / f"{symbol}-{timeframe}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing data file: {p}")
    df = pd.read_csv(p, parse_dates=['time'], index_col='time')
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    else:
        df = df.tz_convert('UTC')
    df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
    return df


def forward_return(df: pd.DataFrame, ts: pd.Timestamp, bars: int, sign: int) -> float:
    """
    Compute signed forward return from timestamp ts to ts+bars.
    sign = +1 for long, -1 for short.
    """
    if ts not in df.index:
        # Snap to next bar if ts falls between bars
        pos = df.index.searchsorted(ts)
        if pos >= len(df.index):
            return np.nan
        ts = df.index[pos]

    i = df.index.get_loc(ts)
    j = i + bars
    if j >= len(df):
        return np.nan

    p0 = df['close'].iloc[i]
    p1 = df['close'].iloc[j]
    return sign * ((p1 - p0) / p0)


def main():
    ap = argparse.ArgumentParser(description="BTC macro-gated session edge evaluation for FX/metals")
    ap.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,XAUUSD")
    ap.add_argument("--btc", default="BTC-USD")
    ap.add_argument("--timeframe", default="5m")
    ap.add_argument("--corr_days", type=float, default=3.0)
    ap.add_argument("--corr_min", type=float, default=0.30)
    ap.add_argument("--asia_q", type=float, default=0.40)
    ap.add_argument("--horiz_bars", default="12,48")  # 1h, 4h @ 5m
    args = ap.parse_args()

    tf = args.timeframe
    targets = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Load BTC and label events
    btc = load_csv(args.btc, tf)
    btc_events = label_btc_events(btc)
    btc_r = btc['close'].pct_change().rename("btc_r")

    results = []
    out_dir = Path("src/data/session_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify London sweep dates for BTC
    # btc_events is indexed by [date, session]; find dates where LDN sweep flags are set
    try:
        sweep_df = btc_events.copy()
        # Only keep London rows
        if isinstance(sweep_df.index, pd.MultiIndex):
            ldn_rows = sweep_df.index.get_level_values(1) == 'LDN'
            sweep_df = sweep_df[ldn_rows]
        # Dates where any sweep flag is 1
        sweep_dates = sweep_df[(sweep_df.get('LDN_sweep_ASIA_H', 0) == 1) | (sweep_df.get('LDN_sweep_ASIA_L', 0) == 1)]
        sweep_dates_set = set(sweep_dates.index.get_level_values(0) if isinstance(sweep_df.index, pd.MultiIndex) else sweep_df.index.date)
    except Exception:
        sweep_dates_set = set()

    horizons = [int(x) for x in args.horiz_bars.split(",")]
    corr_window = int(args.corr_days * 24 * (60 / 5))  # 5m bars

    for sym in targets:
        df = load_csv(sym, tf)
        df_r = df['close'].pct_change().rename("tgt_r")

        # Rolling corr/beta on aligned indices
        corr, beta = rolling_corr_beta(df_r, btc_r, corr_window)

        # Asian width per date for the target
        sr_tgt = session_ranges(df)  # [date, session]
        # Extract Asian width per date
        try:
            asia_w = sr_tgt.xs('ASIA', level=1)['W'].rename('asia_W')
        except Exception:
            # If no ASIA rows found, skip this symbol
            continue

        # Daily Asia width series
        daily = asia_w.reset_index()
        daily.columns = ['date', 'asia_W']
        daily = daily.sort_values('date')
        # Rolling quantile threshold over last 60 trading days (min 20)
        daily['asia_q_thr'] = daily['asia_W'].rolling(60, min_periods=20).quantile(args.asia_q)
        q_map = daily.set_index('date')['asia_q_thr']

        # Build candidate event timestamps from BTC: use London sweep dates
        if not sweep_dates_set:
            continue

        for d in sorted(sweep_dates_set):
            # Gate by correlation at London session open timestamp proxy:
            # Use first London bar timestamp in target data for that date; approximate at 07:00 UTC
            ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=7, minute=0, tz='UTC')
            if ts not in corr.index:
                # Snap to nearest bar
                pos = corr.index.searchsorted(ts)
                if pos >= len(corr.index):
                    continue
                ts = corr.index[pos]

            if pd.isna(corr.loc[ts]) or corr.loc[ts] < args.corr_min:
                continue

            # Asia width filter for that date
            if d not in q_map.index:
                continue
            thr = q_map.loc[d]
            # Last Asia width for that date
            try:
                asiaW = float(asia_w.loc[(d,)])
            except Exception:
                # Some dates may not have ASIA session (holidays)
                continue
            if pd.isna(thr) or pd.isna(asiaW) or not (asiaW <= thr):
                continue

            # Direction from BTC: if LDN_sweep_ASIA_H=1 → fade short; if LDN_sweep_ASIA_L=1 → fade long
            try:
                row = btc_events.loc[(d, 'LDN')]
                sweep_hi = int(row['LDN_sweep_ASIA_H'] > 0) if 'LDN_sweep_ASIA_H' in row else 0
                sweep_lo = int(row['LDN_sweep_ASIA_L'] > 0) if 'LDN_sweep_ASIA_L' in row else 0
            except Exception:
                # Fallback: if MultiIndex missing, skip date
                continue
            if not (sweep_hi or sweep_lo):
                continue

            sign = -1 if sweep_hi else +1

            for hb in horizons:
                fr = forward_return(df, ts, hb, sign)
                if pd.isna(fr):
                    continue
                results.append({
                    'symbol': sym,
                    'date': d.isoformat(),
                    'horizon_bars': hb,
                    'fade_return': fr,
                    'corr': float(corr.loc[ts]) if not pd.isna(corr.loc[ts]) else np.nan,
                    'beta': float(beta.loc[ts]) if ts in beta.index and not pd.isna(beta.loc[ts]) else np.nan,
                    'asiaW': float(asiaW),
                    'asiaW_thr': float(thr),
                    'sweep_hi': sweep_hi,
                    'sweep_lo': sweep_lo
                })

    if not results:
        print("No gated events matched criteria. Check data coverage, timezone alignment, and thresholds.")
        return

    out = pd.DataFrame(results)
    fn = out_dir / "btc_macro_fx_eval.csv"
    out.to_csv(fn, index=False)

    # Summary table
    summ = out.groupby(['symbol', 'horizon_bars']).agg(
        n=('fade_return', 'count'),
        winrate=('fade_return', lambda s: float(np.mean(s > 0))),
        ev=('fade_return', 'mean'),
        ev_std=('fade_return', 'std')
    ).reset_index()
    print(summ.to_string(index=False))
    print(f"\nSaved detailed results: {fn}")


if __name__ == "__main__":
    main()


