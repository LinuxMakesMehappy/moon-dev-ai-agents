import pandas as pd
import numpy as np
from typing import Optional
from src.skills.session_features import london_sweeps_asia, label_sessions


def label_btc_events(btc_df: pd.DataFrame, vol_window_bars: int = 96, vol_k: float = 2.5) -> pd.DataFrame:
    """
    Label BTC session events and volatility spikes.
    Returns a time-indexed DataFrame with columns:
      - VOL_SPIKE: |ret| > vol_k * rolling_std over vol_window_bars
      - LDN_sweep_ASIA_H: 1 at first London bar of dates where London swept Asian high, else 0
      - LDN_sweep_ASIA_L: 1 at first London bar of dates where London swept Asian low, else 0
    Assumes btc_df has ['close'] and datetime index (tz-aware or naive UTC).
    """
    if btc_df.index.tz is None:
        btc_df = btc_df.tz_localize('UTC')
    else:
        btc_df = btc_df.tz_convert('UTC')

    # Compute VOL_SPIKE on time index
    returns = btc_df['close'].pct_change()
    rolling_vol = returns.rolling(vol_window_bars).std()
    vol_spike = (returns.abs() > (vol_k * rolling_vol)).astype(int).rename("VOL_SPIKE")

    # Compute London sweep flags per date/session using session_features,
    # then align them to the first London bar timestamp for each date in btc_df.
    sr = london_sweeps_asia(btc_df)  # MultiIndex [date, session]
    # Extract dates where sweep happened
    dates_hi = set(sr[sr['LDN_sweep_ASIA_H'] == 1].index.get_level_values(0)) if 'LDN_sweep_ASIA_H' in sr.columns else set()
    dates_lo = set(sr[sr['LDN_sweep_ASIA_L'] == 1].index.get_level_values(0)) if 'LDN_sweep_ASIA_L' in sr.columns else set()

    # Find first London bar per date in btc_df using labeled sessions
    dfl = label_sessions(btc_df)
    dfl = dfl[['close', 'session']].copy()
    # Group by date and pick first index where session == 'LDN'
    ldn_first_index = {}
    for date_val, group in dfl.groupby(dfl.index.date):
        g_ldn = group[group['session'] == 'LDN']
        if not g_ldn.empty:
            ldn_first_index[date_val] = g_ldn.index[0]

    # Build time-indexed sweep series
    sweep_hi = pd.Series(0, index=btc_df.index, dtype=int)
    sweep_lo = pd.Series(0, index=btc_df.index, dtype=int)
    for d, ts in ldn_first_index.items():
        if d in dates_hi:
            sweep_hi.loc[ts] = 1
        if d in dates_lo:
            sweep_lo.loc[ts] = 1
    sweep_hi = sweep_hi.rename('LDN_sweep_ASIA_H')
    sweep_lo = sweep_lo.rename('LDN_sweep_ASIA_L')

    out = pd.concat([vol_spike, sweep_hi, sweep_lo], axis=1).fillna(0)
    return out


