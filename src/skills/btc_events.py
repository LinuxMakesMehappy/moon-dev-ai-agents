import pandas as pd
import numpy as np
from typing import Optional
from src.skills.session_features import london_sweeps_asia


def label_btc_events(btc_df: pd.DataFrame, vol_window_bars: int = 96, vol_k: float = 2.5) -> pd.DataFrame:
    """
    Label BTC session events and volatility spikes.
    - LDN_sweep_ASIA_H / LDN_sweep_ASIA_L from session features
    - VOL_SPIKE: |ret| > vol_k * rolling_std over vol_window_bars
    Assumes btc_df has ['close'] and datetime index (tz-aware or naive UTC).
    """
    if btc_df.index.tz is None:
        btc_df = btc_df.tz_localize('UTC')
    else:
        btc_df = btc_df.tz_convert('UTC')

    session_labels = london_sweeps_asia(btc_df)  # MultiIndex [date, session]

    returns = btc_df['close'].pct_change()
    rolling_vol = returns.rolling(vol_window_bars).std()
    vol_spike = (returns.abs() > (vol_k * rolling_vol)).astype(int).rename("VOL_SPIKE")

    # Join session labels to a time index by forward/backward mapping is non-trivial.
    # For event timing, we keep session_labels as-is (per-date, per-session) and provide VOL_SPIKE on time index.
    out = session_labels.copy()
    # Attach a representative timestamp per row: we'll use the last bar index within that session by reindex trick if needed downstream.
    return out.join(vol_spike.to_frame(), how='outer').fillna(0)


