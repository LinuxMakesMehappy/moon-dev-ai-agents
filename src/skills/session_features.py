import pandas as pd
import numpy as np
from typing import Dict, Tuple

# UTC session hour windows (inclusive lower bound, exclusive upper bound)
SESSIONS: Dict[str, Tuple[int, int]] = {
    'ASIA': (0, 7),
    'LDN': (7, 13),
    'NY': (13, 22),
}


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        return df.tz_localize('UTC')
    return df.tz_convert('UTC')


def label_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'session' column labeling each bar as ASIA, LDN, NY, or OFF based on UTC hour.
    Expects df indexed by datetime, with any tz or naive (assumed UTC).
    """
    dfl = _ensure_utc_index(df.copy())
    hours = dfl.index.hour
    session_labels = np.full(len(dfl), 'OFF', dtype=object)
    for name, (start_h, end_h) in SESSIONS.items():
        session_labels[(hours >= start_h) & (hours < end_h)] = name
    dfl['session'] = session_labels
    return dfl


def session_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-day, per-session high/low/close and width.
    Requires columns: 'high', 'low', 'close'.
    Index must be datetime-like.
    """
    dfl = label_sessions(df)
    grp = dfl.groupby([dfl.index.normalize(), 'session'])
    out = grp['high'].max().to_frame('H').join(
        grp['low'].min().to_frame('L')
    ).join(
        grp['close'].last().to_frame('C')
    )
    out['W'] = out['H'] - out['L']
    return out


def london_sweeps_asia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes whether London session swept the prior Asian session's high/low and reclosed back inside.
    Returns a DataFrame with columns:
      - LDN_sweep_ASIA_H (1/0)
      - LDN_sweep_ASIA_L (1/0)
      - W (current session width)
    """
    sr = session_ranges(df)
    # Prior session per day: shift within the same date level
    sr['H_prev'] = sr.groupby(level=0)['H'].shift(1)
    sr['L_prev'] = sr.groupby(level=0)['L'].shift(1)
    idx_session = sr.index.get_level_values(1)

    # London swept Asian high if: London H > Asian H AND London C <= Asian H
    sweep_hi = (idx_session == 'LDN') & (sr['H'] > sr['H_prev']) & (sr['C'] <= sr['H_prev'])
    # London swept Asian low if: London L < Asian L AND London C >= Asian L
    sweep_lo = (idx_session == 'LDN') & (sr['L'] < sr['L_prev']) & (sr['C'] >= sr['L_prev'])

    sr['LDN_sweep_ASIA_H'] = sweep_hi.astype(int)
    sr['LDN_sweep_ASIA_L'] = sweep_lo.astype(int)
    return sr[['LDN_sweep_ASIA_H', 'LDN_sweep_ASIA_L', 'W']]


