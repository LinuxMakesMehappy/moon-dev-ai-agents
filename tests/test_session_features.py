import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.skills.session_features import label_sessions, session_ranges, london_sweeps_asia


def make_series(start: datetime, minutes: int):
    idx = pd.date_range(start=start, periods=minutes, freq="T", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["open"] = 100.0
    df["high"] = 100.0
    df["low"] = 100.0
    df["close"] = 100.0
    df["volume"] = 1.0
    return df


def test_label_sessions_basic():
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    df = make_series(start, minutes=60)
    out = label_sessions(df)
    assert "session" in out.columns
    assert set(out["session"].unique()) <= {"ASIA", "LDN", "NY", "OFF"}


def test_london_sweep_of_asian_high():
    # Build one day with ASIA then LDN
    d = datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)
    # Asia 00:00-06:59 => 420 minutes
    asia = make_series(d, minutes=420)
    asia.loc[:, "low"] = 99.0
    asia.loc[:, "high"] = 101.0
    asia.loc[:, "close"] = 100.5
    # London 07:00-12:59 => 360 minutes
    ldn = make_series(d.replace(hour=7), minutes=360)
    # Sweep: make high > 101 and close <= 101
    ldn.loc[:, "high"] = 102.0
    ldn.loc[:, "low"] = 98.0
    ldn.loc[:, "close"] = 101.0
    df = pd.concat([asia, ldn]).sort_index()

    sr = london_sweeps_asia(df)
    # Find the London row for that date
    # MultiIndex level 1 'LDN'
    assert ("LDN" in sr.index.get_level_values(1))
    ldn_rows = sr.xs("LDN", level=1)
    # At least one LDN row with sweep flag
    assert int(ldn_rows["LDN_sweep_ASIA_H"].max()) == 1


