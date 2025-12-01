import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.skills.btc_events import label_btc_events


def test_btc_vol_spike_label():
    # Build simple series across a day with a spike
    idx = pd.date_range(datetime(2025, 1, 3, tzinfo=timezone.utc), periods=600, freq="T")
    close = pd.Series(100.0, index=idx)
    close.iloc[300] = 110.0  # spike
    df = pd.DataFrame({
        "open": close.shift(1).fillna(100.0),
        "high": close,
        "low": close,
        "close": close,
        "volume": 1.0
    }, index=idx)
    out = label_btc_events(df, vol_window_bars=30, vol_k=2.0)
    assert "VOL_SPIKE" in out.columns
    # Some spike row exists
    assert out["VOL_SPIKE"].sum() >= 1


