import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.skills.cross_asset import rolling_corr_beta


def test_rolling_corr_beta_linear_relation():
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=timezone.utc), periods=100, freq="T")
    drv = pd.Series(np.linspace(0, 1, len(idx)), index=idx).pct_change().fillna(0)
    tgt = 2.0 * drv  # perfect beta=2, corr=1
    corr, beta = rolling_corr_beta(tgt, drv, window=10)
    # Ignore initial NaNs
    c = corr.dropna().tail(10).mean()
    b = beta.dropna().tail(10).mean()
    assert c > 0.99
    assert 1.9 < b < 2.1


