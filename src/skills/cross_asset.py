import pandas as pd
import numpy as np
from typing import Tuple


def rolling_corr_beta(target_returns: pd.Series, driver_returns: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling correlation and beta of target vs driver.
    Inputs are return series indexed by timestamp, possibly with NaNs.
    """
    xy = pd.concat([target_returns.rename("tgt"), driver_returns.rename("drv")], axis=1).dropna()
    if xy.empty:
        idx = target_returns.index.union(driver_returns.index)
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)

    tgt = xy["tgt"]
    drv = xy["drv"]
    cov = tgt.rolling(window).cov(drv)
    var_drv = drv.rolling(window).var()
    beta = cov / var_drv.replace(0, np.nan)
    corr = tgt.rolling(window).corr(drv)

    # Reindex back to full union to simplify joins downstream
    idx = target_returns.index.union(driver_returns.index)
    return corr.reindex(idx), beta.reindex(idx)


