from typing import List, Dict, Callable, Any
import pandas as pd
from src.skills.session_features import london_sweeps_asia


class SessionRangeAgent:
    """
    Produces session-context features per symbol:
      - Last Asian session width
      - Whether last London session swept prior Asian high/low
    The agent expects a data_loader callable: (symbol, timeframe, days_back) -> DataFrame
    with index as datetime and columns: open, high, low, close, volume.
    """

    def __init__(self, data_loader: Callable[[str, str, int], pd.DataFrame]):
        self.data_loader = data_loader

    def run(self, symbols: List[str], timeframe: str = '15m', days_back: int = 60) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        for symbol in symbols:
            df = self.data_loader(symbol, timeframe, days_back)
            sr = london_sweeps_asia(df)

            # Last known Asian width
            try:
                asia_last_width = float(sr.xs('ASIA', level=1)['W'].dropna().tail(1).values[0])
            except Exception:
                asia_last_width = None

            # London sweep flags (last available)
            ldn_sweep_hi = int(sr['LDN_sweep_ASIA_H'].dropna().tail(1).values[0]) if 'LDN_sweep_ASIA_H' in sr else 0
            ldn_sweep_lo = int(sr['LDN_sweep_ASIA_L'].dropna().tail(1).values[0]) if 'LDN_sweep_ASIA_L' in sr else 0

            context[symbol] = {
                'asia_last_width': asia_last_width,
                'ldn_swept_asia_high': ldn_sweep_hi,
                'ldn_swept_asia_low': ldn_sweep_lo,
            }

        return {'session_context': context}


def example_csv_loader(symbol: str, timeframe: str, days_back: int) -> pd.DataFrame:
    """
    Example local loader. Replace with your actual data pipeline (e.g., CoinGecko/BirdEye).
    Assumes a CSV at src/data/{symbol}_{timeframe}.csv with 'time' column.
    """
    path = f"src/data/{symbol.replace('/', '_')}_{timeframe}.csv"
    df = pd.read_csv(path, parse_dates=['time'], index_col='time')
    return df


