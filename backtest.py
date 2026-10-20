"""Vectorised backtester with fees, next-bar execution and standard metrics."""
import numpy as np
import pandas as pd

PERIODS = 365  # crypto trades every day


def run(df: pd.DataFrame, position: pd.Series, fee: float = 0.001, long_only: bool = True) -> pd.Series:
    """Return the strategy's per-bar return series. Signal at close t is held over bar t+1."""
    pos = position.reindex(df.index).fillna(0).clip(0 if long_only else -1, 1)
    held = pos.shift().fillna(0)
    ret = df.close.pct_change().fillna(0)
    cost = held.diff().abs().fillna(held.abs()) * fee
    return held * ret - cost


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / PERIODS
    sd = r.std()
    downside = r[r < 0].std()
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    return {
        "total_%": 100 * (eq.iloc[-1] - 1),
        "cagr_%": 100 * cagr,
        "sharpe": r.mean() / sd * np.sqrt(PERIODS) if sd > 0 else 0.0,
        "sortino": r.mean() / downside * np.sqrt(PERIODS) if downside > 0 else 0.0,
        "max_dd_%": 100 * dd,
        "calmar": cagr / abs(dd) if dd < 0 else 0.0,
        "exposure_%": 100 * (r != 0).mean(),
    }
