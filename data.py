"""Market data: Binance public klines (no API key) with CSV cache, plus a synthetic fallback."""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
BINANCE = "https://data-api.binance.vision/api/v3/klines"


def fetch_binance(symbol: str, interval: str = "1d", start: str = "2019-01-01") -> pd.DataFrame:
    """Download full OHLCV history for a Binance spot symbol, e.g. 'BTCUSDT'."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows = []
    while True:
        url = f"{BINANCE}?symbol={symbol}&interval={interval}&startTime={start_ms}&limit=1000"
        with urllib.request.urlopen(url, timeout=20) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume", *range(6)])
    df = df[["t", "open", "high", "low", "close", "volume"]].astype(float)
    df.index = pd.to_datetime(df.pop("t"), unit="ms")
    return df.iloc[:-1]  # drop the still-open candle


def synthetic(symbol: str, days: int = 2200, seed: int | None = None) -> pd.DataFrame:
    """Regime-switching fat-tailed price series (bull / bear / chop) for offline testing."""
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % 2**32)
    regimes = {0: (0.0025, 0.035), 1: (-0.002, 0.045), 2: (0.0, 0.025)}
    state, rets, vols = 2, [], []
    for _ in range(days):
        if rng.random() < 1 / 60:
            state = int(rng.choice([s for s in regimes if s != state]))
        mu, sig = regimes[state]
        rets.append(mu + sig * rng.standard_t(4) / np.sqrt(2))
        vols.append(sig)
    close = 100 * np.exp(np.cumsum(rets))
    vols = np.array(vols)
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, vols, days)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(10, 0.4, days) * (1 + 20 * np.abs(rets))
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def load(symbol: str, interval: str = "1d", refresh: bool = False, offline: bool = False) -> tuple[pd.DataFrame, str]:
    """Return (ohlcv, source). Uses cache, then Binance, then synthetic data."""
    if offline:
        return synthetic(symbol), "synthetic"
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol}_{interval}.csv")
    if os.path.exists(path) and not refresh and time.time() - os.path.getmtime(path) < 6 * 3600:
        return pd.read_csv(path, index_col=0, parse_dates=True), "binance(cache)"
    try:
        df = fetch_binance(symbol, interval)
        df.to_csv(path)
        return df, "binance"
    except Exception as e:  # network blocked, symbol missing, etc.
        print(f"  ! {symbol}: Binance unavailable ({e.__class__.__name__}), using synthetic data")
        return synthetic(symbol), "synthetic"
