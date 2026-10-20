"""Market data: Binance public klines (no API key) with CSV cache, plus a synthetic fallback."""
import json
import os
import time
import urllib.request
import zlib

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


def synthetic(symbol: str, days: int = 2200, seed: int | None = None, freq: str = "D") -> pd.DataFrame:
    """Regime-switching fat-tailed price series (bull / bear / chop) for offline testing."""
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(symbol.encode()))
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
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq=freq)
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


def fetch_yahoo(ticker: str, start: str = "2022-01-01") -> pd.DataFrame:
    """Daily OHLCV for stocks/ETFs/crypto from Yahoo Finance, e.g. 'NVDA' or 'BTC-USD'."""
    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if df.empty:
        raise ValueError(f"no data for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    df.index = df.index.tz_localize(None).normalize()
    if ticker.endswith("-USD"):  # crypto: today's candle is still open
        df = df[df.index < pd.Timestamp.now("UTC").tz_localize(None).normalize()]
    return df


def _hourly_bars(df: pd.DataFrame, session_close: str | None) -> pd.DataFrame:
    """Re-index raw hourly bars (labelled by start time) to the UTC hour at which each bar has closed."""
    end = df.index + pd.Timedelta(hours=1)
    if session_close:  # the last stock bar of the day is short (15:30-16:00 New York)
        local = end.tz_convert("America/New_York")
        cap = local.normalize() + pd.Timedelta(session_close)
        end = local.where(local <= cap, cap).tz_convert("UTC")
    df = df.copy()
    df.index = end.tz_convert("UTC").ceil("h").tz_localize(None)
    df = df[df.index <= pd.Timestamp.now("UTC").tz_localize(None)]  # drop the bar that is still forming
    return df.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})


def fetch_yahoo_hourly(ticker: str, days: int = 720) -> pd.DataFrame:
    """Hourly OHLCV from Yahoo (max ~730 days), indexed by the UTC hour each bar closed."""
    import yfinance as yf
    start = (pd.Timestamp.now("UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    df = yf.Ticker(ticker).history(start=start, interval="1h", auto_adjust=True)
    if df.empty:
        raise ValueError(f"no hourly data for {ticker}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    return _hourly_bars(df, None if ticker.endswith("-USD") else "16:00:00")


def synthetic_hourly(symbol: str, days: int = 729, stock: bool = False) -> pd.DataFrame:
    """Synthetic hourly bars; stocks only trade 14:00-20:00 UTC on weekdays."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    end = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
    idx = pd.date_range(end=end, periods=days * 24, freq="h")
    if stock:
        idx = idx[(idx.dayofweek < 5) & (idx.hour >= 14) & (idx.hour <= 20)]
    per_day = 7 if stock else 24
    regimes = {0: (0.0025, 0.035), 1: (-0.002, 0.045), 2: (0.0, 0.025)}
    state, rets, vols = 2, [], []
    for _ in range(len(idx)):
        if rng.random() < 1 / (60 * per_day):
            state = int(rng.choice([s for s in regimes if s != state]))
        mu, sig = regimes[state]
        mu, sig = mu / per_day, sig / np.sqrt(per_day)
        rets.append(mu + sig * rng.standard_t(4) / np.sqrt(2))
        vols.append(sig)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, np.array(vols) / 2)) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close,
                         "volume": rng.lognormal(8, 0.4, len(idx))}, index=idx)


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate close-labelled hourly bars into 4h / daily bars labelled at their scheduled close,
    so a bar's signal only exists once the whole bar is finished."""
    if rule == "1h":
        return df
    key = df.index.floor(rule) if rule != "1D" else df.index.floor("D")
    # an hourly bar labelled exactly on a boundary belongs to the bar that ends there
    on_edge = df.index == key
    key = key.where(~on_edge, key - pd.Timedelta(rule if rule != "1D" else "1D"))
    out = df.groupby(key).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    out.index = out.index + pd.Timedelta(rule if rule != "1D" else "1D")
    return out
