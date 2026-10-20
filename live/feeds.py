"""Market data feeds for the live engine.

AlpacaFeed reads real-time and historical bars from Alpaca's market data API (crypto + IEX stocks).
SyntheticFeed produces the same shapes offline, for testing.

All bar frames are indexed by the UTC time the bar CLOSED (naive timestamps), columns open/high/low/close/volume.
"""
import os
import time
import zlib

import numpy as np
import pandas as pd
import requests

DATA_URL = "https://data.alpaca.markets"
STOCK_OPEN, STOCK_CLOSE = pd.Timedelta("09:30:00"), pd.Timedelta("16:00:00")


def alpaca_symbol(ticker):
    return ticker.replace("-USD", "/USD")


def now_utc():
    return pd.Timestamp.now("UTC").tz_localize(None)


class AlpacaFeed:
    def __init__(self, universe, key=None, secret=None):
        self.universe = universe
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": key or os.environ["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": secret or os.environ["ALPACA_SECRET_KEY"],
        })
        self.crypto = [t for t, k in universe.items() if k == "crypto"]
        self.stocks = [t for t, k in universe.items() if k == "stock"]

    def _get(self, path, params):
        for attempt in range(5):
            r = self.session.get(DATA_URL + path, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def _bars(self, path, symbols, timeframe, start, extra):
        out = {s: [] for s in symbols}
        params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": f"{start:%Y-%m-%dT%H:%M:%SZ}",
                  "limit": 10000, "sort": "asc", **extra}
        while True:
            js = self._get(path, params)
            for sym, rows in (js.get("bars") or {}).items():
                out[sym].extend(rows)
            if not js.get("next_page_token"):
                return out
            params["page_token"] = js["next_page_token"]

    def bars(self, minutes, start):
        """Bars of `minutes` size since `start` for the whole universe, keyed by our ticker."""
        tf = f"{minutes}Min" if minutes < 60 else f"{minutes // 60}Hour"
        raw = {}
        if self.crypto:
            got = self._bars("/v1beta3/crypto/us/bars", [alpaca_symbol(t) for t in self.crypto], tf, start, {})
            raw.update({t: got[alpaca_symbol(t)] for t in self.crypto})
        if self.stocks:
            got = self._bars("/v2/stocks/bars", self.stocks, tf, start, {"feed": "iex", "adjustment": "all"})
            raw.update(got)
        return {t: self._frame(rows, minutes, self.universe[t] == "stock") for t, rows in raw.items()}

    @staticmethod
    def _frame(rows, minutes, stock):
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)
        df = pd.DataFrame(rows)
        start = pd.to_datetime(df["t"], utc=True)
        df = pd.DataFrame({"open": df["o"], "high": df["h"], "low": df["l"], "close": df["c"], "volume": df["v"]})
        df = df.astype(float)
        if stock:  # regular session only (an hourly bar starting 09:00 still carries the 09:30 open)
            ny = start.dt.tz_convert("America/New_York")
            tod = ny - ny.dt.normalize()
            first = STOCK_OPEN - pd.Timedelta(minutes=minutes) + pd.Timedelta(minutes=1)
            keep = ((tod >= first) & (tod < STOCK_CLOSE)).to_numpy()
            df, start = df[keep], start[keep]
        df.index = pd.DatetimeIndex(start.dt.tz_localize(None) + pd.Timedelta(minutes=minutes))
        return df[df.index <= now_utc()]

    def latest_prices(self):
        prices = {}
        if self.crypto:
            js = self._get("/v1beta3/crypto/us/latest/trades", {"symbols": ",".join(alpaca_symbol(t) for t in self.crypto)})
            for t in self.crypto:
                trade = (js.get("trades") or {}).get(alpaca_symbol(t))
                if trade:
                    prices[t] = float(trade["p"])
        if self.stocks:
            js = self._get("/v2/stocks/trades/latest", {"symbols": ",".join(self.stocks), "feed": "iex"})
            for t in self.stocks:
                trade = (js.get("trades") or {}).get(t)
                if trade:
                    prices[t] = float(trade["p"])
        return prices


class SyntheticFeed:
    """Deterministic random-walk market with the same interface, for offline tests."""

    def __init__(self, universe, seed=7):
        self.universe = universe
        self.seed = seed
        self._cache = {}

    def _series(self, ticker, minutes):
        if (ticker, minutes) in self._cache:
            return self._cache[(ticker, minutes)]
        end = now_utc().floor(f"{minutes}min")
        days = 900 if minutes >= 60 else 40
        idx = pd.date_range(end=end, periods=days * 1440 // minutes, freq=f"{minutes}min")
        rng = np.random.default_rng(zlib.crc32(f"{ticker}{minutes}{self.seed}".encode()))
        per_day = 1440 / minutes
        drift = np.repeat(rng.choice([0.002, -0.0015, 0.0], size=len(idx) // int(per_day * 30) + 1), int(per_day * 30))
        rets = drift[:len(idx)] / per_day + 0.03 / np.sqrt(per_day) * rng.standard_t(4, len(idx)) / np.sqrt(2)
        close = 100 * np.exp(np.cumsum(rets))
        if minutes < 60:  # join the fine series onto the hourly history so prices line up
            hourly = self._series(ticker, 60)
            close *= hourly.close.asof(idx[0]) / close[0] if len(hourly) else 1.0
        open_ = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.normal(0, 0.01 / np.sqrt(per_day), len(idx))) * close
        df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                           "low": np.minimum(open_, close) - spread, "close": close,
                           "volume": rng.lognormal(8, 0.4, len(idx))}, index=idx)
        if self.universe[ticker] == "stock":
            ny = (idx - pd.Timedelta(minutes=minutes)).tz_localize("UTC").tz_convert("America/New_York")
            tod = ny - ny.normalize()
            df = df[(ny.dayofweek < 5) & (tod >= STOCK_OPEN - pd.Timedelta(minutes=minutes - 1)) & (tod < STOCK_CLOSE)]
        self._cache[(ticker, minutes)] = df
        return df

    def bars(self, minutes, start):
        return {t: self._series(t, minutes)[lambda d: (d.index > start) & (d.index <= now_utc())]
                for t in self.universe}

    def latest_prices(self):
        rng = np.random.default_rng()
        return {t: float(self._series(t, 5).close.iloc[-1] * (1 + rng.normal(0, 0.002))) for t in self.universe}
