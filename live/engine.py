"""Live engine: runs the competition on 5-minute steps with Alpaca market data.

Every 5 minutes it pulls the newest bars, refreshes the signals of every tempo that just closed a bar,
lets the AIs learn when a week has passed, trades, and rebuilds the dashboard.
"""
import logging
import os
import threading

import numpy as np
import pandas as pd

import data
import learning
import paper
from strategies import AIS

from .feeds import now_utc
from .news import NEWS_AIS, news_signals

log = logging.getLogger("engine")

STEP_MINUTES = 5
LIVE_TFS = ("5m", "15m", "1h", "4h", "1D")
RULES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1D": "1D"}
FINE_DAYS = 35        # 5-minute history kept in memory
COARSE_DAYS = 900     # hourly history, so daily strategies have enough warm-up
WINDOW_DAYS = 30      # how far back the AIs look when they learn
MAX_BARS = 2000       # bars fed to a strategy per tempo (more than any warm-up needs)


class LiveEngine:
    def __init__(self, feed, folder, news=None):
        self.feed = feed
        self.news = news
        self.comp = paper.Competition(folder)
        self.fine, self.hourly = {}, {}
        self._sig_cache = {}   # tf -> (last bar labels, {ai: {ticker: Series}})
        self.payload, self.html = None, None
        self.updated = None
        self.lock = threading.Lock()

    # ---------- data ----------
    def bootstrap(self):
        now = now_utc()
        self.hourly = self.feed.bars(60, now - pd.Timedelta(days=COARSE_DAYS))
        self.fine = self.feed.bars(STEP_MINUTES, now - pd.Timedelta(days=FINE_DAYS))
        missing = [t for t, df in self.fine.items() if df.empty]
        if missing:
            raise RuntimeError(f"no market data for {missing}; check the Alpaca keys and symbols")
        log.info("loaded history: %s", {t: len(df) for t, df in self.fine.items()})

    def refresh(self):
        since = min(df.index[-1] for df in self.fine.values()) - pd.Timedelta(minutes=30)
        new = self.feed.bars(STEP_MINUTES, since)
        cutoff = now_utc() - pd.Timedelta(days=FINE_DAYS)
        for t, df in new.items():
            merged = pd.concat([self.fine[t], df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self.fine[t] = merged[merged.index > cutoff]

    def bars(self, t, tf):
        fine = self.fine[t]
        if tf in ("5m", "15m"):
            base = fine
        else:  # hourly history up to where the 5-minute data starts, then hours built from 5-minute bars
            cut = fine.index[0].ceil("h")
            old = self.hourly[t][self.hourly[t].index <= cut]
            base = pd.concat([old, data.resample_bars(fine[fine.index > cut], "1h")])
        return data.resample_bars(base, RULES[tf]).iloc[-MAX_BARS:]

    def frames(self):
        end = max(df.index[-1] for df in self.fine.values())
        start = min(df.index[0] for df in self.fine.values())
        calendar = pd.date_range(start, end, freq=f"{STEP_MINUTES}min")
        close = pd.DataFrame({t: df.close for t, df in self.fine.items()}).reindex(calendar)
        fresh = close.notna()
        close = close.ffill().bfill()
        low = pd.DataFrame({t: df.low for t, df in self.fine.items()}).reindex(calendar).where(fresh, close)
        return calendar, close[list(paper.UNIVERSE)], low[list(paper.UNIVERSE)], fresh[list(paper.UNIVERSE)]

    def signals(self, calendar):
        out = {}
        for tf in LIVE_TFS:
            bars = {t: self.bars(t, tf) for t in self.fine}
            labels = tuple(b.index[-1] if len(b) else None for b in bars.values())
            cached = self._sig_cache.get(tf)
            if cached is None or cached[0] != labels:
                cached = (labels, {ai: {t: fn(b).fillna(0).clip(0, 1) for t, b in bars.items()}
                                   for ai, fn in AIS.items()})
                self._sig_cache[tf] = cached
            for ai, cols in cached[1].items():
                out[(ai, tf)] = pd.DataFrame({t: s.reindex(calendar, method="ffill") for t, s in cols.items()}).fillna(0)
        return out

    def vol(self, calendar):
        cols = {}
        for t in self.fine:
            n = 365 if paper.UNIVERSE[t] == "crypto" else 252
            daily = self.bars(t, "1D").close
            cols[t] = (daily.pct_change().rolling(30).std() * np.sqrt(n)).reindex(calendar, method="ffill")
        return pd.DataFrame(cols)

    def add_news_ais(self, signals, calendar, close):
        """Give the news AIs their signals (and accounts, the first time news is ready)."""
        if self.news is not None and self.news.ready:
            joined = self.comp.add_accounts(NEWS_AIS, calendar[-1])
            if joined:
                log.info("news AIs joined: %s", joined)
            frames = news_signals(self.news.pressure(calendar, STEP_MINUTES), close, STEP_MINUTES)
        else:  # news unavailable: any news AI already in the game stays in cash
            frames = {ai: pd.DataFrame(0.0, index=calendar, columns=close.columns) for ai in NEWS_AIS}
        for ai, frame in frames.items():
            if ai in self.comp.state["accounts"]:
                for tf in LIVE_TFS:
                    signals[(ai, tf)] = frame

    def benchmark(self, close, index):
        """Equal-weight buy & hold of the universe since the start, stored per step so it outlives the
        35 days of 5-minute prices kept in memory."""
        path = os.path.join(self.comp.folder, "benchmark.csv")
        if "bench_base" not in self.comp.state:
            start = pd.Timestamp(self.comp.state["start"])
            base = close.loc[start] if start in close.index else close.iloc[0]
            self.comp.state["bench_base"] = {t: float(v) for t, v in base.items()}
        base = pd.Series(self.comp.state["bench_base"])
        stored = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0] if os.path.exists(path) else pd.Series(dtype=float)
        missing = [d for d in index if d not in stored.index and d in close.index]
        if missing:
            new = pd.Series({d: paper.START_CASH * float((close.loc[d] / base).mean()) for d in missing}, name="benchmark")
            new.to_csv(path, mode="a", header=not os.path.exists(path), index_label="date")
            stored = pd.concat([stored, new])
        return stored[~stored.index.duplicated()].reindex(index).ffill().bfill()

    # ---------- trading ----------
    def step(self):
        self.refresh()
        calendar, close, low, fresh = self.frames()
        signals = self.signals(calendar)
        if not hasattr(self.comp, "state"):
            try:
                self.comp.load()
            except FileNotFoundError:
                self.comp.reset(calendar[-1], pd.Timedelta(minutes=STEP_MINUTES))
                log.info("new competition from %s", self.comp.state["start"])
        self.add_news_ais(signals, calendar, close)
        ctx = learning.Context(paper.UNIVERSE, signals, close, low, self.vol(calendar), fresh,
                               [k == "crypto" for k in paper.UNIVERSE.values()],
                               [paper.FEES[k] for k in paper.UNIVERSE.values()],
                               step_minutes=STEP_MINUTES, window_days=WINDOW_DAYS)
        todo = calendar[calendar > pd.Timestamp(self.comp.state["last_date"])]
        eq_rows, lev_rows, trade_rows = [], [], []
        for d in todo:
            i = calendar.get_loc(d)
            self.comp.learn(i, d, ctx)
            e, l_, t = self.comp.step(i, d, close, low, fresh, ctx)
            eq_rows.append(e)
            lev_rows.append(l_)
            trade_rows.extend(t)
        if len(todo):
            self.comp.state["last_date"] = learning.stamp(todo[-1])
        self.comp.save(eq_rows, lev_rows, trade_rows)
        log.info("processed %d step(s), %d trade(s)", len(todo), len(trade_rows))

        lb, eq, lev, trades, _ = paper.leaderboard(self.comp, close, STEP_MINUTES)
        bench = self.benchmark(close, eq.index)
        payload = paper.build_payload(lb, eq, lev, trades, bench, self.comp, close, demo=False)
        if self.news is not None:
            payload["news"] = self.news.recent()
            payload["newsCost"] = round(self.news.cost["usd"], 2)
        with self.lock:
            self.payload = payload
            self.html = paper.render_dashboard(payload, live=True)
            self.updated = payload["updated"]
        return len(todo)
