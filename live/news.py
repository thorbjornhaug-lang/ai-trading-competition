"""News-reading AIs.

The news desk pulls finance news for our tickers from Alpaca (Benzinga feed), has Claude rate every
article (sentiment -1..1 and impact 0..1 per ticker, plus a one-line reason in Norwegian), stores the
ratings, and turns them into a decaying "news pressure" score per ticker. Two AIs trade on it:

- ai21_news_reader:  buys what the news is clearly positive about, sells when the good news fades.
- ai22_news_guard:   follows the price trend, but steps aside the moment the news turns negative.
"""
import json
import logging
import os
import threading
import time

import anthropic
import numpy as np
import pandas as pd

from .feeds import DATA_URL, now_utc

log = logging.getLogger("news")

MODEL = "claude-opus-5-5"
BATCH = 20                 # articles per Claude request
BACKFILL_DAYS = 31         # enough for the AIs' 30-day learning window
MAX_BACKFILL = 3000        # cap on articles rated at first start (cost guard)
MAX_TAGGED = 5             # skip round-ups tagged with many symbols
POLL_SECONDS = 120
HALF_LIFE_HOURS = 12       # how fast a story's effect fades
PRICE_PER_MTOK = (4.0, 20.0)  # input, output USD for MODEL

NEWS_AIS = ("ai21_news_reader", "ai22_news_guard")

SYSTEM = (
    "You are a financial news analyst. For each article, rate its likely short-term effect "
    "(hours to a few days) on the price of each listed ticker. sentiment: -1 (clearly bearish) to 1 "
    "(clearly bullish), 0 if neutral. impact: 0 to 1 for how much the story matters to that price - "
    "routine updates, promotional pieces and market round-ups are about 0.1, earnings surprises, "
    "regulation, hacks, lawsuits or major deals 0.6 and above. Judge only from the article text, as if "
    "you were reading it at its publish time; do not use knowledge of later events. reason: one short "
    "sentence in Norwegian (bokmål) explaining the rating."
)


def news_symbol(ticker):
    return ticker.replace("-", "")  # BTC-USD -> BTCUSD


class NewsDesk:
    def __init__(self, universe, folder, session=None, client=None):
        self.universe = list(universe)
        self.by_symbol = {news_symbol(t): t for t in self.universe}
        self.session = session
        self.client = client
        self.path = os.path.join(folder, "news.jsonl")
        self.items = {}
        self.lock = threading.Lock()
        self.ready = False
        self.error = None
        self.cost = {"input_tokens": 0, "output_tokens": 0, "usd": 0.0}
        os.makedirs(folder, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    it = json.loads(line)
                    self.items[it["id"]] = it
        self.ready = bool(self.items)  # ratings saved before a restart are usable right away

    # ---------- fetching ----------
    def fetch(self, start):
        """Articles since `start` that mention at least one of our tickers, oldest first."""
        out, params = [], {"symbols": ",".join(self.by_symbol), "start": f"{start:%Y-%m-%dT%H:%M:%SZ}",
                           "limit": 50, "sort": "desc", "include_content": "false"}
        while True:
            r = self.session.get(DATA_URL + "/v1beta1/news", params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            for a in js.get("news", []):
                tickers = [self.by_symbol[s] for s in a.get("symbols", []) if s in self.by_symbol]
                if tickers and len(a.get("symbols", [])) <= MAX_TAGGED:
                    out.append({"id": str(a["id"]), "t": a["created_at"], "headline": a.get("headline", ""),
                                "summary": (a.get("summary") or "")[:500], "source": a.get("source", ""),
                                "url": a.get("url", ""), "tickers": tickers})
            if not js.get("next_page_token") or len(out) >= MAX_BACKFILL:
                return out[::-1]
            params["page_token"] = js["next_page_token"]

    # ---------- rating ----------
    def rate(self, batch):
        """Ask Claude to rate a batch of articles. Returns {id: {"scores": {ticker: [sent, impact]}, "reason": str}}."""
        schema = {
            "type": "object",
            "properties": {"articles": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "reason": {"type": "string"},
                    "scores": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string", "enum": self.universe},
                                       "sentiment": {"type": "number"}, "impact": {"type": "number"}},
                        "required": ["ticker", "sentiment", "impact"], "additionalProperties": False}},
                },
                "required": ["id", "reason", "scores"], "additionalProperties": False}}},
            "required": ["articles"], "additionalProperties": False,
        }
        articles = [{"id": a["id"], "published": a["t"], "headline": a["headline"], "summary": a["summary"],
                     "tickers": a["tickers"]} for a in batch]
        response = self.client.beta.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM,
            messages=[{"role": "user", "content": "Rate these articles:\n" + json.dumps(articles, ensure_ascii=False)}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = response.usage
        self.cost["input_tokens"] += usage.input_tokens
        self.cost["output_tokens"] += usage.output_tokens
        self.cost["usd"] = (self.cost["input_tokens"] * PRICE_PER_MTOK[0]
                            + self.cost["output_tokens"] * PRICE_PER_MTOK[1]) / 1e6
        if response.stop_reason == "refusal":
            log.warning("rating refused for batch starting %s", batch[0]["id"])
            return {}
        text = next(b.text for b in response.content if b.type == "text")
        rated = {}
        for a in json.loads(text)["articles"]:
            rated[a["id"]] = {"reason": a["reason"], "scores": {
                s["ticker"]: [float(np.clip(s["sentiment"], -1, 1)), float(np.clip(s["impact"], 0, 1))]
                for s in a["scores"]}}
        return rated

    def ingest(self, articles):
        new = [a for a in articles if a["id"] not in self.items]
        for i in range(0, len(new), BATCH):
            batch = new[i:i + BATCH]
            try:
                rated = self.rate(batch)
            except anthropic.RateLimitError:
                time.sleep(30)
                rated = self.rate(batch)
            with self.lock, open(self.path, "a") as f:
                for a in batch:
                    if a["id"] in rated:
                        item = {**a, **rated[a["id"]]}
                        self.items[a["id"]] = item
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return len(new)

    def run(self):
        while True:
            try:
                since = now_utc() - pd.Timedelta(days=BACKFILL_DAYS)
                if self.items:
                    since = max(since, pd.Timestamp(max(i["t"] for i in self.items.values())).tz_localize(None)
                                - pd.Timedelta(hours=1))
                n = self.ingest(self.fetch(since))
                if n:
                    log.info("rated %d new article(s), total %d, spent $%.2f", n, len(self.items), self.cost["usd"])
                self.ready, self.error = True, None
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                log.exception("news update failed")
            time.sleep(POLL_SECONDS)

    # ---------- signals ----------
    def pressure(self, calendar, step_minutes):
        """Decaying sum of sentiment x impact per ticker, known at each calendar step (news published up to then)."""
        with self.lock:
            items = list(self.items.values())
        x = pd.DataFrame(0.0, index=calendar, columns=self.universe)
        for it in items:
            slot = calendar.searchsorted(pd.Timestamp(it["t"]).tz_localize(None).ceil(f"{step_minutes}min"))
            if slot >= len(calendar):
                continue
            for t, (sent, impact) in it["scores"].items():
                x.iat[slot, x.columns.get_loc(t)] += sent * impact
        decay = 0.5 ** (step_minutes / (HALF_LIFE_HOURS * 60))
        alpha = 1 - decay
        return x.ewm(alpha=alpha, adjust=False).mean() / alpha

    def recent(self, n=60):
        with self.lock:
            items = sorted(self.items.values(), key=lambda i: i["t"], reverse=True)[:n]
        return [{"t": i["t"][:16].replace("T", " "), "headline": i["headline"], "source": i["source"], "url": i["url"],
                 "reason": i["reason"], "scores": {t: round(s * m, 2) for t, (s, m) in i["scores"].items()}}
                for i in items]


def news_signals(pressure, close, step_minutes):
    """Target exposure in [0, 1] per ticker for the two news AIs."""
    from strategies import hold

    reader = pd.DataFrame({t: hold(pressure[t] > 0.6, pressure[t] < 0.1) for t in pressure.columns})
    per_hour = 60 // step_minutes
    fast, slow = close.ewm(span=12 * per_hour).mean(), close.ewm(span=26 * per_hour).mean()
    guard = ((fast > slow) & (pressure > -0.3)).astype(float)
    return {"ai21_news_reader": reader, "ai22_news_guard": guard}


class SyntheticNews(NewsDesk):
    """Offline stand-in: random headlines and ratings, no network or API calls."""

    def fetch(self, start):
        rng = np.random.default_rng(int(start.timestamp()) // 3600)
        times = pd.date_range(start, now_utc(), freq="45min")
        return [{"id": f"syn-{int(t.timestamp())}", "t": f"{t:%Y-%m-%dT%H:%M:%SZ}",
                 "headline": f"Testnyhet om {tk}", "summary": "", "source": "synthetic", "url": "",
                 "tickers": [tk]} for t in times for tk in [self.universe[rng.integers(len(self.universe))]]]

    def rate(self, batch):
        rng = np.random.default_rng(len(self.items))
        return {a["id"]: {"reason": "Syntetisk vurdering for test.",
                          "scores": {a["tickers"][0]: [float(rng.uniform(-1, 1)), float(rng.uniform(0, 1))]}}
                for a in batch}
