"""Web server: runs the live engine in the background and serves the real-time dashboard.

    ALPACA_API_KEY=... ALPACA_SECRET_KEY=... DATA_DIR=/data uvicorn live.server:app --host 0.0.0.0 --port 8000
    SYNTHETIC=1 uvicorn live.server:app      # offline test with a fake market
"""
import asyncio
import json
import logging
import os
import threading
import time
import traceback

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import paper

from .engine import STEP_MINUTES, LiveEngine
from .feeds import AlpacaFeed, SyntheticFeed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("server")

PRICE_EVERY = 2          # seconds between live price polls
STEP_DELAY = 20          # seconds after a 5-minute boundary before pulling the new bar

feed = SyntheticFeed(paper.UNIVERSE) if os.environ.get("SYNTHETIC") else AlpacaFeed(paper.UNIVERSE)
engine = LiveEngine(feed, os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "state_live")))
status = {"started": time.time(), "last_step": None, "last_error": None, "prices_at": None}
live = {"prices": {}, "t": None}


def trading_loop():
    while True:
        try:
            if not engine.fine:
                engine.bootstrap()
            engine.step()
            status["last_step"], status["last_error"] = engine.updated, None
        except Exception:
            status["last_error"] = traceback.format_exc(limit=3)
            log.exception("step failed")
        now = time.time()
        period = STEP_MINUTES * 60
        time.sleep(period - now % period + STEP_DELAY)


def price_loop():
    while True:
        try:
            live["prices"], live["t"] = feed.latest_prices(), pd.Timestamp.now("UTC").strftime("%H:%M:%S")
            status["prices_at"] = live["t"]
        except Exception as e:
            log.warning("price poll failed: %s", e)
        time.sleep(PRICE_EVERY)


app = FastAPI(title="AI Trading-konkurranse")


@app.on_event("startup")
def start_threads():
    threading.Thread(target=trading_loop, daemon=True).start()
    threading.Thread(target=price_loop, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with engine.lock:
        html = engine.html
    if html:
        return html
    return HTMLResponse("<meta http-equiv='refresh' content='10'><body style='font-family:sans-serif;padding:24px'>"
                        "<h2>Starter opp…</h2><p>Henter kurshistorikk og lar AI-ene velge regler. "
                        "Siden laster seg selv på nytt.</p></body>", status_code=503)


@app.get("/api/state")
def state():
    with engine.lock:
        return JSONResponse(engine.payload or {})


@app.get("/live")
async def live_stream():
    async def events():
        while True:
            yield f"data: {json.dumps({'t': live['t'], 'prices': live['prices'], 'step': engine.updated})}\n\n"
            await asyncio.sleep(PRICE_EVERY)
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health():
    ok = status["last_error"] is None and engine.updated is not None
    return JSONResponse({"ok": ok, **status}, status_code=200 if ok else 503)
