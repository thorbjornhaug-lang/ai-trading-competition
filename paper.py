"""Paper-trading competition: 20 AIs, $500 each, trading crypto and stocks on virtual margin accounts.

    python paper.py                         # hourly run: process new hours, trade, update LEADERBOARD.md + docs/index.html
    python paper.py --reset                 # start a fresh competition now
    python paper.py --reset --start "2026-10-01 14:00"   # replay from a past hour (UTC), then keep going live
    python paper.py --offline --reset --start 2026-09-01  # synthetic data, separate state folder

Rules
- Runs every hour. Crypto trades 24/7; stocks only while the New York market is open.
- Each AI picks its own tempo (1h, 4h or daily bars) and changes it when learning says so.
- Long only. Signals use closed bars only; a signal from hour t is traded at the asset's next hourly price.
- Base weights: signal / max(sum of signals, 4)  ->  max 25% of the book per asset.
- Leverage on crypto only (stocks max 1x). Each AI has a risk profile (see RISK) capping its leverage at 1x, 3x or 10x;
  leverage = target_vol / portfolio_vol (or always max), scaled by the AI's own learned lev_mult.
- Every week each AI reviews its own last 120 days and rewrites its rules (see learning.py).
- Borrowed cash pays 8% a year. Fees: 0.10% crypto, 0.05% stocks.
- Liquidation: if equity at the day's lows falls to 5% of exposure or less, the account is wiped out and is out.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import data
import learning
from strategies import AIS

UNIVERSE = {
    "BTC-USD": "crypto", "ETH-USD": "crypto", "SOL-USD": "crypto", "XRP-USD": "crypto",
    "SPY": "stock", "QQQ": "stock", "NVDA": "stock", "AAPL": "stock", "MSFT": "stock", "TSLA": "stock",
}
FEES = {"crypto": 0.001, "stock": 0.0005}
START_CASH = 500.0
REBAL_BAND = 0.03     # skip trades smaller than 3% of equity x leverage (avoids fee churn)
BORROW_RATE = learning.BORROW_RATE
MAINT_MARGIN = learning.MAINT_MARGIN
MAX_CHART_POINTS = 2000
HOURLY_TFS = ("1h", "4h", "1D")  # tempos available when running on hourly bars (GitHub Actions)
HISTORY_DAYS = 720   # Yahoo serves up to 730 days of hourly bars

# profile: (max leverage, target annual volatility or None = always max leverage)
PROFILES = {
    "Forsiktig 1x": (1.0, None),
    "Moderat ≤3x": (3.0, 0.70),
    "Aggressiv ≤10x": (10.0, 1.50),
    "Degen 10x": (10.0, None),
    "Nyhets-AI ≤3x": (3.0, 0.70),
}
RISK = {
    "ai04_rsi_reversion": "Forsiktig 1x", "ai05_bollinger_reversion": "Forsiktig 1x",
    "ai09_dual_momentum": "Forsiktig 1x", "ai10_vol_target_trend": "Forsiktig 1x", "ai16_ichimoku": "Forsiktig 1x",
    "ai01_sma_cross": "Moderat ≤3x", "ai02_ema_cross": "Moderat ≤3x", "ai07_donchian_turtle": "Moderat ≤3x",
    "ai12_stochastic": "Moderat ≤3x", "ai17_obv_breakout": "Moderat ≤3x",
    "ai03_macd": "Aggressiv ≤10x", "ai08_tsmom": "Aggressiv ≤10x", "ai13_zscore": "Aggressiv ≤10x",
    "ai14_adx_trend": "Aggressiv ≤10x", "ai19_ml_ridge": "Aggressiv ≤10x",
    "ai06_bollinger_breakout": "Degen 10x", "ai11_keltner": "Degen 10x", "ai15_supertrend": "Degen 10x",
    "ai18_vol_breakout": "Degen 10x", "ai20_kalman_trend": "Degen 10x",
    "ai21_news_reader": "Nyhets-AI ≤3x", "ai22_news_guard": "Nyhets-AI ≤3x",
}


def load_prices(offline, days=HISTORY_DAYS):
    dfs = {}
    for t, kind in UNIVERSE.items():
        if offline:
            dfs[t] = data.synthetic_hourly(t, days=days, stock=kind == "stock")
            continue
        for attempt in range(3):
            try:
                dfs[t] = data.fetch_yahoo_hourly(t, days)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))
    return dfs


def build_signals(dfs, calendar, tfs=HOURLY_TFS):
    """signals[(ai, tf)] = DataFrame(hour x ticker) of target exposure in [0, 1] from that AI's strategy
    run on tf-sized bars; a bar's signal appears at the hour the bar closes."""
    bars = {(t, tf): data.resample_bars(df, tf) for t, df in dfs.items() for tf in tfs}
    out = {}
    for name, fn in AIS.items():
        for tf in tfs:
            cols = {t: fn(bars[(t, tf)]).fillna(0).clip(0, 1).reindex(calendar, method="ffill") for t in dfs}
            out[(name, tf)] = pd.DataFrame(cols).fillna(0)
    return out


def build_vol(dfs, calendar):
    """Annualised 30-day volatility per asset from daily bars, known at each day's close."""
    cols = {}
    for t, df in dfs.items():
        n = 365 if UNIVERSE[t] == "crypto" else 252
        daily = data.resample_bars(df, "1D").close
        cols[t] = (daily.pct_change().rolling(30).std() * np.sqrt(n)).reindex(calendar, method="ffill")
    return pd.DataFrame(cols)


class Competition:
    def __init__(self, folder):
        self.folder = folder
        self.state_path = os.path.join(folder, "competition.json")
        self.equity_path = os.path.join(folder, "equity.csv")
        self.leverage_path = os.path.join(folder, "leverage.csv")
        self.trades_path = os.path.join(folder, "trades.csv")

    def reset(self, start, step=pd.Timedelta(hours=1)):
        os.makedirs(self.folder, exist_ok=True)
        for p in (self.equity_path, self.leverage_path, self.trades_path):
            if os.path.exists(p):
                os.remove(p)
        self.state = {
            "start": learning.stamp(start),
            "last_date": learning.stamp(start - step),
            "last_learn": learning.stamp(start - pd.Timedelta(days=learning.LEARN_EVERY_DAYS)),
            "journal": [],
            "accounts": {ai: {"cash": START_CASH, "units": {}, "status": "active", "interest": 0.0, "fees": 0.0,
                              "rules": json.loads(json.dumps(learning.DEFAULT_RULES)), "peak": {}, "stopped": []}
                         for ai in AIS},
        }

    def load(self):
        with open(self.state_path) as f:
            self.state = json.load(f)

    def add_accounts(self, names, when):
        """Let new AIs join a running competition with fresh $500 accounts."""
        new = [n for n in names if n not in self.state["accounts"]]
        for n in new:
            self.state["accounts"][n] = {"cash": START_CASH, "units": {}, "status": "active", "interest": 0.0,
                                         "fees": 0.0, "rules": json.loads(json.dumps(learning.DEFAULT_RULES)),
                                         "peak": {}, "stopped": [], "joined": learning.stamp(when)}
        for path in (self.equity_path, self.leverage_path):
            if new and os.path.exists(path):
                df = pd.read_csv(path, index_col=0)
                for n in new:
                    df[n] = np.nan
                df.to_csv(path)
        return new

    def save(self, equity_rows, lev_rows, trade_rows):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=1)
        for rows, path in ((equity_rows, self.equity_path), (lev_rows, self.leverage_path)):
            if rows:
                pd.DataFrame(rows).set_index("date").to_csv(path, mode="a", header=not os.path.exists(path))
        if trade_rows:
            pd.DataFrame(trade_rows).to_csv(self.trades_path, mode="a", index=False,
                                            header=not os.path.exists(self.trades_path))

    def step(self, i, date, close, low, fresh, ctx):
        """Process calendar day i for every account. Returns (equity row, leverage row, trades)."""
        today = learning.stamp(date)
        eq_row, lev_row, trades = {"date": today}, {"date": today}, []
        px, lo = close.loc[date], low.loc[date]

        def log(ai, t, side, usd, price=0.0, fee=0.0, lev=0.0, note=""):
            trades.append({"date": today, "ai": ai, "ticker": t, "side": side, "usd": round(usd, 2),
                           "price": round(price, 6), "fee": round(fee, 4), "leverage": round(lev, 2), "note": note})

        for ai, acct in self.state["accounts"].items():
            if acct["status"] != "active":
                eq_row[ai], lev_row[ai] = 0.0, 0.0
                continue
            units, rules = acct["units"], acct["rules"]

            if acct["cash"] < 0:  # interest on borrowed cash, per step
                interest = -acct["cash"] * ctx.borrow_per_step
                acct["cash"] -= interest
                acct["interest"] += interest

            # liquidation check at the day's lows (conservative: assumes all lows hit together)
            worst_exposure = sum(u * lo[t] for t, u in units.items())
            if units and acct["cash"] + worst_exposure <= MAINT_MARGIN * worst_exposure:
                log(ai, "ALL", "LIQUIDATED", worst_exposure)
                acct.update(cash=0.0, units={}, status=f"liquidated {today}")
                eq_row[ai], lev_row[ai] = 0.0, 0.0
                continue

            # trailing stop-loss on today's close
            stop_notes = {}
            for t in units:
                if fresh.loc[date, t]:
                    acct["peak"][t] = max(acct["peak"].get(t, px[t]), px[t])
                    if rules["stop"] and px[t] < acct["peak"][t] * (1 - rules["stop"]) and t not in acct["stopped"]:
                        acct["stopped"].append(t)
                        stop_notes[t] = f"stop-loss {rules['stop']:.0%}"

            equity = acct["cash"] + sum(u * px[t] for t, u in units.items())
            max_lev, target_vol = PROFILES[RISK[ai]]
            if i == 0:
                w, lev = np.zeros(len(ctx.tickers)), 0.0
            else:
                raw = ctx.sig[(ai, rules["tf"])][i - 1]
                acct["stopped"] = [t for t in acct["stopped"] if raw[ctx.tickers.index(t)] > 0]
                stopped = np.array([t in acct["stopped"] for t in ctx.tickers])
                banned = learning.banned_mask(ctx, rules["bans"], today)
                sig = learning.effective_signal(ctx, ai, i - 1, rules, banned, stopped)
                w, lev = learning.target_weights(ctx, i - 1, sig, max_lev, target_vol, rules["lev_mult"])
            w = pd.Series(w, index=ctx.tickers)

            orders = {}
            band = REBAL_BAND * equity * max(lev, 1.0)
            for t in UNIVERSE:
                if not fresh.loc[date, t] or np.isnan(px[t]):
                    continue
                held = units.get(t, 0.0) * px[t]
                diff = w[t] * equity - held
                closing = w[t] == 0 and held > 0
                if closing or abs(diff) > band:
                    orders[t] = -held if closing else diff
            # sells first, then buys limited by buying power (max_lev x equity)
            for t, value in sorted(orders.items(), key=lambda kv: kv[1]):
                fee_rate = FEES[UNIVERSE[t]]
                if value > 0:
                    gross = sum(u * px[k] for k, u in units.items())
                    eq_now = acct["cash"] + gross
                    value = min(value, (max_lev * eq_now - gross) / (1 + fee_rate * max_lev))
                    if value < 1:
                        continue
                fee = abs(value) * fee_rate
                units[t] = units.get(t, 0.0) + value / px[t]
                acct["cash"] -= value + fee
                acct["fees"] += fee
                if units[t] * px[t] < 0.01:
                    units.pop(t)
                    acct["peak"].pop(t, None)
                note = stop_notes.get(t, "") or ("bannet" if value < 0 and t in rules["bans"] and w[t] == 0 else "")
                trade_lev = lev if UNIVERSE[t] == "crypto" else min(lev, learning.STOCK_MAX_LEV)
                log(ai, t, "BUY" if value > 0 else "SELL", abs(value), px[t], fee, trade_lev, note)

            gross = sum(u * px[t] for t, u in units.items())
            equity = acct["cash"] + gross
            if units and equity <= MAINT_MARGIN * gross:
                log(ai, "ALL", "LIQUIDATED", gross)
                acct.update(cash=0.0, units={}, status=f"liquidated {today}")
                equity, gross = 0.0, 0.0
            eq_row[ai] = equity
            lev_row[ai] = gross / equity if equity > 0 else 0.0
        return eq_row, lev_row, trades

    def learn(self, i, date, ctx):
        """Weekly: every active AI reviews its own recent results and rewrites its rules."""
        today = learning.stamp(date)
        if date - pd.Timestamp(self.state["last_learn"]) < pd.Timedelta(days=learning.LEARN_EVERY_DAYS):
            return
        self.state["last_learn"] = today
        for ai, acct in self.state["accounts"].items():
            if acct["status"] != "active":
                continue
            max_lev, target_vol = PROFILES[RISK[ai]]
            new, notes, before, after = learning.learn(ctx, ai, acct["rules"], i, today, max_lev, target_vol)
            acct["rules"] = new
            if notes:
                self.state["journal"].append({"date": today, "ai": ai, "changes": notes,
                                              "score_before": round(before, 3), "score_after": round(after, 3)})


def rules_text(rules, date):
    parts = [rules["tf"], f"giring {rules['lev_mult']:.0%}"]
    if rules["confirm"]:
        parts.append(f"bekreft {rules['confirm']}")
    if rules["stop"]:
        parts.append(f"stop {rules['stop']:.0%}")
    bans = [t for t, until in rules["bans"].items() if until > date]
    if bans:
        parts.append("pause " + "/".join(bans))
    return ", ".join(parts)


def leaderboard(comp, close, step_minutes=60):
    eq = pd.read_csv(comp.equity_path, index_col=0, parse_dates=True)
    lev = pd.read_csv(comp.leverage_path, index_col=0, parse_dates=True)
    trades = pd.read_csv(comp.trades_path) if os.path.exists(comp.trades_path) else pd.DataFrame(columns=["ai"])
    last_px = close.loc[eq.index[-1]]
    rows = []
    for ai in eq.columns:
        e = eq[ai].dropna()
        r = e.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        acct = comp.state["accounts"][ai]
        hold = sorted(((u * last_px[t], t) for t, u in acct["units"].items()), reverse=True)
        active_lev = lev[ai][lev[ai] > 0].dropna()
        rows.append({
            "AI": ai,
            "Profil": RISK[ai],
            "Equity $": e.iloc[-1],
            "Return %": 100 * (e.iloc[-1] / START_CASH - 1),
            "Max DD %": 100 * (e / e.cummax() - 1).min(),
            "Sharpe": r.mean() / r.std() * np.sqrt(learning.MINUTES_PER_YEAR / step_minutes) if len(r) > 48 and r.std() > 0 else np.nan,
            "Giring nå": float(np.nan_to_num(lev[ai].iloc[-1])),
            "Snitt giring": active_lev.mean() if len(active_lev) else 0.0,
            "Trades": int(((trades.ai == ai) & (trades.side != "LIQUIDATED")).sum()),
            "Tempo": learning.TF_NAMES[acct["rules"]["tf"]],
            "Status": acct["status"],
            "Regler": rules_text(acct["rules"], comp.state["last_date"]),
            "Lærte": sum(1 for j in comp.state["journal"] if j["ai"] == ai),
            "Top holdings": ", ".join(t for _, t in hold[:3]) or "-",
        })
    lb = pd.DataFrame(rows).sort_values("Equity $", ascending=False).reset_index(drop=True)
    lb.index += 1

    start = pd.Timestamp(comp.state["start"])
    window = close[close.index >= start].bfill()
    bench = START_CASH * (window / window.iloc[0]).mean(axis=1)
    return lb, eq, lev, trades, bench.reindex(eq.index)


def write_markdown(lb, eq, bench, comp, path):
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    fmt = lb.copy()
    fmt.insert(0, "#", [medals.get(i, str(i)) for i in fmt.index])
    for c, d in [("Equity $", 2), ("Return %", 1), ("Max DD %", 1), ("Sharpe", 2), ("Giring nå", 1),
                 ("Snitt giring", 1)]:
        fmt[c] = fmt[c].map(lambda v: "-" if pd.isna(v) else f"{v:,.{d}f}")
    header = "| " + " | ".join(fmt.columns) + " |\n|" + "---|" * len(fmt.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in fmt.itertuples(index=False))
    beat = int((lb["Equity $"] > bench.iloc[-1]).sum())
    dead = int(lb.Status.str.startswith("liquidated").sum())
    text = (f"# 🏆 AI Trading Competition\n\n"
            f"{len(lb)} self-learning AIs · ${START_CASH:.0f} each · crypto + stocks · up to 10x leverage on crypto · paper money\n\n"
            f"**Start:** {comp.state['start']} UTC · **Updated:** {learning.stamp(eq.index[-1])} UTC · "
            f"**Days:** {len(eq) / 24:.1f} · "
            f"**Liquidated:** {dead}\n\n"
            f"**Benchmark** (equal-weight buy & hold, no leverage): **${bench.iloc[-1]:,.2f}** "
            f"→ {beat}/{len(lb)} AIs ahead\n\n{header}{body}\n"
            f"Universe: {', '.join(UNIVERSE)}\n")
    with open(path, "w") as f:
        f.write(text)


def build_payload(lb, eq, lev, trades, bench, comp, close, demo):
    last_px = close.loc[eq.index[-1]]
    holdings = {}
    for ai, acct in comp.state["accounts"].items():
        holdings[ai] = sorted(([t, round(u * last_px[t], 2)] for t, u in acct["units"].items()),
                              key=lambda x: -x[1])
    accounts = {ai: {k: round(v, 2) if isinstance(v, float) else v for k, v in acct.items()
                     if k in ("cash", "status", "interest", "fees")}
                for ai, acct in comp.state["accounts"].items()}
    step = max(1, -(-len(eq) // MAX_CHART_POINTS))  # thin long series for the charts
    keep = sorted(set(range(len(eq) - 1, -1, -step)))
    return {
        "demo": demo,
        "start": comp.state["start"],
        "updated": learning.stamp(eq.index[-1]),
        "startCash": START_CASH,
        "profiles": list(PROFILES),
        "risk": RISK,
        "universe": UNIVERSE,
        "dates": [learning.stamp(d) for d in eq.index[keep]],
        "equity": {ai: [None if pd.isna(v) else round(v, 2) for v in eq[ai].iloc[keep]] for ai in eq.columns},
        "leverage": {ai: [None if pd.isna(v) else round(v, 2) for v in lev[ai].iloc[keep]] for ai in lev.columns},
        "benchmark": [None if pd.isna(v) else round(v, 2) for v in bench.iloc[keep]],
        "tfNames": learning.TF_NAMES,
        "leaderboard": json.loads(lb.to_json(orient="records")),
        "holdings": holdings,
        "positions": {ai: {"cash": acct["cash"], "units": acct["units"]} for ai, acct in comp.state["accounts"].items()},
        "stepPrices": {t: float(v) for t, v in last_px.items()},
        "accounts": accounts,
        "trades": trades.astype({"date": str}).fillna("").values.tolist() if len(trades) else [],
        "journal": comp.state["journal"],
        "rules": {ai: acct["rules"] for ai, acct in comp.state["accounts"].items()},
        "tradeCols": list(trades.columns),
    }


def render_dashboard(payload, live=False):
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "dashboard_template.html")) as f:
        html = f.read()
    html = html.replace("/*__LIVE__*/false", "true" if live else "false")
    return html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), ensure_ascii=False))


def write_dashboard(lb, eq, lev, trades, bench, comp, close, path, demo):
    html = render_dashboard(build_payload(lb, eq, lev, trades, bench, comp, close, demo))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(html)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="start a new competition (wipes state)")
    p.add_argument("--start", help="competition start (UTC) for --reset (default: latest hour)")
    p.add_argument("--offline", action="store_true", help="synthetic data, state in state_offline/")
    p.add_argument("--days", type=int, default=HISTORY_DAYS, help="days of hourly history to load")
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    comp = Competition(os.path.join(root, "state_offline" if args.offline else "state"))

    dfs = load_prices(args.offline, args.days)
    calendar = pd.DatetimeIndex(sorted(set().union(*(df.index for df in dfs.values()))))
    close = pd.DataFrame({t: df.close for t, df in dfs.items()}).reindex(calendar)
    fresh = close.notna()
    close = close.ffill()
    low = pd.DataFrame({t: df.low for t, df in dfs.items()}).reindex(calendar).where(fresh, close)
    signals = build_signals(dfs, calendar)
    vol = build_vol(dfs, calendar)

    if args.reset or not os.path.exists(comp.state_path):
        start = pd.Timestamp(args.start) if args.start else calendar[-1]
        comp.reset(calendar[calendar.searchsorted(start)])
        print(f"New competition from {comp.state['start']}: 20 AIs x ${START_CASH:.0f}")
    else:
        comp.load()

    ctx = learning.Context(UNIVERSE, signals, close, low, vol, fresh,
                           [k == "crypto" for k in UNIVERSE.values()], [FEES[k] for k in UNIVERSE.values()])
    last = pd.Timestamp(comp.state["last_date"])
    todo = calendar[calendar > last]
    equity_rows, lev_rows, trade_rows = [], [], []
    for d in todo:
        i = calendar.get_loc(d)
        comp.learn(i, d, ctx)
        e, l, t = comp.step(i, d, close, low, fresh, ctx)
        equity_rows.append(e)
        lev_rows.append(l)
        trade_rows.extend(t)
    if len(todo):
        comp.state["last_date"] = learning.stamp(todo[-1])
    comp.save(equity_rows, lev_rows, trade_rows)
    print(f"Processed {len(todo)} new hour(s), {len(trade_rows)} trade(s).")

    if not os.path.exists(comp.equity_path):
        return
    lb, eq, lev, trades, bench = leaderboard(comp, close)
    suffix = "_offline" if args.offline else ""
    write_markdown(lb, eq, bench, comp, os.path.join(root, f"LEADERBOARD{suffix}.md"))
    write_dashboard(lb, eq, lev, trades, bench, comp, close,
                    os.path.join(root, "docs_offline" if args.offline else "docs", "index.html"), args.offline)
    pd.set_option("display.width", 220)
    print(f"\nBenchmark buy & hold: ${bench.iloc[-1]:,.2f}\n")
    print(lb.round(2).to_string())


if __name__ == "__main__":
    main()
