"""Paper-trading competition: 20 AIs, $500 each, trading crypto and stocks on virtual margin accounts.

    python paper.py                         # daily run: process new bars, trade, update LEADERBOARD.md + docs/index.html
    python paper.py --reset                 # start a fresh competition today
    python paper.py --reset --start 2026-01-02   # replay from a past date, then keep going live
    python paper.py --offline --reset --start 2026-01-02   # synthetic data, separate state folder

Rules
- Long only. Signals use closed bars only; a signal from bar t is traded at the close of the asset's next bar.
- Base weights: signal / max(sum of signals, 4)  ->  max 25% of the book per asset.
- Leverage up to 10x. Each AI has its own risk profile (see RISK) that decides how much it uses:
  gross exposure = base weights x leverage, where leverage = target_vol / portfolio_vol, capped at max_lev.
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
from strategies import AIS

UNIVERSE = {
    "BTC-USD": "crypto", "ETH-USD": "crypto", "SOL-USD": "crypto", "XRP-USD": "crypto",
    "SPY": "stock", "QQQ": "stock", "NVDA": "stock", "AAPL": "stock", "MSFT": "stock", "TSLA": "stock",
}
FEES = {"crypto": 0.001, "stock": 0.0005}
START_CASH = 500.0
MIN_SPLIT = 4
REBAL_BAND = 0.03     # skip trades smaller than 3% of equity x leverage (avoids fee churn)
BORROW_RATE = 0.08    # yearly interest on borrowed cash
MAINT_MARGIN = 0.05   # liquidated when equity <= 5% of exposure
HISTORY_START = "2022-01-01"

# profile: (max leverage, target annual volatility or None = always max leverage)
PROFILES = {
    "Forsiktig 1x": (1.0, None),
    "Moderat ≤3x": (3.0, 0.70),
    "Aggressiv ≤10x": (10.0, 1.50),
    "Degen 10x": (10.0, None),
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
}


def load_prices(offline):
    dfs = {}
    for t, kind in UNIVERSE.items():
        if offline:
            dfs[t] = data.synthetic(t, days=1500, freq="D" if kind == "crypto" else "B")
        else:
            for attempt in range(3):
                try:
                    dfs[t] = data.fetch_yahoo(t, HISTORY_START)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(10 * (attempt + 1))
    return dfs


def build_signals(dfs, calendar):
    """signals[ai] = DataFrame(calendar x ticker) of target exposure in [0, 1], decided at that bar's close."""
    out = {}
    for name, fn in AIS.items():
        cols = {t: fn(df).fillna(0).clip(0, 1) for t, df in dfs.items()}
        out[name] = pd.DataFrame(cols).reindex(calendar).ffill().fillna(0)
    return out


def build_vol(dfs, calendar):
    """Annualised 30-bar volatility per asset, on each asset's own trading calendar."""
    cols = {}
    for t, df in dfs.items():
        n = 365 if UNIVERSE[t] == "crypto" else 252
        cols[t] = df.close.pct_change().rolling(30).std() * np.sqrt(n)
    return pd.DataFrame(cols).reindex(calendar).ffill()


class Competition:
    def __init__(self, folder):
        self.folder = folder
        self.state_path = os.path.join(folder, "competition.json")
        self.equity_path = os.path.join(folder, "equity.csv")
        self.leverage_path = os.path.join(folder, "leverage.csv")
        self.trades_path = os.path.join(folder, "trades.csv")

    def reset(self, start):
        os.makedirs(self.folder, exist_ok=True)
        for p in (self.equity_path, self.leverage_path, self.trades_path):
            if os.path.exists(p):
                os.remove(p)
        self.state = {
            "start": str(start.date()),
            "last_date": str((start - pd.Timedelta(days=1)).date()),
            "accounts": {ai: {"cash": START_CASH, "units": {}, "status": "active", "interest": 0.0, "fees": 0.0}
                         for ai in AIS},
        }

    def load(self):
        with open(self.state_path) as f:
            self.state = json.load(f)

    def save(self, equity_rows, lev_rows, trade_rows):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=1)
        for rows, path in ((equity_rows, self.equity_path), (lev_rows, self.leverage_path)):
            if rows:
                pd.DataFrame(rows).set_index("date").to_csv(path, mode="a", header=not os.path.exists(path))
        if trade_rows:
            pd.DataFrame(trade_rows).to_csv(self.trades_path, mode="a", index=False,
                                            header=not os.path.exists(self.trades_path))

    def step(self, date, prev, close, low, fresh, signals, vol):
        """Process one calendar day for every account. Returns (equity row, leverage row, trades)."""
        eq_row, lev_row, trades = {"date": date.date()}, {"date": date.date()}, []
        px, lo = close.loc[date], low.loc[date]
        for ai, acct in self.state["accounts"].items():
            if acct["status"] != "active":
                eq_row[ai], lev_row[ai] = 0.0, 0.0
                continue
            units = acct["units"]

            if acct["cash"] < 0:  # interest on borrowed cash, per calendar day
                interest = -acct["cash"] * BORROW_RATE / 365
                acct["cash"] -= interest
                acct["interest"] += interest

            # liquidation check at the day's lows (conservative: assumes all lows hit together)
            worst_exposure = sum(u * lo[t] for t, u in units.items())
            if units and acct["cash"] + worst_exposure <= MAINT_MARGIN * worst_exposure:
                trades.append({"date": date.date(), "ai": ai, "ticker": "ALL", "side": "LIQUIDATED",
                               "usd": round(worst_exposure, 2), "price": 0, "fee": 0, "leverage": 0})
                acct.update(cash=0.0, units={}, status=f"liquidated {date.date()}")
                eq_row[ai], lev_row[ai] = 0.0, 0.0
                continue

            equity = acct["cash"] + sum(u * px[t] for t, u in units.items())
            max_lev, target_vol = PROFILES[RISK[ai]]
            if prev is None:
                sig = pd.Series(0.0, index=px.index)
                lev = 0.0
            else:
                sig = signals[ai].loc[prev]
                w0 = sig / max(sig.sum(), MIN_SPLIT)
                pvol = float((w0 * vol.loc[prev].fillna(1.0)).sum())
                lev = max_lev if target_vol is None or pvol <= 0 else min(max_lev, target_vol / pvol)
            w = sig / max(sig.sum(), MIN_SPLIT) * lev

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
                trades.append({"date": date.date(), "ai": ai, "ticker": t, "side": "BUY" if value > 0 else "SELL",
                               "usd": round(abs(value), 2), "price": round(px[t], 6), "fee": round(fee, 4),
                               "leverage": round(lev, 2)})

            gross = sum(u * px[t] for t, u in units.items())
            equity = acct["cash"] + gross
            if units and equity <= MAINT_MARGIN * gross:
                trades.append({"date": date.date(), "ai": ai, "ticker": "ALL", "side": "LIQUIDATED",
                               "usd": round(gross, 2), "price": 0, "fee": 0, "leverage": 0})
                acct.update(cash=0.0, units={}, status=f"liquidated {date.date()}")
                equity, gross = 0.0, 0.0
            eq_row[ai] = equity
            lev_row[ai] = gross / equity if equity > 0 else 0.0
        return eq_row, lev_row, trades


def leaderboard(comp, close):
    eq = pd.read_csv(comp.equity_path, index_col=0, parse_dates=True)
    lev = pd.read_csv(comp.leverage_path, index_col=0, parse_dates=True)
    trades = pd.read_csv(comp.trades_path) if os.path.exists(comp.trades_path) else pd.DataFrame(columns=["ai"])
    last_px = close.loc[eq.index[-1]]
    rows = []
    for ai in eq.columns:
        e = eq[ai]
        r = e.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        acct = comp.state["accounts"][ai]
        hold = sorted(((u * last_px[t], t) for t, u in acct["units"].items()), reverse=True)
        active_lev = lev[ai][lev[ai] > 0]
        rows.append({
            "AI": ai,
            "Profil": RISK[ai],
            "Equity $": e.iloc[-1],
            "Return %": 100 * (e.iloc[-1] / START_CASH - 1),
            "Max DD %": 100 * (e / e.cummax() - 1).min(),
            "Sharpe": r.mean() / r.std() * np.sqrt(365) if len(r) > 20 and r.std() > 0 else np.nan,
            "Giring nå": lev[ai].iloc[-1],
            "Snitt giring": active_lev.mean() if len(active_lev) else 0.0,
            "Trades": int(((trades.ai == ai) & (trades.side != "LIQUIDATED")).sum()),
            "Status": acct["status"],
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
            f"20 AIs · ${START_CASH:.0f} each · crypto + stocks · up to 10x leverage · paper money\n\n"
            f"**Start:** {comp.state['start']} · **Updated:** {eq.index[-1].date()} · **Days:** {len(eq)} · "
            f"**Liquidated:** {dead}\n\n"
            f"**Benchmark** (equal-weight buy & hold, no leverage): **${bench.iloc[-1]:,.2f}** "
            f"→ {beat}/20 AIs ahead\n\n{header}{body}\n"
            f"Universe: {', '.join(UNIVERSE)}\n")
    with open(path, "w") as f:
        f.write(text)


def write_dashboard(lb, eq, lev, trades, bench, comp, close, path, demo):
    last_px = close.loc[eq.index[-1]]
    holdings = {}
    for ai, acct in comp.state["accounts"].items():
        holdings[ai] = sorted(([t, round(u * last_px[t], 2)] for t, u in acct["units"].items()),
                              key=lambda x: -x[1])
    accounts = {ai: {k: round(v, 2) if isinstance(v, float) else v for k, v in acct.items() if k != "units"}
                for ai, acct in comp.state["accounts"].items()}
    payload = {
        "demo": demo,
        "start": comp.state["start"],
        "updated": str(eq.index[-1].date()),
        "startCash": START_CASH,
        "profiles": list(PROFILES),
        "risk": RISK,
        "universe": UNIVERSE,
        "dates": [str(d.date()) for d in eq.index],
        "equity": {ai: [round(v, 2) for v in eq[ai]] for ai in eq.columns},
        "leverage": {ai: [round(v, 2) for v in lev[ai]] for ai in lev.columns},
        "benchmark": [round(v, 2) for v in bench],
        "leaderboard": json.loads(lb.to_json(orient="records")),
        "holdings": holdings,
        "accounts": accounts,
        "trades": trades.astype({"date": str}).values.tolist() if len(trades) else [],
        "tradeCols": list(trades.columns),
    }
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "dashboard_template.html")) as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(html)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="start a new competition (wipes state)")
    p.add_argument("--start", help="competition start date for --reset (default: latest bar)")
    p.add_argument("--offline", action="store_true", help="synthetic data, state in state_offline/")
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    comp = Competition(os.path.join(root, "state_offline" if args.offline else "state"))

    dfs = load_prices(args.offline)
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

    last = pd.Timestamp(comp.state["last_date"])
    todo = calendar[calendar > last]
    equity_rows, lev_rows, trade_rows = [], [], []
    for d in todo:
        i = calendar.get_loc(d)
        e, l, t = comp.step(d, calendar[i - 1] if i > 0 else None, close, low, fresh, signals, vol)
        equity_rows.append(e)
        lev_rows.append(l)
        trade_rows.extend(t)
    if len(todo):
        comp.state["last_date"] = str(todo[-1].date())
    comp.save(equity_rows, lev_rows, trade_rows)
    print(f"Processed {len(todo)} new day(s), {len(trade_rows)} trade(s).")

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
