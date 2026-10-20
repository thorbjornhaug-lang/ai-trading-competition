"""Paper-trading competition: 20 AIs, $500 each, trading crypto and stocks on virtual accounts.

    python paper.py                         # daily run: process new bars, trade, update LEADERBOARD.md
    python paper.py --reset                 # start a fresh competition today
    python paper.py --reset --start 2026-01-02   # replay from a past date, then keep going live
    python paper.py --offline --reset --start 2026-01-02   # synthetic data, separate state folder

Rules: long only, no leverage, fractional units. Signals use closed bars only. A signal from bar t
is traded at the close of the asset's next bar. Max 25% per asset (4-way minimum split).
Fees: 0.10% crypto, 0.05% stocks, per trade.
"""
import argparse
import json
import os

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
MIN_SPLIT = 4        # weight = signal / max(sum of signals, 4)  ->  max 25% per asset
REBAL_BAND = 0.03    # skip trades smaller than 3% of account equity (avoids fee churn)
HISTORY_START = "2022-01-01"  # warm-up for indicators


def load_prices(offline):
    dfs = {}
    for t, kind in UNIVERSE.items():
        if offline:
            dfs[t] = data.synthetic(t, days=1500, freq="D" if kind == "crypto" else "B")
        else:
            dfs[t] = data.fetch_yahoo(t, HISTORY_START)
    return dfs


def build_signals(dfs, calendar):
    """signals[ai] = DataFrame(calendar x ticker) of target exposure in [0, 1], decided at that bar's close."""
    out = {}
    for name, fn in AIS.items():
        cols = {t: fn(df).fillna(0).clip(0, 1) for t, df in dfs.items()}
        out[name] = pd.DataFrame(cols).reindex(calendar).ffill().fillna(0)
    return out


class Competition:
    def __init__(self, folder):
        self.folder = folder
        self.state_path = os.path.join(folder, "competition.json")
        self.equity_path = os.path.join(folder, "equity.csv")
        self.trades_path = os.path.join(folder, "trades.csv")

    def reset(self, start):
        os.makedirs(self.folder, exist_ok=True)
        for p in (self.equity_path, self.trades_path):
            if os.path.exists(p):
                os.remove(p)
        self.state = {
            "start": str(start.date()),
            "last_date": str((start - pd.Timedelta(days=1)).date()),
            "accounts": {ai: {"cash": START_CASH, "units": {}} for ai in AIS},
        }

    def load(self):
        with open(self.state_path) as f:
            self.state = json.load(f)

    def save(self, equity_rows, trade_rows):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=1)
        if equity_rows:
            eq = pd.DataFrame(equity_rows).set_index("date")
            eq.to_csv(self.equity_path, mode="a", header=not os.path.exists(self.equity_path))
        if trade_rows:
            tr = pd.DataFrame(trade_rows)
            tr.to_csv(self.trades_path, mode="a", index=False, header=not os.path.exists(self.trades_path))

    def step(self, date, prev, close, fresh, signals):
        """Process one calendar day for every account. Returns (equity row, trades)."""
        row, trades = {"date": date.date()}, []
        px = close.loc[date]
        for ai, acct in self.state["accounts"].items():
            units = acct["units"]
            equity = acct["cash"] + sum(u * px[t] for t, u in units.items())
            sig = signals[ai].loc[prev] if prev is not None else pd.Series(0.0, index=px.index)
            w = sig / max(sig.sum(), MIN_SPLIT)

            orders = {}
            for t in UNIVERSE:
                if not fresh.loc[date, t] or np.isnan(px[t]):
                    continue
                held = units.get(t, 0.0) * px[t]
                diff = w[t] * equity - held
                closing = w[t] == 0 and held > 0
                if closing or abs(diff) > REBAL_BAND * equity:
                    orders[t] = -held if closing else diff
            # sells first so buys can use the freed cash
            for t, value in sorted(orders.items(), key=lambda kv: kv[1]):
                fee_rate = FEES[UNIVERSE[t]]
                if value > 0:
                    value = min(value, acct["cash"] / (1 + fee_rate))
                    if value < 1:
                        continue
                fee = abs(value) * fee_rate
                units[t] = units.get(t, 0.0) + value / px[t]
                acct["cash"] -= value + fee
                if units[t] * px[t] < 0.01:
                    units.pop(t)
                trades.append({"date": date.date(), "ai": ai, "ticker": t, "side": "BUY" if value > 0 else "SELL",
                               "usd": round(abs(value), 2), "price": round(px[t], 6), "fee": round(fee, 4)})
            acct["cash"] = max(acct["cash"], 0.0)  # float dust
            row[ai] = acct["cash"] + sum(u * px[t] for t, u in units.items())
        return row, trades


def leaderboard(comp, close):
    eq = pd.read_csv(comp.equity_path, index_col=0, parse_dates=True)
    trades = pd.read_csv(comp.trades_path) if os.path.exists(comp.trades_path) else pd.DataFrame(columns=["ai"])
    last_px = close.loc[eq.index[-1]]
    rows = []
    for ai in eq.columns:
        e = eq[ai]
        r = e.pct_change().dropna()
        acct = comp.state["accounts"][ai]
        hold = sorted(((u * last_px[t], t) for t, u in acct["units"].items()), reverse=True)
        rows.append({
            "AI": ai,
            "Equity $": e.iloc[-1],
            "Return %": 100 * (e.iloc[-1] / START_CASH - 1),
            "Max DD %": 100 * (e / e.cummax() - 1).min(),
            "Sharpe": r.mean() / r.std() * np.sqrt(365) if len(r) > 20 and r.std() > 0 else np.nan,
            "Trades": int((trades.ai == ai).sum()),
            "Cash %": 100 * acct["cash"] / e.iloc[-1],
            "Top holdings": ", ".join(t for _, t in hold[:3]) or "-",
        })
    lb = pd.DataFrame(rows).sort_values("Equity $", ascending=False).reset_index(drop=True)
    lb.index += 1

    start = pd.Timestamp(comp.state["start"])
    base = close[close.index >= start].bfill().iloc[0]
    bench = START_CASH * (last_px / base).mean()
    return lb, eq, bench


def write_markdown(lb, eq, bench, comp, path):
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    fmt = lb.copy()
    fmt.insert(0, "#", [medals.get(i, str(i)) for i in fmt.index])
    for c, d in [("Equity $", 2), ("Return %", 1), ("Max DD %", 1), ("Sharpe", 2), ("Cash %", 0)]:
        fmt[c] = fmt[c].map(lambda v: "-" if pd.isna(v) else f"{v:,.{d}f}")
    header = "| " + " | ".join(fmt.columns) + " |\n|" + "---|" * len(fmt.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in fmt.itertuples(index=False))
    beat = int((lb["Equity $"] > bench).sum())
    text = (f"# 🏆 AI Trading Competition\n\n"
            f"20 AIs · ${START_CASH:.0f} each · crypto + stocks · paper money\n\n"
            f"**Start:** {comp.state['start']} · **Updated:** {eq.index[-1].date()} · "
            f"**Days:** {len(eq)}\n\n"
            f"**Benchmark** (equal-weight buy & hold of all {len(UNIVERSE)} assets): **${bench:,.2f}** "
            f"→ {beat}/20 AIs ahead\n\n{header}{body}\n"
            f"Universe: {', '.join(UNIVERSE)}\n")
    with open(path, "w") as f:
        f.write(text)


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
    signals = build_signals(dfs, calendar)

    if args.reset or not os.path.exists(comp.state_path):
        start = pd.Timestamp(args.start) if args.start else calendar[-1]
        comp.reset(calendar[calendar.searchsorted(start)])
        print(f"New competition from {comp.state['start']}: 20 AIs x ${START_CASH:.0f}")
    else:
        comp.load()

    last = pd.Timestamp(comp.state["last_date"])
    todo = calendar[calendar > last]
    equity_rows, trade_rows = [], []
    for d in todo:
        i = calendar.get_loc(d)
        row, trades = comp.step(d, calendar[i - 1] if i > 0 else None, close, fresh, signals)
        equity_rows.append(row)
        trade_rows.extend(trades)
    if len(todo):
        comp.state["last_date"] = str(todo[-1].date())
    comp.save(equity_rows, trade_rows)
    print(f"Processed {len(todo)} new day(s), {len(trade_rows)} trade(s).")

    if not os.path.exists(comp.equity_path):
        return
    lb, eq, bench = leaderboard(comp, close)
    md = os.path.join(root, "LEADERBOARD_offline.md" if args.offline else "LEADERBOARD.md")
    write_markdown(lb, eq, bench, comp, md)
    pd.set_option("display.width", 200)
    print(f"\nBenchmark buy & hold: ${bench:,.2f}\n")
    print(lb.round(2).to_string())
    print(f"\nWrote {os.path.relpath(md, root)}")


if __name__ == "__main__":
    main()
