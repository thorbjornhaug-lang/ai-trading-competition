"""Run all 20 AIs, rank them honestly (out-of-sample), build the 'Council' ensemble, print today's signals.

    python run.py                          # BTC, ETH, SOL, XRP from Binance
    python run.py --symbols BTCUSDT SOLUSDT --shorts
    python run.py --offline                # synthetic data, no network
"""
import argparse
import os

import pandas as pd

import backtest
import data
from strategies import AIS

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def evaluate(symbol, args):
    df, source = data.load(symbol, refresh=args.refresh, offline=args.offline)
    split = df.index[int(len(df) * (1 - args.oos))]
    positions = {name: fn(df).fillna(0) for name, fn in AIS.items()}

    rows = []
    for name, pos in positions.items():
        r = backtest.run(df, pos, args.fee, long_only=not args.shorts)
        ins, oos = backtest.metrics(r[r.index < split]), backtest.metrics(r[r.index >= split])
        rows.append({"ai": name, "is_sharpe": ins["sharpe"], **oos})
    table = pd.DataFrame(rows).set_index("ai")

    # Council: equal-weight the top-K AIs chosen ONLY on in-sample Sharpe, judged out-of-sample.
    top = table.is_sharpe.nlargest(args.top_k).index
    council_pos = sum(positions[n] for n in top) / len(top)
    rc = backtest.run(df, council_pos, args.fee, long_only=not args.shorts)
    table.loc["COUNCIL(top%d)" % args.top_k] = {"is_sharpe": backtest.metrics(rc[rc.index < split])["sharpe"],
                                                 **backtest.metrics(rc[rc.index >= split])}
    bh = df.close.pct_change().fillna(0)
    table.loc["BUY&HOLD"] = {"is_sharpe": backtest.metrics(bh[bh.index < split])["sharpe"],
                             **backtest.metrics(bh[bh.index >= split])}

    lo = -1 if args.shorts else 0
    today = {n: max(lo, float(p.iloc[-1])) for n, p in positions.items()}
    today["COUNCIL"] = max(lo, float(council_pos.iloc[-1]))
    return table.sort_values("sharpe", ascending=False), today, list(top), source, split, df.index[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    p.add_argument("--fee", type=float, default=0.001, help="per-side cost incl. slippage")
    p.add_argument("--oos", type=float, default=0.3, help="fraction of history held out for testing")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--shorts", action="store_true", help="allow short positions")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    all_tables, signals = [], {}
    for sym in args.symbols:
        table, today, top, source, split, last = evaluate(sym, args)
        print(f"\n=== {sym}  [{source}]  out-of-sample from {split.date()} to {last.date()} ===")
        print(f"Council members (picked in-sample): {', '.join(top)}")
        print(table.round(2).to_string())
        table.insert(0, "symbol", sym)
        all_tables.append(table)
        signals[sym] = today

    combined = pd.concat(all_tables)
    combined.to_csv("results/leaderboard.csv")
    sig = pd.DataFrame(signals).round(2)
    sig.to_csv("results/signals_today.csv")

    ais_only = combined[combined.index.str.startswith("ai")]
    print("\n=== Cross-asset ranking (mean out-of-sample Sharpe) ===")
    agg = ais_only.groupby(level=0)[["sharpe", "cagr_%", "max_dd_%"]].mean()
    agg["beats_bh"] = [
        sum(combined[(combined.index == n) & (combined.symbol == s)].sharpe.iloc[0] >
            combined[(combined.index == "BUY&HOLD") & (combined.symbol == s)].sharpe.iloc[0]
            for s in args.symbols)
        for n in agg.index]
    print(agg.sort_values("sharpe", ascending=False).round(2).to_string())

    print("\n=== Target positions for next bar (1=long, 0=flat, -1=short) ===")
    print(sig.to_string())
    print("\nSaved results/leaderboard.csv and results/signals_today.csv")


if __name__ == "__main__":
    main()
