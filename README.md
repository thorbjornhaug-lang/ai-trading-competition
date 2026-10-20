# 20 Trading AIs

20 strategies (trend, mean reversion, breakout, volatility, ML, Kalman) plus a **Council** ensemble.
They are all backtested on crypto from Binance, with fees and honest out-of-sample testing.

```bash
pip install pandas numpy
python run.py                                   # BTC, ETH, SOL, XRP (daily, since 2019)
python run.py --symbols BTCUSDT SOLUSDT --shorts
python run.py --offline                         # synthetic data, no network
```

Output: per-asset leaderboard, cross-asset ranking, the target position for the next bar, and `results/*.csv`.

**How to read it:** strategies are ranked on the last 30% of history, which they never saw during selection.
The Council picks its top 5 using only in-sample data, so its out-of-sample result is the most honest number here.
If an AI doesn't beat BUY&HOLD on Sharpe on at least 3 of 4 assets, delete it.

Paper-trade only until 3+ months of live signals match the backtest.

## 🏆 Paper-trading competition (`paper.py`)

20 AIs, $500 virtual each, trading 4 cryptos (BTC, ETH, SOL, XRP) and 6 stocks/ETFs (SPY, QQQ, NVDA, AAPL, MSFT, TSLA).
Each AI trades only its own strategy. Long only, max 25% per asset before leverage, real fees, and a signal is traded on the next bar (no look-ahead).

**Leverage up to 10x**, set by each AI's risk profile (5 AIs each): Forsiktig 1x · Moderat ≤3x (targets 70% vol) · Aggressiv ≤10x (targets 150% vol) · Degen 10x (always max). Borrowed money costs 8%/yr. An account is liquidated and out of the game when equity at the day's lows drops to 5% of exposure.

**Dashboard:** `docs/index.html` is rebuilt on every run. It shows the equity race, the leverage-vs-return comparison, the leaderboard, per-AI detail and every trade. Turn on GitHub Pages (Settings → Pages → branch `main`, folder `/docs`) to get it at a live URL.

```bash
python paper.py --reset            # start the competition today
python paper.py                    # run daily: trades new bars, updates LEADERBOARD.md
python paper.py --reset --start 2026-01-02   # replay from a past date, then continue live
```

State lives in `state/` (accounts, equity curve, every trade). Push this repo to GitHub and
`.github/workflows/daily.yml` runs it automatically every day after the US close and commits the leaderboard and dashboard.
