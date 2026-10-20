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
