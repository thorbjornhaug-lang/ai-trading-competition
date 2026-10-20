"""The 20 trading AIs.

Each AI takes an OHLCV DataFrame and returns a target position per bar in [-1, 1]
(1 = fully long, 0 = flat, -1 = fully short). Positions must only use information
available at the close of that bar; the backtester executes them on the next bar.
"""
import numpy as np
import pandas as pd

# ---------- indicator helpers ----------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def hold(enter, exit_, short_enter=None, short_exit=None):
    """Turn entry/exit boolean events into a persistent position (state machine)."""
    pos, out = 0.0, []
    se = short_enter if short_enter is not None else pd.Series(False, index=enter.index)
    sx = short_exit if short_exit is not None else pd.Series(False, index=enter.index)
    for e, x, s, c in zip(enter.values, exit_.values, se.values, sx.values):
        if pos == 0:
            pos = 1.0 if e else (-1.0 if s else 0.0)
        elif pos > 0 and x:
            pos = -1.0 if s else 0.0
        elif pos < 0 and c:
            pos = 1.0 if e else 0.0
        out.append(pos)
    return pd.Series(out, index=enter.index)


# ---------- the 20 AIs ----------

def ai01_sma_cross(df):
    """Trend: 20/50 SMA golden cross."""
    c = df.close
    return np.sign(c.rolling(20).mean() - c.rolling(50).mean())


def ai02_ema_cross(df):
    """Trend: 12/26 EMA cross."""
    return np.sign(ema(df.close, 12) - ema(df.close, 26))


def ai03_macd(df):
    """Trend: MACD line vs signal line."""
    m = ema(df.close, 12) - ema(df.close, 26)
    return np.sign(m - ema(m, 9))


def ai04_rsi_reversion(df):
    """Mean reversion: buy RSI<30, sell RSI>55."""
    r = rsi(df.close)
    return hold(r < 30, r > 55)


def ai05_bollinger_reversion(df):
    """Mean reversion: buy below lower band, exit at middle band."""
    c = df.close
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    return hold(c < mid - 2 * sd, c > mid)


def ai06_bollinger_breakout(df):
    """Breakout: long above upper band, exit below middle."""
    c = df.close
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    return hold(c > mid + 2 * sd, c < mid, c < mid - 2 * sd, c > mid)


def ai07_donchian_turtle(df):
    """Turtle: 20-day high breakout, exit on 10-day low."""
    hi, lo = df.high.rolling(20).max().shift(), df.low.rolling(10).min().shift()
    return hold(df.close > hi, df.close < lo)


def ai08_tsmom(df):
    """Time-series momentum: sign of 90-day return."""
    return np.sign(df.close.pct_change(90))


def ai09_dual_momentum(df):
    """Long only when both 30d and 90d momentum are positive."""
    c = df.close
    return ((c.pct_change(30) > 0) & (c.pct_change(90) > 0)).astype(float)


def ai10_vol_target_trend(df):
    """Above 200 SMA, sized to target 40% annualised volatility."""
    c = df.close
    vol = c.pct_change().rolling(30).std() * np.sqrt(365)
    size = (0.40 / vol).clip(upper=1.0)
    return (c > c.rolling(200).mean()).astype(float) * size


def ai11_keltner(df):
    """Breakout above EMA20 + 2*ATR, exit below EMA20."""
    mid, a = ema(df.close, 20), atr(df, 20)
    return hold(df.close > mid + 2 * a, df.close < mid)


def ai12_stochastic(df):
    """Stochastic %K crosses up from oversold, exit overbought."""
    lo, hi = df.low.rolling(14).min(), df.high.rolling(14).max()
    k = 100 * (df.close - lo) / (hi - lo)
    d = k.rolling(3).mean()
    return hold((k > d) & (k.shift() <= d.shift()) & (k < 25), k > 80)


def ai13_zscore(df):
    """Long/short on 50-day z-score extremes (only in sideways markets), exit at mean."""
    c = df.close
    z = (c - c.rolling(50).mean()) / c.rolling(50).std()
    sideways = c.pct_change(50).abs() < 0.15
    return hold((z < -1.5) & sideways, z > 0, (z > 1.5) & sideways, z < 0)


def ai14_adx_trend(df):
    """Trade DI direction only when ADX > 25 (strong trend)."""
    up, dn = df.high.diff(), -df.low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, 14)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    return np.sign(pdi - mdi) * (adx > 25)


def ai15_supertrend(df, n=10, mult=3.0):
    """Supertrend (ATR trailing stop) direction."""
    hl2, a = (df.high + df.low) / 2, atr(df, n)
    ub, lb = (hl2 + mult * a).values, (hl2 - mult * a).values
    c = df.close.values
    direction = np.zeros(len(c))
    fu, fl, d = ub[0], lb[0], 1
    for i in range(1, len(c)):
        fu = ub[i] if (ub[i] < fu or c[i - 1] > fu) else fu
        fl = lb[i] if (lb[i] > fl or c[i - 1] < fl) else fl
        if c[i] > fu:
            d = 1
        elif c[i] < fl:
            d = -1
        direction[i] = d
    return pd.Series(direction, index=df.index)


def ai16_ichimoku(df):
    """Price above the cloud and Tenkan > Kijun."""
    mid = lambda n: (df.high.rolling(n).max() + df.low.rolling(n).min()) / 2
    tenkan, kijun = mid(9), mid(26)
    span_a, span_b = ((tenkan + kijun) / 2).shift(26), mid(52).shift(26)
    top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    return ((df.close > top) & (tenkan > kijun)).astype(float)


def ai17_obv_breakout(df):
    """Long when price and On-Balance-Volume both make 30-day highs."""
    obv = (np.sign(df.close.diff()).fillna(0) * df.volume).cumsum()
    enter = (df.close >= df.close.rolling(30).max()) & (obv >= obv.rolling(30).max())
    return hold(enter, df.close < ema(df.close, 30))


def ai18_vol_breakout(df):
    """Larry Williams-style: close beyond prior close + 0.5 * prior range."""
    rng = (df.high - df.low).shift()
    up = df.close > df.close.shift() + 0.5 * rng
    return hold(up, df.close < ema(df.close, 10))


def ai19_ml_ridge(df, train=365, retrain=30):
    """ML: walk-forward ridge regression on lagged features predicting next-day return."""
    c = df.close
    r = c.pct_change()
    feats = pd.DataFrame({
        "r1": r, "r5": c.pct_change(5), "r20": c.pct_change(20),
        "vol": r.rolling(20).std(), "rsi": rsi(c) / 100 - 0.5,
        "sma": c / c.rolling(50).mean() - 1,
    })
    y = r.shift(-1)
    X = feats.values
    pos = np.zeros(len(df))
    w = None
    for i in range(train + 60, len(df)):
        if w is None or (i % retrain == 0):
            Xt, yt = X[i - train:i - 1], y.values[i - train:i - 1]  # y[i-1] is r[i], known at i
            ok = ~np.isnan(Xt).any(axis=1) & ~np.isnan(yt)
            Xt, yt = Xt[ok], yt[ok]
            mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
            Z = np.c_[np.ones(len(Xt)), (Xt - mu) / sd]
            w = np.linalg.solve(Z.T @ Z + 10 * np.eye(Z.shape[1]), Z.T @ yt)
            scale = np.std(Z @ w) + 1e-12
        if np.isnan(X[i]).any():
            continue
        pred = np.r_[1, (X[i] - mu) / sd] @ w
        pos[i] = np.clip(pred / (2 * scale), -1, 1)
    return pd.Series(pos, index=df.index)


def ai20_kalman_trend(df, q=1e-5, rv=1e-3):
    """Kalman filter on log price (level + slope); trade the sign of the estimated slope."""
    z = np.log(df.close.values)
    x, P = np.array([z[0], 0.0]), np.eye(2)
    F, H, Q = np.array([[1, 1], [0, 1]]), np.array([1.0, 0.0]), q * np.eye(2)
    slope = np.zeros(len(z))
    for i, zi in enumerate(z):
        x, P = F @ x, F @ P @ F.T + Q
        k = P @ H / (H @ P @ H + rv)
        x, P = x + k * (zi - H @ x), (np.eye(2) - np.outer(k, H)) @ P
        slope[i] = x[1]
    s = pd.Series(slope, index=df.index)
    return np.sign(s) * (s.abs() > s.abs().rolling(100).median())


AIS = {f.__name__: f for f in [
    ai01_sma_cross, ai02_ema_cross, ai03_macd, ai04_rsi_reversion, ai05_bollinger_reversion,
    ai06_bollinger_breakout, ai07_donchian_turtle, ai08_tsmom, ai09_dual_momentum, ai10_vol_target_trend,
    ai11_keltner, ai12_stochastic, ai13_zscore, ai14_adx_trend, ai15_supertrend,
    ai16_ichimoku, ai17_obv_breakout, ai18_vol_breakout, ai19_ml_ridge, ai20_kalman_trend,
]}
