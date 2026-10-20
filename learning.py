"""Self-learning layer on top of each AI's base strategy.

Every AI owns a rulebook it rewrites itself:
- tf:       how often it trades — reads its strategy on 5-minute up to daily bars
- lev_mult: how much of its allowed leverage it actually uses
- confirm:  extra bars a signal must hold before it buys (fewer whipsaws)
- stop:     trailing stop-loss from the peak since entry (None = no stop)
- bans:     assets it refuses to trade for a while because they cost it money

Every LEARN_EVERY_DAYS each AI replays its own strategy over the last WINDOW_DAYS (a "shadow" run
with the same sizing, leverage, fees, interest and liquidation rules, step by step), pauses assets that
lost it money, and tests alternative rulebooks. It switches only if the new rules would have done clearly better.
"""
import itertools

import numpy as np
import pandas as pd

MIN_SPLIT = 4          # base weight = signal / max(sum of signals, 4)  ->  max 25% per asset before leverage
STOCK_MAX_LEV = 1.0    # leverage is for crypto only
BORROW_RATE = 0.08
MAINT_MARGIN = 0.05
MINUTES_PER_YEAR = 525600

TIMEFRAMES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1D": 1440}   # bar size in minutes
TF_NAMES = {"5m": "hvert 5. minutt", "15m": "hvert 15. minutt", "1h": "hver time",
            "4h": "hver 4. time", "1D": "én gang om dagen"}
LEARN_EVERY_DAYS = 7
WINDOW_DAYS = 120
BAN_WINDOW_DAYS = 60
BAN_DAYS = 28
BAN_LOSS = -0.05       # pause an asset that cost more than 5% of the account over the ban window
MAX_BANS = 3
MIN_IMPROVEMENT = 0.02
LEV_MULTS = (0.25, 0.5, 1.0, 1.5)
CONFIRMS = (0, 2)
STOPS = (None, 0.08, 0.15)

DEFAULT_RULES = {"tf": "1D", "lev_mult": 1.0, "confirm": 0, "stop": None, "bans": {}}


def stamp(ts):
    return f"{pd.Timestamp(ts):%Y-%m-%d %H:%M}"


class Context:
    """Numpy views of the market, indexed [step, asset] on the shared calendar of step_minutes bars."""

    def __init__(self, tickers, signals, close, low, vol, fresh, is_crypto, fees, step_minutes=60,
                 window_days=WINDOW_DAYS):
        self.tickers = list(tickers)
        self.step_minutes = step_minutes
        self.window_days = window_days
        self.tfs = sorted({tf for (_, tf) in signals}, key=TIMEFRAMES.get)
        self.borrow_per_step = BORROW_RATE * step_minutes / MINUTES_PER_YEAR
        arrays = {}  # the same frame used for several tempos (news AIs) shares one array
        self.sig = {key: arrays.setdefault(id(s), s[self.tickers].to_numpy(np.float32)) for key, s in signals.items()}
        c = close[self.tickers].to_numpy(float)
        prev = np.vstack([c[:1], c[:-1]])
        self.close = c
        self.ret = np.nan_to_num(c / prev - 1)
        self.lowret = np.nan_to_num(low[self.tickers].to_numpy(float) / prev - 1)
        self.vol = np.nan_to_num(vol[self.tickers].to_numpy(float), nan=1.0)
        self.fresh = fresh[self.tickers].to_numpy(bool)
        self.is_crypto = np.asarray(is_crypto, bool)
        self.fees = np.asarray(fees, float)


def effective_signal(ctx, ai, i, rules, banned, stopped):
    """Signal decided at step i, after the AI's own rules (timeframe, confirmation, bans, stops)."""
    s = ctx.sig[(ai, rules["tf"])]
    c = rules["confirm"] * TIMEFRAMES[rules["tf"]] // ctx.step_minutes
    sig = s[max(0, i - c):i + 1].min(axis=0).astype(float) if c else s[i].astype(float)
    sig[banned | stopped] = 0.0
    return sig


def target_weights(ctx, i, sig, max_lev, target_vol, lev_mult):
    """Target exposure per asset as a fraction of equity, and the leverage chosen."""
    w0 = sig / max(sig.sum(), MIN_SPLIT)
    pvol = float((w0 * ctx.vol[i]).sum())
    lev = max_lev if target_vol is None or pvol <= 0 else min(max_lev, target_vol / pvol)
    lev = min(max_lev, lev * lev_mult)
    return w0 * np.where(ctx.is_crypto, lev, min(lev, STOCK_MAX_LEV)), lev


def banned_mask(ctx, bans, now):
    return np.array([t in bans and bans[t] > now for t in ctx.tickers])


def shadow(ctx, ai, rules, banned, start, end, max_lev, target_vol):
    """Replay rules over steps [start, end). Returns (score, per-asset P&L as fraction of equity)."""
    n = len(ctx.tickers)
    w = np.zeros(n)
    peaks = np.full(n, np.nan)
    stopped = np.zeros(n, bool)
    eq, top, max_dd = 1.0, 1.0, 0.0
    asset_pnl = np.zeros(n)
    raw = ctx.sig[(ai, rules["tf"])]
    for i in range(max(start, 1), end):
        gross = w.sum()
        if gross > 0:
            low_r = ctx.lowret[i]
            if 1 + (w * low_r).sum() <= MAINT_MARGIN * (w * (1 + low_r)).sum():
                return -10.0, asset_pnl - 1.0 / n
            pnl = w * ctx.ret[i]
            asset_pnl += pnl * eq
            eq *= 1 + pnl.sum() - ctx.borrow_per_step * max(gross - 1, 0)
            top = max(top, eq)
            max_dd = max(max_dd, 1 - eq / top)
            if eq <= 0:
                return -10.0, asset_pnl
            if rules["stop"]:
                held = w > 0
                peaks = np.where(held, np.fmax(peaks, ctx.close[i]), np.nan)
                stopped |= held & (ctx.close[i] < peaks * (1 - rules["stop"]))
        stopped &= raw[i - 1] > 0
        sig = effective_signal(ctx, ai, i - 1, rules, banned, stopped)
        new_w, _ = target_weights(ctx, i - 1, sig, max_lev, target_vol, rules["lev_mult"])
        new_w = np.where(ctx.fresh[i], new_w, w)
        eq *= 1 - (ctx.fees * np.abs(new_w - w)).sum()
        w = new_w
    return float(np.log(max(eq, 1e-9)) - max_dd), asset_pnl


def describe(old, new, max_lev):
    out = []
    if new["tf"] != old["tf"]:
        out.append(f"Bytter tempo: handler nå {TF_NAMES[new['tf']]} (før {TF_NAMES[old['tf']]})")
    if new["lev_mult"] != old["lev_mult"] and max_lev > 1:
        more = new["lev_mult"] > old["lev_mult"]
        out.append(f"{'Øker' if more else 'Kutter'} giringen: bruker nå {new['lev_mult']:.0%} av giringen sin "
                   f"(før {old['lev_mult']:.0%}), maks {min(max_lev, max_lev * new['lev_mult']):.1f}x på krypto")
    elif new["lev_mult"] != old["lev_mult"]:
        out.append(f"Investerer nå {new['lev_mult']:.0%} av det signalene sier (før {old['lev_mult']:.0%})")
    if new["confirm"] != old["confirm"]:
        out.append(f"Venter nå {new['confirm']} ekstra candles på bekreftet signal før kjøp" if new["confirm"]
                   else "Kjøper nå med en gang signalet kommer")
    if new["stop"] != old["stop"]:
        out.append(f"Ny stop-loss: selger hvis kursen faller {new['stop']:.0%} fra toppen" if new["stop"]
                   else "Fjerner stop-loss")
    return out


def learn(ctx, ai, rules, i, now, max_lev, target_vol):
    """One learning session at step i (uses data up to step i-1). Returns (new rules, journal lines, scores)."""
    per_day = 1440 // ctx.step_minutes
    start = max(1, i - ctx.window_days * per_day)
    ban_days = min(BAN_WINDOW_DAYS, ctx.window_days)
    notes = []
    bans = {t: until for t, until in rules["bans"].items() if until > now}
    for t in rules["bans"]:
        if t not in bans:
            notes.append(f"Ferdig med pausen fra {t}, prøver igjen")

    current = dict(rules, bans=bans)
    banned = banned_mask(ctx, bans, now)
    before, _ = shadow(ctx, ai, current, banned, start, i, max_lev, target_vol)

    # learn from assets that lost it money recently
    _, pnl = shadow(ctx, ai, current, banned, max(1, i - ban_days * per_day), i, max_lev, target_vol)
    until = stamp(pd.Timestamp(now) + pd.Timedelta(days=BAN_DAYS))
    for k in np.argsort(pnl):
        t = ctx.tickers[k]
        if len(bans) >= MAX_BANS or pnl[k] >= BAN_LOSS:
            break
        if t not in bans:
            bans[t] = until
            notes.append(f"Tar pause fra {t} i {BAN_DAYS} dager (kostet {-pnl[k]:.1%} av kontoen "
                         f"de siste {ban_days} dagene)")
    banned = banned_mask(ctx, bans, now)
    current = dict(current, bans=bans)
    base, _ = shadow(ctx, ai, current, banned, start, i, max_lev, target_vol)

    # stage 1: pick the tempo; stage 2: tune leverage, confirmation and stop-loss for that tempo
    best, best_score = current, base
    for tf in ctx.tfs:
        cand = dict(current, tf=tf)
        if cand != current and ctx.sig[(ai, tf)] is not ctx.sig[(ai, current["tf"])]:
            score, _ = shadow(ctx, ai, cand, banned, start, i, max_lev, target_vol)
            if score > best_score:
                best, best_score = cand, score
    mults = sorted({m for m in LEV_MULTS if m <= 1.0 or max_lev > 1})
    stage1 = best
    for m, c, s in itertools.product(mults, CONFIRMS, STOPS):
        cand = dict(stage1, lev_mult=m, confirm=c, stop=s)
        if cand == stage1:
            continue
        score, _ = shadow(ctx, ai, cand, banned, start, i, max_lev, target_vol)
        if score > best_score:
            best, best_score = cand, score
    if best_score < base + MIN_IMPROVEMENT or best["tf"] not in ctx.tfs:  # only switch when clearly better
        best, best_score = current, base
    notes += describe(current, best, max_lev)
    return best, notes, before, best_score
