"""Self-learning layer on top of each AI's base strategy.

Every AI owns a rulebook it rewrites itself:
- lev_mult: how much of its allowed leverage it actually uses
- confirm:  extra days a signal must hold before it buys (fewer whipsaws)
- stop:     trailing stop-loss from the peak since entry (None = no stop)
- bans:     assets it refuses to trade for a while because they cost it money

Every LEARN_EVERY days each AI replays its own strategy over the last WINDOW days (a "shadow" run
with the same sizing, leverage, fees, interest and liquidation rules), bans assets that lost it money,
and tests alternative rulebooks. It switches only if the new rules would have done clearly better.
"""
import itertools

import numpy as np

MIN_SPLIT = 4          # base weight = signal / max(sum of signals, 4)  ->  max 25% per asset before leverage
STOCK_MAX_LEV = 1.0    # leverage is for crypto only
BORROW_RATE = 0.08
MAINT_MARGIN = 0.05

LEARN_EVERY = 7
WINDOW = 120
BAN_WINDOW = 60
BAN_DAYS = 28
BAN_LOSS = -0.05       # ban an asset that cost more than 5% of the account over BAN_WINDOW
MAX_BANS = 3
MIN_IMPROVEMENT = 0.02
LEV_MULTS = (0.25, 0.5, 1.0, 1.5)
CONFIRMS = (0, 2)
STOPS = (None, 0.08, 0.15)

DEFAULT_RULES = {"lev_mult": 1.0, "confirm": 0, "stop": None, "bans": {}}


class Context:
    """Numpy views of the market, indexed [day, asset] on the shared calendar."""

    def __init__(self, tickers, signals, close, low, vol, fresh, is_crypto, fees):
        self.tickers = list(tickers)
        self.sig = {ai: s[self.tickers].to_numpy(float) for ai, s in signals.items()}
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
    """Signal decided at bar i, after the AI's own rules (confirmation, bans, stops)."""
    s = ctx.sig[ai]
    c = rules["confirm"]
    sig = s[max(0, i - c):i + 1].min(axis=0) if c else s[i].copy()
    sig[banned | stopped] = 0.0
    return sig


def target_weights(ctx, i, sig, max_lev, target_vol, lev_mult):
    """Target exposure per asset as a fraction of equity, and the leverage chosen."""
    w0 = sig / max(sig.sum(), MIN_SPLIT)
    pvol = float((w0 * ctx.vol[i]).sum())
    lev = max_lev if target_vol is None or pvol <= 0 else min(max_lev, target_vol / pvol)
    lev = min(max_lev, lev * lev_mult)
    return w0 * np.where(ctx.is_crypto, lev, min(lev, STOCK_MAX_LEV)), lev


def banned_mask(ctx, bans, date):
    return np.array([t in bans and bans[t] > date for t in ctx.tickers])


def shadow(ctx, ai, rules, banned, start, end, max_lev, target_vol):
    """Replay rules over days [start, end). Returns (score, per-asset P&L as fraction of equity)."""
    n = len(ctx.tickers)
    w = np.zeros(n)
    peaks = np.full(n, np.nan)
    stopped = np.zeros(n, bool)
    eq, top, max_dd = 1.0, 1.0, 0.0
    asset_pnl = np.zeros(n)
    for i in range(start, end):
        gross = w.sum()
        if gross > 0:
            low_r = ctx.lowret[i]
            if 1 + (w * low_r).sum() <= MAINT_MARGIN * (w * (1 + low_r)).sum():
                return -10.0, asset_pnl - 1.0 / n
            pnl = w * ctx.ret[i]
            day = pnl.sum() - BORROW_RATE / 365 * max(gross - 1, 0)
            asset_pnl += pnl * eq
            eq *= 1 + day
            top = max(top, eq)
            max_dd = max(max_dd, 1 - eq / top)
            if eq <= 0:
                return -10.0, asset_pnl
        if rules["stop"]:
            held = w > 0
            peaks = np.where(held, np.fmax(peaks, ctx.close[i]), np.nan)
            stopped |= held & (ctx.close[i] < peaks * (1 - rules["stop"]))
        if i == 0:
            continue
        stopped &= ctx.sig[ai][i - 1] > 0
        sig = effective_signal(ctx, ai, i - 1, rules, banned, stopped)
        new_w, _ = target_weights(ctx, i - 1, sig, max_lev, target_vol, rules["lev_mult"])
        new_w = np.where(ctx.fresh[i], new_w, w)
        eq *= 1 - (ctx.fees * np.abs(new_w - w)).sum()
        w = new_w
    return float(np.log(max(eq, 1e-9)) - max_dd), asset_pnl


def describe(old, new, max_lev):
    out = []
    if new["lev_mult"] != old["lev_mult"] and max_lev > 1:
        more = new["lev_mult"] > old["lev_mult"]
        out.append(f"{'Øker' if more else 'Kutter'} giringen: bruker nå {new['lev_mult']:.0%} av giringen sin "
                   f"(før {old['lev_mult']:.0%}), maks {min(max_lev, max_lev * new['lev_mult']):.1f}x på krypto")
    elif new["lev_mult"] != old["lev_mult"]:
        out.append(f"Investerer nå {new['lev_mult']:.0%} av det signalene sier (før {old['lev_mult']:.0%})")
    if new["confirm"] != old["confirm"]:
        out.append(f"Venter nå {new['confirm']} ekstra dager på bekreftet signal før kjøp" if new["confirm"]
                   else "Kjøper nå med en gang signalet kommer")
    if new["stop"] != old["stop"]:
        out.append(f"Ny stop-loss: selger hvis kursen faller {new['stop']:.0%} fra toppen" if new["stop"]
                   else "Fjerner stop-loss")
    return out


def learn(ctx, ai, rules, i, date, max_lev, target_vol):
    """One learning session at bar i (uses data up to bar i-1). Returns (new rules, journal lines, scores)."""
    start = max(1, i - WINDOW)
    notes = []
    bans = {t: until for t, until in rules["bans"].items() if until > date}
    for t in rules["bans"]:
        if t not in bans:
            notes.append(f"Ferdig med pausen fra {t}, prøver igjen")

    current = dict(rules, bans=bans)
    banned = banned_mask(ctx, bans, date)
    before, _ = shadow(ctx, ai, current, banned, start, i, max_lev, target_vol)

    # learn from losing assets in the recent past
    _, pnl = shadow(ctx, ai, current, banned, max(1, i - BAN_WINDOW), i, max_lev, target_vol)
    until = str((np.datetime64(date) + np.timedelta64(BAN_DAYS, "D")))
    for k in np.argsort(pnl):
        t = ctx.tickers[k]
        if len(bans) >= MAX_BANS or pnl[k] >= BAN_LOSS:
            break
        if t not in bans:
            bans[t] = until
            notes.append(f"Slutter å handle {t} i {BAN_DAYS} dager (kostet {-pnl[k]:.1%} av kontoen "
                         f"de siste {BAN_WINDOW} dagene)")
    banned = banned_mask(ctx, bans, date)
    current = dict(current, bans=bans)
    base, _ = shadow(ctx, ai, current, banned, start, i, max_lev, target_vol)

    # try alternative rulebooks
    mults = sorted({m for m in LEV_MULTS if m <= 1.0 or max_lev > 1})
    best, best_score = current, base
    for m, c, s in itertools.product(mults, CONFIRMS, STOPS):
        cand = dict(current, lev_mult=m, confirm=c, stop=s)
        if cand == current:
            continue
        score, _ = shadow(ctx, ai, cand, banned, start, i, max_lev, target_vol)
        if score > best_score:
            best, best_score = cand, score
    if best_score < base + MIN_IMPROVEMENT:  # only switch when clearly better
        best, best_score = current, base
    notes += describe(current, best, max_lev)
    return best, notes, before, best_score
