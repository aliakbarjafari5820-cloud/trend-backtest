"""کتابخونهٔ استراتژی‌های اسکالپ + موتور اجرا (یکسان برای همه).
هر استراتژی: سیگنال خرید/فروش روی کلوز کندل بسته‌شده → ورود روی اپن کندل بعد.
خروج: حد ضرر و حد سود بر اساس ATR(14) لحظهٔ سیگنال، خروج سفارشی (روی کلوز → اپن بعد)، یا سقف زمانی.
هزینه‌ها: ورود و حد سود با سفارش لیمیت (Maker ۰.۰۲٪)، حد ضرر/خروج زمانی/سفارشی با مارکت (Taker ۰.۰۵٪ + اسلیپیج ۰.۰۲٪)،
فاندینگ ۰.۰۱٪ هر ۸ ساعت به ضرر معامله."""
import numpy as np
import pandas as pd

MAKER, TAKER, SLIP, FUND8H = 0.0002, 0.0005, 0.0002, 0.0001
try:
    from numba import njit
except Exception:                         # بدون numba هم کار می‌کنه (کندتر)
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


# ---------------- اندیکاتورها (همه فقط از گذشته) ----------------
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values
def sma(x, n): return pd.Series(x).rolling(n).mean().values

def rsi(c, n):
    d = np.diff(c, prepend=c[0]); up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1/n, adjust=False).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1/n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).values

def atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    return pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(n).mean().values

def bbands(c, n=20, k=2.0):
    m = pd.Series(c).rolling(n).mean(); s = pd.Series(c).rolling(n).std()
    return (m - k * s).values, m.values, (m + k * s).values

def stoch(h, l, c, n=14, d=3):
    ll = pd.Series(l).rolling(n).min(); hh = pd.Series(h).rolling(n).max()
    k = (100 * (pd.Series(c) - ll) / (hh - ll)).rolling(d).mean()
    return k.values, k.rolling(d).mean().values

def macd(c, f=12, s=26, g=9):
    m = ema(c, f) - ema(c, s); return m, ema(m, g)

def vwap_day(t, h, l, c, v):
    day = (t // 86_400_000).astype(np.int64); tp = (h + l + c) / 3
    df = pd.DataFrame(dict(day=day, pv=tp * v, v=v, pv2=tp * tp * v))
    g = df.groupby("day"); cv = g.v.cumsum().replace(0, np.nan)
    vw = g.pv.cumsum() / cv; var = (g.pv2.cumsum() / cv - vw ** 2).clip(lower=0)
    return vw.values, np.sqrt(var).values

def supertrend(h, l, c, n=10, m=3.0):
    a = atr(h, l, c, n); hl2 = (h + l) / 2; up, dn = hl2 - m * a, hl2 + m * a
    dirn = np.ones(len(c)); fu, fd = up.copy(), dn.copy()
    for i in range(1, len(c)):
        if np.isnan(a[i]): continue
        fu[i] = max(up[i], fu[i-1]) if c[i-1] > fu[i-1] else up[i]
        fd[i] = min(dn[i], fd[i-1]) if c[i-1] < fd[i-1] else dn[i]
        dirn[i] = 1 if c[i] > fd[i-1] else (-1 if c[i] < fu[i-1] else dirn[i-1])
    return dirn

def htf_trend(df, minutes=240):
    """روند ۴ ساعته برای هر کندل، فقط از کندل‌های ۴ ساعتهٔ بسته‌شده: ۱ = کلوز بالای ابر ایچیموکو و EMA50 > EMA200، ۱- برعکس."""
    t = df["t"].values.astype(np.int64); step = int(np.median(np.diff(t))) if len(t) > 1 else 300_000
    g = df.groupby((df.t // (minutes * 60_000)).astype(np.int64))
    H = pd.DataFrame(dict(t=g.t.first(), h=g.h.max(), l=g.l.min(), c=g.c.last(), last=g.t.last())).reset_index(drop=True)
    Hs, Ls, cc = pd.Series(H.h), pd.Series(H.l), H.c.values
    ten = ((Hs.rolling(9).max() + Ls.rolling(9).min()) / 2).values; kij = ((Hs.rolling(26).max() + Ls.rolling(26).min()) / 2).values
    sa = pd.Series((ten + kij) / 2).shift(26).values; sb = ((Hs.rolling(52).max() + Ls.rolling(52).min()) / 2).shift(26).values
    e50, e200 = ema(cc, 50), ema(cc, 200)
    reg = np.where((cc > np.fmax(sa, sb)) & (e50 > e200), 1, np.where((cc < np.fmin(sa, sb)) & (e50 < e200), -1, 0))
    full = ((H["last"] + step) >= (H.t + minutes * 60_000)).values
    reg = np.where(full, reg, 0)
    idx = np.searchsorted(H.t.values + minutes * 60_000, t + step, side="right") - 1
    return np.where(idx >= 0, reg[np.clip(idx, 0, None)], 0)


def prev(x):
    """مقدار کندل قبل (برای کندل اول، خودش) — بدون چرخش آرایه از انتها."""
    x = np.asarray(x); return np.r_[x[:1], x[:-1]]
def cross_up(a, b): return (a > b) & (prev(a) <= prev(b))
def cross_dn(a, b): return (a < b) & (prev(a) >= prev(b))


# ---------------- موتور خروج ----------------
@njit(cache=True)
def _sim(o, h, l, c, t, A, le, se, xl, xs, sl_m, tp_m, max_bars, eod):
    n = len(c); out = []; i = 1
    while i < n - 2:
        d = 1 if le[i] else (-1 if se[i] else 0)
        if d == 0 or not (A[i] > 0):
            i += 1; continue
        k = i + 1; e = o[k]; risk = sl_m * A[i]
        sl = e - d * risk; tp = e + d * tp_m * A[i] if tp_m > 0 else np.nan
        xi = -1; xp = 0.0; kind = 0                   # kind: 0=SL 1=TP 2=سفارشی/زمانی
        for j in range(k, n):
            if j > k and ((d == 1 and o[j] <= sl) or (d == -1 and o[j] >= sl)):
                xi, xp, kind = j, o[j], 0; break
            if (d == 1 and l[j] <= sl) or (d == -1 and h[j] >= sl):
                xi, xp, kind = j, sl, 0; break
            if tp_m > 0 and ((d == 1 and h[j] >= tp) or (d == -1 and l[j] <= tp)):
                xi, xp, kind = j, tp, 1; break
            if j + 1 < n:
                cust = (d == 1 and xl[j]) or (d == -1 and xs[j])
                timeup = (j - k + 1) >= max_bars
                dayend = eod and (t[j + 1] // 86_400_000) != (t[k] // 86_400_000)
                if cust or timeup or dayend:
                    xi, xp, kind = j + 1, o[j + 1], 2; break
        if xi < 0: break
        out.append((k, xi, d, e, xp, risk, kind))
        i = xi                                         # یک معامله در لحظه
    return out


def simulate(df, le, se, sl=1.5, tp=3.0, max_bars=48, xl=None, xs=None, eod=False):
    o, h, l, c, t = [df[x].values.astype(np.float64) for x in "ohlct"]
    A = atr(h, l, c); n = len(c)
    z = np.zeros(n, dtype=np.bool_)
    le = np.nan_to_num(le).astype(np.bool_); se = np.nan_to_num(se).astype(np.bool_)
    xl = z if xl is None else np.nan_to_num(xl).astype(np.bool_); xs = z if xs is None else np.nan_to_num(xs).astype(np.bool_)
    rows = _sim(o, h, l, c, t.astype(np.int64), np.nan_to_num(A), le, se, xl, xs, float(sl), float(tp if tp else 0), int(max_bars), bool(eod))
    if not rows: return pd.DataFrame()
    R = pd.DataFrame(rows, columns=["k", "xi", "side", "entry", "exit", "risk", "kind"])
    R["entry_t"] = t[R.k.values].astype(np.int64); R["exit_t"] = t[R.xi.values].astype(np.int64)
    stop_pct = R.risk / R.entry
    exit_cost = np.where(R.kind == 1, MAKER, TAKER + SLIP)
    hours = (R.exit_t - R.entry_t) / 3.6e6
    R["R_gross"] = (R.exit - R.entry) * R.side / R.risk
    R["cost_R"] = (MAKER + exit_cost + FUND8H * hours / 8) / stop_pct
    R["R_net"] = R.R_gross - R.cost_R
    R["stop_pct"] = stop_pct; R["hours"] = hours
    return R.drop(columns=["k", "xi"])


# ---------------- استراتژی‌ها ----------------
def strategies(df):
    """هر مورد: (نام, تابع سازندهٔ سیگنال‌ها). پارامترها همون مقادیر رایج/پیش‌فرض منابع‌ان و بهینه نشدن."""
    o, h, l, c, t = [df[x].values.astype(np.float64) for x in "ohlct"]
    v = df["v"].values.astype(np.float64) if "v" in df else np.ones(len(c))
    E9, E20, E21, E50, E200 = ema(c, 9), ema(c, 20), ema(c, 21), ema(c, 50), ema(c, 200)
    R2, R14 = rsi(c, 2), rsi(c, 14); S5 = sma(c, 5)
    bl, bm, bu = bbands(c); A = atr(h, l, c)
    bw = (bu - bl) / bm; bw_min = pd.Series(bw).rolling(120).min().values
    kl, ku = ema(c, 20) - 1.5 * A, ema(c, 20) + 1.5 * A
    M, MS = macd(c); K, D = stoch(h, l, c)
    VW, VS = vwap_day(t, h, l, c, v)
    ST = supertrend(h, l, c)
    hi20 = pd.Series(h).rolling(20).max().shift().values; lo20 = pd.Series(l).rolling(20).min().shift().values
    Hs, Ls = pd.Series(h), pd.Series(l)
    ten = ((Hs.rolling(9).max() + Ls.rolling(9).min()) / 2).values; kij = ((Hs.rolling(26).max() + Ls.rolling(26).min()) / 2).values
    sa = pd.Series((ten + kij) / 2).shift(26).values; sb = ((Hs.rolling(52).max() + Ls.rolling(52).min()) / 2).shift(26).values
    ctop, cbot = np.fmax(sa, sb), np.fmin(sa, sb)
    bull, bear = c > o, c < o
    up, dn = (E50 > E200) & (c > E200), (E50 < E200) & (c < E200)
    pc = np.r_[c[0], c[:-1]]
    body = abs(c - o); lw = np.minimum(o, c) - l; uw = h - np.maximum(o, c); rng_ = (h - l).clip(1e-12)
    # فیبوناچی: سقف/کف ۴۸ کندل اخیر
    H48 = pd.Series(h).rolling(48).max().shift().values; L48 = pd.Series(l).rolling(48).min().shift().values
    fibL = H48 - 0.618 * (H48 - L48); fibS = L48 + 0.618 * (H48 - L48)
    # بازهٔ آغازین: ساعت اول روز UTC و ساعت اول بازار نیویورک (۱۳:۳۰ تا ۱۴:۳۰ UTC)
    mins = (t // 60_000) % 1440; day = t // 86_400_000
    def orb(start, end):
        inwin = (mins >= start) & (mins < end)
        g = pd.DataFrame(dict(day=day, h=np.where(inwin, h, np.nan), l=np.where(inwin, l, np.nan)))
        rh = g.groupby("day").h.transform("max").values; rl = g.groupby("day").l.transform("min").values
        after = mins >= end
        first_l = pd.Series(after & (c > rh)).groupby(day).cumsum().values == 1
        first_s = pd.Series(after & (c < rl)).groupby(day).cumsum().values == 1
        return after & (c > rh) & first_l, after & (c < rl) & first_s
    orb_utc, orb_ny = orb(0, 60), orb(810, 870)
    rnd = np.random.default_rng(0).random(len(c))
    # بازهٔ ۱۰ صبح تا ۱۰ شب تهران (UTC+3:30) = ۰۶:۳۰ تا ۱۸:۳۰ UTC؛ سیگنال روی کلوز کندل، پس زمان بسته شدن کندل ملاکه
    step = int(np.median(np.diff(t))) if len(t) > 1 else 300_000
    cmin = ((t + step) // 60_000) % 1440
    teh = (cmin > 390) & (cmin <= 1110)
    HT = htf_trend(df)
    bigstop = (2.0 * A / c) >= 0.005          # حد ضرر (۲ ATR) حداقل ۰.۵٪ قیمت → هزینه‌ها حداکثر حدود ۰.۱۸R

    S = {
        # --- برگشت به میانگین ---
        "RSI2_MR":          dict(le=(R2 < 10) & (c > E200), se=(R2 > 90) & (c < E200), xl=c > S5, xs=c < S5, sl=2.0, tp=0, max_bars=24),
        "RSI_BB_MR":        dict(le=(c < bl) & (R14 < 30), se=(c > bu) & (R14 > 70), xl=c >= bm, xs=c <= bm, sl=1.5, tp=0, max_bars=36),
        "BB_REENTRY":       dict(le=(pc < prev(bl)) & (c > bl), se=(pc > prev(bu)) & (c < bu), xl=c >= bm, xs=c <= bm, sl=1.5, tp=0, max_bars=36),
        "STOCH_X_TREND":    dict(le=cross_up(K, D) & (K < 20) & (c > E50), se=cross_dn(K, D) & (K > 80) & (c < E50), sl=1.5, tp=2.0, max_bars=36),
        "VWAP_REVERSION":   dict(le=(pc < prev(VW - 2 * VS)) & (c > VW - 2 * VS), se=(pc > prev(VW + 2 * VS)) & (c < VW + 2 * VS),
                                 xl=c >= VW, xs=c <= VW, sl=1.5, tp=0, max_bars=48),
        "PINBAR_BB":        dict(le=(l <= bl) & (lw > 2 * body) & ((c - l) / rng_ > 0.66), se=(h >= bu) & (uw > 2 * body) & ((h - c) / rng_ > 0.66), sl=1.5, tp=2.0, max_bars=36),
        # --- روندی / شکست ---
        "EMA9_21_X":        dict(le=cross_up(E9, E21) & (c > E200), se=cross_dn(E9, E21) & (c < E200), sl=1.5, tp=3.0, max_bars=48),
        "EMA9_21_X_TP1.5":  dict(le=cross_up(E9, E21) & (c > E200), se=cross_dn(E9, E21) & (c < E200), sl=1.0, tp=1.5, max_bars=24),
        "EMA20_PULLBACK":   dict(le=(E20 > E50) & up & (l <= E20) & (c > E20) & bull, se=(E20 < E50) & dn & (h >= E20) & (c < E20) & bear, sl=1.5, tp=3.0, max_bars=48),
        "MACD_X":           dict(le=cross_up(M, MS) & (M < 0) & (c > E200), se=cross_dn(M, MS) & (M > 0) & (c < E200), sl=1.5, tp=3.0, max_bars=48),
        "VWAP_TREND":       dict(le=(c > VW) & (E9 > E21) & (l <= VW) & bull, se=(c < VW) & (E9 < E21) & (h >= VW) & bear, sl=1.5, tp=3.0, max_bars=48),
        "BB_SQUEEZE_BO":    dict(le=(prev(bw) <= 1.1 * prev(bw_min)) & (c > bu), se=(prev(bw) <= 1.1 * prev(bw_min)) & (c < bl), sl=1.5, tp=3.0, max_bars=48),
        "KELTNER_BO":       dict(le=(c > ku) & up, se=(c < kl) & dn, sl=1.5, tp=3.0, max_bars=48),
        "DONCHIAN20_BO":    dict(le=c > hi20, se=c < lo20, sl=2.0, tp=4.0, max_bars=96),
        "SUPERTREND":       dict(le=(ST == 1) & (prev(ST) == -1), se=(ST == -1) & (prev(ST) == 1), xl=ST == -1, xs=ST == 1, sl=2.0, tp=0, max_bars=288),
        "ICHIMOKU_TK":      dict(le=cross_up(ten, kij) & (c > ctop), se=cross_dn(ten, kij) & (c < cbot), sl=1.5, tp=3.0, max_bars=48),
        "FIB_618":          dict(le=up & (l <= fibL) & (c > fibL) & bull, se=dn & (h >= fibS) & (c < fibS) & bear, sl=1.5, tp=2.0, max_bars=48),
        "ORB_UTC":          dict(le=orb_utc[0], se=orb_utc[1], sl=1.5, tp=3.0, max_bars=10**6, eod=True),
        "ORB_NEWYORK":      dict(le=orb_ny[0], se=orb_ny[1], sl=1.5, tp=3.0, max_bars=10**6, eod=True),
        # --- فقط ساعت ۱۰ صبح تا ۱۰ شب تهران (الف: ورود در بازه / ب: ورود در بازه + بستن ساعت ۱۰ شب)
        "SUPERTREND_TEH":         dict(le=(ST == 1) & (prev(ST) == -1) & teh, se=(ST == -1) & (prev(ST) == 1) & teh, xl=ST == -1, xs=ST == 1, sl=2.0, tp=0, max_bars=288),
        "SUPERTREND_TEH_CLOSE":   dict(le=(ST == 1) & (prev(ST) == -1) & teh, se=(ST == -1) & (prev(ST) == 1) & teh, xl=(ST == -1) | ~teh, xs=(ST == 1) | ~teh, sl=2.0, tp=0, max_bars=288),
        "ICHIMOKU_TK_TEH":        dict(le=cross_up(ten, kij) & (c > ctop) & teh, se=cross_dn(ten, kij) & (c < cbot) & teh, sl=1.5, tp=3.0, max_bars=48),
        "ICHIMOKU_TK_TEH_CLOSE":  dict(le=cross_up(ten, kij) & (c > ctop) & teh, se=cross_dn(ten, kij) & (c < cbot) & teh, xl=~teh, xs=~teh, sl=1.5, tp=3.0, max_bars=48),
        # --- استراتژی طراحی‌شده (پیش از تست ثابت شده؛ ۴ نسخه): روند ۴ ساعته + ساعت ۱۰ تا ۲۲ تهران + حد ضرر ≥ ۰.۵٪ قیمت
        #     ورود: پایان پولبک روی تایم پایین (سوپرترند یا کراس تنکان/کیجون دوباره هم‌جهت روند ۴ ساعته)
        "CLAUDE_ST_3R":     dict(le=(ST == 1) & (prev(ST) == -1) & (HT == 1) & teh & bigstop, se=(ST == -1) & (prev(ST) == 1) & (HT == -1) & teh & bigstop,
                                 sl=2.0, tp=6.0, max_bars=192),
        "CLAUDE_ST_TRAIL":  dict(le=(ST == 1) & (prev(ST) == -1) & (HT == 1) & teh & bigstop, se=(ST == -1) & (prev(ST) == 1) & (HT == -1) & teh & bigstop,
                                 xl=(ST == -1) | (HT == -1), xs=(ST == 1) | (HT == 1), sl=2.0, tp=0, max_bars=10**6),
        "CLAUDE_TK_3R":     dict(le=cross_up(ten, kij) & (c > ctop) & (HT == 1) & teh & bigstop, se=cross_dn(ten, kij) & (c < cbot) & (HT == -1) & teh & bigstop,
                                 sl=2.0, tp=6.0, max_bars=192),
        "CLAUDE_TK_TRAIL":  dict(le=cross_up(ten, kij) & (c > ctop) & (HT == 1) & teh & bigstop, se=cross_dn(ten, kij) & (c < cbot) & (HT == -1) & teh & bigstop,
                                 xl=(c < kij) | (HT == -1), xs=(c > kij) | (HT == 1), sl=2.0, tp=0, max_bars=10**6),
        # --- معیار مقایسه: ورود تصادفی با همون مدیریت خروج ---
        "RANDOM_BASELINE":  dict(le=rnd < 0.01, se=(rnd >= 0.01) & (rnd < 0.02), sl=1.5, tp=3.0, max_bars=48),
    }
    return S
