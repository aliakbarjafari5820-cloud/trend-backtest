"""موتور بک‌تست یکسان برای سه استراتژی: پرایس اکشن (PA)، اسمارت مانی (SMC) و ترکیبی (HYBRID).
همهٔ تعریف‌های عددی در PARAMS هستن و برای همهٔ ارزها و تایم‌ها یکسان‌ان (بدون بهینه‌سازی برای هر ارز)."""
import numpy as np
import pandas as pd

PARAMS = dict(
    swing_k=2,            # سوئینگ: ۲ کندل قبل و ۲ کندل بعد (طبق متن)؛ تأیید بعد از بسته شدن کندل دوم
    atr_n=14,             # ATR برای تلورانس‌ها
    brk_min_atr=0.50,     # حداقل قدرت شکست: کلوز حداقل ۰.۵ ATR اون‌ور سطح (برای شکست PA و BOS)
    retest_tol_atr=0.25,  # تلورانس پولبک: لو تا سطح + ۰.۲۵ ATR
    fail_close_atr=0.50,  # اگه کلوز ۰.۵ ATR اون‌ور سطح بسته بشه، شکست باطله
    pa_wait=20,           # حداکثر ۲۰ کندل برای پولبک + تأیید بعد از شکست (PA)
    sweep_reclaim=2,      # اعتبار سوئیپ: کلوز دوباره این‌ور سطح، حداکثر طی ۲ کندل
    bos_wait=20,          # بعد از سوئیپ، حداکثر ۲۰ کندل تا BOS/CHoCH
    zone_wait=20,         # اعتبار بازگشت به OB/FVG: حداکثر ۲۰ کندل بعد از BOS
    sl_buffer_atr=0.10,   # فاصلهٔ محافظتی حد ضرر = ۰.۱ ATR
    rr=2.0,               # حد سود = ۲R خالص (بعد از کارمزد و اسلیپیج ورود و خروج)
    sl_min_atr=0.10,      # فاصلهٔ حد ضرر باید بین ۰.۱ و ۳ ATR باشه، وگرنه معامله رد می‌شه
    sl_max_atr=3.00,
    min_stop_pct=0.0005,  # حد ضرر کمتر از ۰.۰۵٪ قیمت = معامله رد می‌شه (غیرقابل اجرا)
)
# کندل تأیید: کندل صعودی (کلوز > اپن) که بالای سطح/لبهٔ پایین ناحیه بسته بشه (برای فروش برعکس).

COSTS = dict(fee=0.0005, slip=0.0002, funding_8h=0.0001, max_lev=20.0)
# کارمزد Taker ۰.۰۵٪ + اسلیپیج ۰.۰۲٪ هر طرف؛ فاندینگ ۰.۰۱٪ هر ۸ ساعت (به ضرر هر دو جهت، محافظه‌کارانه)؛
# اهرم حداکثر ۲۰x: اگه ۱٪ ریسک اهرم بیشتری بخواد، حجم کم می‌شه (ریسک واقعی < ۱٪).


def atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


def swing_flags(h, l, k):
    """سوئینگ‌های بالا/پایین (کندل j از k کندل قبل و بعد بالاتر/پایین‌تر). در ایندکس j+k تأیید می‌شن."""
    # اکیداً بالاتر از k کندل قبل و بالاتر یا مساوی k کندل بعد (برای جلوگیری از سوئینگ تکراری در قله‌های مساوی)
    H, L = pd.Series(h), pd.Series(l)
    left_h = H.shift(1).rolling(k).max().values; right_h = H[::-1].shift(1).rolling(k).max()[::-1].values
    left_l = L.shift(1).rolling(k).min().values; right_l = L[::-1].shift(1).rolling(k).min()[::-1].values
    sh = (h > left_h) & (h >= right_h)
    sl = (l < left_l) & (l <= right_l)
    return np.nan_to_num(sh, nan=False).astype(bool), np.nan_to_num(sl, nan=False).astype(bool)


class Structure:
    """سوئینگ‌های تأییدشده تا کندل فعلی + روند پرایس اکشن (HH/HL یا LH/LL)."""
    def __init__(self, h, l, k):
        self.h, self.l, self.k = h, l, k
        self.sh, self.sl = swing_flags(h, l, k)
        self.highs, self.lows = [], []      # (قیمت, ایندکس)

    def update(self, i):
        j = i - self.k
        if j >= self.k:
            if self.sh[j]: self.highs.append((self.h[j], j))
            if self.sl[j]: self.lows.append((self.l[j], j))

    def trend(self):
        if len(self.highs) < 2 or len(self.lows) < 2: return 0
        (h1, _), (h2, _) = self.highs[-2], self.highs[-1]
        (l1, _), (l2, _) = self.lows[-2], self.lows[-1]
        if h2 > h1 and l2 > l1: return 1
        if h2 < h1 and l2 < l1: return -1
        return 0


def simulate_exit(o, h, l, c, t, k, d, sl, tp, trail=None):
    """از کندل ورود k جلو می‌ره؛ اگه SL و TP توی یه کندل بودن → SL (محافظه‌کارانه). گپ از SL → خروج روی اپن.
    trail (اختیاری): آرایهٔ کیجون؛ اگه کلوز اون‌ور کیجون بسته بشه → خروج روی اپن کندل بعد (حد سود ثابت نداره)."""
    n = len(c)
    for i in range(k, n):
        if d == 1:
            if o[i] <= sl and i > k: return i, o[i]
            if l[i] <= sl: return i, sl
            if trail is None and h[i] >= tp: return i, tp
        else:
            if o[i] >= sl and i > k: return i, o[i]
            if h[i] >= sl: return i, sl
            if trail is None and l[i] <= tp: return i, tp
        if trail is not None and i + 1 < n and (c[i] - trail[i]) * d < 0:
            return i + 1, o[i + 1]
    return None, None


def make_trade(df, k, d, sl, P, kind, setup, atr_val, trail=None):
    o, h, l, c, t = df["o"].values, df["h"].values, df["l"].values, df["c"].values, df["t"].values
    e = o[k]; risk = (e - sl) * d
    if risk <= 0 or risk / e < P["min_stop_pct"]: return None
    if not (P["sl_min_atr"] * atr_val <= risk <= P["sl_max_atr"] * atr_val): return None
    tp = e + d * (P["rr"] * risk + 2 * (COSTS["fee"] + COSTS["slip"]) * e)   # ۲R خالص بعد از هزینه‌ها
    if trail is not None: tp = np.nan
    xi, xp = simulate_exit(o, h, l, c, t, k, d, sl, tp, trail)
    if xi is None: return None
    stop_pct = risk / e
    R_gross = (xp - e) * d / risk
    hours = (t[xi] - t[k]) / 3.6e6
    fee_R = 2 * (COSTS["fee"] + COSTS["slip"]) / stop_pct
    flat_funding_R = COSTS["funding_8h"] * hours / 8 / stop_pct   # فقط اگه دادهٔ فاندینگ واقعی نباشه استفاده می‌شه
    risk_frac = min(0.01, COSTS["max_lev"] * stop_pct)       # سقف اهرم ۲۰x
    return dict(entry_t=int(t[k]), exit_t=int(t[xi]), side=d, entry=e, sl=sl, tp=tp, exit=xp,
                stop_pct=stop_pct, R_gross=R_gross, fee_R=fee_R, funding_R=-flat_funding_R,
                R_net=R_gross - fee_R - flat_funding_R, risk_frac=risk_frac,
                hours=hours, strategy=kind, setup=setup)


def run_pa(df, P=PARAMS):
    o, h, l, c = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    A = atr(h, l, c, P["atr_n"]); n = len(c); S = Structure(h, l, P["swing_k"])
    trades, pend, busy_until, used = [], [], -1, set()
    for i in range(1, n - 1):
        S.update(i); tr = S.trend()
        # شکست سقف/کف معتبر قبلی در جهت روند (فقط با کلوز)
        # هر سطح فقط اولین بار که کلوز با قدرت ≥ ۰.۵ ATR ازش رد بشه حساب می‌شه (اگه روند هم‌جهت نبود، سطح می‌سوزه)
        if S.highs:
            lv, j = S.highs[-1]; th = lv + P["brk_min_atr"] * A[i]
            if ("H", j) not in used and c[i] >= th and c[i - 1] < th:
                used.add(("H", j))
                if tr == 1: pend.append(dict(d=1, lv=lv, bi=i, touched=False))
        if S.lows:
            lv, j = S.lows[-1]; th = lv - P["brk_min_atr"] * A[i]
            if ("L", j) not in used and c[i] <= th and c[i - 1] > th:
                used.add(("L", j))
                if tr == -1: pend.append(dict(d=-1, lv=lv, bi=i, touched=False))
        keep = []
        for p in pend:
            d, lv = p["d"], p["lv"]
            if i - p["bi"] > P["pa_wait"]: continue
            if (c[i] - lv) * d < -P["fail_close_atr"] * A[i]: continue
            if i > p["bi"] and ((d == 1 and l[i] <= lv + P["retest_tol_atr"] * A[i]) or
                                (d == -1 and h[i] >= lv - P["retest_tol_atr"] * A[i])):
                p["touched"] = True
            conf = p["touched"] and S.trend() == d and ((d == 1 and c[i] > o[i] and c[i] > lv) or
                                                         (d == -1 and c[i] < o[i] and c[i] < lv))
            if conf and i + 1 > busy_until:
                ref = S.lows if d == 1 else S.highs
                if ref:
                    sl = ref[-1][0] - d * P["sl_buffer_atr"] * A[i]
                    tdict = make_trade(df, i + 1, d, sl, P, "PA", "BRK_PULLBACK", A[i])
                    if tdict:
                        trades.append(tdict)
                        busy_until = int(np.searchsorted(df["t"].values, tdict["exit_t"]))
                continue
            keep.append(p)
        pend = keep
    return trades


def find_zone(o, h, l, c, a, b, d, fvg_only=False):
    """OB = آخرین کندل خلاف جهت بین سوئیپ (a) و BOS (b)؛ FVG = آخرین گپ سه‌کندلی در همین بازه.
    اگه هم‌پوشانی داشتن → ناحیهٔ مشترک (OB+FVG)، وگرنه OB، وگرنه FVG."""
    ob = fvg = None
    for j in range(b, a - 1, -1):
        if (d == 1 and c[j] < o[j]) or (d == -1 and c[j] > o[j]):
            ob = (l[j], h[j]); break
    for j in range(b, max(a + 2, 2) - 1, -1):
        if d == 1 and l[j] > h[j - 2]: fvg = (h[j - 2], l[j]); break
        if d == -1 and h[j] < l[j - 2]: fvg = (h[j], l[j - 2]); break
    if fvg_only: return (fvg, "FVG") if fvg else (None, None)
    if ob and fvg:
        lo, hi = max(ob[0], fvg[0]), min(ob[1], fvg[1])
        if lo < hi: return (lo, hi), "OB+FVG"
    if ob: return ob, "OB"
    if fvg: return fvg, "FVG"
    return None, None


def ichimoku(h, l, c, tn=9, kn=26, bn=52):
    H, L = pd.Series(h), pd.Series(l)
    ten = ((H.rolling(tn).max() + L.rolling(tn).min()) / 2).values
    kij = ((H.rolling(kn).max() + L.rolling(kn).min()) / 2).values
    span_a = pd.Series((ten + kij) / 2).shift(kn).values           # ابر ۲۶ کندل جلوتر رسم می‌شه → مقدار امروز از ۲۶ کندل قبل
    span_b = ((H.rolling(bn).max() + L.rolling(bn).min()) / 2).shift(kn).values
    return ten, kij, span_a, span_b


def ichimoku_htf_trend(df, htf_min=240):
    """روند ایچیموکو ۴ ساعته برای هر کندل: ۱ = کلوز بالای ابر و تنکان > کیجون، ۱- = برعکس، ۰ = بقیه.
    فقط از کندل‌های ۴ ساعتهٔ کاملاً بسته‌شده تا پایان کندل فعلی استفاده می‌شه (بدون نگاه به آینده)."""
    t = df["t"].values; step = np.median(np.diff(t)) if len(t) > 1 else 3_600_000
    g = df.groupby((df.t // (htf_min * 60_000)).astype(np.int64))
    H4 = pd.DataFrame(dict(t=g.t.first(), h=g.h.max(), l=g.l.min(), c=g.c.last(), last=g.t.last())).reset_index(drop=True)
    full = (H4["last"] + step) >= (H4.t + htf_min * 60_000)          # فقط کندل ۴ ساعتهٔ کامل
    ten, kij, sa, sb = ichimoku(H4.h.values, H4.l.values, H4.c.values)
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb); cc = H4.c.values
    reg = np.where((cc > top) & (ten > kij), 1, np.where((cc < bot) & (ten < kij), -1, 0))
    reg = np.where(full.values, reg, 0)
    close_t = H4.t.values + htf_min * 60_000
    idx = np.searchsorted(close_t, t + step, side="right") - 1          # آخرین کندل ۴ ساعته‌ای که تا پایان این کندل بسته شده
    out = np.where(idx >= 0, reg[np.clip(idx, 0, None)], 0)
    return out


def run_smc(df, P=PARAMS, hybrid=False, ichi=False, zones=None, trail=False, kind=None):
    o, h, l, c = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    A = atr(h, l, c, P["atr_n"]); n = len(c); S = Structure(h, l, P["swing_k"])
    kind = kind or ("HYBRID" if hybrid else "SMC")
    IT = ichimoku_htf_trend(df) if ichi else None
    KJ = ichimoku(h, l, c)[1] if trail else None
    trades, busy_until, used = [], -1, set()
    st = {1: None, -1: None}          # یه ستاپ فعال برای هر جهت
    for i in range(1, n - 1):
        S.update(i)
        for d in (1, -1):
            s = st[d]
            liq = S.lows if d == 1 else S.highs
            # ۱) سوئیپ نقدینگی: نفوذ به اون‌ور آخرین کف/سقف معتبر
            if s is None and liq:
                lv, j = liq[-1]
                if ("S", d, j) not in used and ((d == 1 and l[i] < lv) or (d == -1 and h[i] > lv)):
                    used.add(("S", d, j))
                    s = st[d] = dict(stage="pierce", lv=lv, si=i, ext=l[i] if d == 1 else h[i])
            if s is None: continue
            if s["stage"] == "pierce":
                s["ext"] = min(s["ext"], l[i]) if d == 1 else max(s["ext"], h[i])
                if (c[i] - s["lv"]) * d > 0:                       # کلوز دوباره این‌ور سطح
                    s["stage"] = "bos"; s["ri"] = i
                elif i - s["si"] >= P["sweep_reclaim"]:
                    st[d] = None
                continue
            if (c[i] - s["ext"]) * d < 0:                           # کلوز اون‌ور نقطهٔ سوئیپ → باطل
                st[d] = None; continue
            if s["stage"] == "bos":
                if i - s["ri"] > P["bos_wait"]: st[d] = None; continue
                ref = S.highs if d == 1 else S.lows
                if ref and ref[-1][1] < i and (c[i] - ref[-1][0]) * d >= P["brk_min_atr"] * A[i]:   # BOS/CHoCH: کلوز با قدرت ≥ ۰.۵ ATR
                    zone, zt = find_zone(o, h, l, c, s["si"], i, d, fvg_only=(zones == "FVG"))
                    if zone is None: st[d] = None; continue
                    s.update(stage="zone", bi=i, zone=zone, zt=zt, touched=False)
                continue
            if s["stage"] == "zone":
                if i - s["bi"] > P["zone_wait"]: st[d] = None; continue
                lo, hi = s["zone"]
                if (d == 1 and l[i] <= hi) or (d == -1 and h[i] >= lo): s["touched"] = True
                conf = s["touched"] and ((d == 1 and c[i] > o[i] and c[i] > lo) or
                                         (d == -1 and c[i] < o[i] and c[i] < hi))
                if conf and hybrid and S.trend() != d: conf = False
                if conf and ichi and IT[i] != d: conf = False
                if conf:
                    st[d] = None
                    if i + 1 > busy_until:
                        sl = s["ext"] - d * P["sl_buffer_atr"] * A[i]
                        tdict = make_trade(df, i + 1, d, sl, P, kind, s["zt"], A[i], KJ)
                        if tdict:
                            trades.append(tdict)
                            busy_until = int(np.searchsorted(df["t"].values, tdict["exit_t"]))
    return trades


STRATEGIES = {"PA": run_pa, "SMC": lambda df: run_smc(df), "HYBRID": lambda df: run_smc(df, hybrid=True),
              # ترکیب ایچیموکو: روند ۴ ساعتهٔ ایچیموکو + ساختار پرایس اکشن + ورود اسمارت مانی فقط از FVG
              "ICHI_2R": lambda df: run_smc(df, hybrid=True, ichi=True, zones="FVG", kind="ICHI_2R"),
              "ICHI_KIJUN": lambda df: run_smc(df, hybrid=True, ichi=True, zones="FVG", trail=True, kind="ICHI_KIJUN")}
