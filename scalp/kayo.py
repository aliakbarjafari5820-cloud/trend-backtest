"""هستهٔ کدپذیر سیستم کایوموکو (آکادمی نکوموکو، ۱۴۰۱) — تعریف‌ها قبل از تست ثابت شدن و بعدش عوض نمی‌شن.

ایچیموکو استاندارد ۹/۲۶/۵۲. همهٔ محاسبه‌ها فقط با کندل‌های بسته‌شده؛ ورود روی اپن کندل بعد.

کومونیکو (ورود):
  خرید: کندل صعودی (کلوز > اپن) که کلوزش از زیر به بالای «سنکو B فعلی» (ابرِ زیر کندل فعلی) عبور کنه،
        و در ۱۰ کندل اخیر قیمت به یک «سطح مهم» رسیده باشه = Low حداکثر ۰.۵ ATR بالای آخرین کف تأییدشده (۳ کندل هر طرف).
  فروش: قرینه (عبور به زیر سنکو B، کندل نزولی، نزدیک آخرین سقف تأییدشده).
  تارگت کومونیکو = ۲.۵ × |سنکو A − سنکو B|؛ اگه این فاصله «زیاد» باشه (> ۲ ATR) → ۱ برابر (طبق نکتهٔ جزوه).
  حد ضرر = نصف فاصلهٔ تارگت (طبق موکوهیو: «حد ضرر نصف تارگت») → نسبت ریسک به ریوارد همیشه ۱ به ۲.

خط طلایی (تأیید): کیجون با دورهٔ ۵۲. خرید فقط وقتی کلوز بالای خط طلاییه؛ فروش برعکس.
KJC (تأیید زمانی): از کندل سیگنال، ۹ سیکل ۲۶ کندلی به عقب؛ کندل هر سیکل «روی فلت کیجون» حساب می‌شه
  اگه کیجون اون کندل با کندل قبلش برابر باشه. حداقل ۸ از ۹ و سیکل خطا نباید ۳، ۶ یا ۹ باشه.
موکوهیو (خروج): باکس = از سقف تا کف ابر در ۲۶ کندل اخیر (فاصلهٔ زمانی کندل‌ها تا چیکو).
  تارگت = کل باکس (تارگت دوم)؛ اگه باکس کوچیک باشه (< ۲ ATR) → ۲ برابر باکس. حد ضرر = نصف تارگت.
فیلتر هزینه: اگه فاصلهٔ حد ضرر کمتر از ۰.۲٪ قیمت باشه معامله انجام نمی‌شه (کارمزد بیشتر از ~۰.۴R می‌شد)."""
import numpy as np
import pandas as pd
from scalp import atr, prev


def ichi(h, l, tn=9, kn=26, bn=52):
    H, L = pd.Series(h), pd.Series(l)
    mid = lambda n: ((H.rolling(n).max() + L.rolling(n).min()) / 2).values
    ten, kij, gold = mid(tn), mid(kn), mid(52)
    sa_now = pd.Series((ten + kij) / 2).shift(kn).values      # ابر زیر کندل فعلی (۲۶ کندل قبل محاسبه شده)
    sb_now = pd.Series(mid(bn)).shift(kn).values
    return ten, kij, gold, sa_now, sb_now


def kjc_ok(kij, min_hits=8):
    """برای هر کندل: آیا ۹ سیکل ۲۶‌تایی گذشته روی فلت کیجون بودن؟"""
    n = len(kij); flat = np.r_[False, kij[1:] == kij[:-1]]
    hits = np.zeros(n, int); bad_key = np.zeros(n, bool)
    for k in range(1, 10):
        j = np.arange(n) - 26 * k; ok = (j >= 1) & flat[np.clip(j, 0, None)]
        hits += ok
        if k in (3, 6, 9): bad_key |= ~ok
    return (hits >= min_hits) & ~bad_key


def last_swing_levels(h, l, k=3):
    """آخرین سقف/کف تأییدشده تا هر کندل (تأیید k کندل بعد از نقطه)."""
    H, L = pd.Series(h), pd.Series(l); w = 2 * k + 1
    is_h = (H == H.rolling(w, center=True).max()).values; is_l = (L == L.rolling(w, center=True).min()).values
    sh = pd.Series(np.where(is_h, h, np.nan)).shift(k).ffill().values    # در کندل j+k تأیید می‌شه
    sl = pd.Series(np.where(is_l, l, np.nan)).shift(k).ffill().values
    return sh, sl


def kayo_strategies(df):
    o, h, l, c = [df[x].values.astype(np.float64) for x in "ohlc"]
    A = atr(h, l, c)
    ten, kij, gold, sa, sb = ichi(h, l)
    sh, sl_lv = last_swing_levels(h, l)
    near_sup = pd.Series((l - sl_lv) <= 0.5 * A).rolling(10).max().fillna(0).values.astype(bool)
    near_res = pd.Series((sh - h) <= 0.5 * A).rolling(10).max().fillna(0).values.astype(bool)
    kn_long = (c > o) & (c > sb) & (prev(c) <= prev(sb)) & near_sup
    kn_short = (c < o) & (c < sb) & (prev(c) >= prev(sb)) & near_res
    dist = np.abs(sa - sb)
    kn_tp = np.where(dist > 2 * A, 1.0, 2.5) * dist
    ctop = pd.Series(np.fmax(sa, sb)).rolling(26).max().values; cbot = pd.Series(np.fmin(sa, sb)).rolling(26).min().values
    box = ctop - cbot
    mh_tp = np.where(box < 2 * A, 2 * box, box)
    gl_long, gl_short = c > gold, c < gold
    kjc = kjc_ok(kij)
    def plan(tp):
        sld = tp / 2
        return np.where(sld / c >= 0.002, sld, 0.0), np.where(sld / c >= 0.002, tp, 0.0)
    kn_sld, kn_tpd = plan(kn_tp); mh_sld, mh_tpd = plan(mh_tp)
    B = dict(max_bars=10**6)
    return {
        "KAYO_KN":         dict(le=kn_long, se=kn_short, sld=kn_sld, tpd=kn_tpd, **B),
        "KAYO_KN_GL":      dict(le=kn_long & gl_long, se=kn_short & gl_short, sld=kn_sld, tpd=kn_tpd, **B),
        "KAYO_KN_GL_KJC":  dict(le=kn_long & gl_long & kjc, se=kn_short & gl_short & kjc, sld=kn_sld, tpd=kn_tpd, **B),
        "KAYO_KN_GL_MH":   dict(le=kn_long & gl_long, se=kn_short & gl_short, sld=mh_sld, tpd=mh_tpd, **B),
    }
