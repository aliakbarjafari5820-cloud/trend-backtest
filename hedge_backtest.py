"""بک‌تست ایدهٔ لانگ + شورت همزمان: روزی ۵ جفت معامله روی بیت‌کوین، حد ضرر ۱ ATR و تارگت ۲ ATR."""
import os, sys, time, requests, numpy as np, pandas as pd
TF = sys.argv[1]; MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}[TF]
SYM = "BTCUSDT"; FEE = 0.0005; PAIRS_PER_DAY = 5; RR = 2.0
URL = "https://data-api.binance.vision/api/v3/klines"
END = int(time.time()*1000); START = END - 5*365*86400*1000
rows, t = [], START
while t < END:
    for _ in range(5):
        try:
            d = requests.get(URL, params={"symbol": SYM, "interval": TF, "startTime": t, "limit": 1000}, timeout=20).json(); break
        except Exception: time.sleep(2)
    if not isinstance(d, list) or not d: t += MIN*60*1000*1000; continue
    rows += [r[:5] for r in d]; t = int(d[-1][0]) + 1
df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]).astype(float).drop_duplicates("t").reset_index(drop=True)
t, o, h, l, c = [df[x].values for x in "tohlc"]
pc = np.r_[c[0], c[:-1]]
atr = pd.Series(np.maximum(h-l, np.maximum(abs(h-pc), abs(l-pc)))).rolling(14).mean().shift().values  # فقط از کندل‌های قبلی

# زمان‌های ورود: هر روز ۵ بار با فاصلهٔ مساوی، اولین کندلی که بعد از هر نوبت باز می‌شه
day = 86400*1000
slots = np.array([d0 + k*day/PAIRS_PER_DAY for d0 in range((int(t[0])//day+1)*day, int(t[-1]), day) for k in range(PAIRS_PER_DAY)])
entries = np.unique(np.searchsorted(t, slots))
entries = entries[(entries > 20) & (entries < len(t)-1)]

def first_true(mask_fn, start):
    """اولین ایندکسی که شرط برقرار می‌شه (در پنجره‌های ۵۰۰۰تایی)."""
    i = start
    while i < len(t):
        j = min(i+5000, len(t)); m = mask_fn(i, j)
        if m.any(): return i + int(np.argmax(m))
        i = j
    return len(t)

legs = []   # (زمان ورود, جهت, R بدون کارمزد, R با کارمزد, مدت به ساعت)
for i in entries:
    e, r = o[i], atr[i]
    if not r > 0: continue
    fee_R = 2*FEE*e/r
    for d in (1, -1):
        sl, tp = e - d*r, e + d*RR*r
        if d == 1:
            hs = first_true(lambda a, b: l[a:b] <= sl, i); ht = first_true(lambda a, b: h[a:b] >= tp, i)
        else:
            hs = first_true(lambda a, b: h[a:b] >= sl, i); ht = first_true(lambda a, b: l[a:b] <= tp, i)
        if hs >= len(t) and ht >= len(t): continue          # هنوز باز مونده
        R = -1.0 if hs <= ht else RR                         # اگه هر دو توی یه کندل بود، ضرر حساب می‌شه (محافظه‌کارانه)
        legs.append((t[i], d, R, R - fee_R, (t[min(hs, ht)] - t[i])/3.6e6))
L = pd.DataFrame(legs, columns=["t", "dir", "R", "Rfee", "hours"])
L["year"] = pd.to_datetime(L.t, unit="ms").dt.year
pairs = L.groupby("t").agg(R=("R", "sum"), Rfee=("Rfee", "sum"), n=("R", "size")); pairs = pairs[pairs.n == 2]
print(f"\n===== بیت‌کوین | تایم {TF} | ۵ سال =====")
print(f"تعداد معامله: {len(L)}  (جفت کامل: {len(pairs)})  | میانگین ATR به درصد قیمت: {np.nanmean(atr/c)*100:.3f}%")
print(f"درصد معاملات برنده: {(L.R > 0).mean()*100:.1f}%   میانگین مدت معامله: {L.hours.mean():.1f} ساعت")
print(f"نتیجهٔ جفت‌ها بدون کارمزد: +1R={((pairs.R>0)&(pairs.R<2)).mean()*100:.0f}%  |  -2R={(pairs.R==-2).mean()*100:.0f}%  |  +4R={(pairs.R==4).mean()*100:.0f}%")
print(f"جمع کل بدون کارمزد: {L.R.sum():+.0f}R   (میانگین هر معامله {L.R.mean():+.3f}R)")
print(f"جمع کل با کارمزد:    {L.Rfee.sum():+.0f}R   (میانگین هر معامله {L.Rfee.mean():+.3f}R | کارمزد هر معامله ≈ {(L.R-L.Rfee).mean():.2f}R)")
print("سال به سال (R با کارمزد):", L.groupby("year").Rfee.sum().round(0).to_dict())
print("لانگ‌ها / شورت‌ها (R با کارمزد):", L.groupby("dir").Rfee.sum().round(0).to_dict())
