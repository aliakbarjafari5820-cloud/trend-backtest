"""لانگ + شورت همزمان، فقط وقتی قیمت سقف یا کف قبلی همون تایم رو می‌شکنه (بیت‌کوین، ۵ سال)."""
import sys, time, requests, numpy as np, pandas as pd
TF = sys.argv[1]; MIN = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}[TF]
SYM = "BTCUSDT"; FEE = 0.0005; RR = 2.0; K = 3
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
t, o, h, l, c = [df[x].values for x in "tohlc"]; n = len(c)
pc = np.r_[c[0], c[:-1]]
tr = np.maximum(h-l, np.maximum(abs(h-pc), abs(l-pc)))
atr = pd.Series(tr).rolling(14).mean().values
up = np.r_[0, np.diff(h)]; dn = np.r_[0, -np.diff(l)]
pdm = np.where((up > dn) & (up > 0), up, 0); mdm = np.where((dn > up) & (dn > 0), dn, 0)
a = pd.Series(tr).ewm(alpha=1/14).mean()
pdi = 100*pd.Series(pdm).ewm(alpha=1/14).mean()/a; mdi = 100*pd.Series(mdm).ewm(alpha=1/14).mean()/a
adx = (100*(pdi-mdi).abs()/(pdi+mdi)).ewm(alpha=1/14).mean().values

# شکست‌ها: آخرین سقف/کف تأییدشده (K کندل هر طرف) و اولین کلوز اون‌ورش
events = []   # (ایندکس کندل شکست, جهت شکست)
last_hi = last_lo = None
for i in range(2*K, n-1):
    j = i - K
    if h[j] == h[j-K:j+K+1].max(): last_hi = h[j]
    if l[j] == l[j-K:j+K+1].min(): last_lo = l[j]
    if last_hi is not None and c[i] > last_hi: events.append((i, 1)); last_hi = None
    elif last_lo is not None and c[i] < last_lo: events.append((i, -1)); last_lo = None

def first_true(fn, s):
    i = s
    while i < n:
        j = min(i+5000, n); m = fn(i, j)
        if m.any(): return i + int(np.argmax(m))
        i = j
    return n

legs = []
for i, bd in events:
    k = i + 1; e, r = o[k], atr[i]
    if not r > 0: continue
    for d in (1, -1):
        sl, tp = e - d*r, e + d*RR*r
        if d == 1: hs = first_true(lambda a, b: l[a:b] <= sl, k); ht = first_true(lambda a, b: h[a:b] >= tp, k)
        else:      hs = first_true(lambda a, b: h[a:b] >= sl, k); ht = first_true(lambda a, b: l[a:b] <= tp, k)
        if hs >= n and ht >= n: continue
        R = -1.0 if hs <= ht else RR
        legs.append((i, bd, d, R, R - 2*FEE*e/r, adx[i] > 25))
L = pd.DataFrame(legs, columns=["i", "brk", "dir", "R", "Rfee", "trend"])
L["with_break"] = L.brk == L.dir
print(f"\n===== بیت‌کوین | تایم {TF} | ۵ سال | {len(events)} شکست =====")
for name, X in [("همهٔ شکست‌ها", L), ("فقط با روند (ADX>25)", L[L.trend])]:
    w, a_ = X[X.with_break], X[~X.with_break]
    print(f"[{name}] معامله: {len(X)} | بدون کارمزد: {X.R.sum():+.0f}R | با کارمزد: {X.Rfee.sum():+.0f}R (هر معامله {X.Rfee.mean():+.3f}R)")
    print(f"    هم‌جهت شکست: برد {(w.R>0).mean()*100:.1f}% | بدون کارمزد {w.R.sum():+.0f}R | با کارمزد {w.Rfee.sum():+.0f}R")
    print(f"    خلاف شکست:  برد {(a_.R>0).mean()*100:.1f}% | بدون کارمزد {a_.R.sum():+.0f}R | با کارمزد {a_.Rfee.sum():+.0f}R")
