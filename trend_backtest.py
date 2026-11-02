"""بک‌تست پنج‌سالهٔ دنبال کردن روند (فقط خرید) روی ۵ ارز و ۵ تایم‌فریم.
اجرا:  python trend_backtest.py
داده‌ها یک بار دانلود و توی پوشهٔ data ذخیره می‌شن. نتیجه توی results.csv هم ذخیره می‌شه."""
import os, sys, time, requests, numpy as np, pandas as pd

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
ALL_TFS = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
TFS = {k: v for k, v in ALL_TFS.items() if len(sys.argv) < 2 or k in sys.argv[1:]}
YEARS = 5
FEE = 0.0005                     # کارمزد هر طرف معامله (۰.۰۵٪)
URL = "https://data-api.binance.vision/api/v3/klines"
END = int(time.time()*1000); START = END - YEARS*365*86400*1000
os.makedirs("data", exist_ok=True)

def get(sym, tf):
    path = f"data/{sym}_{tf}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    rows, t, step = [], START, ALL_TFS[tf]*60*1000*1000
    while t < END:
        for _ in range(5):
            try:
                d = requests.get(URL, params={"symbol": sym, "interval": tf, "startTime": t, "limit": 1000}, timeout=20).json()
                break
            except Exception:
                time.sleep(2)
        if not isinstance(d, list) or not d:
            t += step; continue
        rows += [r[:5] for r in d]; t = int(d[-1][0]) + 1
        pass
    print()
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]).astype(float).drop_duplicates("t")
    df.to_csv(path, index=False)
    return df

def run(df, kind, n_entry, n_exit=None, atr_mult=3):
    o, h, l, c, t = [df[x].values for x in "ohlct"]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h-l, np.maximum(abs(h-pc), abs(l-pc)))
    atr = pd.Series(tr).rolling(14).mean().values
    cs = pd.Series(c)
    hi = cs.rolling(n_entry).max().shift().values
    lo = cs.rolling(n_exit).min().shift().values if kind == "turtle" else None
    pos, eq, entry, ext = 0, 1.0, 0.0, 0.0
    curve = np.ones(len(c)); trades = []
    for i in range(n_entry, len(c)-1):
        sig = pos
        if pos == 0:
            if c[i] > hi[i]: sig, ext = 1, c[i]                 # شکست سقف → خرید
        elif kind == "turtle":
            if c[i] < lo[i]: sig = 0                            # شکست کف کوتاه‌تر → خروج
        else:
            ext = max(ext, c[i])
            if c[i] < ext - atr_mult*atr[i]: sig = 0            # حد ضرر متحرک ATR → خروج
        if sig != pos:
            px = o[i+1]                                         # اجرا روی باز شدن کندل بعد
            if pos == 1:
                trades.append(px/entry - 1 - 2*FEE); eq *= (px/entry)*(1-FEE)
            if sig == 1:
                entry = px; eq *= (1-FEE)
            pos = sig
        curve[i+1] = eq*(c[i+1]/entry) if pos == 1 else eq
    return pd.Series(curve, index=pd.to_datetime(t, unit="ms")), np.array(trades)

def configs(tf):
    per_day = 1440 // ALL_TFS[tf]
    return {
        "Turtle 20/10 کندل":        ("turtle", 20, 10),
        "Chandelier 20 کندل":       ("chandelier", 20, None),
        "Turtle 20/10 روز":         ("turtle", 20*per_day, 10*per_day),
        "Chandelier 20 روز":        ("chandelier", 20*per_day, None),
    }

rows = []
for tf in TFS:
    for name in configs(tf):
        curves, all_tr = [], []
        for sym in SYMS:
            df = get(sym, tf)
            kind, ne, nx = configs(tf)[name]
            cv, tr = run(df, kind, ne, nx)
            curves.append(cv.resample("1D").last().ffill()); all_tr.append(tr)
            c1 = cv / cv.iloc[0]
            rows.append(dict(tf=tf, strategy=name, coin=sym[:-4], total_pct=round((c1.iloc[-1]-1)*100),
                             maxDD_pct=round((1-c1/c1.cummax()).max()*100), trades=len(tr),
                             win_pct=round((tr > 0).mean()*100) if len(tr) else 0))
        port = pd.concat(curves, axis=1).ffill().dropna().mean(axis=1); port /= port.iloc[0]
        tr = np.concatenate(all_tr)
        yearly = port.resample("YE").last(); yearly = (yearly/yearly.shift().fillna(1)-1)*100
        rows.append(dict(tf=tf, strategy=name, coin="سبد ۵ ارز", total_pct=round((port.iloc[-1]-1)*100),
                         maxDD_pct=round((1-port/port.cummax()).max()*100), trades=len(tr),
                         win_pct=round((tr > 0).mean()*100) if len(tr) else 0,
                         PF=round(tr[tr > 0].sum()/max(1e-9, -tr[tr < 0].sum()), 2),
                         **{str(k.year): round(v) for k, v in yearly.items()}))
        print(f"{tf:>4} | {name:<20} | سود کل {rows[-1]['total_pct']:>6}% | افت {rows[-1]['maxDD_pct']:>3}% | معامله {len(tr):>5} | PF {rows[-1]['PF']}")

res = pd.DataFrame(rows)
res.to_csv(f"results_{'_'.join(TFS)}.csv", index=False, encoding="utf-8-sig")
print("\nخرید و نگه داشتن (سبد ۵ ارز):")
bh = pd.concat([(lambda d: pd.Series(d.c.values, index=pd.to_datetime(d.t, unit="ms")))(get(s, list(TFS)[-1])).resample("1D").last()
                for s in SYMS], axis=1).ffill().dropna()
bh = (bh/bh.iloc[0]).mean(axis=1)
print(f"سود کل {round((bh.iloc[-1]-1)*100)}%  | افت {round((1-bh/bh.cummax()).max()*100)}%")
print("\nتموم شد. جدول کامل توی results.csv ذخیره شد.")
