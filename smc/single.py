"""تست یک استراتژی روی یک ارز و یک تایم (برای بررسی جداگانهٔ هر استراتژی).
از داخل فایل‌های strategies/0X_*.py صدا زده می‌شه."""
import io, sys, zipfile, datetime, requests, numpy as np, pandas as pd

def months(n=24):
    d = datetime.date.today().replace(day=1); out = []
    for _ in range(n):
        d = (d - datetime.timedelta(days=1)).replace(day=1); out.append(d.strftime("%Y-%m"))
    return sorted(out)

def load(symbol, tf="1h", n_months=24):
    url = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/5m/{s}-5m-{m}.zip"
    parts = []
    for m in months(n_months):
        r = requests.get(url.format(s=symbol, m=m), timeout=30)
        if r.status_code != 200: continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]), header=None).iloc[:, :5]
        parts.append(df[pd.to_numeric(df[0], errors="coerce").notna()].astype(float))
    df = pd.concat(parts); df.columns = ["t", "o", "h", "l", "c"]
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    mins = {"5m": 5, "15m": 15, "1h": 60}[tf]
    if mins == 5: return df
    g = df.groupby((df.t // (mins * 60_000)).astype(np.int64))
    return pd.DataFrame(dict(t=g.t.first(), o=g.o.first(), h=g.h.max(), l=g.l.min(), c=g.c.last())).reset_index(drop=True)

def report(trades, name):
    if not trades: print(f"{name}: هیچ معامله‌ای"); return
    T = pd.DataFrame(trades); R = T.R_net.values; w, l = R[R > 0], R[R <= 0]
    print(f"{name}: {len(R)} معامله | برد {100*len(w)/len(R):.1f}% | PF {w.sum()/max(1e-9,-l.sum()):.2f} | "
          f"Expectancy {R.mean():+.3f}R | جمع {R.sum():+.1f}R (قبل از هزینه {T.R_gross.sum():+.1f}R)")
    print(T[["entry_t", "side", "setup", "entry", "sl", "tp", "exit", "R_net"]].tail(10).to_string(index=False))

def cli(fn, name):
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"; tf = sys.argv[2] if len(sys.argv) > 2 else "1h"
    df = load(sym, tf); print(f"{sym} {tf}: {len(df)} کندل"); report(fn(df), f"{name} {sym} {tf}")
