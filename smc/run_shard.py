"""یک تکه از بازار: دادهٔ ۵ دقیقه‌ای ۲ سالهٔ قراردادهای دائمی USDT رو از data.binance.vision می‌گیره،
۱۵ دقیقه و ۱ ساعته رو از روش می‌سازه و هر سه استراتژی رو اجرا می‌کنه.  python run_shard.py SHARD NSHARDS"""
import io, os, sys, zipfile, json, requests, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
from engine import STRATEGIES

SHARD, NSH = int(sys.argv[1]), int(sys.argv[2])
MONTHS = json.load(open("months.json")); SYMS = json.load(open("symbols.json"))[SHARD::NSH]
BASE = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/5m/{s}-5m-{m}.zip"
TFS = {"5m": 5, "15m": 15, "1h": 60}
sess = requests.Session()

def month(args):
    s, m = args
    for _ in range(3):
        try:
            r = sess.get(BASE.format(s=s, m=m), timeout=30)
            if r.status_code == 404: return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4])
            df = df[pd.to_numeric(df[0], errors="coerce").notna()].astype(float)
            df.columns = ["t", "o", "h", "l", "c"]; return df
        except Exception: pass
    return None

def resample(df, minutes):
    if minutes == 5: return df
    g = df.groupby((df.t // (minutes * 60_000)).astype(np.int64))
    return pd.DataFrame(dict(t=g.t.first(), o=g.o.first(), h=g.h.max(), l=g.l.min(), c=g.c.last())).reset_index(drop=True)

rows = []; done = 0
with ThreadPoolExecutor(8) as ex:
    for s in SYMS:
        parts = [p for p in ex.map(month, [(s, m) for m in MONTHS]) if p is not None]
        if not parts: continue
        df5 = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
        if len(df5) < 2000: continue
        for tf, mins in TFS.items():
            df = resample(df5, mins)
            for name, fn in STRATEGIES.items():
                for tr in fn(df):
                    tr.update(symbol=s, tf=tf); rows.append(tr)
        done += 1
        print(f"{s}: {len(df5)} candles", flush=True)
os.makedirs("out", exist_ok=True)
pd.DataFrame(rows).to_csv(f"out/trades_{SHARD:02d}.csv", index=False)
print(f"shard {SHARD}: {done}/{len(SYMS)} symbols, {len(rows)} trades")
