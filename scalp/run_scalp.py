"""اجرای همهٔ استراتژی‌های اسکالپ روی یک ارز: ۳ سال دادهٔ ۵ دقیقه‌ای فیوچرز بایننس، تایم ۵ و ۱۵ دقیقه.  python run_scalp.py BTCUSDT"""
import io, sys, os, zipfile, datetime, requests, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scalp import strategies, simulate

SYM = sys.argv[1]; N_MONTHS = 36
URL = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/5m/{s}-5m-{m}.zip"
d = datetime.date.today().replace(day=1); months = []
for _ in range(N_MONTHS):
    d = (d - datetime.timedelta(days=1)).replace(day=1); months.append(d.strftime("%Y-%m"))

def get(m):
    for _ in range(3):
        try:
            r = requests.get(URL.format(s=SYM, m=m), timeout=30)
            if r.status_code == 404: return None
            z = zipfile.ZipFile(io.BytesIO(r.content))
            x = pd.read_csv(z.open(z.namelist()[0]), header=None).iloc[:, :6]
            return x[pd.to_numeric(x[0], errors="coerce").notna()].astype(float)
        except Exception: pass
    return None

with ThreadPoolExecutor(8) as ex: parts = [p for p in ex.map(get, sorted(months)) if p is not None]
df5 = pd.concat(parts); df5.columns = ["t", "o", "h", "l", "c", "v"]
df5 = df5.drop_duplicates("t").sort_values("t").reset_index(drop=True)
print(SYM, len(df5), "candles", len(parts), "months")
g = df5.groupby((df5.t // 900_000).astype(np.int64))
df15 = pd.DataFrame(dict(t=g.t.first(), o=g.o.first(), h=g.h.max(), l=g.l.min(), c=g.c.last(), v=g.v.sum())).reset_index(drop=True)
out = []
for tf, df in (("5m", df5), ("15m", df15)):
    for name, cfg in strategies(df).items():
        R = simulate(df, **cfg)
        if len(R): R["strategy"], R["tf"], R["symbol"] = name, tf, SYM; out.append(R)
        print(f"  {tf} {name}: {len(R)} trades, net {R.R_net.mean() if len(R) else 0:+.3f}R", flush=True)
os.makedirs("out", exist_ok=True)
pd.concat(out).to_csv(f"out/scalp_{SYM}.csv", index=False)
