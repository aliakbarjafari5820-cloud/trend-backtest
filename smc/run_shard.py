"""یک تکه از بازار: دادهٔ ۵ دقیقه‌ای ۲ سالهٔ قراردادهای دائمی USDT + نرخ فاندینگ واقعی رو از data.binance.vision می‌گیره،
۱۵ دقیقه و ۱ ساعته رو از روی ۵ دقیقه می‌سازه، هر سه استراتژی رو اجرا می‌کنه و کیفیت داده رو ثبت می‌کنه.
python run_shard.py SHARD NSHARDS"""
import io, os, sys, zipfile, json, requests, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor
from engine import STRATEGIES, COSTS

SHARD, NSH = int(sys.argv[1]), int(sys.argv[2])
MONTHS = json.load(open("months.json")); SYMS = json.load(open("symbols.json"))[SHARD::NSH]
K_URL = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/5m/{s}-5m-{m}.zip"
F_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"
TFS = {"5m": 5, "15m": 15, "1h": 60}
MIN_CANDLES = 8640          # کمتر از ۳۰ روز دادهٔ ۵ دقیقه‌ای = «تاریخچهٔ ناکافی»، بررسی نمی‌شه
sess = requests.Session()


def fetch(url):
    for _ in range(3):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 404: return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return pd.read_csv(z.open(z.namelist()[0]), header=None)
        except Exception:
            pass
    return "error"


def klines(args):
    s, m = args
    df = fetch(K_URL.format(s=s, m=m))
    if df is None or isinstance(df, str): return m, df
    df = df.iloc[:, :5]; df = df[pd.to_numeric(df[0], errors="coerce").notna()].astype(float)
    df.columns = ["t", "o", "h", "l", "c"]; return m, df


def funding(args):
    s, m = args
    df = fetch(F_URL.format(s=s, m=m))
    if df is None or isinstance(df, str): return None
    df = df[pd.to_numeric(df[0], errors="coerce").notna()]
    return pd.DataFrame(dict(t=df[0].astype(float), rate=df[df.columns[-1]].astype(float)))


def resample(df, minutes):
    if minutes == 5: return df
    g = df.groupby((df.t // (minutes * 60_000)).astype(np.int64))
    return pd.DataFrame(dict(t=g.t.first(), o=g.o.first(), h=g.h.max(), l=g.l.min(), c=g.c.last())).reset_index(drop=True)


rows, quality = [], []
with ThreadPoolExecutor(8) as ex:
    for s in SYMS:
        res = list(ex.map(klines, [(s, m) for m in MONTHS]))
        got = [m for m, d in res if isinstance(d, pd.DataFrame)]
        errs = [m for m, d in res if isinstance(d, str)]
        parts = [d for m, d in res if isinstance(d, pd.DataFrame)]
        q = dict(symbol=s, months_found=len(got), months_error=len(errs), first_month=got[0] if got else "",
                 last_month=got[-1] if got else "", candles=0, missing_candles=0, gaps_over_1h=0,
                 funding_data=False, status="")
        if not parts:
            q["status"] = "no_data_in_period"; quality.append(q); continue
        df5 = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
        dt = np.diff(df5.t.values) / 300_000
        q.update(candles=len(df5), missing_candles=int(np.clip(dt - 1, 0, None).sum()), gaps_over_1h=int((dt > 12).sum()))
        if len(df5) < MIN_CANDLES:
            q["status"] = "insufficient_history"; quality.append(q); continue
        q["status"] = "ok" if len(got) == len(MONTHS) and q["gaps_over_1h"] == 0 else "partial"
        fr = [f for f in ex.map(funding, [(s, m) for m in got]) if f is not None]
        F = pd.concat(fr).sort_values("t") if fr else None
        q["funding_data"] = F is not None and len(F) > 0
        if q["funding_data"]:
            ft, cum = F.t.values, np.r_[0, np.cumsum(F.rate.values)]
        for tf, mins in TFS.items():
            df = resample(df5, mins)
            for name, fn in STRATEGIES.items():
                for tr in fn(df):
                    if q["funding_data"]:
                        # فاندینگ واقعی: مجموع نرخ‌ها بین ورود و خروج؛ لانگ نرخ مثبت رو می‌پردازه، شورت می‌گیره
                        a, b = np.searchsorted(ft, tr["entry_t"], "right"), np.searchsorted(ft, tr["exit_t"], "right")
                        fsum = cum[b] - cum[a]
                        tr["funding_R"] = -tr["side"] * fsum / tr["stop_pct"]
                        tr["R_net"] = tr["R_gross"] - tr["fee_R"] + tr["funding_R"]
                    tr.update(symbol=s, tf=tf); rows.append(tr)
        quality.append(q)
        print(f"{s}: {len(df5)} candles, {q['status']}", flush=True)

os.makedirs("out", exist_ok=True)
pd.DataFrame(rows).to_csv(f"out/trades_{SHARD:02d}.csv", index=False)
pd.DataFrame(quality).to_csv(f"out/quality_{SHARD:02d}.csv", index=False)
print(f"shard {SHARD}: {sum(q['status'] in ('ok', 'partial') for q in quality)}/{len(SYMS)} symbols tested, {len(rows)} trades")
