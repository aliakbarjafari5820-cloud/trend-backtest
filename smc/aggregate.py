"""جمع‌بندی علمی: شبیه‌سازی سبد با ۱۰٬۰۰۰ USDT، ریسک ۱٪، حداکثر ۳ معاملهٔ همزمان، تقسیم ۶۰/۲۰/۲۰،
Walk-Forward، بازار صعودی/نزولی (بر اساس روند بیت‌کوین)، ریسک لیکویید، و جواب ۷ سؤال."""
import glob, json, os, time, requests, numpy as np, pandas as pd

CAP0, MAX_OPEN, MMR = 10_000.0, 3, 0.005          # سرمایهٔ اولیه، حداکثر معاملهٔ همزمان، مارجین نگهداری
rng = np.random.default_rng(42)
os.makedirs("results", exist_ok=True); os.makedirs("results/by_tf", exist_ok=True)

T = pd.concat([pd.read_csv(f) for f in glob.glob("all/**/trades_*.csv", recursive=True) if os.path.getsize(f) > 5], ignore_index=True)
Q = pd.concat([pd.read_csv(f) for f in glob.glob("all/**/quality_*.csv", recursive=True)], ignore_index=True)
Q.sort_values("symbol").to_csv("results/data_quality.csv", index=False)
T["day"] = pd.to_datetime(T.entry_t, unit="ms").dt.floor("D")

# --- ریسک لیکویید (مارجین ایزوله): اهرم = ریسک/فاصلهٔ حد ضرر؛ اگه ضرر واقعی (مثلاً با گپ) از فاصلهٔ لیکویید بیشتر بشه، کل مارجین از دست رفته
lev = (T.risk_frac / T.stop_pct).clip(lower=1e-9)
liq_dist = 1 / lev - MMR
T["liq_R"] = -liq_dist / T.stop_pct
T["liquidated"] = (T.R_gross < T.liq_R) & (liq_dist > 0)
T.loc[T.liquidated, "R_net"] = T.loc[T.liquidated, "liq_R"] - T.loc[T.liquidated, "fee_R"]
T["leverage"] = lev

# --- رژیم بازار: روزانهٔ بیت‌کوین؛ صعودی = کلوز و SMA50 بالای SMA200، نزولی = هر دو زیرش، بقیه خنثی
t0 = int(T.entry_t.min()) - 260 * 86_400_000; rows = []
try:
    while t0 < int(time.time() * 1000):
        d = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": "BTCUSDT", "interval": "1d", "startTime": t0, "limit": 1000}, timeout=30).json()
        if not d: break
        rows += d; t0 = d[-1][0] + 1
except Exception as e:
    print("BTC regime data unavailable:", e)
if not rows:   # بدون دادهٔ بیت‌کوین، رژیم «نامشخص»
    rows = [[int(x), 0, 0, 0, 1] for x in range(int(T.entry_t.min()) - 260 * 86_400_000, int(T.exit_t.max()) + 86_400_000, 86_400_000)]
btc = pd.DataFrame(rows).iloc[:, [0, 4]].astype(float); btc.columns = ["t", "c"]
btc["day"] = pd.to_datetime(btc.t, unit="ms").dt.floor("D"); s50, s200 = btc.c.rolling(50).mean(), btc.c.rolling(200).mean()
btc["regime"] = np.where((btc.c > s200) & (s50 > s200), "bull", np.where((btc.c < s200) & (s50 < s200), "bear", "neutral"))
T = T.merge(btc[["day", "regime"]].drop_duplicates("day"), on="day", how="left"); T["regime"] = T.regime.fillna("unknown")

tmin, tmax = T.entry_t.min(), T.exit_t.max()
cut1, cut2 = tmin + 0.6 * (tmax - tmin), tmin + 0.8 * (tmax - tmin)
T["segment"] = np.where(T.entry_t < cut1, "1_dev_60", np.where(T.entry_t < cut2, "2_val_20", "3_test_20"))


def trade_stats(R):
    R = np.asarray(R, float); w, ls = R[R > 0], R[R <= 0]
    return dict(trades=len(R), win_pct=round(100 * len(w) / max(1, len(R)), 1),
                PF=round(w.sum() / max(1e-9, -ls.sum()), 2), expectancy_R=round(R.mean(), 3) if len(R) else 0.0,
                avg_win_R=round(w.mean(), 2) if len(w) else 0.0, avg_loss_R=round(ls.mean(), 2) if len(ls) else 0.0)


def portfolio(X):
    """سبد مشترک: سیگنال‌ها به ترتیب زمان (هم‌زمان‌ها تصادفی)، اگه کمتر از ۳ معاملهٔ باز بود قبول می‌شه.
    حجم = ریسک ۱٪ اکوئیتی لحظهٔ ورود (یا کمتر به‌خاطر سقف اهرم). سود/ضرر موقع خروج ثبت می‌شه."""
    X = X.assign(r=rng.random(len(X))).sort_values(["entry_t", "r"])
    eq, open_, taken, curve = CAP0, [], [], [(int(X.entry_t.min()) if len(X) else 0, CAP0)]
    for row in X.itertuples():
        due = sorted(p for p in open_ if p[0] <= row.entry_t)
        for p in due: eq += p[1]; curve.append((p[0], eq))
        open_ = [p for p in open_ if p[0] > row.entry_t]
        if len(open_) >= MAX_OPEN or eq <= 0: continue
        open_.append((row.exit_t, eq * row.risk_frac * row.R_net)); taken.append(row.Index)
    for p in sorted(open_): eq += p[1]; curve.append((p[0], eq))
    c = pd.Series([v for _, v in curve], index=pd.to_datetime([k for k, _ in curve], unit="ms"))
    return c, X.loc[taken]


def curve_metrics(c):
    if len(c) < 2: return dict(net_profit_usdt=0, net_profit_pct=0, maxDD_pct=0, sharpe=0, sortino=0, calmar=0)
    daily = c.groupby(c.index.floor("D")).last()
    daily = daily.reindex(pd.date_range(daily.index[0], daily.index[-1], freq="D")).ffill()
    r = daily.pct_change().dropna()
    dd = (1 - c / c.cummax()).max()
    years = max((c.index[-1] - c.index[0]).days / 365, 1e-9)
    cagr = (c.iloc[-1] / CAP0) ** (1 / years) - 1 if c.iloc[-1] > 0 else -1
    neg = r[r < 0]
    return dict(net_profit_usdt=round(c.iloc[-1] - CAP0), net_profit_pct=round((c.iloc[-1] / CAP0 - 1) * 100, 1),
                maxDD_pct=round(dd * 100, 1),
                sharpe=round(r.mean() / r.std() * np.sqrt(365), 2) if r.std() > 0 else 0,
                sortino=round(r.mean() / neg.std() * np.sqrt(365), 2) if len(neg) > 1 and neg.std() > 0 else 0,
                calmar=round(cagr / dd, 2) if dd > 0 else 0)


main, seg_rows, reg_rows, coin_rows, setup_rows, equity = [], [], [], [], [], {}
for (s, tf), X in T.groupby(["strategy", "tf"]):
    c, tk = portfolio(X); equity[(s, tf)] = c
    st, cm = trade_stats(tk.R_net), curve_metrics(c)
    per_coin = X.groupby("symbol").R_net.agg(["count", "sum", "mean"])
    main.append(dict(strategy=s, tf=tf, **cm, **st,
                     long_expR=trade_stats(tk[tk.side == 1].R_net)["expectancy_R"],
                     short_expR=trade_stats(tk[tk.side == -1].R_net)["expectancy_R"],
                     hold_hours=round(tk.hours.mean(), 1), max_leverage=round(tk.leverage.max(), 1),
                     liquidations=int(tk.liquidated.sum()),
                     all_signals=len(X), all_signals_expR=round(X.R_net.mean(), 3),
                     expR_before_costs=round(X.R_gross.mean(), 3),
                     coins_tested=len(per_coin), coins_profitable_pct=round((per_coin["sum"] > 0).mean() * 100, 1)))
    for seg, Y in X.groupby("segment"):
        cs, tks = portfolio(Y)
        seg_rows.append(dict(strategy=s, tf=tf, segment=seg, **curve_metrics(cs), **trade_stats(tks.R_net)))
    for reg, Y in X.groupby("regime"):
        reg_rows.append(dict(strategy=s, tf=tf, regime=reg, **trade_stats(Y.R_net)))
    for sym, Y in X.groupby("symbol"):
        coin_rows.append(dict(strategy=s, tf=tf, symbol=sym, **trade_stats(Y.R_net), sum_R=round(Y.R_net.sum(), 2)))
    for su, Y in X.groupby("setup"):
        setup_rows.append(dict(strategy=s, tf=tf, setup=su, **trade_stats(Y.R_net)))

M = pd.DataFrame(main).sort_values(["tf", "strategy"]); SEG = pd.DataFrame(seg_rows); REG = pd.DataFrame(reg_rows)
COIN = pd.DataFrame(coin_rows); SET = pd.DataFrame(setup_rows)
M.to_csv("results/summary.csv", index=False); SEG.to_csv("results/split_60_20_20.csv", index=False)
REG.to_csv("results/by_regime.csv", index=False); COIN.to_csv("results/by_coin.csv", index=False)
SET.to_csv("results/by_setup.csv", index=False)
for tf, Y in M.groupby("tf"): Y.to_csv(f"results/by_tf/summary_{tf}.csv", index=False)

# --- Walk-Forward: هر بار ۶ ماه آموزش → بهترین ترکیب (استراتژی × تایم) بر اساس Calmar → ۳ ماه بعد اجرا
start = pd.to_datetime(tmin, unit="ms").normalize(); end = pd.to_datetime(tmax, unit="ms")
wf, wf_eq, p = [], CAP0, start
while p + pd.DateOffset(months=9) <= end + pd.Timedelta(days=1):
    tr0, tr1, te1 = p, p + pd.DateOffset(months=6), p + pd.DateOffset(months=9)
    ms = lambda x: int(x.value // 1_000_000)
    best, bestv = None, -1e9
    for (s, tf), X in T.groupby(["strategy", "tf"]):
        Y = X[(X.entry_t >= ms(tr0)) & (X.entry_t < ms(tr1))]
        if len(Y) < 30: continue
        v = curve_metrics(portfolio(Y)[0])["calmar"]
        if v > bestv: best, bestv = (s, tf), v
    if best is None: break
    X = T[(T.strategy == best[0]) & (T.tf == best[1])]
    Z = X[(X.entry_t >= ms(tr1)) & (X.entry_t < ms(te1))]
    cz, tz = portfolio(Z); ret = cz.iloc[-1] / CAP0 - 1 if len(cz) else 0
    wf_eq *= 1 + ret
    wf.append(dict(train=f"{tr0.date()}→{tr1.date()}", test=f"{tr1.date()}→{te1.date()}", chosen=f"{best[0]} {best[1]}",
                   train_calmar=round(bestv, 2), test_return_pct=round(ret * 100, 1), test_trades=len(tz),
                   test_expR=trade_stats(tz.R_net)["expectancy_R"]))
    p += pd.DateOffset(months=3)
WF = pd.DataFrame(wf); WF.to_csv("results/walk_forward.csv", index=False)

# --- جواب سؤال‌ها
def best(col, df=M, high=True):
    r = df.sort_values(col, ascending=not high).iloc[0]; return f"{r.strategy} {r.tf} ({col}={r[col]})"
test = SEG[SEG.segment == "3_test_20"]
bull, bear = REG[REG.regime == "bull"], REG[REG.regime == "bear"]
tf_score = M.groupby("tf")[["net_profit_pct", "calmar", "expectancy_R"]].mean().round(2)
answers = {
    "1_most_profitable": best("net_profit_pct"),
    "2_lowest_drawdown": best("maxDD_pct", high=False),
    "3_most_coins_positive": best("coins_profitable_pct"),
    "4_best_timeframe_by_avg_calmar": tf_score.calmar.idxmax(),
    "5_best_in_bull_by_expR": best("expectancy_R", bull) if len(bull) else "n/a",
    "6_best_in_bear_by_expR": best("expectancy_R", bear) if len(bear) else "n/a",
    "7_profitable_out_of_sample": test[test.net_profit_pct > 0][["strategy", "tf", "net_profit_pct"]].to_dict("records"),
    "walk_forward_final_equity_usdt": round(wf_eq),
}
json.dump(answers, open("results/answers.json", "w"), ensure_ascii=False, indent=1)

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
out = []
out.append("=== DATA QUALITY ===\n" + Q.status.value_counts().to_string())
out.append("=== MAIN (portfolio 10k USDT, 1% risk, max 3 open) ===\n" + M.to_string(index=False))
out.append("=== 60/20/20 SPLIT ===\n" + SEG[["strategy", "tf", "segment", "net_profit_pct", "maxDD_pct", "sharpe", "calmar", "trades", "win_pct", "PF", "expectancy_R"]].to_string(index=False))
out.append("=== REGIME (all signals) ===\n" + REG.pivot_table(index=["strategy", "tf"], columns="regime", values="expectancy_R").round(3).to_string())
out.append("=== SETUP ===\n" + SET.to_string(index=False))
out.append("=== WALK FORWARD ===\n" + WF.to_string(index=False))
out.append("=== TIMEFRAME AVG ===\n" + tf_score.to_string())
out.append("=== ANSWERS ===\n" + json.dumps(answers, ensure_ascii=False, indent=1))
open("results/report.txt", "w").write("\n\n".join(out)); print("\n\n".join(out))
