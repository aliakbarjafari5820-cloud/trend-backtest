"""جمع‌بندی: شبیه‌سازی سبد (ریسک ۱٪، حداکثر ۳ معاملهٔ همزمان، با کارمزد/اسلیپیج/فاندینگ) + آمار همهٔ سیگنال‌ها."""
import glob, numpy as np, pandas as pd
T = pd.concat([pd.read_csv(f) for f in glob.glob("all/**/trades_*.csv", recursive=True) if open(f).read(10)], ignore_index=True)
T["year"] = pd.to_datetime(T.entry_t, unit="ms").dt.year
rng = np.random.default_rng(42)

def stats(R):
    R = np.asarray(R); w, ls = R[R > 0], R[R <= 0]
    return dict(n=len(R), win=round(100*len(w)/max(1, len(R)), 1),
                PF=round(w.sum()/max(1e-9, -ls.sum()), 2), expR=round(R.mean(), 3) if len(R) else 0,
                avgW=round(w.mean(), 2) if len(w) else 0, avgL=round(ls.mean(), 2) if len(ls) else 0)

def portfolio(X, max_open=3):
    X = X.assign(r=rng.random(len(X))).sort_values(["entry_t", "r"])
    eq, peak, mdd, open_ = 1.0, 1.0, 0.0, []; taken = []
    for row in X.itertuples():
        for p in sorted([p for p in open_ if p[0] <= row.entry_t]):
            eq += p[1]; peak = max(peak, eq); mdd = max(mdd, 1 - eq/peak)
        open_ = [p for p in open_ if p[0] > row.entry_t]
        if len(open_) >= max_open or eq <= 0: continue
        open_.append((row.exit_t, eq*row.risk_frac*row.R_net)); taken.append(row.Index)
    for p in sorted(open_):
        eq += p[1]; peak = max(peak, eq); mdd = max(mdd, 1 - eq/peak)
    return eq, mdd, X.loc[taken]

rows = []; setup_rows = []
for (s, tf), X in T.groupby(["strategy", "tf"]):
    eq, mdd, tk = portfolio(X)
    st = stats(tk.R_net); allst = stats(X.R_net)
    sym_pos = (X.groupby("symbol").R_net.sum() > 0).mean()*100
    yr = X.groupby("year").R_net.mean().round(3).to_dict()
    rows.append({"strategy": s, "tf": tf, "net_profit_%": round((eq-1)*100, 1), "maxDD_%": round(mdd*100, 1),
                 "trades": st["n"], "win_%": st["win"], "PF": st["PF"], "expectancy_R": st["expR"],
                 "avg_win_R": st["avgW"], "avg_loss_R": st["avgL"],
                 "long_expR": stats(tk[tk.side == 1].R_net)["expR"], "short_expR": stats(tk[tk.side == -1].R_net)["expR"],
                 "hold_h": round(tk.hours.mean(), 1), "ALL_signals": allst["n"], "ALL_PF": allst["PF"],
                 "ALL_expR": allst["expR"], "ALL_expR_gross": round(X.R_gross.mean(), 3),
                 "coins_profitable_%": round(sym_pos, 1), "ALL_expR_by_year": yr})
    for st_, Y in X.groupby("setup"):
        setup_rows.append({"strategy": s, "tf": tf, "setup": st_, **stats(Y.R_net)})
R = pd.DataFrame(rows).sort_values(["tf", "strategy"])
R.to_csv("summary.csv", index=False); pd.DataFrame(setup_rows).to_csv("by_setup.csv", index=False)
coin = T.groupby(["strategy", "tf", "symbol"]).R_net.agg(["count", "sum", "mean"]).round(3).reset_index()
coin.to_csv("by_coin.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(R.drop(columns=["ALL_expR_by_year"]).to_string(index=False))
print(pd.DataFrame(setup_rows).to_string(index=False))
print(R[["strategy", "tf", "ALL_expR_by_year"]].to_string(index=False))
print("symbols:", T.symbol.nunique(), "| total signals:", len(T))
