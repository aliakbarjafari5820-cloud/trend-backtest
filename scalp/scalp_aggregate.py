"""جمع‌بندی اسکالپ با پروتکل ازپیش‌تعیین‌شده برای جلوگیری از «شانسِ آماری»:
۶۰٪ اول (توسعه): PF > ۱.۱، میانگین > ۰.۰۵R خالص، حداقل ۱۰۰ معامله  →  ۲۰٪ بعد (اعتبارسنجی): میانگین > ۰
→ ۲۰٪ آخر (تست نهایی، فقط یک بار دیده می‌شه): میانگین > ۰، PF > ۱.۰۵، سودده روی حداقل ۳ ارز از ۵."""
import glob, json, os, numpy as np, pandas as pd
T = pd.concat([pd.read_csv(f) for f in glob.glob("all/**/scalp_*.csv", recursive=True)], ignore_index=True)
t0, t1 = T.entry_t.min(), T.entry_t.max(); c1, c2 = t0 + 0.6*(t1-t0), t0 + 0.8*(t1-t0)
T["seg"] = np.where(T.entry_t < c1, "dev", np.where(T.entry_t < c2, "val", "test"))
os.makedirs("results", exist_ok=True)

def st(X):
    R = X.R_net.values; w, l = R[R > 0], R[R <= 0]
    eq = np.cumprod(1 + 0.01*X.sort_values("exit_t").R_net.values) if len(R) else np.array([1.0])
    return pd.Series(dict(n=len(R), win=round(100*len(w)/max(1, len(R)), 1), PF=round(w.sum()/max(1e-9, -l.sum()), 2),
                          expR=round(R.mean(), 3) if len(R) else 0, expR_gross=round(X.R_gross.mean(), 3) if len(R) else 0,
                          cost_R=round(X.cost_R.mean(), 3) if len(R) else 0,
                          coins_pos=int((X.groupby("symbol").R_net.sum() > 0).sum()),
                          ret_pct=round((eq[-1]-1)*100, 1), maxDD=round((1-eq/np.maximum.accumulate(eq)).max()*100, 1)))

rows = []
for (s, tf), X in T.groupby(["strategy", "tf"]):
    r = {"strategy": s, "tf": tf}
    for seg in ("all", "dev", "val", "test"):
        Y = X if seg == "all" else X[X.seg == seg]
        for k, v in st(Y).items(): r[f"{seg}_{k}"] = v
    r["pass_dev"] = r["dev_PF"] > 1.1 and r["dev_expR"] > 0.05 and r["dev_n"] >= 100
    r["pass_val"] = r["pass_dev"] and r["val_expR"] > 0
    r["pass_test"] = r["pass_val"] and r["test_expR"] > 0 and r["test_PF"] > 1.05 and r["test_coins_pos"] >= 3
    rows.append(r)
A = pd.DataFrame(rows).sort_values("all_expR", ascending=False)
A.to_csv("results/scalp_summary.csv", index=False)
coin = T.groupby(["strategy", "tf", "symbol", "seg"]).R_net.agg(["count", "mean", "sum"]).round(3).reset_index()
coin.to_csv("results/scalp_by_coin.csv", index=False)
ctx = dict(configs=len(A), positive_all_net=int((A.all_expR > 0).sum()), positive_all_gross=int((A.all_expR_gross > 0).sum()),
           positive_test=int((A.test_expR > 0).sum()), passed_dev=A[A.pass_dev][["strategy", "tf"]].values.tolist(),
           passed_val=A[A.pass_val][["strategy", "tf"]].values.tolist(), passed_test=A[A.pass_test][["strategy", "tf"]].values.tolist(),
           period=[str(pd.to_datetime(t0, unit="ms").date()), str(pd.to_datetime(t1, unit="ms").date())],
           cut_val=str(pd.to_datetime(c1, unit="ms").date()), cut_test=str(pd.to_datetime(c2, unit="ms").date()))
json.dump(ctx, open("results/scalp_verdict.json", "w"), indent=1)
# منحنی R تجمعی برای نمودار
curves = {}
for (s, tf), X in T.groupby(["strategy", "tf"]):
    Y = X.sort_values("exit_t"); d = pd.to_datetime(Y.exit_t, unit="ms").dt.floor("D")
    cs = Y.R_net.groupby(d.values).sum().cumsum()
    curves[f"{s} {tf}"] = [[str(k.date()), round(v, 1)] for k, v in cs.resample("W").last().ffill().items()]
json.dump(curves, open("results/scalp_curves.json", "w"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["strategy", "tf", "all_n", "all_win", "all_PF", "all_expR", "all_expR_gross", "all_cost_R", "all_coins_pos",
        "dev_expR", "val_expR", "test_expR", "test_PF", "test_coins_pos", "pass_dev", "pass_val", "pass_test"]
print(A[cols].to_string(index=False)); print(json.dumps(ctx, indent=1))
