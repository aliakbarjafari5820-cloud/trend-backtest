"""تست واحد اسکالپ: (۱) هیچ سیگنالی به آینده نگاه نکنه (۲) روی قیمت تصادفی مزیت قبل از هزینه نداشته باشه (۳) هزینه‌ها درست."""
import numpy as np, pandas as pd, sys
from scalp import strategies, simulate

def rw(n=40000, seed=1):
    rng = np.random.default_rng(seed); c = 100*np.exp(np.cumsum(rng.normal(0, 0.002, n))); o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.001, n))*c
    return pd.DataFrame(dict(t=1.7e12 + np.arange(n)*300_000., o=o, h=np.maximum(o, c)+w, l=np.minimum(o, c)-w, c=c, v=rng.random(n)*100+1))

def test_no_lookahead():
    df = rw(); full = strategies(df); cut = 25000
    part = strategies(df.iloc[:cut].reset_index(drop=True))
    for name in full:
        for key in ("le", "se", "xl", "xs"):
            if key not in full[name]: continue
            a = np.nan_to_num(np.asarray(full[name][key][:cut - 1], float)); b = np.nan_to_num(np.asarray(part[name][key][:cut - 1], float))
            assert (a == b).all(), (name, key, int((a != b).sum()))

def test_random_walk_no_edge():
    df = rw(seed=3); bad = []
    for name, cfg in strategies(df).items():
        R = simulate(df, **cfg)
        if len(R) < 30: continue
        m, se = R.R_gross.mean(), R.R_gross.std()/np.sqrt(len(R))
        if m > 4*se + 0.03: bad.append((name, round(m, 3), round(se, 3)))
    assert not bad, bad

def test_costs():
    df = rw(); R = simulate(df, **strategies(df)["EMA9_21_X"])
    assert (R.R_net < R.R_gross).all() and len(R) > 10
    tp = R[R.kind == 1]; assert np.allclose(tp.R_gross, 2.0, atol=1e-9)    # TP = ۳ ATR / ریسک ۱.۵ ATR = ۲R

def test_htf_no_lookahead():
    from scalp import htf_trend
    df = rw(60000, seed=5); full = htf_trend(df)
    for cut in (20001, 40017):
        assert (htf_trend(df.iloc[:cut].reset_index(drop=True)) == full[:cut]).all(), cut

if __name__ == "__main__":
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("OK  ", n)
            except AssertionError as e: fails += 1; print("FAIL", n, e)
    sys.exit(1 if fails else 0)
