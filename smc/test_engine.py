"""تست واحد موتور: هر بخش جدا با داده‌های ساختگی که جواب درستشون معلومه.  اجرا: python -m pytest smc -q"""
import numpy as np, pandas as pd
from engine import swing_flags, simulate_exit, find_zone, run_pa, run_smc, Structure, PARAMS


def bars(closes, wick=0.1, opens=None):
    c = np.array(closes, float)
    o = np.r_[c[0], c[:-1]] if opens is None else np.array(opens, float)
    h = np.maximum(o, c) + wick; l = np.minimum(o, c) - wick
    return pd.DataFrame(dict(t=np.arange(len(c)) * 3_600_000, o=o, h=h, l=l, c=c))


def path(points, steps=4):
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        out += list(np.linspace(a, b, steps + 1)[1:])
    return out


def test_swings_confirmed_two_bars_later():
    h = np.array([1, 2, 5, 2, 1, 1, 1.]); l = h - 0.5
    sh, sl = swing_flags(h, l, 2)
    assert sh[2] and sh.sum() >= 1
    S = Structure(h, l, 2)
    for i in range(4): S.update(i)
    assert S.highs == []                      # هنوز تأیید نشده
    S.update(4); assert S.highs[0] == (5.0, 2)  # در کندل ۴ تأیید می‌شه


def test_trend_hh_hl():
    df = bars(path([100, 110, 105, 115, 108, 120, 112]))
    S = Structure(df.h.values, df.l.values, 2)
    for i in range(len(df)): S.update(i)
    assert S.trend() == 1


def test_exit_same_bar_counts_as_stop():
    o = np.array([100, 100.]); h = np.array([100, 103.]); l = np.array([100, 98.]); c = o; t = np.array([0, 1])
    i, px = simulate_exit(o, h, l, c, t, 1, 1, 99, 102)
    assert px == 99


def test_exit_gap_through_stop_uses_open():
    o = np.array([100, 100, 95.]); h = np.array([100, 100.5, 96]); l = np.array([100, 99.5, 94]); c = o; t = np.arange(3)
    i, px = simulate_exit(o, h, l, c, t, 1, 1, 99, 102)
    assert i == 2 and px == 95


def test_find_zone_ob_and_fvg():
    o = np.array([10, 10.5, 10.4, 11.0, 12.0]); c = np.array([10.5, 10.4, 11.0, 12.0, 12.5])
    h = np.maximum(o, c) + 0.05; l = np.minimum(o, c) - 0.05
    zone, zt = find_zone(o, h, l, c, 0, 4, 1)
    assert zt in ("OB", "OB+FVG", "FVG") and zone[0] < zone[1]
    # کندل ۱ نزولیه → OB = [10.35, 10.55]
    z2, t2 = find_zone(o, h, l, c, 0, 2, 1)
    assert t2 == "OB" and np.isclose(z2[0], 10.35) and np.isclose(z2[1], 10.55)


def PA_DF():
    return bars(path([100, 110, 105, 115, 108, 120, 116, 122.5, 120.3], 3) + [121.5, 123, 126, 130, 134, 138, 142, 146, 150])


def test_pa_long_breakout_pullback():
    df = PA_DF()
    tr = run_pa(df)
    assert len(tr) >= 1, "باید یه معاملهٔ خرید پیدا کنه"
    t0 = tr[0]
    # حد ضرر زیر آخرین کف (۱۱۶) با ۰.۱ ATR فاصله، و حد سود دقیقاً ۲R خالص بعد از کارمزد و اسلیپیج
    assert t0["side"] == 1 and 115 < t0["sl"] < 116 and abs(t0["R_gross"] - t0["fee_R"] - 2.0) < 1e-9


def test_smc_long_sweep_bos_ob():
    pts = [110, 104, 107, 100, 106, 103]                    # کف ۱۰۰، سقف ۱۰۶
    closes = path(pts) + [101.5, 100.5]                     # نزدیک کف
    opens = None
    df = bars(closes)
    # کندل سوئیپ: سایه تا ۹۸، کلوز ۱۰۱
    sweep = pd.DataFrame(dict(t=[0], o=[100.6], h=[101.2], l=[98.0], c=[101.0]))
    bear = pd.DataFrame(dict(t=[0], o=[101.0], h=[101.1], l=[100.4], c=[100.5]))   # OB
    up = bars([100.5, 102.5, 104.5, 107.0, 107.5], wick=0.05).iloc[1:]
    back = bars([107.5, 104, 101.5, 100.8], wick=0.05).iloc[1:]                      # برگشت به OB
    conf = pd.DataFrame(dict(t=[0], o=[100.8], h=[101.6], l=[100.7], c=[101.5]))     # کندل تأیید
    rally = bars([101.5, 104, 107, 110, 113, 116], wick=0.05).iloc[1:]
    df = pd.concat([df, sweep, bear, up, back, conf, rally], ignore_index=True)
    df["t"] = np.arange(len(df)) * 3_600_000
    tr = run_smc(df)
    longs = [x for x in tr if x["side"] == 1]
    assert longs, "باید یه معاملهٔ خرید SMC پیدا کنه"
    x = longs[0]
    assert x["sl"] < 98.0 and x["setup"] in ("OB", "OB+FVG", "FVG")


def test_weak_breakout_ignored():
    # شکست با کلوز کمتر از ۰.۵ ATR بالای سقف نباید معامله بسازه
    df = bars(path([100, 110, 105, 115, 108, 120, 116, 120.4, 119.9], 3) + [120.3, 120.2, 120.4, 120.1])
    assert run_pa(df) == []


def test_stop_outside_atr_range_rejected():
    from engine import make_trade, PARAMS
    df = PA_DF()
    assert make_trade(df, 20, 1, df.o[20] - 10, PARAMS, "PA", "x", 1.0) is None     # ۱۰ ATR > ۳ ATR
    assert make_trade(df, 20, 1, df.o[20] - 0.05, PARAMS, "PA", "x", 1.0) is None   # ۰.۰۵ ATR < ۰.۱ ATR


def test_costs_reduce_R():
    t0 = run_pa(PA_DF())[0]
    assert t0["R_net"] < t0["R_gross"]


def mirror(df, base=300.0):
    return pd.DataFrame(dict(t=df.t, o=base-df.o, h=base-df.l, l=base-df.h, c=base-df.c))


def test_pa_short_is_mirror_of_long():
    s = [x for x in run_pa(mirror(PA_DF())) if x["side"] == -1]
    assert s and abs(s[0]["R_gross"] - s[0]["fee_R"] - 2.0) < 1e-9


def test_random_walk_no_lookahead_edge():
    # روی قیمت تصادفی، بدون کارمزد میانگین R نباید به‌طور معنادار مثبت باشه (اگه بود یعنی نگاه به آینده)
    rng = np.random.default_rng(1); c = 100*np.exp(np.cumsum(rng.normal(0, 0.004, 60000)))
    o = np.r_[c[0], c[:-1]]; w = np.abs(rng.normal(0, 0.002, len(c)))*c
    df = pd.DataFrame(dict(t=np.arange(len(c))*3_600_000, o=o, h=np.maximum(o, c)+w, l=np.minimum(o, c)-w, c=c))
    for name, fn in [("PA", run_pa), ("SMC", run_smc)]:
        R = np.array([x["R_gross"] for x in fn(df)])
        assert len(R) > 100, name
        se = R.std()/np.sqrt(len(R))
        assert R.mean() < 3*se, (name, R.mean(), se)


if __name__ == "__main__":
    import sys, inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try: fn(); print("OK  ", name)
            except AssertionError as e: fails += 1; print("FAIL", name, e)
    sys.exit(1 if fails else 0)
