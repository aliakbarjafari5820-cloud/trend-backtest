"""استراتژی ۳ — Hybrid (Price Action + Smart Money).  قوانین: 03_HYBRID_PA_SMC.md   |   تعریف‌های مشترک: 00_BACKTEST_PROTOCOL.md
اجرای جداگانه روی یک ارز:  python strategies/03_HYBRID_PA_SMC.py BTCUSDT 1h"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smc"))
from engine import run_smc, PARAMS, COSTS


def run(df):
    """جهت با پرایس اکشن (HH/HL یا LH/LL) + ورود اسمارت مانی."""
    return run_smc(df, hybrid=True)


if __name__ == "__main__":
    from single import cli
    cli(run, "HYBRID")
