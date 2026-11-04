"""استراتژی ۲ — Smart Money Concepts.  قوانین: 02_SMART_MONEY.md   |   تعریف‌های مشترک: 00_BACKTEST_PROTOCOL.md
اجرای جداگانه روی یک ارز:  python strategies/02_SMART_MONEY.py BTCUSDT 1h"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smc"))
from engine import run_smc, PARAMS, COSTS


def run(df):
    """سوئیپ نقدینگی → BOS/CHoCH → OB/FVG → برگشت → کندل تأیید."""
    return run_smc(df, hybrid=False)


if __name__ == "__main__":
    from single import cli
    cli(run, "SMART_MONEY")
