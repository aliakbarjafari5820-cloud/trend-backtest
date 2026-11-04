"""استراتژی ۱ — Classic Price Action.  قوانین: 01_PRICE_ACTION.md   |   تعریف‌های مشترک: 00_BACKTEST_PROTOCOL.md
اجرای جداگانه روی یک ارز:  python strategies/01_PRICE_ACTION.py BTCUSDT 1h"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smc"))
from engine import run_pa, PARAMS, COSTS


def run(df):
    """df: ستون‌های t (میلی‌ثانیه), o, h, l, c — خروجی: فهرست معاملات با R خالص بعد از کارمزد/اسلیپیج/فاندینگ."""
    return run_pa(df)


if __name__ == "__main__":
    from single import cli
    cli(run, "PRICE_ACTION")
