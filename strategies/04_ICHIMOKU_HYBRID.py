"""استراتژی ۴ — ترکیب ایچیموکو ۴ ساعته + پرایس اکشن + ورود FVG اسمارت مانی.  قوانین: 04_ICHIMOKU_HYBRID.md
اجرای جداگانه روی یک ارز:  python strategies/04_ICHIMOKU_HYBRID.py BTCUSDT 1h [kijun]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smc"))
from engine import run_smc


def run(df, exit_mode="2R"):
    """exit_mode: "2R" (حد سود ثابت ۲R خالص) یا "kijun" (خروج با کلوز اون‌ور کیجون)."""
    trail = exit_mode == "kijun"
    return run_smc(df, hybrid=True, ichi=True, zones="FVG", trail=trail, kind="ICHI_KIJUN" if trail else "ICHI_2R")


if __name__ == "__main__":
    from single import cli
    mode = "kijun" if "kijun" in sys.argv[3:] else "2R"
    cli(lambda df: run(df, mode), f"ICHIMOKU_HYBRID_{mode}")
