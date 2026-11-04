"""فهرست همهٔ قراردادهای دائمی USDT (شامل حذف‌شده‌ها، برای جلوگیری از سوگیری بقا) و ۲۴ ماه آخر."""
import json, re, datetime, requests
U = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
syms, marker = [], ""
while True:
    x = requests.get(U, params={"delimiter": "/", "prefix": "data/futures/um/monthly/klines/", "marker": marker}, timeout=30).text
    syms += re.findall(r"<Prefix>data/futures/um/monthly/klines/([^/<]+)/</Prefix>", x)
    if "<IsTruncated>true</IsTruncated>" not in x: break
    marker = re.findall(r"<NextMarker>([^<]+)</NextMarker>", x)[0]
syms = sorted({s for s in syms if s.endswith("USDT") and "_" not in s})
today = datetime.date.today().replace(day=1)
months = []
y, m = today.year, today.month
for _ in range(24):
    m -= 1
    if m == 0: y, m = y - 1, 12
    months.append(f"{y}-{m:02d}")
months = sorted(months)
json.dump(syms, open("symbols.json", "w")); json.dump(months, open("months.json", "w"))
print(len(syms), "symbols |", months[0], "→", months[-1])
