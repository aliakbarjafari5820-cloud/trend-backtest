"""SCALP_REPORT.html — جدول همهٔ استراتژی‌های اسکالپ، نتیجهٔ پروتکل ۶۰/۲۰/۲۰ و نمودار R تجمعی."""
import json, html, pandas as pd
A = pd.read_csv("results/scalp_summary.csv"); V = json.load(open("results/scalp_verdict.json")); C = json.load(open("results/scalp_curves.json"))
try:
    import requests; CJS = "<script>" + requests.get("https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js", timeout=30).text + "</script>"
except Exception:
    CJS = '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'
cols = [("strategy", "استراتژی"), ("tf", "تایم"), ("all_n", "معاملات"), ("all_win", "برد %"), ("all_PF", "PF"), ("all_expR", "میانگین خالص R"),
        ("all_expR_gross", "قبل از هزینه R"), ("all_cost_R", "هزینه R"), ("all_coins_pos", "ارز سودده (از ۵)"), ("all_ret_pct", "بازده با ریسک ۱٪"),
        ("all_maxDD", "افت %"), ("dev_expR", "توسعه ۶۰٪"), ("val_expR", "اعتبارسنجی ۲۰٪"), ("test_expR", "تست ۲۰٪"), ("pass_test", "قبول نهایی")]
def cell(k, v):
    if isinstance(v, (bool,)) or k.startswith("pass"): return f'<td class="{"ok" if v else "bad"}">{"✓" if v else "✗"}</td>'
    if isinstance(v, float): return f'<td class="{"neg" if v < 0 else ""}">{v:,.3f}</td>' if "expR" in k or "cost" in k else f'<td class="{"neg" if v < 0 else ""}">{v:,.1f}</td>'
    return f"<td>{html.escape(str(v))}</td>"
rows = "".join("<tr>" + "".join(cell(k, r[k]) for k, _ in cols) + "</tr>" for _, r in A.iterrows())
top = A.sort_values("dev_expR", ascending=False).head(6)
keys = [f"{r.strategy} {r.tf}" for _, r in top.iterrows()] + [k for k in C if k.startswith("RANDOM_BASELINE")]
passed = V["passed_test"]
verdict = (f"<p class='good'><b>استراتژی‌هایی که همهٔ مراحل رو قبول شدن:</b> {'، '.join(' '.join(x) for x in passed)}</p>" if passed else
           "<p class='warn'><b>هیچ استراتژی اسکالپی همهٔ مراحل (توسعه ← اعتبارسنجی ← تست نهایی) رو قبول نشد.</b></p>")
page = f"""<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scalp Report</title>{CJS}
<style>:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e5e4df;--pos:#1f7a4d;--neg:#b3392f;--warnbg:#fff4e5;--goodbg:#e9f6ef}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1e1e1c;--ink:#ecebe6;--mut:#a3a29c;--line:#33332f;--pos:#5cc48f;--neg:#ef7d72;--warnbg:#3a2c16;--goodbg:#173326}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.7 Tahoma,sans-serif}}main{{max-width:1200px;margin:auto;padding:24px 16px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}}.tw{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}th,td{{padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}}th{{color:var(--mut)}}
.neg{{color:var(--neg)}}.ok{{color:var(--pos);text-align:center}}.bad{{color:var(--neg);text-align:center}}.warn{{background:var(--warnbg);padding:12px;border-radius:8px}}.good{{background:var(--goodbg);padding:12px;border-radius:8px}}
.sub{{color:var(--mut)}}canvas{{max-height:380px}}</style></head><body><main>
<h1>گزارش تست اسکالپ</h1><p class="sub">BTC، ETH، SOL، BNB، XRP — فیوچرز بایننس، {V['period'][0]} تا {V['period'][1]}، تایم ۵ و ۱۵ دقیقه، ورود و حد سود با سفارش لیمیت (Maker ۰.۰۲٪)، حد ضرر با مارکت (۰.۰۵٪ + اسلیپیج ۰.۰۲٪)، فاندینگ ۰.۰۱٪ هر ۸ ساعت.</p>
<div class="card">{verdict}<ul>
<li>از {V['configs']} ترکیب (استراتژی × تایم)، <b>{V['positive_all_gross']}</b> تا قبل از هزینه‌ها مثبت بودن و <b>{V['positive_all_net']}</b> تا بعد از هزینه‌ها.</li>
<li>مرحلهٔ ۱ (توسعه، تا {V['cut_val']}): {len(V['passed_dev'])} قبول — مرحلهٔ ۲ (اعتبارسنجی، تا {V['cut_test']}): {len(V['passed_val'])} قبول — مرحلهٔ ۳ (تست نهایی): {len(passed)} قبول.</li>
<li>معیار مقایسه «RANDOM_BASELINE» ورود تصادفیه؛ هر استراتژی‌ای که بهتر از اون نباشه، مزیتی نداره.</li></ul></div>
<div class="card"><canvas id="ch"></canvas><p class="sub">R خالص تجمعی (جمع ۵ ارز) — ۶ استراتژی برتر در دورهٔ توسعه + ورود تصادفی. خط‌چین: شروع اعتبارسنجی و تست.</p></div>
<div class="card tw"><table><thead><tr>{''.join(f'<th>{h}</th>' for _, h in cols)}</tr></thead><tbody>{rows}</tbody></table></div>
<p class="sub">پروتکل: توسعه PF&gt;۱.۱ و میانگین&gt;۰.۰۵R و ≥۱۰۰ معامله ← اعتبارسنجی میانگین&gt;۰ ← تست نهایی میانگین&gt;۰، PF&gt;۱.۰۵، سودده در ≥۳ ارز. نتایج گذشته تضمین آینده نیست.</p>
</main><script>
const C={json.dumps({k: C[k] for k in keys if k in C})};const pal=["#2f5bd3","#c2410c","#15803d","#9333ea","#db2777","#0891b2","#6b7280","#6b7280"];
const labels=[...new Set(Object.values(C).flat().map(p=>p[0]))].sort();
if(typeof Chart!=="undefined")new Chart(document.getElementById("ch"),{{type:"line",data:{{labels,datasets:Object.keys(C).map((k,i)=>{{const m=Object.fromEntries(C[k]);let last=0;return {{label:k,data:labels.map(d=>(m[d]!==undefined?(last=m[d]):last)),borderColor:pal[i%pal.length],borderDash:k.startsWith("RANDOM")?[4,4]:[],pointRadius:0,borderWidth:1.6}}}})}},
options:{{animation:false,scales:{{x:{{ticks:{{maxTicksLimit:10}}}},y:{{title:{{display:true,text:"R"}}}}}}}}}});
</script></body></html>"""
open("results/SCALP_REPORT.html", "w", encoding="utf-8").write(page); print("ok")
