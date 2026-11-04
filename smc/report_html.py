"""FINAL_COMPARISON_REPORT.html — گزارش نهایی با نمودار، جدول و جزئیات معاملات (از روی پوشهٔ results)."""
import json, os, sys, html
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "FINAL_COMPARISON_REPORT.html")
rd = lambda f: pd.read_csv(os.path.join(D, f))
M, SEG, REG, SET, COIN = rd("summary.csv"), rd("split_60_20_20.csv"), rd("by_regime.csv"), rd("by_setup.csv"), rd("by_coin.csv")
WF = rd("walk_forward.csv") if os.path.getsize(os.path.join(D, "walk_forward.csv")) > 2 else pd.DataFrame()
Q, EQ = rd("data_quality.csv").reindex(columns=["symbol", "status", "months_found", "candles", "missing_candles", "gaps_over_1h"]), pd.read_csv(os.path.join(D, "equity_daily.csv"), index_col=0)
TR = rd("trades_sample.csv"); ANS = json.load(open(os.path.join(D, "answers.json")))
CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
try:   # Chart.js داخل خود فایل گذاشته می‌شه تا گزارش بدون اینترنت هم باز بشه
    import requests; CHARTJS_TAG = "<script>" + requests.get(CDN, timeout=30).text + "</script>"
except Exception:
    CHARTJS_TAG = f'<script src="{CDN}"></script>'

NAMES = {"PA": "پرایس اکشن", "SMC": "اسمارت مانی", "HYBRID": "ترکیبی"}
M["name"] = M.strategy.map(NAMES) + " · " + M.tf

# --- قابل اتکا بودن: همهٔ شرط‌ها باید برقرار باشن
test = SEG[SEG.segment == "3_test_20"].set_index(["strategy", "tf"])
def checks(r):
    k = (r.strategy, r.tf); t = test.loc[k] if k in test.index else None
    return {"سود کل مثبت": r.net_profit_pct > 0, "PF ≥ ۱.۲": r.PF >= 1.2,
            "سودده در بیش از نصف ارزها": r.coins_profitable_pct > 50,
            "تست خارج از نمونه مثبت": t is not None and t.net_profit_pct > 0,
            "افت کمتر از ۳۰٪": r.maxDD_pct < 30, "حداقل ۲۰۰ معامله": r.trades >= 200}
M["checks"] = M.apply(checks, axis=1); M["score"] = M.checks.apply(lambda c: sum(c.values()))
reliable = M[M.score == 6]

def table(df, cols, heads, fmt=None, cls=""):
    fmt = fmt or {}
    h = "".join(f"<th>{x}</th>" for x in heads)
    rows = ""
    for _, r in df.iterrows():
        tds = ""
        for c in cols:
            v = r[c]; s = fmt[c](v) if c in fmt else (f"{v:,.2f}" if isinstance(v, float) else html.escape(str(v)))
            neg = isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0 and c not in ("avg_loss_R",)
            tds += f'<td class="{"neg" if neg else ""}">{s}</td>'
        rows += f"<tr>{tds}</tr>"
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table></div>'

pct = lambda v: f"{v:+.1f}%"; usd = lambda v: f"{v:+,.0f}"
main_tbl = table(M.sort_values(["score", "calmar"], ascending=False),
    ["name", "net_profit_usdt", "net_profit_pct", "maxDD_pct", "PF", "win_pct", "expectancy_R", "trades", "avg_win_R", "avg_loss_R",
     "sharpe", "sortino", "calmar", "coins_profitable_pct", "liquidations", "score"],
    ["استراتژی · تایم", "سود خالص (USDT)", "سود %", "حداکثر افت %", "PF", "برد %", "Expectancy (R)", "معاملات", "میانگین برد R",
     "میانگین باخت R", "Sharpe", "Sortino", "Calmar", "ارزهای سودده %", "لیکویید", "امتیاز اتکا (از ۶)"],
    {"net_profit_usdt": usd, "net_profit_pct": pct, "maxDD_pct": lambda v: f"{v:.1f}%", "win_pct": lambda v: f"{v:.1f}%",
     "coins_profitable_pct": lambda v: f"{v:.0f}%", "trades": lambda v: f"{v:,}", "liquidations": str, "score": str})
chk_rows = "".join(f"<tr><td>{r['name']}</td>" + "".join(f'<td class="{"ok" if v else "bad"}">{"✓" if v else "✗"}</td>' for v in r.checks.values()) + "</tr>"
                   for _, r in M.sort_values("score", ascending=False).iterrows())
chk_tbl = f'<div class="tw"><table><thead><tr><th>استراتژی · تایم</th>{"".join(f"<th>{k}</th>" for k in M.checks.iloc[0])}</tr></thead><tbody>{chk_rows}</tbody></table></div>'
SEG["name"] = SEG.strategy.map(NAMES) + " · " + SEG.tf
seg_p = SEG.pivot_table(index="name", columns="segment", values="net_profit_pct").reset_index()
seg_p.columns = ["name"] + [c for c in seg_p.columns[1:]]
seg_tbl = table(SEG, ["name", "segment", "net_profit_pct", "maxDD_pct", "PF", "win_pct", "expectancy_R", "trades", "sharpe", "calmar"],
                ["استراتژی · تایم", "بخش", "سود %", "افت %", "PF", "برد %", "Expectancy", "معاملات", "Sharpe", "Calmar"],
                {"net_profit_pct": pct, "maxDD_pct": lambda v: f"{v:.1f}%", "win_pct": lambda v: f"{v:.1f}%", "trades": lambda v: f"{v:,}",
                 "segment": lambda v: {"1_dev_60": "توسعه ۶۰٪", "2_val_20": "اعتبارسنجی ۲۰٪", "3_test_20": "تست نهایی ۲۰٪"}.get(v, v)})
REG["name"] = REG.strategy.map(NAMES) + " · " + REG.tf
reg_p = REG.pivot_table(index="name", columns="regime", values="expectancy_R").round(3).reset_index()
reg_tbl = table(reg_p, list(reg_p.columns), ["استراتژی · تایم"] + [{"bull": "صعودی", "bear": "نزولی", "neutral": "خنثی", "unknown": "نامشخص"}.get(c, c) for c in reg_p.columns[1:]])
SET["name"] = SET.strategy.map(NAMES) + " · " + SET.tf
set_tbl = table(SET, ["name", "setup", "trades", "win_pct", "PF", "expectancy_R"], ["استراتژی · تایم", "نوع ستاپ", "معاملات", "برد %", "PF", "Expectancy"],
                {"win_pct": lambda v: f"{v:.1f}%", "trades": lambda v: f"{v:,}"})
wf_tbl = table(WF, list(WF.columns), ["دورهٔ آموزش", "دورهٔ تست", "انتخاب‌شده", "Calmar آموزش", "بازده تست %", "معاملات تست", "Expectancy تست"]) if len(WF) else "<p>دادهٔ کافی نبود.</p>"
best = M.sort_values(["score", "calmar"], ascending=False).iloc[0]
cb = COIN[(COIN.strategy == best.strategy) & (COIN.tf == best.tf)].sort_values("sum_R", ascending=False)
coin_tbl = table(pd.concat([cb.head(15), cb.tail(15)]), ["symbol", "trades", "win_pct", "PF", "expectancy_R", "sum_R"],
                 ["نماد", "معاملات", "برد %", "PF", "Expectancy", "جمع R"], {"win_pct": lambda v: f"{v:.1f}%"})
q = Q.status.value_counts().to_dict()
qlabel = {"ok": "کامل", "partial": "ناقص (ماه یا کندل گم‌شده)", "insufficient_history": "تاریخچهٔ ناکافی (کنار گذاشته شد)", "no_data_in_period": "بدون داده در بازه"}
q_html = "".join(f"<li><b>{v:,}</b> {qlabel.get(k, k)}</li>" for k, v in q.items())
bad_q = Q[Q.status.isin(["insufficient_history", "partial"])].sort_values("status")[["symbol", "status", "months_found", "candles", "missing_candles", "gaps_over_1h"]]

def answer_text():
    test_ok = ANS.get("7_profitable_out_of_sample") or []
    t = [("۱. سودآورترین", ANS["1_most_profitable"]), ("۲. کمترین افت سرمایه", ANS["2_lowest_drawdown"]),
         ("۳. سودده در بیشترین ارزها", ANS["3_most_coins_positive"]), ("۴. بهترین تایم‌فریم (میانگین Calmar)", ANS["4_best_timeframe_by_avg_calmar"]),
         ("۵. بهترین در بازار صعودی", ANS["5_best_in_bull_by_expR"]), ("۶. بهترین در بازار نزولی", ANS["6_best_in_bear_by_expR"]),
         ("۷. سودده در تست خارج از نمونه", "، ".join(f"{NAMES.get(x['strategy'], x['strategy'])} {x['tf']} ({x['net_profit_pct']:+.1f}%)" for x in test_ok) or "هیچ‌کدام"),
         ("Walk-Forward (۱۰٬۰۰۰ USDT →)", f"{ANS['walk_forward_final_equity_usdt']:,} USDT")]
    lab = {"net_profit_pct=": "سود ٪", "maxDD_pct=": "افت ٪", "coins_profitable_pct=": "ارزهای سودده ٪", "expectancy_R=": "Expectancy R "}
    for k, v in NAMES.items(): t = [(a, str(b).replace(k + " ", v + " ")) for a, b in t]
    for k, v in lab.items(): t = [(a, b.replace(k, v)) for a, b in t]
    return "".join(f"<div class='qa'><span>{a}</span><b>{html.escape(str(b))}</b></div>" for a, b in t)

verdict = (f"<p class='good'>استراتژی‌هایی که همهٔ ۶ شرط قابل اتکا بودن رو دارن: <b>{'، '.join(reliable.name)}</b></p>" if len(reliable) else
           "<p class='warn'><b>هیچ‌کدوم از ۹ ترکیب همهٔ شرط‌های «قابل اتکا» رو نداره.</b> یعنی با این قوانین و هزینه‌های واقعی، هیچ‌کدوم برای معاملهٔ واقعی فیوچرز توصیه نمی‌شه.</p>")
eq_json = {c: [[i, round(v, 2)] for i, v in EQ[c].dropna().items()] for c in EQ.columns}
TR["entry_time"] = pd.to_datetime(TR.entry_t, unit="ms").dt.strftime("%Y-%m-%d %H:%M")
TR["exit_time"] = pd.to_datetime(TR.exit_t, unit="ms").dt.strftime("%Y-%m-%d %H:%M")
tr_cols = ["strategy", "tf", "symbol", "side", "setup", "entry_time", "exit_time", "entry", "sl", "tp", "exit", "leverage", "R_gross", "fee_R", "funding_R", "R_net", "hours", "regime", "segment"]
tr_json = TR[tr_cols].round(5).values.tolist()

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e5e4df;--pos:#1f7a4d;--neg:#b3392f;--acc:#2f5bd3;--warnbg:#fff4e5;--goodbg:#e9f6ef}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--card:#1e1e1c;--ink:#ecebe6;--mut:#a3a29c;--line:#33332f;--pos:#5cc48f;--neg:#ef7d72;--acc:#7ea0ff;--warnbg:#3a2c16;--goodbg:#173326}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.7 Vazirmatn,Tahoma,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px}
.sub{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;direction:rtl}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--card)}td.neg{color:var(--neg)}td.ok{color:var(--pos);text-align:center}td.bad{color:var(--neg);text-align:center}
.warn{background:var(--warnbg);padding:12px;border-radius:8px}.good{background:var(--goodbg);padding:12px;border-radius:8px}
.qa{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid var(--line)}.qa span{color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}canvas{width:100%!important;height:280px!important}
select,input{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
.pager{display:flex;gap:8px;align-items:center;margin:8px 0}button{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
ul{margin:4px 0}code{font-size:12px}
"""
JS = """
const EQ=%s, TR=%s, NM={"PA":"پرایس اکشن","SMC":"اسمارت مانی","HYBRID":"ترکیبی"}, COL={"PA":"#2f5bd3","SMC":"#c2410c","HYBRID":"#15803d"};
if(typeof Chart==='undefined'){window.Chart=function(){};}
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function charts(){["5m","15m","1h"].forEach(tf=>{
  const ds=Object.keys(EQ).filter(k=>k.endsWith(" "+tf)).map(k=>({label:NM[k.split(" ")[0]],data:EQ[k].map(p=>({x:p[0],y:p[1]})),borderColor:COL[k.split(" ")[0]],pointRadius:0,borderWidth:1.6}));
  new Chart(document.getElementById("eq_"+tf),{type:"line",data:{datasets:ds},options:{animation:false,parsing:true,scales:{x:{type:"category",labels:EQ[Object.keys(EQ).find(k=>k.endsWith(" "+tf))].map(p=>p[0]),ticks:{maxTicksLimit:8,color:css("--mut")}},y:{ticks:{color:css("--mut")},title:{display:true,text:"USDT"}}},plugins:{legend:{labels:{color:css("--ink")}},title:{display:true,text:"اکوئیتی — "+tf,color:css("--ink")}}}});
  const dd=Object.keys(EQ).filter(k=>k.endsWith(" "+tf)).map(k=>{let pk=0;return {label:NM[k.split(" ")[0]],data:EQ[k].map(p=>{pk=Math.max(pk,p[1]);return -100*(1-p[1]/pk)}),borderColor:COL[k.split(" ")[0]],pointRadius:0,borderWidth:1.4}});
  new Chart(document.getElementById("dd_"+tf),{type:"line",data:{labels:EQ[Object.keys(EQ).find(k=>k.endsWith(" "+tf))].map(p=>p[0]),datasets:dd},options:{animation:false,scales:{x:{ticks:{maxTicksLimit:8,color:css("--mut")}},y:{ticks:{color:css("--mut")},title:{display:true,text:"افت %%"}}},plugins:{legend:{labels:{color:css("--ink")}},title:{display:true,text:"افت سرمایه — "+tf,color:css("--ink")}}}});
});}
let page=0;const PS=50;
function rows(){const s=document.getElementById("fs").value,t=document.getElementById("ft").value,q=document.getElementById("fq").value.toUpperCase();
  return TR.filter(r=>(!s||r[0]==s)&&(!t||r[1]==t)&&(!q||r[2].includes(q)));}
function draw(){const R=rows(),n=Math.max(1,Math.ceil(R.length/PS));page=Math.min(page,n-1);
  document.getElementById("pg").textContent=`صفحهٔ ${page+1} از ${n} — ${R.length.toLocaleString()} معامله`;
  document.getElementById("tb").innerHTML=R.slice(page*PS,page*PS+PS).map(r=>"<tr>"+r.map((v,i)=>`<td class="${i==15&&v<0?"neg":""}">${i==0?NM[v]:i==3?(v==1?"خرید":"فروش"):v}</td>`).join("")+"</tr>").join("");}
window.addEventListener("load",()=>{charts();draw();["fs","ft","fq"].forEach(id=>document.getElementById(id).addEventListener("input",()=>{page=0;draw()}));
  document.getElementById("pv").onclick=()=>{page=Math.max(0,page-1);draw()};document.getElementById("nx").onclick=()=>{page++;draw()};});
""" % (json.dumps(eq_json), json.dumps(tr_json))

tr_heads = ["استراتژی", "تایم", "نماد", "جهت", "ستاپ", "ورود", "خروج", "قیمت ورود", "حد ضرر", "حد سود", "قیمت خروج", "اهرم", "R ناخالص", "کارمزد R", "فاندینگ R", "R خالص", "ساعت", "رژیم", "بخش"]
page = f"""<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Final Comparison Report</title>
{CHARTJS_TAG}<style>{CSS}</style></head><body><main>
<h1>گزارش نهایی مقایسهٔ سه استراتژی{(" — " + os.environ["SYMBOLS"].replace("USDT","").replace(",", "، ")) if os.environ.get("SYMBOLS") else ""}</h1>
<div class="sub">پرایس اکشن کلاسیک · اسمارت مانی · ترکیبی — همهٔ قراردادهای دائمی USDT بایننس، تایم‌های ۵ و ۱۵ دقیقه و ۱ ساعته، سرمایهٔ اولیه ۱۰٬۰۰۰ USDT، ریسک ۱٪، حداکثر ۳ معاملهٔ همزمان، با کارمزد، اسلیپیج، فاندینگ واقعی و ریسک لیکویید.</div>
<h2>نتیجهٔ نهایی</h2><div class="card">{verdict}{answer_text()}</div>
<h2>جدول مقایسه</h2><div class="card">{main_tbl}<p class="sub">امتیاز اتکا = تعداد شرط‌های برقرار از ۶ شرط جدول بعدی. مرتب‌سازی: امتیاز، بعد Calmar.</p></div>
<h2>شرط‌های قابل اتکا بودن</h2><div class="card">{chk_tbl}</div>
<h2>اکوئیتی و افت سرمایه</h2><div class="grid">{"".join(f'<div class="card"><canvas id="eq_{t}"></canvas></div><div class="card"><canvas id="dd_{t}"></canvas></div>' for t in ["5m","15m","1h"])}</div>
<h2>تقسیم داده: ۶۰٪ توسعه · ۲۰٪ اعتبارسنجی · ۲۰٪ تست نهایی</h2><div class="card">{seg_tbl}</div>
<h2>Walk-Forward (۶ ماه آموزش ← ۳ ماه تست)</h2><div class="card">{wf_tbl}</div>
<h2>بازار صعودی، نزولی و خنثی (Expectancy بر حسب R، همهٔ سیگنال‌ها)</h2><div class="card">{reg_tbl}<p class="sub">رژیم با روزانهٔ بیت‌کوین: صعودی = کلوز و SMA50 بالای SMA200، نزولی = هر دو زیر SMA200.</p></div>
<h2>عملکرد نوع ناحیهٔ ورود</h2><div class="card">{set_tbl}</div>
<h2>بهترین و بدترین ارزها — {NAMES[best.strategy]} {best.tf}</h2><div class="card">{coin_tbl}</div>
<h2>کیفیت داده</h2><div class="card"><ul>{q_html}</ul><details><summary>فهرست نمادهای ناقص یا کنار گذاشته‌شده ({len(bad_q)})</summary>{table(bad_q, list(bad_q.columns), ["نماد","وضعیت","ماه‌های موجود","کندل","کندل گم‌شده","گپ بالای ۱ ساعت"])}</details></div>
<h2>جزئیات معاملات (معاملات انجام‌شده در سبد، حداکثر ۱۵۰۰ معاملهٔ آخر هر ترکیب)</h2><div class="card">
<div class="pager"><select id="fs"><option value="">همهٔ استراتژی‌ها</option><option value="PA">پرایس اکشن</option><option value="SMC">اسمارت مانی</option><option value="HYBRID">ترکیبی</option></select>
<select id="ft"><option value="">همهٔ تایم‌ها</option><option>5m</option><option>15m</option><option>1h</option></select><input id="fq" placeholder="نماد…" size="10">
<button id="pv">قبلی</button><button id="nx">بعدی</button><span id="pg" class="sub"></span></div>
<div class="tw"><table><thead><tr>{"".join(f"<th>{h}</th>" for h in tr_heads)}</tr></thead><tbody id="tb"></tbody></table></div></div>
<p class="sub">روش کامل: <code>strategies/00_BACKTEST_PROTOCOL.md</code>. نتایج گذشته تضمینی برای آینده نیست.</p>
</main><script>{JS}</script></body></html>"""
open(OUT, "w", encoding="utf-8").write(page); print("written", OUT, round(len(page) / 1e6, 2), "MB")
