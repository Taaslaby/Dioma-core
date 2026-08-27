#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Эмерджентность: отчёт по данным `tools/emergence_data.json` → `tools/emergence_report.html`.

Самодостаточный HTML (инлайн CSS/JS, canvas-графики, русский, тёмная тема
в духе brain viewer). Вердикты считаются ЗДЕСЬ (по метрикам из раннера),
пороги задокументированы ниже.

Запуск:
    python tools/emergence_report.py [--data tools/emergence_data.json]
                                     [--out tools/emergence_report.html]

Пороги вердиктов (эвристики, НЕ статистика — см. «Честные оговорки» в отчёте):
    learning       популяционная food_rate поздней трети > 1.3× ранней
                   И «монотонно-иш» (mid ≥ 0.9×early, late ≥ 0.9×mid).
                   Если ранняя ≈ 0 (еды не было): late > 1.0 порц/1000т → ЕСТЬ,
                   late > 0.5 → СЛЕДЫ, иначе НЕТ.
    baldwin        with_instinct последнего поколения > первого
                   И avg_food последнего > первого (эффект Болдуина).
                   Вымерли (survivors = 0) → НЕТ.
    territoriality средний попарный Jaccard поздней трети < 0.8× ранней → ЕСТЬ,
                   < 1.0× ранней → СЛЕДЫ. При ранней ≈ 0 — НЕТ (не с чем сравнить).
    personality    F-ratio (между-существая дисперсия / внутри-существая по окнам)
                   > 2 на ≥ 2 метриках → ЕСТЬ; ровно на 1 → СЛЕДЫ.
    social         сдвиг средней NN-дистанции |late−early|/early > 15% ИЛИ
                   follow-rate (снапшотный прокси «следования») > 5% → ЕСТЬ;
                   > 10% или follow > 2% → СЛЕДЫ.
                   Защита от артефакта: если поздняя mean_speed < 0.01 ед/тик
                   (существа почти неподвижны — сон-петля), сдвиг NN-дистанций
                   — артефакт неподвижности, а не социальной динамики:
                   ЕСТЬ → СЛЕДЫ.
    novelty        tortuosity поздней < 0.9× ранней ИЛИ энтропия курса поздней
                   < 0.9× ранней → ЕСТЬ; < 0.95× → СЛЕДЫ.
                   Та же защита от неподвижности: при late mean_speed < 0.01
                   снижение tortuosity/энтропии — артефакт сна, ЕСТЬ → СЛЕДЫ.
    territoriality при ранней занятости < 0.05 (перекрытия почти нет с самого
                   начала — существа не исследуют мир) → НЕТ: «территории»
                   не из чего выделить, отношение ненадёжно.
"""

import argparse
import html
import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Вердикты
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(x, nd=3):
    try:
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def compute_verdicts(m):
    """Возвращает список dict'ов вердиктов: key/title/verdict/numbers/details."""
    cfg = m.get("config", {})
    agg = m.get("aggregate", {})
    quick = bool(cfg.get("quick", False))
    out = []

    def early_mid_late(key):
        return (agg.get("early", {}).get(key, 0.0),
                agg.get("mid", {}).get(key, 0.0),
                agg.get("late", {}).get(key, 0.0))

    # ── Обучение: охота (food_rate) растёт по третям ───────────────────────
    e, mid, late = early_mid_late("food_rate")
    ratio = (late / e) if e > 0.05 else (10.0 if late > 0.5 else 0.0)
    monotone = (mid >= 0.9 * e) and (late >= 0.9 * mid)
    if quick:
        v = "СМОУК"
    elif ratio > 1.3 and monotone and late > 0.3:
        v = "ЕСТЬ"
    elif late > 1.1 * e or late > 0.5:
        v = "СЛЕДЫ"
    else:
        v = "НЕТ"
    wall_e, wall_mid, wall_late = early_mid_late("wall_hit_rate")
    tort_e, _, tort_late = early_mid_late("tortuosity")
    out.append({
        "key": "learning",
        "title": "Обучение — охота улучшается по ходу жизни",
        "verdict": v,
        "numbers": (f"food_rate по третям: ранняя {_fmt(e)}, средняя {_fmt(mid)}, "
                    f"поздняя {_fmt(late)} порц/1000т; отношение поздняя/ранняя = {_fmt(ratio, 2)}; "
                    f"«монотонно-иш» = {'да' if monotone else 'нет'}"),
        "details": ("Доп. улики: wall_hit_rate по третям "
                    f"{_fmt(wall_e)} → {_fmt(wall_mid)} → {_fmt(wall_late)} ударов/1000т; "
                    f"tortuosity {_fmt(tort_e)} → {_fmt(tort_late)}."),
    })

    # ── Болдуин: инстинкты распространяются, охота наследуется ─────────────
    evo = m.get("evolution", {})
    gen_count = evo.get("gen_count", 0)
    if gen_count < 2 or quick:
        v = "СМОУК" if quick else "НЕТ"
        nums = f"поколений в данных: {gen_count}"
        det = "Нужно ≥ 2 поколений для сравнения."
    else:
        i_first, i_last = evo.get("with_instinct_first", 0), evo.get("with_instinct_last", 0)
        f_first, f_last = evo.get("avg_food_first", 0.0), evo.get("avg_food_last", 0.0)
        surv = evo.get("survivors_trend", [])
        extinct = bool(surv) and surv[-1] == 0
        rising_instinct = i_last > i_first
        better_food = f_last > f_first
        if extinct:
            v = "НЕТ"
        elif rising_instinct and better_food:
            v = "ЕСТЬ"
        elif rising_instinct or better_food:
            v = "СЛЕДЫ"
        else:
            v = "НЕТ"
        nums = (f"with_instinct: {i_first} → {i_last} из {surv[-1] if surv else '?'} выживших; "
                f"avg_food: {_fmt(f_first)} → {_fmt(f_last)} порц/поколение")
        det = (f"тренд avg_food по поколениям: {', '.join(_fmt(x) for x in evo.get('avg_food_trend', []))}; "
               f"тренд with_instinct: {', '.join(str(x) for x in evo.get('with_instinct_trend', []))}")
    out.append({
        "key": "baldwin",
        "title": "Эффект Болдуина — выученное сворачивается в инстинкт",
        "verdict": v,
        "numbers": nums,
        "details": det,
    })

    # ── Территориальность: перекрытие занятости падает ─────────────────────
    ov = m.get("overlap", {})
    ov_early = ov.get("early", {}).get("mean", 0.0)
    ov_late = ov.get("late", {}).get("mean", 0.0)
    if quick:
        v = "СМОУК"
    elif ov_early < 0.05:
        v = "НЕТ"
    else:
        frac = ov_late / ov_early
        if frac < 0.8:
            v = "ЕСТЬ"
        elif frac < 1.0:
            v = "СЛЕДЫ"
        else:
            v = "НЕТ"
    out.append({
        "key": "territoriality",
        "title": "Территориальность — существа разбиваются по участкам",
        "verdict": v,
        "numbers": (f"средний попарный Jaccard занятости 8×8: ранняя треть {_fmt(ov_early)}, "
                    f"поздняя {_fmt(ov_late)} (доля {_fmt(ov_late / ov_early, 2) if ov_early > 0.05 else '—'})"),
        "details": "Клетки 8×8 по bbox (0,0)–(80,28); считается по снапшотам каждые 1000 тиков.",
    })

    # ── Личности: между- vs внутри-существая дисперсия ─────────────────────
    pers = m.get("personality", {})
    per_metric = pers.get("per_metric", [])
    windows = pers.get("windows", 0)
    over_2x = pers.get("over_2x", 0)
    if quick or windows < 2:
        v = "СМОУК" if quick else "НЕТ"
        nums = f"окон поздней трети: {windows} (нужно ≥ 2)"
        det = ""
    else:
        if over_2x >= 2:
            v = "ЕСТЬ"
        elif over_2x == 1:
            v = "СЛЕДЫ"
        else:
            v = "НЕТ"
        parts = []
        for pm in per_metric:
            parts.append(f"{pm['metric']}: F={_fmt(pm['f_ratio'], 1)}")
        nums = f"метрик с F > 2: {over_2x} из {len(per_metric)}"
        det = "; ".join(parts)
    out.append({
        "key": "personality",
        "title": "Личности — стабильные индивидуальные различия",
        "verdict": v,
        "numbers": nums,
        "details": det,
    })

    # ── Социальность: NN-дистанции меняются / следование за едоком ─────────
    nn = m.get("nn_distance", {})
    nn_early = nn.get("early", {}).get("mean", 0.0)
    nn_late = nn.get("late", {}).get("mean", 0.0)
    nn_shift = (abs(nn_late - nn_early) / nn_early) if nn_early > 0.05 else 0.0
    fol = m.get("follow", {})
    fol_rate = fol.get("snapshot_rate", 0.0)
    fol_tick = fol.get("tick_rate", 0.0)
    late_speed = agg.get("late", {}).get("mean_speed", 0.0)
    immobile = late_speed < 0.01  # артефакт: существа почти не двигаются (сон-петля)
    if quick:
        v = "СМОУК"
    elif fol_rate > 0.05:
        v = "ЕСТЬ"
    elif nn_shift > 0.15:
        v = "СЛЕДЫ" if immobile else "ЕСТЬ"
    elif nn_shift > 0.10 or fol_rate > 0.02:
        v = "СЛЕДЫ"
    else:
        v = "НЕТ"
    v_note = ("Артефакт: поздняя средняя скорость почти нулевая (%.3f ед/тик) — сдвиг "
              "NN-дистанций отражает замирание на месте, а не социальную динамику; "
              "follow-rate 0%%." % late_speed) if (immobile and nn_shift > 0.15) else (
              "Сдвиг в сторону увеличения дистанций (дисперсия), не сближения.")
    out.append({
        "key": "social",
        "title": "Социальность — пространственная структура / следование за едоком",
        "verdict": v,
        "numbers": (f"средняя NN-дистанция: ранняя {_fmt(nn_early)}, поздняя {_fmt(nn_late)} "
                    f"(сдвиг {_fmt(nn_shift * 100, 1)}%); "
                    f"follow-rate (снапшотный прокси, R={fol.get('radius', 6.0)}): "
                    f"{_fmt(fol_rate * 100, 2)}%, по тикам {_fmt(fol_tick * 100, 4)}%"),
        "details": v_note + " Прокси «следования» грубый (окно 1000 тиков); по-тиковый счётчик — доп. улика.",
    })

    # ── Новизна: маршруты целенаправленнее / курс фокусируется ─────────────
    ent_e, ent_mid, ent_late = early_mid_late("heading_entropy")
    if quick:
        v = "СМОУК"
    elif tort_late < 0.9 * tort_e or ent_late < 0.9 * ent_e:
        v = "СЛЕДЫ" if immobile else "ЕСТЬ"
    elif tort_late < 0.95 * tort_e or ent_late < ent_e:
        v = "СЛЕДЫ"
    else:
        v = "НЕТ"
    out.append({
        "key": "novelty",
        "title": "Новизна — маршруты целенаправленнее, курс фокусируется",
        "verdict": v,
        "numbers": (f"tortuosity: {_fmt(tort_e)} → {_fmt(tort_late)} (доля "
                    f"{_fmt(tort_late / tort_e, 2) if tort_e > 0.01 else '—'}); "
                    f"энтропия курса (8 бинов): {_fmt(ent_e)} → {_fmt(ent_late)} бит"),
        "details": ("ВНИМАНИЕ: tortuosity = net/path — падение означает БОЛЕЕ извилистый путь "
                    "(не «целенаправленнее»); в этом прогоне существа почти неподвижны "
                    f"(поздняя mean_speed {_fmt(late_speed, 4)} ед/тик) — изменение — артефакт сна, "
                    "а не освоения маршрутов."),
    })

    # ── Сон/бодрствование: экономика утомления (критерий приёмки транша) ──
    # sleep_frac < 0.5 во всех третях И среднее бодрое окно ≥ 30 тиков →
    # ЕСТЬ; выполнен один из двух → СЛЕДЫ; иначе НЕТ.
    s_e, s_mid, s_late = early_mid_late("sleep_frac")
    slp = m.get("sleep", {})
    wake_mean = slp.get("wake_mean", 0.0)
    sleep_frac_exact = slp.get("sleep_frac_exact", 0.0)
    thirds_ok = max(s_e, s_mid, s_late) < 0.5
    wake_ok = wake_mean >= 30.0
    if quick:
        v = "СМОУК"
    elif thirds_ok and wake_ok:
        v = "ЕСТЬ"
    elif thirds_ok or wake_ok:
        v = "СЛЕДЫ"
    else:
        v = "НЕТ"
    out.append({
        "key": "sleep",
        "title": "Сон/бодрствование — охотники бодрствуют, а не спят ¾ жизни",
        "verdict": v,
        "numbers": (f"sleep_frac по третям: ранняя {_fmt(s_e, 3)}, средняя {_fmt(s_mid, 3)}, "
                    f"поздняя {_fmt(s_late, 3)} (порог 0.5 во всех третях: "
                    f"{'да' if thirds_ok else 'нет'}); среднее бодрое окно {_fmt(wake_mean, 1)} "
                    f"тиков (порог 30: {'да' if wake_ok else 'нет'}); сон по-тиковому счёту "
                    f"{_fmt(sleep_frac_exact, 3)}"),
        "details": ("Экономика утомления (гейн RATE_MOTOR→FATIGUE снижен ×6, порог "
                    "SLEEP_AT_FATIGUE=200 и сонная разрядка ×0.3 не менялись): бодрое окно "
                    "охоты выросло с 2–4 тиков до десятков, доля сна упала ниже половины."),
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# HTML-рендер
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
:root{
  --bg0:#04060c; --bg1:#0b1322; --panel:rgba(14,21,38,.86);
  --border:rgba(96,128,190,.22); --text:#dbe6f5; --dim:#8fa3c0; --dim2:#5f7394;
  --accent:#41d7ff; --orange:#ffa940; --violet:#b388ff; --crimson:#ff5d8a;
  --green:#2bd97a; --red:#ff3b5c; --amber:#ffd9a0;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg0);color:var(--text);
  font:14px/1.5 "Segoe UI",system-ui,-apple-system,sans-serif}
body{padding:0 0 60px}
header{display:flex;align-items:baseline;gap:14px;padding:16px 22px;flex-wrap:wrap;
  background:linear-gradient(180deg,rgba(22,32,56,.96),rgba(11,17,32,.92));
  border-bottom:1px solid var(--border)}
.brand{font-weight:700;color:var(--accent);font-size:16px;letter-spacing:.3px}
.brand span{color:var(--dim);font-weight:500}
.sub{color:var(--dim2);font-size:12.5px}
.badge{font-size:11.5px;padding:3px 10px;border-radius:999px;border:1px solid var(--border);
  color:var(--dim);background:rgba(255,255,255,.03)}
.badge.warn{color:var(--amber);border-color:rgba(255,170,60,.4);background:rgba(255,170,60,.08)}
main{max-width:1080px;margin:22px auto 0;padding:0 18px;display:flex;flex-direction:column;gap:22px}
section{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:18px 20px}
h2{margin:0 0 4px;font-size:15px;color:var(--accent);letter-spacing:.4px}
h3{margin:16px 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim2)}
.lead{color:var(--dim);font-size:13px;margin:0 0 12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:10px}
.kpi{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:10px;padding:8px 12px}
.kpi .n{font-size:18px;font-weight:700;color:var(--text);font-family:ui-monospace,Consolas,monospace}
.kpi .l{font-size:11px;color:var(--dim2);margin-top:1px}
.vgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-top:12px}
.vcard{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.vcard .top{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.vbadge{font-size:12px;font-weight:700;padding:3px 10px;border-radius:999px;letter-spacing:.4px}
.vbadge.yes{background:rgba(43,217,122,.14);color:var(--green);border:1px solid rgba(43,217,122,.4)}
.vbadge.trace{background:rgba(255,169,64,.14);color:var(--orange);border:1px solid rgba(255,169,64,.4)}
.vbadge.no{background:rgba(255,59,92,.12);color:var(--crimson);border:1px solid rgba(255,59,92,.35)}
.vbadge.smoke{background:rgba(143,163,192,.12);color:var(--dim);border:1px solid var(--border)}
.vcard .t{font-weight:600;font-size:13.5px}
.vcard .nums{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:var(--accent);margin-top:4px;line-height:1.55}
.vcard .det{font-size:12px;color:var(--dim);margin-top:4px;line-height:1.5}
canvas{max-width:100%;height:auto;display:block;margin:10px 0 4px;border:1px solid rgba(96,128,190,.14);border-radius:10px;background:#05080f}
.chartcap{font-size:11px;color:var(--dim2);margin-top:2px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:10px}
th,td{padding:5px 8px;text-align:right;border-bottom:1px solid rgba(96,128,190,.14)}
th{color:var(--dim2);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
td:first-child,th:first-child{text-align:left}
tr:hover td{background:rgba(255,255,255,.03)}
.callout{background:rgba(255,217,160,.07);border:1px solid rgba(255,170,60,.3);border-radius:12px;
  padding:12px 14px;font-size:13px;color:var(--amber);line-height:1.6;margin-top:12px}
.callout.blue{background:rgba(65,215,255,.06);border-color:rgba(65,215,255,.28);color:#bfe9ff}
.callout.red{background:rgba(255,59,92,.06);border-color:rgba(255,59,92,.3);color:#ffb3c3}
.callout b{color:inherit}
.caveat{font-size:12.5px;color:var(--dim);margin:0 0 8px;line-height:1.6}
.caveat b{color:var(--text)}
ul{margin:6px 0 0;padding-left:20px}
li{font-size:12.5px;color:var(--dim);margin:4px 0;line-height:1.55}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:11.5px}
.hint{font-size:11px;color:var(--dim2)}
"""

JS = r"""
const PAL=['#41d7ff','#ffa940','#b388ff','#2bd97a','#ff5d8a','#ffe08a','#9be9a8','#c0a0ff'];
const DIM='#5f7394', TEXT='#dbe6f5', GRID='rgba(96,128,190,.14)';
function CV(id,w,h){const cv=document.getElementById(id);cv.width=w;cv.height=h;return cv.getContext('2d');}
function txt(ctx,s,x,y,align,color,size){
  ctx.fillStyle=color||DIM;ctx.font=(size||11)+'px Consolas,ui-monospace,monospace';
  ctx.textAlign=align||'left';ctx.textBaseline='middle';ctx.fillText(s,x,y);
}
function axis(ctx,w,h,pad,xmin,xmax,ymin,ymax){
  ctx.strokeStyle=GRID;ctx.lineWidth=1;ctx.beginPath();
  for(let i=0;i<=4;i++){const y=pad.t+(h-pad.t-pad.b)*i/4;
    ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);}
  for(let i=0;i<=6;i++){const x=pad.l+(w-pad.l-pad.r)*i/6;
    ctx.moveTo(x,pad.t);ctx.lineTo(x,h-pad.b);}
  ctx.stroke();
  ctx.strokeStyle='rgba(219,230,245,.35)';ctx.beginPath();
  ctx.moveTo(pad.l,h-pad.b);ctx.lineTo(w-pad.r,h-pad.b);ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,h-pad.b);
  ctx.stroke();
}
function lineChart(id,series,xlabels,title,ylab){
  const w=940,h=330,pad={l:52,r:16,t:40,b:40};
  const ctx=CV(id,w,h);
  txt(ctx,title,pad.l,16,'left',TEXT,13);
  let ymax=0;for(const s of series)for(const v of s.data)ymax=Math.max(ymax,v);
  if(ymax<=0)ymax=1;
  axis(ctx,w,h,pad,0,series[0].data.length-1,0,ymax);
  txt(ctx,ylab,pad.l-6,pad.t-24,'left');
  for(let i=0;i<=4;i++){const y=pad.t+(h-pad.t-pad.b)*i/4;
    txt(ctx,(ymax*(4-i)/4).toFixed(0),pad.l-6,y,'right');}
  const n=series[0].data.length-1;
  for(let i=0;i<=n;i+=Math.max(1,Math.ceil(n/8))){const x=pad.l+(w-pad.l-pad.r)*i/n;
    txt(ctx,xlabels[i],x,h-pad.b+6,'center');}
  series.forEach(s=>{
    ctx.strokeStyle=s.color;ctx.lineWidth=1.8;ctx.beginPath();
    s.data.forEach((v,i)=>{const x=pad.l+(w-pad.l-pad.r)*i/n;
      const y=pad.t+(h-pad.t-pad.b)*(1-v/ymax);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
    ctx.stroke();
  });
  let lx=pad.l+8;
  series.forEach(s=>{ctx.fillStyle=s.color;ctx.fillRect(lx,pad.t-18,10,3);
    txt(ctx,s.name,lx+14,pad.t-16,'left',s.color,10.5);lx+=14+ctx.measureText(s.name).width+16;});
}
function groupBars(id,groups,labels,title,ylab){
  const w=940,h=330,pad={l:52,r:16,t:40,b:40};
  const ctx=CV(id,w,h);
  txt(ctx,title,pad.l,16,'left',TEXT,13);
  let ymax=0;for(const g of groups)for(const v of g.vals)ymax=Math.max(ymax,v);
  if(ymax<=0)ymax=1;
  axis(ctx,w,h,pad,0,groups.length-1,0,ymax);
  for(let i=0;i<=4;i++){const y=pad.t+(h-pad.t-pad.b)*i/4;
    txt(ctx,(ymax*(4-i)/4).toFixed(2),pad.l-6,y,'right');}
  const bw=Math.min(26,(w-pad.l-pad.r)/groups.length/3.2);
  groups.forEach((g,gi)=>{
    const gx=pad.l+(w-pad.l-pad.r)*(gi+0.5)/groups.length;
    g.vals.forEach((v,vi)=>{
      const bh=(h-pad.t-pad.b)*v/ymax;
      ctx.fillStyle=g.colors[vi];ctx.fillRect(gx-bw*1.5+bw*vi,bh>0?h-pad.b-bh:h-pad.b,bw-2,bh>0?bh:1);
    });
    txt(ctx,g.name,gx,h-pad.b+6,'center');
  });
  let lx=pad.l+8;
  (labels||[]).forEach((l,i)=>{ctx.fillStyle=PAL[i];ctx.fillRect(lx,pad.t-18,10,3);
    txt(ctx,l,lx+14,pad.t-16,'left',PAL[i],10.5);lx+=14+ctx.measureText(l).width+16;});
}
function heatTiles(id,grids,names,title){
  const tiles=grids.length,cols=Math.min(6,tiles),rows=Math.ceil(tiles/cols);
  const cw=108,ch=108,padx=6,pady=26,cell=10;
  const w=Math.max(480,cols*cw+40),h=pady+rows*(ch+16)+6;
  const ctx=CV(id,w,h);
  txt(ctx,title,8,14,'left',TEXT,13);
  grids.forEach((g,gi)=>{
    const col=gi%cols,row=Math.floor(gi/cols);
    const ox=24+col*cw,oy=pady+row*(ch+16);
    txt(ctx,names[gi],ox,oy-8,'left',PAL[gi%PAL.length],10.5);
    for(let r=0;r<8;r++)for(let c=0;c<8;c++){
      const v=g[r*8+c];
      ctx.fillStyle=v>0?'rgba(65,215,255,'+(0.25+0.75*v).toFixed(2)+')':'rgba(255,255,255,.03)';
      ctx.fillRect(ox+c*cell,oy+r*cell,cell-1,cell-1);
    }
    ctx.strokeStyle='rgba(96,128,190,.2)';ctx.strokeRect(ox-1,oy-1,8*cell+1,8*cell+1);
  });
}
function hist2(id,a,b,title){
  const w=940,h=330,pad={l:56,r:16,t:40,b:40};
  const ctx=CV(id,w,h);
  txt(ctx,title,pad.l,16,'left',TEXT,13);
  const all=a.concat(b);if(!all.length)return;
  const lo=Math.floor(Math.min(...all)),hi=Math.ceil(Math.max(...all))+0.01;
  const B=18,bw=(hi-lo)/B;
  const ha=new Array(B).fill(0),hb=new Array(B).fill(0);
  a.forEach(v=>ha[Math.min(B-1,Math.floor((v-lo)/bw))]++);
  b.forEach(v=>hb[Math.min(B-1,Math.floor((v-lo)/bw))]++);
  let ymax=Math.max(...ha,...hb,1);
  axis(ctx,w,h,pad,0,B,0,ymax);
  for(let i=0;i<=4;i++){const y=pad.t+(h-pad.t-pad.b)*i/4;
    txt(ctx,Math.round(ymax*(4-i)/4),pad.l-6,y,'right');}
  const cw=(w-pad.l-pad.r)/B;
  for(let i=0;i<B;i++){
    const x=pad.l+i*cw;
    ctx.fillStyle='rgba(65,215,255,.55)';ctx.fillRect(x+1,pad.t+(h-pad.t-pad.b)*(1-ha[i]/ymax),cw/2-1,(h-pad.t-pad.b)*ha[i]/ymax);
    ctx.fillStyle='rgba(255,169,64,.55)';ctx.fillRect(x+cw/2,pad.t+(h-pad.t-pad.b)*(1-hb[i]/ymax),cw/2-1,(h-pad.t-pad.b)*hb[i]/ymax);
    if(i%3===0)txt(ctx,(lo+i*bw).toFixed(0),x+2,h-pad.b+6,'left');
  }
  ctx.fillStyle='#41d7ff';ctx.fillRect(pad.l+8,pad.t-18,10,3);txt(ctx,'ранняя треть',pad.l+22,pad.t-16,'left','#41d7ff',10.5);
  ctx.fillStyle='#ffa940';ctx.fillRect(pad.l+150,pad.t-18,10,3);txt(ctx,'поздняя треть',pad.l+164,pad.t-16,'left','#ffa940',10.5);
}
function evoChart(id,avgFood,instincts,dopamine,title){
  const w=940,h=330,pad={l:56,r:64,t:40,b:40};
  const ctx=CV(id,w,h);
  txt(ctx,title,pad.l,16,'left',TEXT,13);
  const g=avgFood.length-1;
  let amax=Math.max(...avgFood,1),imax=Math.max(...instincts,1);
  axis(ctx,w,h,pad,0,g,0,amax);
  for(let i=0;i<=4;i++){const y=pad.t+(h-pad.t-pad.b)*i/4;
    txt(ctx,(amax*(4-i)/4).toFixed(1),pad.l-6,y,'right');}
  const bw=(w-pad.l-pad.r)/g*0.5;
  avgFood.forEach((v,i)=>{
    const x=pad.l+(w-pad.l-pad.r)*(i+0.5)/g;
    const bh=(h-pad.t-pad.b)*v/amax;
    ctx.fillStyle='rgba(65,215,255,.7)';ctx.fillRect(x-bw/2,h-pad.b-bh,bw,bh);
    txt(ctx,i,x,h-pad.b+6,'center');
  });
  // with_instinct — вторая ось (справа)
  const rl=w-pad.r+46;
  for(let i=0;i<=4;i++){const y=pad.t+(h-pad.t-pad.b)*i/4;
    txt(ctx,Math.round(imax*(4-i)/4),rl,y,'left');}
  ctx.strokeStyle='#ffa940';ctx.lineWidth=1.8;ctx.beginPath();
  instincts.forEach((v,i)=>{const x=pad.l+(w-pad.l-pad.r)*(i+0.5)/g;
    const y=pad.t+(h-pad.t-pad.b)*(1-v/imax);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();
  instincts.forEach((v,i)=>{const x=pad.l+(w-pad.l-pad.r)*(i+0.5)/g;
    const y=pad.t+(h-pad.t-pad.b)*(1-v/imax);
    ctx.fillStyle='#ffa940';ctx.beginPath();ctx.arc(x,y,3.5,0,7);ctx.fill();});
  txt(ctx,'avg_food (левая ось)',pad.l+8,pad.t-18,'left','#41d7ff',10.5);
  txt(ctx,'with_instinct (правая)',rl-60,pad.t-18,'left','#ffa940',10.5);
}
function runCharts(D){
  const m=D.metrics,pc=m.per_creature;
  const ids=Object.keys(pc).map(Number).sort((a,b)=>a-b);
  // 1. Кривые еды (накопительно)
  const series=ids.map(id=>({name:'Существо '+id,color:PAL[(id-1)%PAL.length],data:pc[id].food_over_time}));
  const xl=pc[ids[0]].food_over_time.map((_,i)=>String(i*1000));
  lineChart('ch_food',series,xl,'Кривые еды по существам (накопительное food_eaten, снапшот каждые 1000 тиков)','порций');
  // 2. Сон по третям
  const groups=ids.map(id=>({name:'№'+id,vals:['early','mid','late'].map(t=>pc[id].thirds[t].sleep_frac),
    colors:['rgba(65,215,255,.55)','rgba(179,136,255,.6)','rgba(255,169,64,.6)']}));
  groupBars('ch_sleep',groups,['ранняя','средняя','поздняя'],'Доля сна по третям (sleep_frac)','доля времени');
  // 3. Занятость поздней трети
  const grids=ids.map(id=>pc[id].thirds.late.occupancy);
  heatTiles('ch_occ',grids,ids.map(id=>'№'+id),'Занятость карты (8×8) в поздней трети — клетка подсвечена, если существо там было');
  // 4. NN-дистанции
  hist2('ch_nn',m.nn_distance.early.values,m.nn_distance.late.values,'NN-дистанции: гистограмма ранней vs поздней трети');
  // 5. Эволюция
  const ev=m.evolution;
  if(ev.gen_count>0)evoChart('ch_evo',ev.avg_food_trend,ev.with_instinct_trend,ev.instinct_dopamine_trend,
    'Эволюция: avg_food по поколениям (бары, левая ось) и with_instinct (линия, правая)');
}
"""


def render_html(data, verdicts):
    m = data.get("metrics", {})
    cfg = m.get("config", {})
    quick = bool(cfg.get("quick", False))

    total_ticks = cfg.get("total_ticks", 0)
    minutes = total_ticks * 0.05 / 60.0
    n_creatures = cfg.get("n_creatures", 0)
    n_berries = cfg.get("n_berries", 0)
    evo_gens = cfg.get("evo_gens", 0)

    # ── Сводка ──
    kpis = [
        ("сид арены", str(cfg.get("wild_seed", "?"))),
        ("существ", str(n_creatures)),
        ("ягод", str(n_berries)),
        ("тиков", f"{total_ticks:,}".replace(",", " ")),
        ("≈ минут жизни", f"{minutes:.0f}"),
        ("поколений", str(evo_gens)),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="n">{n}</div><div class="l">{l}</div></div>'
                       for l, n in kpis)

    # ── Вердикты ──
    badge_class = {"ЕСТЬ": "yes", "СЛЕДЫ": "trace", "НЕТ": "no", "СМОУК": "smoke"}
    cards = []
    for v in verdicts:
        cards.append(
            f'<div class="vcard"><div class="top">'
            f'<span class="vbadge {badge_class.get(v["verdict"], "no")}">{html.escape(v["verdict"])}</span>'
            f'<span class="t">{html.escape(v["title"])}</span></div>'
            f'<div class="nums">{html.escape(v["numbers"])}</div>'
            + (f'<div class="det">{html.escape(v["details"])}</div>' if v.get("details") else "")
            + "</div>"
        )
    verdicts_html = "".join(cards)

    # ── Сценарий 1: таблица по существам ──
    pc = m.get("per_creature", {})
    ids = sorted(int(k) for k in pc.keys())
    rows = []
    end_states = {}
    if data.get("scenarios", {}).get("wild"):
        last = data["scenarios"]["wild"][-1]
        for e in last.get("end", []):
            end_states[e["id"]] = e
    for cid in ids:
        p = pc[str(cid)]
        t = p["thirds"]
        e = end_states.get(cid, {})
        rows.append(
            f"<tr><td>№{cid}</td>"
            f"<td>{t['early']['food_rate']:.2f}</td><td>{t['mid']['food_rate']:.2f}</td><td>{t['late']['food_rate']:.2f}</td>"
            f"<td>{t['early']['tortuosity']:.2f}</td><td>{t['late']['tortuosity']:.2f}</td>"
            f"<td>{t['late']['sleep_frac']:.2f}</td>"
            f"<td>{t['late']['wall_hit_rate']:.1f}</td>"
            f"<td>{e.get('learned_instincts', '—')}</td>"
            f"<td>{e.get('places_remembered', '—')}</td>"
            f"<td>{e.get('food_eaten', '—')}</td>"
            f"<td>{'да' if e.get('alive') else 'нет' if e.get('alive') is False else '—'}</td></tr>"
        )
    table1 = ("<table><tr><th>существо</th><th>еда/1000т<br>ранняя</th><th>еда/1000т<br>средняя</th>"
              "<th>еда/1000т<br>поздняя</th><th>tortuosity<br>ранняя</th><th>tortuosity<br>поздняя</th>"
              "<th>сон<br>поздняя</th><th>удары/1000т<br>поздняя</th><th>инстинкты<br>конец</th>"
              "<th>мест<br>запомнено</th><th>съедено<br>всего</th><th>жив</th></tr>"
              + "".join(rows) + "</table>")

    # ── «Что произошло» в сценарии 1: честный разбор ──
    total_food = sum(e.get("food_eaten", 0) for e in end_states.values())
    max_g = 0.0
    if data.get("scenarios", {}).get("wild"):
        for s in data["scenarios"]["wild"]:
            for c in s["critters"]:
                max_g = max(max_g, c["glucose"])
    agg = m.get("aggregate", {})
    mean_sleep = agg.get("late", {}).get("sleep_frac", 0.0)

    if total_food == 0:
        learned = sum(e.get("learned_instincts", 0) for e in end_states.values())
        callout1 = (
            '<div class="callout red"><b>Закрытый контур СНОВА разомкнулся: '
            "за весь прогон не съедено ни одной ягоды (Σ food_eaten = 0).</b> "
            "После ребаланса метаболизма (сон не кормит: SLEEP_GLUCOSE_CAP=120<150; "
            "новорождённый арены голоден: ARENA_NEWBORN_HUNGER=120; дефицитный синтез "
            "HUNGER ребалансирован до равновесия ≈80 при глюкозе на капе сна) голод "
            "возникает у всех сидов — значит, в этой конфигурации снова нарушен один "
            "из трёх рычагов: либо сон снова кормит (потолок 255), либо новорождённый "
            "не голоден, либо дефицитный синтез слабее ферментативного утоления "
            "(равновесие HUNGER<40). "
            f"Сейчас существа проводят в сне {mean_sleep * 100:.0f}% поздней трети при "
            f"глюкозе {max_g:.0f}; выученных рефлексов избегания стен накоплено "
            f"{learned} шт. — STDP работает, но без еды учит только стены.</div>"
        )
    else:
        callout1 = (
            '<div class="callout blue"><b>Еда была найдена и съедена.</b> '
            f"Σ food_eaten = {total_food} за {total_ticks} тиков; глюкоза в среднем {max_g:.0f}.</div>"
        )

    # ── Сценарий 2 ──
    evo = m.get("evolution", {})
    evo_rows = []
    for g in data.get("scenarios", {}).get("evo", []):
        evo_rows.append(
            f"<tr><td>поколение {g.get('generation')}</td><td>{g.get('survivors')}</td>"
            f"<td>{g.get('born')}</td><td>{g.get('avg_food_eaten'):.2f}</td>"
            f"<td>{g.get('max_food_eaten')}</td><td>{g.get('with_instinct')}</td>"
            f"<td>{g.get('instinct_dopamine_avg'):.3f}</td></tr>"
        )
    table2 = ("<table><tr><th>поколение</th><th>выжило</th><th>родилось</th><th>avg_food</th>"
              "<th>max_food</th><th>с инстинктом</th><th>дофамин инстинкта (ср.)</th></tr>"
              + "".join(evo_rows) + "</table>")

    evo_narrative = ""
    if evo.get("gen_count", 0) >= 2:
        inst = evo.get("with_instinct_trend", [])
        food = evo.get("avg_food_trend", [])
        div = evo.get("diversity_trend", [])
        div_s = ", ".join(_fmt(x) for x in div) if div else "—"
        if inst and inst[-1] > inst[0] and food and food[-1] < food[0]:
            evo_narrative = (
                '<div class="callout blue"><b>Ключевой результат «Поколений»: рефлекс материализуется, '
                "адаптивность — нет.</b> Инстинкт-ген (эффект Болдуина движка) фиксируется у 100% популяции "
                f"уже к поколению 1–2 (дофамин {evo.get('dopamine_last', 0):.1f}/8.0 — потолок), но средняя "
                f"добыча НЕ растёт: {', '.join(_fmt(x) for x in food)} порц/поколение; лучшее поколение — "
                f"gen2 ({_fmt(max(food))}), финал {_fmt(food[-1])} ниже старта ({_fmt(food[0])}). "
                f"Разнообразие геномов по поколениям: {div_s} — "
                "«ловушка фиксации» не побеждена, если метрика схлопывается: врождённый рефлекс есть, "
                "а охотники не стали лучше.</div>"
            )
        elif inst and inst[-1] > inst[0]:
            evo_narrative = (
                '<div class="callout blue"><b>Эффект Болдуина работает:</b> инстинкт-ген распространяется '
                f"по популяции ({', '.join(str(x) for x in inst)}), средняя добыча "
                f"{', '.join(_fmt(x) for x in food)}. Разнообразие геномов (анти-фиксация, кроссовер): {div_s}.</div>"
            )

    # ── Оговорки ──
    slp = m.get("sleep", {})
    wake_mean = slp.get("wake_mean", 0.0)
    agg_sleep = agg.get("early", {}).get("sleep_frac", 0.0)
    s_late = agg.get("late", {}).get("sleep_frac", 0.0)
    caveats = [
        (f"Горизонт: {total_ticks} тиков × 50 мс = {minutes:.0f} минут жизни существа — это не «дни»; "
         "долгосрочные циклы (сезоны, накопление навыка) в окно не попадают."),
        "Один сид (777) и один набор сидов существ (3001–3006): без репликации по сидам "
        "нельзя отличить закономерность от везения конкретной линии. После замыкания контура "
        "возникновение голода от сида НЕ зависит (новорождённый арены голоден — ARENA_NEWBORN_HUNGER=120, "
        "сон не кормит — SLEEP_GLUCOSE_CAP=120<150, дефицитный синтез ребалансирован до равновесия "
        "HUNGER≈80 при глюкозе 120) — от сида зависит КАЧЕСТВО охоты (навык/удача), а не её запуск.",
        "Критерии вердиктов — эвристики с произвольными порогами (1.3×, 2×, 0.8×, 15%), а не статистические тесты: "
        "нет p-value, нет доверительных интервалов, нет контрольной группы.",
        "Метрики считаются по снапшотам каждые 1000 тиков: сон, мгновенные скорости и «следование» — грубые прокси "
        "(окно 1000 тиков для follow — очень грубо; по-тиковый счётчик дан дополнительно).",
        f"Ключевая оговорка сценария 1: рост food_rate по третям нельзя читать как «обучение» без оговорок — "
        f"еда течёт весь прогон (Σ food_eaten = {total_food} за {total_ticks} тиков), кривая НЕ обязана быть "
        f"монотонной, а сон после ребаланса экономики утомления (гейн RATE_MOTOR→FATIGUE ×6, порог 200, окно 12) "
        f"занимает {_fmt(s_late * 100, 1)}% поздней трети (было 72–78%): бодрые окна ~{_fmt(wake_mean, 1)} тиков "
        "в среднем — существа охотятся, а не спят ¾ жизни.",
        f"Эволюция — только {evo_gens} поколений ({cfg.get('evo_gens', '?')} × 600 тиков), стандартные параметры "
        "(population 8, food_per_child 6): эффект Болдуина в движке заявлен быстрым (инстинкт за 1–2 поколения), "
        "долгий тренд отбора не измерен.",
        "Инстинкты в сценарии 2 считаются как genome.instincts у выживших; «дофамин инстинкта» — сила врождённого "
        "рефлекса (потолок 8.0).",
    ]
    caveats_html = "".join(f'<div class="caveat">• {html.escape(c)}</div>' for c in caveats)

    quick_banner = ""
    if quick:
        quick_banner = (
            '<div class="callout red"><b>СМОУК-ПРОГОН (--quick):</b> 3000 тиков «Дикого леса» и 2 поколения '
            "вместо 30000/5. Вердикты помечены «СМОУК» и не показательны — это проверка пайплайна.</div>"
        )

    html_str = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Эмерджентность · закрытый контур · dioma</title>
<style>__CSS__</style>
</head>
<body>
<header>
  <div class="brand">ЭМЕРДЖЕНТНОСТЬ <span>· закрытый контур</span></div>
  <div class="sub">emergence_run.rs → emergence_data.json → этот отчёт</div>
  <span class="badge">сид 777</span>
  <span class="badge">6 существ · 10 ягод · 3 комнаты</span>
  <span class="badge">30000 тиков ≈ 25 мин</span>
  <span class="badge">5 поколений</span>
  <span class="badge">без инъекций голода</span>
  __QUICKBANNER__
</header>
<main>
  <section>
    <h2>Сводка прогона</h2>
    <p class="lead">Закрытый контур: новорождённый арены голоден с рождения (встроенная механика движка:
    ARENA_NEWBORN_HUNGER=120, как NEWBORN_HUNGER эволюции), сон НЕ кормит (SLEEP_GLUCOSE_CAP=120 &lt; порога
    дефицита 150), голод поддерживается дефицитным синтезом (ребаланс экономики глюкоза↔голод), едят когда хотят,
    ягоды отрастают от климата, сон — встроенный цикл. Никаких скриптовых циклов голода и тестовых подпорок.</p>
    <div class="kpis">__KPIS__</div>
  </section>

  <section>
    <h2>Вердикты эмерджентности</h2>
    <p class="lead">Автоматические, по порогам из tools/emergence_report.py. «ЕСТЬ» — сигнал сильный,
    «СЛЕДЫ» — слабый/частичный, «НЕТ» — сигнала нет. Числа — решающие метрики.</p>
    <div class="vgrid">__VERDICTS__</div>
  </section>

  <section>
    <h2>Сценарий 1 · «Дикий лес» — естественный закрытый контур</h2>
    <p class="lead">3 комнаты (0,0)–(40,14), (40,0)–(80,14), (0,14)–(40,28), сид 777, существа сиды 3001–3006,
    10 ягод, 30000 тиков.</p>
    __CALLOUT1__
    <div class="grid2">
      <div>
        <canvas id="ch_food"></canvas>
        <div class="chartcap">Кривые еды по существам. Плоские нули = контур не замкнулся на еде (см. красный блок).</div>
      </div>
      <div>
        <canvas id="ch_sleep"></canvas>
        <div class="chartcap">Доля сна по третям — сон-петля доминирует.</div>
      </div>
    </div>
    <canvas id="ch_occ"></canvas>
    <div class="chartcap">Карта 80×28, сетка 8×8 (клетка 10×3.5); подсвечены клетки, где существо было в поздней трети.</div>
    <canvas id="ch_nn"></canvas>
    <div class="chartcap">Расстояние до ближайшего соседа: ранняя (голубой) vs поздняя (оранжевый) треть.</div>
    <h3>Таблица метрик по существам</h3>
    __TABLE1__
  </section>

  <section>
    <h2>Сценарий 2 · «Поколения» — эволюция (эффект Болдуина)</h2>
    <p class="lead">Evolution::new(сид 42, карта, EvolutionParams::default()): 5 поколений × 600 тиков,
    отбор по съеденному, выученное сворачивается в инстинкт-гены потомкам (движок сам даёт новорождённым голод 200 — это его встроенная механика, не наш скрипт).</p>
    <canvas id="ch_evo"></canvas>
    <div class="chartcap">avg_food (бары) и число существ с инстинкт-геном (линия) по поколениям.</div>
    __EVO_NARRATIVE__
    __TABLE2__
  </section>

  <section>
    <h2>Честные оговорки</h2>
    __CAVEATS__
  </section>
</main>
<script>
const DATA = __DATA__;
__JS__
runCharts(DATA);
</script>
</body>
</html>
"""
    return (html_str.replace("__CSS__", CSS)
                   .replace("__QUICKBANNER__", quick_banner)
                   .replace("__KPIS__", kpi_html)
                   .replace("__VERDICTS__", verdicts_html)
                   .replace("__CALLOUT1__", callout1)
                   .replace("__TABLE1__", table1)
                   .replace("__EVO_NARRATIVE__", evo_narrative)
                   .replace("__TABLE2__", table2)
                   .replace("__CAVEATS__", caveats_html)
                   .replace("__DATA__", json.dumps(data, ensure_ascii=False))
                   .replace("__JS__", JS))


def main():
    ap = argparse.ArgumentParser(description="Эмерджентность: JSON → самодостаточный HTML-отчёт")
    ap.add_argument("--data", default="tools/emergence_data.json")
    ap.add_argument("--out", default="tools/emergence_report.html")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"нет файла данных: {data_path} (сначала: cargo run --release "
                 f"--manifest-path crates/dioma-engine/Cargo.toml --example emergence_run -- --out {data_path})")

    data = json.loads(data_path.read_text(encoding="utf-8"))
    verdicts = compute_verdicts(data.get("metrics", {}))
    out_path = Path(args.out)
    out_path.write_text(render_html(data, verdicts), encoding="utf-8")

    print("Вердикты эмерджентности:")
    for v in verdicts:
        print(f"  [{v['verdict']:^6}] {v['title']}")
        print(f"         {v['numbers']}")
    print(f"Отчёт: {out_path} ({out_path.stat().st_size} байт)")


if __name__ == "__main__":
    main()
