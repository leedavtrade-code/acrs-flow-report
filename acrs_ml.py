#!/usr/bin/env python3
"""ACRS ML review: post-mortem, deflated Sharpe and a record-only meta-label test.

Steps 3, 5 and 6 of the research -> code -> backtest -> live -> post-mortem ->
fine-tune loop, built to the same standard as the rest of ACRS: it reads the
data the builder already embeds in ACRS_Simulation.html and changes nothing in
the live model. Rule changes still go through the builder's same-universe A/B.

    python ml/acrs_ml.py                 # writes ACRS_ML_Review.html + ml/ml_review.json

Needs numpy only.
"""
import argparse
import html
import json
import math
import re
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NORM = NormalDist()
EULER_GAMMA = 0.5772156649015329

FEATURES = ['f_rs13', 'f_atrp', 'f_rvol', 'f_ext50', 'f_ext150', 'f_ma_spread',
            'f_to_pivot', 'f_dv20', 'f_fip', 'f_fip252', 'f_secrank']
FEATURE_LABEL = {
    'f_rs13': '13-wk RS', 'f_atrp': 'ATR %', 'f_rvol': 'Relative volume',
    'f_ext50': '% above MA50', 'f_ext150': '% above MA150', 'f_ma_spread': 'MA50-MA150 spread',
    'f_to_pivot': '% to pivot', 'f_dv20': '20d $ volume', 'f_fip': 'FIP (13w)',
    'f_fip252': 'FIP (1y)', 'f_secrank': 'Sector rank'}

# Pre-registered before the first run (2026-10-08). Do not loosen after seeing a result.
PROMOTION = {'min_fold_win_share': 0.75, 'min_mean_ic': 0.05, 'forward_weeks': 26}


# ---------------------------------------------------------------- loading

def load_embedded(path):
    """The `const D = {...}` object the builder writes into each page."""
    text = Path(path).read_text(encoding='utf-8')
    m = re.search(r'const D\s*=\s*', text)
    if not m:
        raise ValueError(f'no embedded data object in {path}')
    obj, _ = json.JSONDecoder().raw_decode(text[m.end():])
    return obj


def r_of(trades):
    return np.array([t['realized_R'] for t in trades], dtype=float)


def profit_factor(r):
    loss = -r[r < 0].sum()
    return float(r[r > 0].sum() / loss) if loss > 0 else float('inf')


# ---------------------------------------------------------------- step 5: post-mortem

def bootstrap_mean_gap(a, b, rng, n=20000):
    """P(mean(a) <= mean(b)) under resampling of each group."""
    ia = rng.integers(0, len(a), (n, len(a)))
    ib = rng.integers(0, len(b), (n, len(b)))
    return float(np.mean(a[ia].mean(1) - b[ib].mean(1) <= 0))


def post_mortem(trades, rng):
    r = r_of(trades)
    years = sorted({t['into'][:4] for t in trades})
    by_setup = []
    for s in sorted({t['setup'] for t in trades}):
        sub = [t for t in trades if t['setup'] == s]
        rs = np.sort(r_of(sub))[::-1]
        per_year = {y: round(float(sum(t['realized_R'] for t in sub if t['into'][:4] == y)), 1) for y in years}
        by_setup.append({
            'setup': s, 'n': len(sub), 'share': round(len(sub) / len(trades), 3),
            'win': round(float(np.mean(rs > 0)), 3), 'mean_R': round(float(rs.mean()), 2),
            'median_R': round(float(np.median(rs)), 2), 'total_R': round(float(rs.sum()), 1),
            'total_R_ex_top3': round(float(rs[3:].sum()), 1),
            'years_positive': sum(v > 0 for v in per_year.values()), 'years': len(years),
            'by_year': per_year})
    by_exit = []
    for c in sorted({t['exit_cause'] for t in trades}):
        rs = r_of([t for t in trades if t['exit_cause'] == c])
        by_exit.append({'exit': c, 'n': len(rs), 'mean_R': round(float(rs.mean()), 2),
                        'total_R': round(float(rs.sum()), 1)})

    stop_out = [t for t in trades if t['exit_cause'] != 'MA50_CLOSE']
    normal = [t for t in trades if t['exit_cause'] == 'MA50_CLOSE']
    stop_profile = {f: [round(float(np.nanmean([t[f] for t in stop_out])), 2),
                        round(float(np.nanmean([t[f] for t in normal])), 2)]
                    for f in ('f_ext50', 'f_rvol', 'f_to_pivot', 'f_atrp')}

    gross = np.sort(r[r > 0])[::-1]
    top = max(1, int(round(0.05 * len(r))))
    best = max(by_setup, key=lambda x: x['total_R'])
    rest = r_of([t for t in trades if t['setup'] != best['setup']])
    return {
        'n': len(trades), 'total_R': round(float(r.sum()), 1),
        'profit_factor': round(profit_factor(r), 2),
        'top5pct_share_of_gross': round(float(gross[:top].sum() / gross.sum()), 3),
        'by_setup': by_setup, 'by_exit': by_exit, 'stop_profile': stop_profile,
        'stop_leak_R': round(float(r_of(stop_out).sum()), 1),
        'carrier': best['setup'],
        'p_carrier_not_better': bootstrap_mean_gap(r_of([t for t in trades if t['setup'] == best['setup']]), rest, rng),
    }


# ---------------------------------------------------------------- step 3: deflated Sharpe

def daily_returns(curve):
    eq = np.array([v for _, v in curve], dtype=float)
    return eq[1:] / eq[:-1] - 1


def moments(x):
    m, s = x.mean(), x.std(ddof=1)
    z = (x - m) / s
    return float(m / s), float(np.mean(z ** 3)), float(np.mean(z ** 4))


def psr(sr, sr_star, t, skew, kurt):
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012), per-period SRs."""
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return NORM.cdf((sr - sr_star) * math.sqrt(t - 1) / den)


def expected_max_sr(n_trials, var):
    """E[max SR] of n_trials zero-edge strategies (False Strategy Theorem)."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var) * ((1 - EULER_GAMMA) * NORM.inv_cdf(1 - 1 / n_trials)
                             + EULER_GAMMA * NORM.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(curve, n_trials, trial_sharpes_annual=None, grid=(1, 5, 10, 20, 50, 100, 200)):
    ret = daily_returns(curve)
    t = len(ret)
    sr, skew, kurt = moments(ret)
    null_var = 1 / (t - 1)    # sampling variance of a zero-edge SR estimate: independent trials
    rows = []
    for n in sorted(set(grid) | {n_trials}):
        sr0 = expected_max_sr(n, null_var)
        rows.append({'n': n, 'hurdle_annual': round(sr0 * math.sqrt(252), 2),
                     'dsr': round(psr(sr, sr0, t, skew, kurt), 3), 'registry': n == n_trials})
    budget = next((n for n in range(1, 5001) if psr(sr, expected_max_sr(n, null_var), t, skew, kurt) < 0.95), None)
    out = {'sharpe_annual': round(sr * math.sqrt(252), 2), 'days': t,
           'skew': round(skew, 2), 'kurtosis': round(kurt, 2),
           'psr_vs_zero': round(psr(sr, 0, t, skew, kurt), 4),
           'n_trials': n_trials, 'table': rows,
           'dsr': next(r['dsr'] for r in rows if r['registry']),
           'first_n_below_95': budget}
    if trial_sharpes_annual and len(trial_sharpes_annual) > 1:
        v = float(np.var(np.array(trial_sharpes_annual) / math.sqrt(252), ddof=1))
        out['dsr_grid_dispersion'] = round(psr(sr, expected_max_sr(n_trials, v), t, skew, kurt), 3)
    return out


# ---------------------------------------------------------------- step 4->5: is live consistent with the backtest?

def ks_stat(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, 'right') / len(a)
                               - np.searchsorted(b, grid, 'right') / len(b))))


def live_check(backtest, live, inception, window, rng, n_boot=20000, n_perm=2000):
    """Live closed trades vs the pre-inception backtest. A trade that CLOSED inside a
    `window`-day live record held at most `window` days, so the comparison pool is
    restricted the same way; otherwise the backtest's 100-day+ winners (still open
    in a young book) make any young book look broken."""
    pre = [t for t in backtest if t['into'] < inception]
    pool = [t for t in pre if t['hold'] <= window]
    if not live or len(pool) < 30:
        return None
    rl, live_pf = r_of(live), profit_factor(r_of(live))

    def p_this_bad(trs):
        samp = r_of(trs)[rng.integers(0, len(trs), (n_boot, len(rl)))]
        loss = np.where(samp < 0, -samp, 0).sum(1)
        pf = np.divide(np.where(samp > 0, samp, 0).sum(1), loss, out=np.full(n_boot, np.inf), where=loss > 0)
        return round(float(np.mean(samp.sum(1) <= rl.sum())), 3), round(float(np.mean(pf <= live_pf)), 3)

    p_r, p_pf = p_this_bad(pool)
    p_r_naive, _ = p_this_bad(pre)

    p_a = float(np.mean([t['setup'] == 'A' for t in pre]))
    k = sum(t['setup'] == 'A' for t in live)
    p_mix = sum(math.comb(len(live), i) * p_a ** i * (1 - p_a) ** (len(live) - i) for i in range(k + 1))

    drift = []
    for f in FEATURES:
        a = np.array([t[f] for t in pre if t[f] is not None], float)
        b = np.array([t[f] for t in live if t[f] is not None], float)
        if len(b) < 5:
            continue
        d = ks_stat(a, b)
        both = np.concatenate([a, b])
        hits = 0
        for _ in range(n_perm):
            rng.shuffle(both)
            hits += ks_stat(both[:len(a)], both[len(a):]) >= d
        drift.append({'feature': f, 'label': FEATURE_LABEL[f], 'ks': round(d, 3),
                      'p': round((hits + 1) / (n_perm + 1), 4),
                      'backtest_median': round(float(np.median(a)), 2),
                      'live_median': round(float(np.median(b)), 2)})
    return {
        'n_live': len(live), 'n_pre': len(pre), 'window_days': window, 'n_pool': len(pool),
        'live_total_R': round(float(rl.sum()), 2), 'live_pf': round(live_pf, 2),
        'p_total_R_this_bad': p_r, 'p_pf_this_bad': p_pf, 'p_total_R_this_bad_uncensored': p_r_naive,
        'live_A': k, 'expected_A': round(p_a * len(live), 1), 'p_this_few_A': round(p_mix, 3),
        'live_mix': {s: sum(t['setup'] == s for t in live) for s in 'ABC'},
        'drift': drift, 'drift_bonferroni': round(0.05 / max(1, len(drift)), 4),
    }


# ---------------------------------------------------------------- step 6: meta-label filter (record-only)

def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):                      # average ties
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float(np.corrcoef(rankdata(np.asarray(a)), rankdata(np.asarray(b)))[0, 1])


def design(trades, mu=None, sd=None):
    x = np.array([[np.nan if t[f] is None else float(t[f]) for f in FEATURES] for t in trades])
    j = FEATURES.index('f_dv20')
    x[:, j] = np.log1p(np.clip(x[:, j], 0, None))
    if mu is None:
        mu, sd = np.nanmean(x, 0), np.nanstd(x, 0)
        sd[sd == 0] = 1
    z = np.clip(np.nan_to_num((x - mu) / sd), -3, 3)
    setup = np.array([[t['setup'] == 'A', t['setup'] == 'B'] for t in trades], float)
    return np.c_[np.ones(len(trades)), z, setup], mu, sd


def fit_ridge(x, y, lam):
    pen = lam * np.eye(x.shape[1])
    pen[0, 0] = 0
    return np.linalg.solve(x.T @ x + pen, x.T @ y)


def fit_logit(x, y, lam, iters=50):
    w = np.zeros(x.shape[1])
    pen = lam * np.eye(x.shape[1])
    pen[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(x @ w)))
        step = np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]) + pen + 1e-9 * np.eye(len(w)),
                               x.T @ (p - y) + pen @ w)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def meta_label(trades, keep=0.5, min_train=100):
    """Annual walk-forward. Training uses only trades CLOSED before the test year
    starts (purge), so no label overlaps the test window. Score = model rank;
    the filter keeps the top `keep` share of each test year's entries."""
    folds = []
    for y in sorted({int(t['into'][:4]) for t in trades}):
        start, end = f'{y}-01-01', f'{y + 1}-01-01'
        train = [t for t in trades if t['out'] < start]
        test = [t for t in trades if start <= t['into'] < end]
        if len(train) < min_train or len(test) < 10:
            continue
        xtr, mu, sd = design(train)
        xte, _, _ = design(test, mu, sd)
        rtr, rte = r_of(train), r_of(test)
        scores = {
            'ridge': xte @ fit_ridge(xtr, np.clip(rtr, -2, 5), 10.0),   # winsorised: one 47R trade must not set the fit
            'logit': xte @ fit_logit(xtr, (rtr > 0).astype(float), 5.0),
        }
        row = {'year': y, 'n_train': len(train), 'n_test': len(test), 'take_all_R': round(float(rte.sum()), 1)}
        for name, s in scores.items():
            kept = s >= np.quantile(s, 1 - keep)
            row[name + '_R'] = round(float(rte[kept].sum()), 1)
            row[name + '_ic'] = round(spearman(s, rte), 3)
        folds.append(row)
    models = {}
    for name in ('ridge', 'logit'):
        wins = sum(f[name + '_R'] >= f['take_all_R'] for f in folds)
        ic = float(np.mean([f[name + '_ic'] for f in folds])) if folds else float('nan')
        passed = bool(folds) and wins / len(folds) >= PROMOTION['min_fold_win_share'] and ic >= PROMOTION['min_mean_ic']
        models[name] = {'filtered_R': round(sum(f[name + '_R'] for f in folds), 1),
                        'folds_beating_take_all': wins, 'mean_ic': round(ic, 3),
                        'passes_backtest_rule': passed}
    return {'keep': keep, 'folds': folds, 'take_all_R': round(sum(f['take_all_R'] for f in folds), 1),
            'models': models, 'promotion_rule': PROMOTION,
            'status': ('candidate for forward paper test' if any(m['passes_backtest_rule'] for m in models.values())
                       else 'rejected - stays record-only')}


# ---------------------------------------------------------------- registry

def load_registry(path):
    reg = json.loads(Path(path).read_text(encoding='utf-8'))
    tested = [e for e in reg['experiments'] if e['status'] != 'proposed']
    n = 1 + sum(int(e.get('variants') or 0) for e in tested) + int(reg.get('untracked_prior_trials') or 0)
    return reg, n


# ---------------------------------------------------------------- critic

def findings(pm, dsr, live, ml, reg):
    out = []
    c = next(s for s in pm['by_setup'] if s['setup'] == pm['carrier'])
    others = [s for s in pm['by_setup'] if s['setup'] != c['setup']]
    out.append(
        f"Setup {c['setup']} carries the system: {c['n']} of {pm['n']} trades ({c['share']:.0%}) produce "
        f"{c['total_R']:+.1f}R of {pm['total_R']:+.1f}R. Without its 3 best trades it still makes "
        f"{c['total_R_ex_top3']:+.1f}R and it was positive in {c['years_positive']} of {c['years']} years "
        f"(bootstrap P(not better than the rest) = {pm['p_carrier_not_better']:.3f}).")
    for s in others:
        if abs(s['total_R']) < 0.1 * max(1, abs(pm['total_R'])):
            out.append(
                f"Setup {s['setup']} is {s['n']} trades ({s['share']:.0%} of all entries) for {s['total_R']:+.1f}R in total "
                f"({s['total_R_ex_top3']:+.1f}R without its best 3). It occupies slots and risk budget for roughly zero edge; "
                f"this is a hypothesis to test in the builder at portfolio level, not a rule change.")
    sp = pm['stop_profile']
    out.append(
        f"Disaster-stop and gap exits cost {pm['stop_leak_R']:+.1f}R. Those entries were stretched: "
        f"{sp['f_ext50'][0]:.1f}% above MA50 vs {sp['f_ext50'][1]:.1f}% for normal exits, RVOL {sp['f_rvol'][0]:.2f} vs {sp['f_rvol'][1]:.2f}.")
    out.append(
        f"Returns are tail-driven: the top 5% of trades supply {pm['top5pct_share_of_gross']:.0%} of gross profit. "
        f"Any filter that skips entries has to be judged on R kept, not on hit rate.")
    if live:
        out.append(
            f"Live window: {live['n_live']} closed trades, {live['live_total_R']:+.2f}R, PF {live['live_pf']}. Drawing {live['n_live']} "
            f"backtest trades that could have closed inside a {live['window_days']}-day window gives a result this bad "
            f"{live['p_total_R_this_bad']:.0%} of the time (PF this low: {live['p_pf_this_bad']:.0%}): normal for a young book. "
            f"Comparing against all backtest trades, long winners included, would say {live['p_total_R_this_bad_uncensored']:.0%} "
            f"and wrongly suggest something broke. Only {live['live_A']} Setup A entries vs {live['expected_A']} expected "
            f"(P = {live['p_this_few_A']:.2f}); the live book has mostly been fed the setups that earn ~0R in the backtest too.")
        sig = [d for d in live['drift'] if d['p'] < live['drift_bonferroni']]
        out.append('Entry-feature drift vs backtest: ' + (
            ', '.join(f"{d['label']} (median {d['backtest_median']} -> {d['live_median']}, p={d['p']})" for d in sig)
            if sig else f"none significant after Bonferroni (p < {live['drift_bonferroni']}). The machine is buying the same kind of stocks."))
    m = ml['models']
    out.append(
        f"ML entry filter (keep top {ml['keep']:.0%} by model score), purged annual walk-forward: take-all {ml['take_all_R']:+.1f}R vs "
        f"ridge {m['ridge']['filtered_R']:+.1f}R (IC {m['ridge']['mean_ic']:+.3f}) and logistic {m['logit']['filtered_R']:+.1f}R "
        f"(IC {m['logit']['mean_ic']:+.3f}). Status: {ml['status']}.")
    out.append(
        f"Deflated Sharpe: the 5-year curve's Sharpe {dsr['sharpe_annual']} gives DSR {dsr['dsr']:.2f} at the registry's "
        f"{dsr['n_trials']} trials" + (f"; it drops below 0.95 at about {dsr['first_n_below_95']} independent trials." if dsr['first_n_below_95'] else '.')
        + " This is before the survivorship look-ahead the simulation page already measures, which lowers it further.")
    return out


# ---------------------------------------------------------------- render

CSS = """
:root{--bg:#0e1117;--panel:#161b24;--line:#232a36;--txt:#dbe2ee;--dim:#8b96a8;
--green:#3ecf8e;--red:#f0647a;--amber:#e8b348;--blue:#5aa2f0}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,'Segoe UI',sans-serif;padding:24px 16px;max-width:1100px;margin:0 auto}
h1{font-size:20px;margin-bottom:2px} h2{font-size:13px;margin:26px 0 10px;color:var(--dim);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
a{color:var(--blue)} .sub{color:var(--dim);font-size:13px;margin-bottom:6px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:8px;overflow-x:auto}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px}
.tile .k{color:var(--dim);font-size:12px}.tile .v{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}.tile .n{color:var(--dim);font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--dim);font-weight:600} th:first-child,td:first-child{text-align:left}
td.l{text-align:left;white-space:normal}
.pos{color:var(--green)}.neg{color:var(--red)}.amb{color:var(--amber)}.dim{color:var(--dim)}
ol.find{padding-left:20px} ol.find li{margin:6px 0}
.loop{display:flex;flex-wrap:wrap;gap:6px;font-size:12px;margin:10px 0}
.loop span{border:1px solid var(--line);border-radius:10px;padding:2px 10px;color:var(--dim)}
.loop span.on{border-color:var(--blue);color:var(--blue)}
"""


def num(v, fmt='{:+.1f}'):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '<span class="dim">–</span>'
    cls = 'pos' if v > 0 else 'neg' if v < 0 else ''
    return f'<span class="{cls}">{fmt.format(v)}</span>'


def table(head, rows):
    h = ''.join(f'<th>{c}</th>' for c in head)
    cell = lambda c: f'<td class="{c[0]}">{c[1]}</td>' if isinstance(c, tuple) else f'<td>{c}</td>'
    b = ''.join('<tr>' + ''.join(cell(c) for c in r) + '</tr>' for r in rows)
    return f'<table><tr>{h}</tr>{b}</table>'


def render(rep):
    e = html.escape
    pm, dsr, live, ml, reg = rep['post_mortem'], rep['deflated_sharpe'], rep['live'], rep['meta_label'], rep['registry']
    years = sorted(pm['by_setup'][0]['by_year'])
    dsr_cls = 'pos' if dsr['dsr'] >= 0.95 else 'amb' if dsr['dsr'] >= 0.8 else 'neg'
    tiles = [
        ('Deflated Sharpe', f'<span class="{dsr_cls}">{dsr["dsr"]:.2f}</span>', f'{dsr["n_trials"]} registered trials · ≥0.95 = significant'),
        ('Live vs backtest', f'{live["p_total_R_this_bad"]:.0%}' if live else '–', 'chance a random backtest sample is this bad'),
        ('ML entry filter', '<span class="neg">rejected</span>' if 'rejected' in ml['status'] else '<span class="amb">candidate</span>',
         f'{ml["take_all_R"]:+.0f}R take-all vs {max(m["filtered_R"] for m in ml["models"].values()):+.0f}R best filter'),
        (f'Setup {pm["carrier"]} share of R', f'{next(s["total_R"] for s in pm["by_setup"] if s["setup"] == pm["carrier"]) / pm["total_R"]:.0%}',
         f'of {pm["total_R"]:+.0f}R from {next(s["share"] for s in pm["by_setup"] if s["setup"] == pm["carrier"]):.0%} of trades'),
    ]
    tiles_html = ''.join(f'<div class="panel tile"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{n}</div></div>' for k, v, n in tiles)

    setup_tbl = table(['Setup', 'Trades', 'Win', 'Mean R', 'Median R', 'Total R', 'ex-top-3', 'Yrs +'] + years,
                      [[s['setup'], s['n'], f"{s['win']:.0%}", num(s['mean_R'], '{:+.2f}'), num(s['median_R'], '{:+.2f}'),
                        num(s['total_R']), num(s['total_R_ex_top3']), f"{s['years_positive']}/{s['years']}"]
                       + [num(s['by_year'][y]) for y in years] for s in pm['by_setup']])
    exit_tbl = table(['Exit', 'Trades', 'Mean R', 'Total R'],
                     [[e(x['exit']), x['n'], num(x['mean_R'], '{:+.2f}'), num(x['total_R'])] for x in pm['by_exit']])
    dsr_tbl = table(['Independent trials N', 'Hurdle Sharpe (annual)', 'DSR', ''],
                    [[r['n'], f"{r['hurdle_annual']:.2f}", ('pos' if r['dsr'] >= 0.95 else 'neg', f"{r['dsr']:.3f}"),
                      '← registry count' if r['registry'] else ''] for r in dsr['table']])
    live_html = '<p class="dim">No live trades yet.</p>'
    if live:
        drift_tbl = table(['Feature', 'Backtest median', 'Live median', 'KS', 'p'],
                          [[d['label'], d['backtest_median'], d['live_median'], d['ks'],
                            f'<span class="{"neg" if d["p"] < live["drift_bonferroni"] else ""}">{d["p"]}</span>'] for d in live['drift']])
        live_html = (f'<p class="sub">{live["n_live"]} live trades ({live["live_mix"]["A"]} A · {live["live_mix"]["B"]} B · {live["live_mix"]["C"]} C) '
                     f'compared with the {live["n_pool"]} pre-inception backtest trades that held ≤ {live["window_days"]} days (the longest a trade closed in the live window can have held). '
                     f'P(a random backtest sample is this bad) = {live["p_total_R_this_bad"]:.0%} on total R, {live["p_pf_this_bad"]:.0%} on profit factor. '
                     f'Below: entry-feature drift against all {live["n_pre"]} pre-inception trades; red p = drifted after Bonferroni (p &lt; {live["drift_bonferroni"]}).</p>' + drift_tbl)
    fold_tbl = table(['Test year', 'Train trades', 'Test trades', 'Take-all R', 'Ridge R', 'Ridge IC', 'Logit R', 'Logit IC'],
                     [[f['year'], f['n_train'], f['n_test'], num(f['take_all_R']), num(f['ridge_R']), num(f['ridge_ic'], '{:+.3f}'),
                       num(f['logit_R']), num(f['logit_ic'], '{:+.3f}')] for f in ml['folds']])
    reg_tbl = table(['Experiment', 'Status', 'Trials', 'Evidence'],
                    [[e(x['id']), f'<span class="{ {"rejected": "neg", "adopted": "pos"}.get(x["status"], "amb") }">{e(x["status"])}</span>',
                      x.get('variants', 0), ('l', e(x['evidence']))] for x in reg['experiments']])
    p = PROMOTION
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ACRS ML Review</title><style>{CSS}</style></head><body>
<p class="sub"><a href="index.html">← Flow report</a> · <a href="ACRS_Simulation.html">Simulation</a></p>
<h1>ACRS · ML Review</h1>
<p class="sub">Data as of {e(rep['asof'])} · generated {e(rep['generated'])} · reads the simulation ledger only · changes nothing in the live model</p>
<div class="loop"><span>1 Research</span><span>2 Code</span><span class="on">3 Backtest → deflated Sharpe</span><span>4 Live</span><span class="on">5 Post-mortem</span><span class="on">6 Fine-tune → registry + record-only ML</span></div>
<div class="tiles">{tiles_html}</div>

<h2>Critic: what the ledger says</h2>
<div class="panel"><ol class="find">{''.join(f'<li>{e(f)}</li>' for f in rep['findings'])}</ol></div>

<h2>Step 5 · Where the R comes from, and where it bleeds</h2>
<div class="panel"><p class="sub">{pm['n']} backtest round-trips, measured in R (1R = the entry's disaster-stop distance). Per-trade R ignores slot competition, so a setup with ~0R can still matter by keeping a slot warm. Any change here needs a portfolio-level A/B in the builder.</p>{setup_tbl}</div>
<div class="panel">{exit_tbl}</div>

<h2>Step 3 · Deflated Sharpe: how many tries can this backtest afford?</h2>
<div class="panel"><p class="sub">Sharpe {dsr['sharpe_annual']} on {dsr['days']} daily returns (skew {dsr['skew']}, kurtosis {dsr['kurtosis']}); PSR vs zero = {dsr['psr_vs_zero']}.
The hurdle is the Sharpe that the best of N zero-edge variants would show by luck. Every variant you or an AI agent tries raises N. That is why an agent that rewrites the rules and re-backtests in a loop will always find a 'better' strategy: it is spending this budget.
Survivorship look-ahead is not in this number; the simulation page measures that separately.
{f"With the hurdle set from the parameter grid's own Sharpe spread instead, DSR = {dsr['dsr_grid_dispersion']}. That figure is optimistic, because neighbouring parameters are near-copies of each other." if 'dsr_grid_dispersion' in dsr else ''}</p>{dsr_tbl}</div>

<h2>Step 4 → 5 · Is the live book behaving like the backtest?</h2>
<div class="panel">{live_html}</div>

<h2>Step 6 · Meta-label filter (record-only)</h2>
<div class="panel"><p class="sub">Two models score each entry from its f_* features + setup. Each test year trains only on trades already closed before it starts (purged), then keeps the top {ml['keep']:.0%} of entries.
Pre-registered promotion rule: beat take-all R in ≥{p['min_fold_win_share']:.0%} of years <b>and</b> mean out-of-sample rank IC ≥ {p['min_mean_ic']}, then {p['forward_weeks']} weeks of forward paper record before it can touch a live order.
<b>Status: {e(ml['status'])}.</b></p>{fold_tbl}</div>

<h2>Experiment registry ({rep['n_trials']} trials counted)</h2>
<div class="panel"><p class="sub">ml/experiments.json is the database of failed rules. Add a row before running a test. Rejected rows stay because they still used up part of the budget above.{' Untracked earlier variants: unknown. Set <code>untracked_prior_trials</code>.' if reg.get('untracked_prior_trials') is None else ''}</p>{reg_tbl}</div>
<p class="sub" style="margin-top:16px">Research tooling, not investment advice.</p>
</body></html>
"""


# ---------------------------------------------------------------- main

def build(sim_path, registry_path, seed=7):
    D = load_embedded(sim_path)
    rng = np.random.default_rng(seed)
    reg, n_trials = load_registry(registry_path)
    trades = sorted(D['trades'], key=lambda t: t['into'])
    grid = [v['sharpe'] for k in ('rs_threshold', 'exit_ma', 'atr_mult')
            for v in D.get('robust', {}).get('sensitivity', {}).get(k, [])]
    rep = {
        'asof': D['asof'], 'generated': date.today().isoformat(), 'n_trials': n_trials,
        'post_mortem': post_mortem(trades, rng),
        'deflated_sharpe': deflated_sharpe(D['curve'], n_trials, grid),
        'live': live_check(trades, D.get('since', {}).get('trades', []), D['inception'],
                           len(D.get('since', {}).get('curve', [])), rng),
        'meta_label': meta_label(trades),
        'registry': reg,
    }
    rep['findings'] = findings(rep['post_mortem'], rep['deflated_sharpe'], rep['live'], rep['meta_label'], reg)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sim', default=ROOT / 'ACRS_Simulation.html')
    ap.add_argument('--registry', default=ROOT / 'ml' / 'experiments.json')
    ap.add_argument('--out-html', default=ROOT / 'ACRS_ML_Review.html')
    ap.add_argument('--out-json', default=ROOT / 'ml' / 'ml_review.json')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    rep = build(a.sim, a.registry, a.seed)
    Path(a.out_json).write_text(json.dumps({k: v for k, v in rep.items() if k != 'registry'}, indent=1), encoding='utf-8')
    Path(a.out_html).write_text(render(rep), encoding='utf-8')
    for f in rep['findings']:
        print('-', f)


if __name__ == '__main__':
    main()
