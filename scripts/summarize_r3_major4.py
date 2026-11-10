#!/usr/bin/env python3
"""Aggregate R3-Major-4 CIFAR logs → Acc + AUC + LLM cost (Table VI draft).

Parses:
  - Average Test Accuracy of model …, Average AUC: …
  - LLM cost cumulative: {...}

Usage:
  python scripts/summarize_r3_major4.py --log-dir Logs/r3_major4/<stamp>
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict


AVG_ACC_RE = re.compile(
    r'Average Test Accuracy of model (\d+): ([\d.]+)%, Average AUC: ([\d.]+)'
)
CUM_COST_RE = re.compile(r'LLM cost cumulative:\s*(\{.*\})')
AGENT_RE = re.compile(
    r'Agent modes: AGENT_MODE=([^,]+), DCSA_MODE=([^,]+), AFWA_MODE=([^,]+),'
)


def _fmt(val, prec=2, empty='n/a'):
    if val is None:
        return empty
    return f'{val:.{prec}f}'


def _parse_log(path: str):
    text = open(path, encoding='utf-8', errors='replace').read()
    rounds = {}
    for m in re.finditer(r'Communication Epoch (\d+)/(\d+)', text):
        rounds[m.start()] = int(m.group(1))

    # epoch -> list of (acc%, auc) per model average line
    epoch_metrics = defaultdict(list)
    for m in AVG_ACC_RE.finditer(text):
        pos = m.start()
        ep = None
        for start, epoch in sorted(rounds.items()):
            if start <= pos:
                ep = epoch
            else:
                break
        if ep is not None:
            epoch_metrics[ep].append((float(m.group(2)), float(m.group(3))))

    best_ep, best_acc, best_auc = None, -1.0, None
    last_ep, last_acc, last_auc = None, None, None
    for ep, pairs in sorted(epoch_metrics.items()):
        mean_acc = sum(a for a, _ in pairs) / len(pairs)
        mean_auc = sum(u for _, u in pairs) / len(pairs)
        last_ep, last_acc, last_auc = ep, mean_acc, mean_auc
        if mean_acc > best_acc:
            best_ep, best_acc, best_auc = ep, mean_acc, mean_auc

    cum = None
    m = CUM_COST_RE.search(text)
    if m:
        try:
            cum = ast.literal_eval(m.group(1))
        except Exception:
            try:
                cum = json.loads(m.group(1).replace("'", '"'))
            except Exception:
                cum = None

    modes = AGENT_RE.search(text)
    agent = modes.group(1).strip() if modes else '?'
    dcsa = modes.group(2).strip() if modes else '?'
    afwa = modes.group(3).strip() if modes else '?'

    base = os.path.basename(path)
    tag = base
    seed = None
    mm = re.match(r'main_(.+)_seed(\d+)\.log$', base)
    if mm:
        tag, seed = mm.group(1), int(mm.group(2))

    return {
        'path': path,
        'tag': tag,
        'seed': seed,
        'agent_mode': agent,
        'dcsa_mode': dcsa,
        'afwa_mode': afwa,
        'best_epoch': best_ep,
        'best_across_acc': best_acc if best_acc >= 0 else None,
        'best_across_auc': best_auc,
        'last_epoch': last_ep,
        'last_across_acc': last_acc,
        'last_across_auc': last_auc,
        'llm_cost': cum,
    }


def _mean_std(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--csv-out', default=None)
    args = parser.parse_args(argv)

    logs = sorted(glob.glob(os.path.join(args.log_dir, 'main_*.log')))
    if not logs:
        print(f'No main_*.log under {args.log_dir}', file=sys.stderr)
        return 1

    rows = [_parse_log(p) for p in logs]
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r['tag']].append(r)

    print('=== Per-run ===')
    for r in rows:
        cost = r['llm_cost'] or {}
        print(
            f"{r['tag']:20s} seed={r['seed']}  "
            f"best_acc={_fmt(r['best_across_acc'])}% "
            f"best_auc={_fmt(r['best_across_auc'], 4)}@ep{r['best_epoch']}  "
            f"last_acc={_fmt(r['last_across_acc'])}% "
            f"last_auc={_fmt(r['last_across_auc'], 4)}  "
            f"calls={cost.get('calls', 0)}  "
            f"lat_ms={float(cost.get('avg_latency_ms', 0) or 0):.1f}  "
            f"fb={100.0 * float(cost.get('fallback_rate', 0) or 0):.2f}%"
        )

    print('\n=== mean ± std by tag (Acc + AUC + cost) ===')
    header = (
        f"{'setting':20s} "
        f"{'acc_mean':>9s} {'acc_std':>7s} "
        f"{'auc_mean':>9s} {'auc_std':>7s} "
        f"{'calls/rnd*':>10s} {'lat_ms':>8s} {'fb_%':>7s}"
    )
    print(header)
    table = []
    for tag, group in sorted(by_tag.items()):
        accs = [g['last_across_acc'] for g in group if g['last_across_acc'] is not None]
        aucs = [g['last_across_auc'] for g in group if g['last_across_auc'] is not None]
        best_accs = [g['best_across_acc'] for g in group if g['best_across_acc'] is not None]
        best_aucs = [g['best_across_auc'] for g in group if g['best_across_auc'] is not None]
        acc_m, acc_s = _mean_std(accs)
        auc_m, auc_s = _mean_std(aucs)
        best_acc_m, best_acc_s = _mean_std(best_accs)
        best_auc_m, best_auc_s = _mean_std(best_aucs)

        calls_per_round = []
        lats = []
        fbs = []
        for g in group:
            c = g['llm_cost'] or {}
            if g['last_epoch'] and c.get('calls'):
                calls_per_round.append(c['calls'] / max(1, g['last_epoch']))
            if c:
                lats.append(float(c.get('avg_latency_ms', 0) or 0))
                fbs.append(100.0 * float(c.get('fallback_rate', 0) or 0))
        cpr_m, _ = _mean_std(calls_per_round)
        lat_m, _ = _mean_std(lats)
        fb_m, _ = _mean_std(fbs)

        print(
            f'{tag:20s} '
            f'{_fmt(acc_m):>9s} {_fmt(acc_s):>7s} '
            f'{_fmt(auc_m, 4):>9s} {_fmt(auc_s, 4):>7s} '
            f'{_fmt(cpr_m) if cpr_m is not None else "0":>10s} '
            f'{_fmt(lat_m, 1) if lat_m is not None else "0":>8s} '
            f'{_fmt(fb_m) if fb_m is not None else "—":>7s}'
        )
        table.append({
            'setting': tag,
            'acc_mean': acc_m,
            'acc_std': acc_s,
            'auc_mean': auc_m,
            'auc_std': auc_s,
            'best_acc_mean': best_acc_m,
            'best_acc_std': best_acc_s,
            'best_auc_mean': best_auc_m,
            'best_auc_std': best_auc_s,
            'calls_per_round': cpr_m,
            'avg_latency_ms': lat_m,
            'fallback_rate_pct': fb_m,
            'n_seeds': len(group),
        })

    out_csv = args.csv_out or os.path.join(args.log_dir, 'r3_major4_summary.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()) if table else [])
        if table:
            writer.writeheader()
            writer.writerows(table)
    print('\nWrote', out_csv)
    print(
        '\nManuscript fillers:\n'
        '  Report Acc as acc_mean ± acc_std (%) and AUC as auc_mean ± auc_std '
        '(last-round mean across models; also best_* columns in CSV).\n'
        '  Table VI: calls/rnd*, lat_ms, fb_% '
        '(*calls cumulative / last epoch).'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
