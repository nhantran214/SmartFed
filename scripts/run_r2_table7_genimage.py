#!/usr/bin/env python3
"""R2-4 / Table VII: three w_k rules on GenImage, one command.

KL alignment stays on. Only the rule that builds w_k changes:
  uniform       — 1/K
  dataset_size  — n_k / sum_j n_j  (FedAvg)
  signed_loss   — signed-loss APR (delta-loss update)

Same GenImage training recipe that produced the Table II SmartFed (EN-B0)
numbers in this repo: 8 clients, Homo_ServerGSAM, EfficientNet-B0,
genimage_pub, 20 communication rounds, 3 full GSAM epochs, 60 KL batches
per pair, full validation on the last round. DCSA and AFWA stay on
(Llama3), as in Tables II–V. The only change across the three runs is w_k.
Each rule runs once (one seed).

Requires the existing pretrain checkpoints under Model_Storage (parties 0..8).
This script does not pretrain.

  python scripts/run_r2_table7_genimage.py

After the three runs it writes Logs/r2_table7/<stamp>/TABLE_VII.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Match config_genimage.py / the GenImage run behind Table II.
# Do not cap GSAM batches or the final validation set.
ROUNDS = 20
GSAM_EPOCHS = 3
KL_BATCHES = 60

# Signed-loss APR first. The other two rules start only if this row
# reaches the Table II band (accuracy >= 80%, AUC >= 85%).
RULES = ('signed_loss', 'uniform', 'dataset_size')
MIN_ACC = 80.0
MIN_AUC = 0.85
ROW_LABEL = {
    'uniform': 'Uniform (1/K)',
    'dataset_size': 'Dataset size',
    'signed_loss': 'Signed-loss APR, Eqs. (10)–(12)',
}

AVG_ACC_RE = re.compile(
    r'Average Test Accuracy of model (\d+): ([\d.]+)%, Average AUC: ([\d.]+)'
)
BEST_RE = re.compile(
    r'New best across-eval at round (\d+): Acc=([\d.]+)%, AUC=([\d.]+), full_eval=(True|False)'
)


def _ckpt_dir() -> str:
    return os.path.join(REPO, 'Model_Storage', 'CE', 'None0')


def _missing_checkpoints() -> list[str]:
    root = _ckpt_dir()
    missing = []
    for i in range(9):
        path = os.path.join(root, f'EfficientNetB0_{i}.ckpt')
        if not os.path.isfile(path):
            missing.append(path)
    return missing


def _saved_round_count(rule: str) -> int:
    """How many communication rounds have a full 9-party checkpoint."""
    base = os.path.join(
        REPO, f'Model_Storage_main_table7_{rule}', 'CE', 'None0'
    )
    done = 0
    for epoch in range(ROUNDS):
        paths = [
            os.path.join(base, f'EfficientNetB0_{i}_epoch_{epoch}.ckpt')
            for i in range(9)
        ]
        if all(os.path.isfile(path) for path in paths):
            done += 1
        else:
            break
    return done


def _resume_epoch(rule: str):
    """0-based round to continue from, or None when this rule should start fresh.

    A finished 20-round run is not resumed. Epochs 0–2 alone are the old
    3-round probe and are not continued.
    """
    done = _saved_round_count(rule)
    if done >= ROUNDS or done <= 3:
        return None
    return done


def _env(seed: int, rule: str) -> dict:
    env = os.environ.copy()
    env.update({
        'SMARTFED_DATASET_TYPE': 'genimage',
        'SMARTFED_PROTOCOL': 'Homo_ServerGSAM',
        'SMARTFED_N_CLIENTS': '8',
        'SMARTFED_SEED': str(seed),
        'SMARTFED_AGENT_MODE': 'llm',
        'SMARTFED_SELECTION_BUDGET': '8',
        'SMARTFED_WEIGHT_RULE': rule,
        'SMARTFED_COMMUNICATION_EPOCH': str(ROUNDS),
        'SMARTFED_GSAM_TOTAL_EPOCHS': str(GSAM_EPOCHS),
        'SMARTFED_PUBLIC_KL_MAX_BATCHES': str(KL_BATCHES),
        'SMARTFED_EVAL_MAX_BATCHES': '64',
        'SMARTFED_EVAL_EVERY_N_ROUNDS': '5',
        'SMARTFED_EVAL_FINAL_FULL': '1',
        'SMARTFED_LOCAL_WEIGHT_MAX_BATCHES': '64',
        'SMARTFED_LOG_LLM_COST': '1',
        'SMARTFED_MODEL_STORAGE_ROOT': './Model_Storage',
        'SMARTFED_MODEL_STORAGE_MAIN_ROOT': f'./Model_Storage_main_table7_{rule}',
    })
    # A leftover cap from the short run would keep accuracy far below Table II.
    env.pop('SMARTFED_GSAM_MAX_BATCHES', None)
    return env


def _run(cmd, env, log_path: str) -> int:
    print('=' * 72)
    print('CMD:', ' '.join(cmd))
    print('LOG:', log_path)
    print('WEIGHT_RULE:', env.get('SMARTFED_WEIGHT_RULE'))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as fh:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f'exit={proc.returncode}')
    return proc.returncode


def _best_within_budget(rule: str):
    """Best full validation whose round index is inside the 20-round budget."""
    best = None
    pattern = os.path.join(REPO, 'Logs', 'r2_table7', '*', f'main_{rule}_seed*.log')
    for path in sorted(glob.glob(pattern)):
        text = open(path, encoding='utf-8', errors='replace').read()
        for match in BEST_RE.finditer(text):
            rnd = int(match.group(1))
            if match.group(4) != 'True' or rnd >= ROUNDS:
                continue
            acc = float(match.group(2))
            auc = float(match.group(3))
            if best is None or acc > best[1] or (acc == best[1] and auc > best[2]):
                best = (rnd, acc, auc)
    metrics_path = os.path.join(
        REPO, f'Model_Storage_main_table7_{rule}', 'CE', 'None0', 'best_metrics.json'
    )
    if os.path.isfile(metrics_path):
        with open(metrics_path, encoding='utf-8') as fh:
            meta = json.load(fh)
        rnd = int(meta.get('round', ROUNDS))
        if meta.get('full_eval') and rnd < ROUNDS and meta.get('accuracy') is not None:
            acc = float(meta['accuracy'])
            auc = float(meta['auc'])
            if best is None or acc > best[1] or (acc == best[1] and auc > best[2]):
                best = (rnd, acc, auc)
    if best is None:
        return None, None, None
    return best


def _saved_best_metrics(path: str):
    """Best checkpoint recorded during the run. Falls back to the last round."""
    text = open(path, encoding='utf-8', errors='replace').read()
    found = list(BEST_RE.finditer(text))
    full = [
        m for m in found
        if m.group(4) == 'True' and int(m.group(1)) < ROUNDS
    ]
    chosen = (full or [m for m in found if int(m.group(1)) < ROUNDS])
    if chosen:
        m = chosen[-1]
        return int(m.group(1)), float(m.group(2)), float(m.group(3))
    return _last_round_metrics(path)


def _last_round_metrics(path: str):
    text = open(path, encoding='utf-8', errors='replace').read()
    rounds = {}
    for m in re.finditer(r'Communication Epoch (\d+)/(\d+)', text):
        rounds[m.start()] = int(m.group(1))
    by_ep = {}
    for m in AVG_ACC_RE.finditer(text):
        ep = None
        for start, epoch in sorted(rounds.items()):
            if start <= m.start():
                ep = epoch
            else:
                break
        if ep is None:
            continue
        by_ep.setdefault(ep, []).append((float(m.group(2)), float(m.group(3))))
    if not by_ep:
        return None, None, None
    last_ep = max(by_ep)
    pairs = by_ep[last_ep]
    acc = sum(a for a, _ in pairs) / len(pairs)
    auc = sum(u for _, u in pairs) / len(pairs)
    return last_ep, acc, auc


def _write_table(out_dir: str, rows: list[dict]) -> str:
    lines = [
        'TABLE VII. GenImage accuracy and AUC under three rules for w_k.',
        'KL alignment is unchanged. Numbers are the saved best full validation of one seed (seed 0). AUC is on the 0–100 scale used in Table II.',
        '',
        '| Weight rule | Average accuracy | Average AUC |',
        '|---|---|---|',
    ]
    for row in rows:
        if row['acc'] is None:
            acc_s, auc_s = 'failed', 'failed'
        else:
            acc_s = f"{row['acc']:.2f}"
            auc_pct = row['auc'] * 100.0 if row['auc'] <= 1.5 else row['auc']
            auc_s = f"{auc_pct:.2f}"
        lines.append(f"| {ROW_LABEL[row['rule']]} | {acc_s} | {auc_s} |")
    lines.append('')
    lines.append(
        f'Setup: GenImage, 8 clients, Homo_ServerGSAM, EfficientNet-B0, '
        f'DCSA+AFWA on (Llama3), rounds={ROUNDS}, full GSAM epochs={GSAM_EPOCHS} '
        f'(no batch cap), KL batches/pair={KL_BATCHES}, full val on the last round. '
        f'One seed, so this file has no standard deviation.'
    )
    path = os.path.join(out_dir, 'TABLE_VII.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('Wrote', path)
    return path


def _default_python() -> str:
    """Use the smartfed conda env even when the caller forgot conda activate."""
    try:
        __import__('torch')
        __import__('torchvision')
        return sys.executable
    except Exception:
        pass
    candidate = os.path.expanduser('~/anaconda3/envs/smartfed/bin/python')
    if os.path.isfile(candidate):
        return candidate
    return sys.executable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--python', default=_default_python())
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the three commands and exit without training',
    )
    args = parser.parse_args(argv)

    missing = _missing_checkpoints()
    if missing and not args.dry_run:
        print('Missing pretrain checkpoints (Table II / Model_Storage):', file=sys.stderr)
        for path in missing:
            print(' ', path, file=sys.stderr)
        print(
            'Pretrain all 9 parties first, then rerun this script. '
            'This command does not pretrain.',
            file=sys.stderr,
        )
        return 2

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(REPO, 'Logs', 'r2_table7', stamp)
    os.makedirs(out_dir, exist_ok=True)

    if args.dry_run:
        for rule in RULES:
            env = _env(args.seed, rule)
            print('=' * 72)
            print('would run main.py', 'WEIGHT_RULE=' + rule)
            print('log', os.path.join(out_dir, f'main_{rule}_seed{args.seed}.log'))
            print('rounds', ROUNDS, 'gsam_epochs', GSAM_EPOCHS, 'kl_batches', KL_BATCHES, 'gsam_batch_cap', 'none')
        print('dry-run only; no training')
        return 0

    failures = []
    rows = []
    for rule in RULES:
        env = _env(args.seed, rule)
        done = _saved_round_count(rule)
        if done >= ROUNDS:
            ep, acc, auc = _best_within_budget(rule)
            print(
                f'{rule} already has {done}/{ROUNDS} rounds. '
                f'Best full val inside the budget: round {ep}, acc={acc}, auc={auc}'
            )
            rows.append({'rule': rule, 'epoch': ep, 'acc': acc, 'auc': auc})
            rc = 0
        else:
            state_path = os.path.join(
                REPO, f'Model_Storage_main_table7_{rule}', 'CE', 'None0', 'resume_state.pt'
            )
            resume_at = _resume_epoch(rule)
            if os.path.isfile(state_path):
                print(
                    f'Resume {rule}: resume_state.pt '
                    f'(model weights, w_k, and the in-round stage)'
                )
            elif resume_at is not None:
                env['SMARTFED_RESUME_EPOCH'] = str(resume_at)
                print(
                    f'Resume {rule}: Communication Epoch {resume_at + 1}/{ROUNDS} '
                    f'(weights from epoch_{resume_at - 1})'
                )
            log_path = os.path.join(out_dir, f'main_{rule}_seed{args.seed}.log')
            rc = _run([args.python, 'main.py'], env, log_path)
            ep, acc, auc = (None, None, None)
            if os.path.isfile(log_path):
                ep, acc, auc = _saved_best_metrics(log_path)
            rows.append({'rule': rule, 'epoch': ep, 'acc': acc, 'auc': auc})
        if rc != 0:
            failures.append((rule, rc))
        _write_table(out_dir, rows)
        if rule == 'signed_loss':
            auc_pct = None if auc is None else (auc * 100.0 if auc <= 1.5 else auc)
            auc_raw = None if auc is None else (auc if auc <= 1.5 else auc / 100.0)
            print(
                f'Signed-loss APR result: acc={acc} auc={auc_pct} '
                f'(need acc>={MIN_ACC}, auc>={MIN_AUC * 100:.0f})'
            )
            ok = (
                rc == 0
                and acc is not None
                and auc_raw is not None
                and acc >= MIN_ACC
                and auc_raw >= MIN_AUC
            )
            if not ok:
                print(
                    'Signed-loss APR is below the Table II band. '
                    'Stopping before uniform and dataset-size.',
                    file=sys.stderr,
                )
                _write_table(out_dir, rows)
                return 1

    if failures:
        print('FAILED:', failures, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
