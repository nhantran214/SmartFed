#!/usr/bin/env python3
"""R3-Major-4 CIFAR-10 experiment runner (Figs. 4–5 + Table VI).

Three operating modes (reviewer chain):
  W/o MA → Rule-based Eq.(26) → LLM-driven (DCSA/AFWA)

Also supports Fig.4 / Fig.5 factorizations:
  --figure 4  : vary DCSA (AFWA fixed to LLM)
  --figure 5  : vary AFWA (DCSA fixed to LLM)
  --figure both (default): full AGENT_MODE ∈ {wo_ma, rule_based, llm}

Example (3 seeds, Fig.4+5):
  python scripts/run_r3_major4_cifar.py --seeds 0,1,2 --figure both

Full LLM only, 8 clients + public 6000/1040 (EfficientNetB0):
  python scripts/run_r3_major4_cifar.py --seeds 0,1,2 --figure llm \\
      --n-clients 8 --public-train 6000 --public-test 1040

Dry-run (print commands only):
  python scripts/run_r3_major4_cifar.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_seeds(text: str):
    return [int(x.strip()) for x in text.split(',') if x.strip() != '']


def _default_budget(figure: str, n_clients: int) -> int:
    # Fig.4 needs a subset; agent-chain / LLM / Fig.5 train all clients.
    if figure in ('4', 'fig4', 'dcsa'):
        return min(4, n_clients)
    return n_clients


_BACKBONE_ALIASES = {
    'efficientnetb0': 'EfficientNetB0',
    'efficientnet_b0': 'EfficientNetB0',
    'efficientnet-b0': 'EfficientNetB0',
    'b0': 'EfficientNetB0',
    'efficientnetb4': 'EfficientNetB4',
    'efficientnet_b4': 'EfficientNetB4',
    'efficientnet-b4': 'EfficientNetB4',
    'b4': 'EfficientNetB4',
}


def _normalize_backbone(name: str) -> str:
    key = name.strip()
    return _BACKBONE_ALIASES.get(key.lower().replace(' ', ''), key)


def _storage_roots(n_clients: int, public_train: int, public_test: int, backbone: str):
    """Isolate non-default splits / backbones so they do not overwrite 16-client B0 ckpts."""
    if (
        n_clients == 16
        and public_train == 5000
        and public_test == 1000
        and backbone == 'EfficientNetB0'
    ):
        return './Model_Storage_cifar10', './Model_Storage_main_cifar10'
    tag = f'n{n_clients}_pub{public_train}_{public_test}_{backbone}'
    return (
        os.path.join('.', f'Model_Storage_cifar10_{tag}'),
        os.path.join('.', f'Model_Storage_main_cifar10_{tag}'),
    )


def _base_env(
    seed: int,
    rounds: int,
    selection_budget: int,
    model: str,
    n_clients: int,
    public_train: int,
    public_test: int,
    backbone: str,
) -> dict:
    ckpt_root, main_root = _storage_roots(
        n_clients, public_train, public_test, backbone
    )
    env = os.environ.copy()
    env.update({
        'SMARTFED_CONFIG': 'cifar',
        'SMARTFED_DATASET_TYPE': 'cifar10',
        'SMARTFED_N_CLIENTS': str(n_clients),
        'SMARTFED_HOMO_BACKBONE': backbone,
        'SMARTFED_CIFAR_PUBLIC_TRAIN': str(public_train),
        'SMARTFED_CIFAR_PUBLIC_TEST': str(public_test),
        'SMARTFED_PROTOCOL': 'Homo_FL',
        'SMARTFED_SEED': str(seed),
        'SMARTFED_COMMUNICATION_EPOCH': str(rounds),
        'SMARTFED_SELECTION_BUDGET': str(selection_budget),
        'SMARTFED_OLLAMA_MODEL': model,
        'SMARTFED_LOG_LLM_COST': '1',
        'SMARTFED_PRIVATE_DATASET_DIR': os.path.join(REPO, 'data', 'cifar'),
        'SMARTFED_PUBLIC_DATASET_DIR': os.path.join(REPO, 'data', 'cifar'),
        'SMARTFED_CIFAR10_PICKLE_DIR': os.path.join(REPO, 'data', 'cifar-10-python'),
        'SMARTFED_MODEL_STORAGE_ROOT': ckpt_root,
        'SMARTFED_MODEL_STORAGE_MAIN_ROOT': main_root,
    })
    return env


def _jobs_for_figure(figure: str):
    """Return list of (tag, env_overrides)."""
    if figure in ('llm', 'full_llm'):
        return [
            ('llm', {'SMARTFED_AGENT_MODE': 'llm'}),
        ]
    if figure in ('agent', 'both', 'all'):
        return [
            ('wo_ma', {'SMARTFED_AGENT_MODE': 'wo_ma'}),
            ('rule_based', {'SMARTFED_AGENT_MODE': 'rule_based'}),
            ('llm', {'SMARTFED_AGENT_MODE': 'llm'}),
        ]
    if figure in ('4', 'fig4', 'dcsa'):
        # Fig.4: AFWA fixed LLM; vary DCSA
        return [
            ('fig4_dcsa_off', {
                'SMARTFED_AGENT_MODE': 'custom',
                'SMARTFED_DCSA_MODE': 'off',
                'SMARTFED_AFWA_MODE': 'llm',
            }),
            ('fig4_dcsa_rule', {
                'SMARTFED_AGENT_MODE': 'custom',
                'SMARTFED_DCSA_MODE': 'rule_based',
                'SMARTFED_AFWA_MODE': 'llm',
            }),
            ('fig4_dcsa_llm', {
                'SMARTFED_AGENT_MODE': 'custom',
                'SMARTFED_DCSA_MODE': 'llm',
                'SMARTFED_AFWA_MODE': 'llm',
            }),
        ]
    if figure in ('5', 'fig5', 'afwa'):
        # Fig.5: DCSA fixed LLM; vary AFWA
        return [
            ('fig5_afwa_off', {
                'SMARTFED_AGENT_MODE': 'custom',
                'SMARTFED_DCSA_MODE': 'llm',
                'SMARTFED_AFWA_MODE': 'off',
            }),
            ('fig5_afwa_rule', {
                'SMARTFED_AGENT_MODE': 'custom',
                'SMARTFED_DCSA_MODE': 'llm',
                'SMARTFED_AFWA_MODE': 'rule_based',
            }),
            ('fig5_afwa_llm', {
                'SMARTFED_AGENT_MODE': 'custom',
                'SMARTFED_DCSA_MODE': 'llm',
                'SMARTFED_AFWA_MODE': 'llm',
            }),
        ]
    raise ValueError(f'Unknown --figure {figure!r}')


def _run(cmd, env, dry_run: bool, log_path: str) -> int:
    print('=' * 72)
    print('CMD:', ' '.join(cmd))
    print('LOG:', log_path)
    print('ENV:', {k: env[k] for k in sorted(env) if k.startswith('SMARTFED_')})
    if dry_run:
        return 0
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='0,1,2', help='Comma-separated seeds')
    parser.add_argument('--rounds', type=int, default=40)
    parser.add_argument(
        '--selection-budget',
        type=int,
        default=None,
        help='Override DCSA budget. Default: all clients (or 4 for Fig.4)',
    )
    parser.add_argument('--n-clients', type=int, default=16)
    parser.add_argument(
        '--public-train',
        type=int,
        default=5000,
        help='Public split training images (rest divided among clients)',
    )
    parser.add_argument(
        '--public-test',
        type=int,
        default=1000,
        help='Public split test images (rest divided among clients)',
    )
    parser.add_argument('--model', default='llama3:8b')
    parser.add_argument(
        '--backbone',
        default='EfficientNetB0',
        help='Homo_FL backbone (EfficientNetB0 or EfficientNetB4)',
    )
    parser.add_argument(
        '--figure',
        default='both',
        choices=[
            'both', 'agent', 'llm', 'full_llm',
            '4', '5', 'fig4', 'fig5', 'dcsa', 'afwa', 'all',
        ],
        help='both/agent = W/o MA→Rule→LLM; llm = full LLM only; 4=DCSA; 5=AFWA',
    )
    parser.add_argument('--skip-pretrain', action='store_true')
    parser.add_argument('--pretrain-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--python',
        default=sys.executable,
        help='Python interpreter (default: current)',
    )
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args.seeds)
    jobs = _jobs_for_figure(args.figure)
    n_clients = int(args.n_clients)
    public_train = int(args.public_train)
    public_test = int(args.public_test)
    backbone = _normalize_backbone(args.backbone)
    selection_budget = (
        args.selection_budget
        if args.selection_budget is not None
        else _default_budget(args.figure, n_clients)
    )
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(REPO, 'Logs', 'r3_major4', stamp)
    os.makedirs(out_dir, exist_ok=True)

    # Ensure image data exists; convert from pickles if train/ empty.
    train_root = os.path.join(REPO, 'data', 'cifar', 'train')
    if not os.path.isdir(train_root) or not os.listdir(train_root):
        convert_cmd = [args.python, os.path.join('scripts', 'convert_cifar10.py')]
        rc = _run(
            convert_cmd,
            os.environ.copy(),
            args.dry_run,
            os.path.join(out_dir, 'convert_cifar10.log'),
        )
        if rc != 0:
            return rc

    # Pretrain once per seed (shared across agent modes for that seed).
    for seed in seeds:
        env = _base_env(
            seed, args.rounds, selection_budget, args.model,
            n_clients, public_train, public_test, backbone,
        )
        if not args.skip_pretrain:
            pre_log = os.path.join(out_dir, f'pretrain_seed{seed}.log')
            rc = _run([args.python, 'pretrain.py'], env, args.dry_run, pre_log)
            if rc != 0:
                print(f'pretrain failed seed={seed}', file=sys.stderr)
                return rc
    if args.pretrain_only:
        print('Pretrain-only done.')
        return 0

    failures = []
    for seed in seeds:
        for tag, overrides in jobs:
            env = _base_env(
                seed, args.rounds, selection_budget, args.model,
                n_clients, public_train, public_test, backbone,
            )
            env.update(overrides)
            log_path = os.path.join(out_dir, f'main_{tag}_seed{seed}.log')
            rc = _run([args.python, 'main.py'], env, args.dry_run, log_path)
            if rc != 0:
                failures.append((tag, seed, rc))

    summary_path = os.path.join(out_dir, 'RUN_SUMMARY.txt')
    with open(summary_path, 'w', encoding='utf-8') as fh:
        fh.write(f'figure={args.figure}\nseeds={seeds}\nrounds={args.rounds}\n')
        fh.write(
            f'n_clients={n_clients}\n'
            f'public_train={public_train}\n'
            f'public_test={public_test}\n'
            f'selection_budget={selection_budget}\n'
            f'backbone={backbone}\n'
        )
        fh.write(f'jobs={[j[0] for j in jobs]}\n')
        fh.write(f'failures={failures}\n')
        fh.write(
            'Next: python scripts/summarize_r3_major4.py '
            f'--log-dir {out_dir}\n'
        )
    print('Wrote', summary_path)
    if failures:
        print('FAILED:', failures, file=sys.stderr)
        return 1
    print('All jobs finished. Summarize with:')
    print(f'  python scripts/summarize_r3_major4.py --log-dir {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
