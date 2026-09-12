"""Frozen G10 experiment: matched payload exposure, one prescribed expansion.

Protocol: docs/g10_equal_budget_predeclared.md. Run one seed per invocation.
No early stopping, split-time search, best-checkpoint selection, or test tuning.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F
import tokenizers

import g6a_subword_scale_lm as base
import g6c_paired_exposure_crossover as frozen
import g7b_balanced_product_key as pk
from paramprobe.growth import clone_page_table, clone_adamw

SEEDS = (47, 48, 49, 50, 51, 52)
CHECKPOINTS = (120, 480, 960, 1920)
METHODS = ('fixed16', 'fixed256', 'clone256', 'balanced_pk256')
SPLIT_HASHES = {
    'train': '6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57',
    'validation': '4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7',
    'test': '173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08',
}


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path: Path, event: dict) -> None:
    line = json.dumps(event, sort_keys=True, allow_nan=False)
    print(line, flush=True)
    with path.open('a', encoding='utf8') as stream:
        stream.write(line + '\n')


def adam(pages):
    return torch.optim.AdamW(pages.parameters(), lr=.004, weight_decay=.0001)


def route_payload(pages, hidden, projection, thresholds, bits, router=None):
    if router is None:
        ids = base.route(hidden, projection, thresholds, bits)
        return pages(ids, hidden), ids
    ids, gate = router(hidden)
    return pages(ids, hidden) * gate.unsqueeze(-1), ids


def diagnostic_eval(model, pages, bank, projection, thresholds, bits, router=None):
    pages.eval()
    nll, count = 0.0, 0
    with torch.no_grad():
        for hidden, y in bank:
            residual, _ = route_payload(pages, hidden, projection, thresholds, bits, router)
            logits = model.head(model.rest(hidden + residual))
            losses = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), reduction='none')
            nll += float(losses.double().sum())
            count += y.numel()
    return nll / count


def full_context_batches(tokens, context=128, batch_size=8):
    """Score every target index 1..len(tokens)-1 exactly once, including tail."""
    if tokens.ndim != 1 or tokens.dtype != torch.long or len(tokens) < 2:
        raise ValueError('one-dimensional long tokens with at least two entries required')
    full, remainder = divmod(len(tokens) - 1, context)
    for start_chunk in range(0, full, batch_size):
        n = min(batch_size, full - start_chunk)
        starts = torch.arange(start_chunk, start_chunk + n) * context
        indices = starts[:, None] + torch.arange(context + 1)[None, :]
        seq = tokens[indices]
        yield starts, seq[:, :-1], seq[:, 1:]
    if remainder:
        start = full * context
        seq = tokens[start:]
        yield torch.tensor([start]), seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)


def main(args):
    if args.seed not in SEEDS:
        raise ValueError(f'G10 seed must be in {SEEDS}')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    out = Path(args.output) / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    event_path = out / 'events.jsonl'
    if event_path.exists():
        raise FileExistsError(f'refusing to overwrite an existing run: {event_path}')
    tokenizer, model = frozen.load_assets(args)
    paths = {'train': args.train_path, 'validation': args.validation_path, 'test': args.test_path}
    for split, path in paths.items():
        actual = sha(path)
        if actual != SPLIT_HASHES[split]:
            raise RuntimeError(f'{split} data hash mismatch: {actual}')
    provenance = {
        'event': 'provenance', 'seed': args.seed,
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'tokenizers': tokenizers.__version__,
        'threads': torch.get_num_threads(), 'platform': platform.platform(),
        'source_sha256': {str(p): sha(p) for p in
                          [Path(__file__), Path('src/paramprobe/growth.py'),
                           Path('experiments/g6a_subword_scale_lm.py'),
                           Path('experiments/g6c_paired_exposure_crossover.py'),
                           Path('experiments/g7b_balanced_product_key.py')]},
        'tokenizer_sha256': sha(args.tokenizer_json),
        'backbone_sha256': sha(args.checkpoint), 'dataset_sha256': SPLIT_HASHES,
        'protocol_commit': 'b9d841a2f4a9b67a5fd696cd0068f008402e27b0',
        'resource_scope': 'resident training; logical one-block inference; no physical I/O timing',
    }
    emit(event_path, provenance)
    started = time.perf_counter()
    train = base.encode_split(tokenizer, args.train_path)
    valid = base.encode_split(tokenizer, args.validation_path)
    if (len(train), len(valid)) != (4254523, 445470):
        raise RuntimeError('frozen tokenization changed')
    projection, thresholds = base.build_balanced_hash(model, train, 8, 40, 8)
    validation_bank = base.make_hidden_bank(model, valid, 40, 8, 1234)
    bank = base.make_hidden_bank(model, train, 1920, 8, 50000 + args.seed)
    emit(event_path, {'event': 'bank_ready', 'seed': args.seed,
                      'hidden_batches': len(bank), 'targets_per_method': 1920*8*128,
                      'setup_seconds': time.perf_counter()-started})
    completed = {}
    diagnostics = {}
    train_seconds = {}
    clone_pages, clone_optimizer = None, None
    warm_seconds = 0.0
    clone120 = None
    pk_router = None
    for method in METHODS:
        router = None
        bits = 4 if method == 'fixed16' else 8
        first_step = 0
        router_seconds = 0.0
        if method == 'clone256':
            pages, optimizer = clone_pages, clone_optimizer
            if pages is None or optimizer is None:
                raise RuntimeError('clone must be created from the fixed16 prefix')
            first_step = 120
            diagnostics[method] = {'120': clone120}
        else:
            pages = frozen.init_pages(16 if method == 'fixed16' else 256, args.seed)
            optimizer = adam(pages)
            diagnostics[method] = {}
        if method == 'balanced_pk256':
            rstart = time.perf_counter()
            router, router_objective = pk.train_balanced_router(model, train, args.seed)
            router_seconds = time.perf_counter() - rstart
            pk_router = router
            emit(event_path, {'event': 'router_trained', 'seed': args.seed,
                              'method': method, 'extra_router_steps': 600,
                              'extra_router_seconds': router_seconds,
                              'objective_components': list(router_objective)})
        step_seconds = 0.0
        for index in range(first_step, 1920):
            hidden, y = bank[index]
            pages.train()
            t0 = time.perf_counter()
            residual, _ = route_payload(pages, hidden, projection, thresholds, bits, router)
            logits = model.head(model.rest(hidden + residual))
            loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite training loss {method} {index+1}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step_seconds += time.perf_counter()-t0
            step = index+1
            if step in CHECKPOINTS:
                val = diagnostic_eval(model, pages, validation_bank, projection, thresholds, bits, router)
                diagnostics[method][str(step)] = val
                emit(event_path, {'event': 'checkpoint', 'seed': args.seed, 'method': method,
                                  'step': step, 'sampled_validation_ce': val,
                                  'cumulative_step_seconds': step_seconds + (warm_seconds if method=='clone256' else 0)})
            if method == 'fixed16' and step == 120:
                clone_pages = clone_page_table(pages, 16)
                clone_optimizer = clone_adamw(optimizer, pages, clone_pages, 16)
                warm_seconds = step_seconds
                with torch.no_grad():
                    parent_ids = base.route(hidden, projection, thresholds, 4)
                    child_ids = base.route(hidden, projection, thresholds, 8)
                    if not torch.equal(child_ids // 16, parent_ids):
                        raise RuntimeError('prefix routing nesting violated')
                    before = pages(parent_ids, hidden)
                    after = clone_pages(child_ids, hidden)
                    max_error = float((before-after).abs().max())
                    if max_error != 0:
                        raise RuntimeError(f'function-preserving clone failed: {max_error}')
                clone120 = diagnostic_eval(model, clone_pages, validation_bank, projection, thresholds, 8)
                if clone120 != diagnostics[method]['120']:
                    raise RuntimeError('full downstream clone equality failed')
                emit(event_path, {'event': 'clone_preservation', 'seed': args.seed,
                                  'max_abs_residual_error': max_error,
                                  'downstream_validation_ce_error': clone120-diagnostics[method]['120']})
        pages.eval()
        completed[method] = (pages, bits, router)
        train_seconds[method] = step_seconds + (warm_seconds if method == 'clone256' else 0)
        state_path = out / f'{method}.pt'
        saved = {'pages': pages.state_dict(), 'projection': projection, 'thresholds': thresholds,
                 'used_bits': bits, 'seed': args.seed, 'payload_steps': 1920}
        if router is not None:
            saved['router'] = router.state_dict()
        torch.save(saved, state_path)
        emit(event_path, {'event': 'method_trained', 'seed': args.seed, 'method': method,
                          'state_sha256': sha(state_path), 'payload_training_seconds': train_seconds[method],
                          'extra_router_seconds': router_seconds})
        if method != 'fixed16':
            del optimizer
    del bank
    # Test evaluation begins only after all prescribed variants are fixed.
    test = base.encode_split(tokenizer, args.test_path)
    totals = {method: 0.0 for method in METHODS}
    totals['frozen_backbone'] = 0.0
    counts = {method: np.zeros(16 if method=='fixed16' else 256, dtype=np.int64) for method in METHODS}
    target_count = 0
    segment_path = out / 'test_segments.jsonl'
    with torch.no_grad(), segment_path.open('w', encoding='utf8') as stream:
        for starts, x, y in full_context_batches(test):
            hidden = model.first(x)
            segment = [{'first_target_index': int(s)+1, 'targets': y.shape[1]} for s in starts]
            for method in (*METHODS, 'frozen_backbone'):
                if method == 'frozen_backbone':
                    h = hidden
                else:
                    pages, bits, router = completed[method]
                    residual, ids = route_payload(pages, hidden, projection, thresholds, bits, router)
                    h = hidden + residual
                    counts[method] += np.bincount(ids.reshape(-1).numpy(), minlength=len(counts[method]))
                logits = model.head(model.rest(h))
                loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), reduction='none')
                loss = loss.reshape(y.shape).double().sum(dim=1)
                totals[method] += float(loss.sum())
                for row, value in zip(segment, loss):
                    row[method+'_nll_sum'] = float(value)
            for row in segment:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            target_count += y.numel()
    if target_count != len(test)-1:
        raise RuntimeError('test target coverage error')
    test_ce = {method: value/target_count for method,value in totals.items()}
    result = {
        'seed': args.seed, 'status': 'complete', 'test_targets': target_count,
        'test_context_policy': 'reset every 128 targets; each target scored once; partial tail included',
        'test_ce': test_ce, 'sampled_validation_ce': diagnostics,
        'payload_training_seconds': train_seconds,
        'test_page_counts': {k:v.tolist() for k,v in counts.items()},
        'payload_optimizer_steps_per_method': 1920,
        'logical_payload_token_assignments_per_method': 1966080,
        'pk_extra_router_optimizer_steps': 600,
        'page_payload_bytes': 15824, 'block_bytes': 16384,
        'page_matrix_macs_per_token': 3840,
        'fixed_router_matrix_macs_per_token': 768,
        'pk_router_matrix_macs_per_token': 672,
        'total_elapsed_seconds': time.perf_counter()-started,
        'physical_storage_benchmark': False,
    }
    temporary = out / 'result.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(out / 'result.json')
    emit(event_path, {'event': 'complete', 'seed': args.seed, 'test_targets': target_count,
                      'test_ce': test_ce, 'total_elapsed_seconds': result['total_elapsed_seconds']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', required=True)
    parser.add_argument('--validation-path', required=True)
    parser.add_argument('--test-path', required=True)
    parser.add_argument('--tokenizer-json', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--threads', type=int, default=2)
    main(parser.parse_args())
