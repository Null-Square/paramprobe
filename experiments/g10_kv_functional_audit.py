"""Post-hoc diagnostics of unchanged G9b training; no new KV tuning.

Reproduces the frozen experiment and instruments completed models only. It
never selects a checkpoint, changes a weight, or feeds validation into training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import g9b_balanced_local_kv as frozen
from paramprobe.soft_memory_audit import StreamingSoftMemoryAudit


def fingerprint(*modules):
    digest = hashlib.sha256()
    for module in modules:
        for name, tensor in module.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collect(router, kv, bank):
    audit = StreamingSoftMemoryAudit(256, 21)
    with torch.no_grad():
        for h, _ in bank:
            ids, _ = router(h)
            p = frozen.local_probabilities(kv, ids, h)
            audit.update(ids.reshape(-1).numpy(), p.numpy())
    return audit


def constant_from_training(audit, kv):
    """Train-only mean probability; zero fallback for unobserved blocks."""
    mean = np.zeros((256, 21), dtype=np.float64)
    observed = audit.counts > 0
    mean[observed] = audit.mass[observed] / audit.counts[observed, None]
    values = kv.values.weight.detach().reshape(256, 21, 96).double().numpy()
    constants = .15 * np.tanh(np.einsum('bs,bsd->bd', mean, values))
    bound = 0.0
    for block in np.flatnonzero(observed):
        bound += audit.counts[block] * audit.constant_residual_mse_bound(int(block), values[block], .15)
    bound /= int(audit.counts.sum())
    return torch.tensor(constants, dtype=kv.values.weight.dtype), float(bound)


def compare_constant(model, router, kv, bank, constant):
    """Compare fixed trained-mean replacement on held-out validation only."""
    losses = {'original_kv': 0.0, 'train_mean_constant': 0.0}
    n, squared_error = 0, 0.0
    with torch.no_grad():
        for h, y in bank:
            ids, gate = router(h)
            residual, _ = kv(ids, h)
            replacement = constant[ids]
            squared_error += float((residual-replacement).double().square().sum())
            for name, r in [('original_kv', residual), ('train_mean_constant', replacement)]:
                logits = model.head(model.rest(h + r * gate.unsqueeze(-1)))
                loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), reduction='none')
                losses[name] += float(loss.double().sum())
            n += y.numel()
    return {'validation_ce': {k: v/n for k,v in losses.items()},
            'validation_ungated_residual_vector_mse': squared_error/n,
            'validation_targets': n,
            'constant_fitted_on': 'training probability means only',
            'unseen_global_block_fallback': 'zero residual',
            'constant_parameter_bytes_per_block': 96*4,
            'original_kv_parameter_bytes_per_block': 21*96*2*4,
            'constant_physical_io_claim': False}


class Instrumentation:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.seed = None
        self.model = None
        self.constant = None
        self.original_pretrain = frozen.pretrain_local_keys
        self.original_train = frozen.train_values_only
        self.original_usage = frozen.local_usage_stats

    def pretrain(self, kv, hidden, global_ids, feature_scale, seed):
        self.seed = seed
        self.constant = None
        return self.original_pretrain(kv, hidden, global_ids, feature_scale, seed)

    def train(self, model, router, kv, bank):
        self.model = model
        return self.original_train(model, router, kv, bank)

    def usage(self, router, kv, bank):
        original = self.original_usage(router, kv, bank)
        rng = torch.get_rng_state().clone()
        before = fingerprint(router, kv)
        split = 'train' if len(bank) == 1920 else 'validation'
        if len(bank) not in (1920, 40):
            raise RuntimeError('unexpected frozen diagnostic bank length')
        audit = collect(router, kv, bank)
        summary = audit.summary()
        observed = audit.counts > 0
        summary['hard_dead_fraction_within_observed_blocks'] = float((audit.hard_counts[observed] == 0).mean())
        summary['global_unobserved_fraction'] = float((~observed).mean())
        if split == 'train':
            self.constant, bound = constant_from_training(audit, kv)
            summary['empirical_train_constant_residual_mse_bound'] = bound
            torch.save({'kv': kv.state_dict(), 'router': router.state_dict(),
                        'train_mean_constant': self.constant, 'seed': self.seed},
                       self.out / f'seed_{self.seed}_kv.pt')
        else:
            if self.constant is None or self.model is None:
                raise RuntimeError('train-only constants must be fitted before validation')
            summary.update(compare_constant(self.model, router, kv, bank, self.constant))
        if not torch.equal(rng, torch.get_rng_state()) or before != fingerprint(router, kv):
            raise RuntimeError('diagnostic mutated the learned state or random stream')
        summary.update({'seed': self.seed, 'split': split, 'model_state_sha256': before,
                        'status': 'post_hoc_diagnostic_no_tuning',
                        'rng_and_model_state_unchanged': True})
        path = self.out / f'seed_{self.seed}_{split}.json'
        if path.exists():
            raise FileExistsError(f'refusing to overwrite {path}')
        path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print('functional_audit=' + json.dumps({k:v for k,v in summary.items() if k != 'per_block'}, sort_keys=True), flush=True)
        return original


def main(args):
    expected = '894350bd8953bcd3479220a29b5fcd92e5b4058aca7e121db551b777ec0f32b7'
    if hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('frozen G9b source identity changed')
    inst = Instrumentation(args.audit_output)
    frozen.pretrain_local_keys = inst.pretrain
    frozen.train_values_only = inst.train
    frozen.local_usage_stats = inst.usage
    try:
        frozen.run(args)
    finally:
        frozen.pretrain_local_keys = inst.original_pretrain
        frozen.train_values_only = inst.original_train
        frozen.local_usage_stats = inst.original_usage


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('train-path', 'validation-path', 'tokenizer-json', 'checkpoint', 'audit-output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--threads', type=int, default=2)
    main(p.parse_args())
