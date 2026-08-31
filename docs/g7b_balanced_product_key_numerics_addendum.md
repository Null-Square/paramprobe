# G7b numerical addendum — predeclared before implementation/run

This addendum fixes numerical details implicit in `docs/g7b_balanced_product_key_predeclared.md`. It is committed before G7b experiment code or workflow execution.

- subkey probability softmax temperature is exactly `1.0`;
- all entropies use natural logarithms;
- probability-log stabilization is exactly `1e-12`;
- `H(p) = -sum(p * log(p + 1e-12))`;
- the 256-way product distribution is formed explicitly as the outer product of the two 16-way subkey softmax distributions and flattened in row-major Cartesian order matching `page_id = idx1*16 + idx2`;
- feature scale for router pretraining is the per-feature population standard deviation of the 81,920-state training-only hidden bank, clamped below at `1e-4`;
- each router pretraining step samples 1,024 bank rows with replacement using `torch.randint`;
- the two noisy views use independent `torch.randn_like` draws from the same fixed RNG stream seeded `71000+s` after router initialization;
- for validation route stability, concatenate the frozen evaluation hidden bank, compute its population per-feature std clamped at `1e-4`, use clean top-1 route versus one perturbed view, and seed that perturbation with `72000+s`;
- the fixed factorized reference stability uses the same evaluation hidden states and same perturbation tensor for that seed;
- BatchNorm running statistics accumulated during the 600 router-only training steps are frozen and retained for inference; no post-training recalibration is performed.

No item in this addendum may change after G7b execution begins.
