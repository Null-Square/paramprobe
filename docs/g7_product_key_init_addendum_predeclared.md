# G7 product-key implementation addendum — predeclared before code/run

This addendum fixes implementation details left implicit in `docs/g7_hard_budget_challenge_predeclared.md`. It is committed before the G7 experiment implementation and before any G7 workflow execution.

For both PEER-style variants and each page/router seed `s`:

- router initialization RNG seed is `60000 + s`;
- query linear weight is initialized Normal(0, `1/sqrt(96)`), bias exactly zero;
- each product-key codebook is initialized Normal(0, `1/sqrt(d_key/2)`);
- query BatchNorm is `BatchNorm1d(d_key, affine=False, eps=1e-5, momentum=0.1)`;
- BatchNorm therefore adds no learned parameters, but its running mean/variance buffers are reported as constant resident state;
- product-key top-1 address is the Cartesian pair of the independent argmax subkeys;
- selected router score is `sigmoid(best_subkey_score_1 + best_subkey_score_2)` with no additional temperature or scaling;
- this scalar gates the selected page residual before it is added to the frozen backbone hidden state;
- no auxiliary balancing, entropy, expert-importance, or routing-consistency loss is used.

These details may not be changed after G7 execution begins.
