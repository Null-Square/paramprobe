# G8 timing addendum — predeclared before implementation

This addendum fixes the one timing detail left implicit in the G8 predeclaration.

- PyTorch intra-op threads are fixed to 8 via `torch.set_num_threads(8)`.
- PyTorch inter-op threads are left at the runtime default because each timed routing call is a single synchronous CPU tensor program; the environment value is recorded.
- Timings remain secondary diagnostics and do not enter the G8 pass/fail classification.

No other G8 protocol detail changes.