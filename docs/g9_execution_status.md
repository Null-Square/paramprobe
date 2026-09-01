# G9 execution status

Status: **PREDECLARED; SCIENTIFIC EXECUTION NOT YET STARTED**

G9 is fully predeclared and implemented:

- protocol: `docs/g9_operator_vs_kv_block_predeclared.md`;
- experiment: `experiments/g9_operator_vs_kv_block.py`;
- workflow: `.github/workflows/g9_operator_vs_kv_block.yml`.

The first GitHub Actions run, `33371680164`, concluded failure before recording any job step. The normalized job payload contains `steps=null`; therefore no dependency installation, asset verification, data loading, router training, payload training, or evaluation executed.

The immediately preceding G8 workflow showed the same zero-step behavior on two attempts. GitHub's public global status API later reported Actions operational, so this repository record does not attribute the failure to a GitHub-wide outage. The exact scheduling/provisioning cause is unknown from available logs.

Accordingly:

- G9 has **no scientific result** yet;
- the zero-step workflow failure must not be classified as an operator or KV failure;
- the predeclared protocol and interpretation remain frozen;
- no G9 hyperparameter may be changed before a successful scientific execution.

The exact frozen G6a asset artifact has also been downloaded independently for recovery work, but the local research environment currently lacks both the checksum-verified WikiText-2 raw train split and the pinned `tokenizers` runtime needed to execute the exact committed G9 protocol. No reconstructed/substitute corpus or tokenizer is authorized for G9.