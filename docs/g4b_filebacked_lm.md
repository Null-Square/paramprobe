# G4b — trained LM pages through file-backed parameter probes

## Status

**FUNCTIONAL STORAGE PATH PASSED; PUBLICATION HARDWARE STUDY STILL PENDING.**

G4b connects the trained Tiny Shakespeare ParamProbe operator to the repository's actual fixed-block storage backends. This closes a gap left by the earlier synthetic/random-byte direct-I/O precheck: the bytes read from storage now encode the learned `48 -> 10 -> 48` LM page operators themselves.

The experiment has two deliberately separate parts:

1. **functional equivalence and exact accounting** for a trained 64-page learned-router model;
2. a **systems-only mirrored-store sweep** that spreads probes across progressively larger files while keeping the learned function exactly unchanged.

The mirrored files do not contain additional unique learned capacity. They are not a capacity/quality experiment.

## Environment

Successful GitHub Actions run: `33328703030` at commit `f3c4386c1334be554afbb523c0c893873d1c6ea0`.

The runner reported:

- Ubuntu 24.04, Linux `6.17.0-1022-azure`;
- Microsoft hypervisor / Azure VM;
- 2 logical CPUs, Intel Xeon 6973P-C;
- block device: `MSFT NVMe Accelerator v1.0`, 75 GiB;
- workspace filesystem: `/dev/nvme0n1p1`, ext4;
- ordinary page-cache bypass for direct reads: Linux `O_DIRECT + preadv`.

Because this is a virtualized hosted runner, latency/throughput values below are diagnostics rather than publication hardware claims. Device/controller caches, storage virtualization, and noisy-neighbor effects are not controlled.

Dataset identity remains canonical Tiny Shakespeare:

- bytes: `1,115,394`;
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`;
- Git blob SHA-1: `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`.

## Trained-page resource envelope

The test trains the frozen G3 learned router and the paired seed-7, 64-page operator configuration, then serializes every page into the exact physical block format used by the stores.

- trained pages: 64;
- active address bits: 6 of the fixed 8-factor router;
- page operator: `48 -> 10 -> 48`;
- learned FP32 page payload: 4,072 bytes;
- physical block: 4,096 bytes;
- inference probes/token: `q=1`;
- logical external parameter bytes/token: exactly 4,096.

Reliability ordering reproduced the G3c/G3f order:

`[4, 7, 6, 2, 5, 0, 3, 1]`.

## Functional equivalence

A deterministic 256-token validation batch was executed four ways:

1. resident PyTorch page table;
2. serialized in-memory blocks decoded through the page codec;
3. `FileParameterStore` using explicit `pread`;
4. `DirectIOParameterStore` using aligned `O_DIRECT + preadv`.

| execution | validation CE | max abs residual error vs resident | probes | external bytes |
|---|---:|---:|---:|---:|
| resident | 2.52785110 | — | — | — |
| encoded blocks | 2.52785110 | `2.98e-8` | — | — |
| pread | 2.52785110 | `2.98e-8` | 256 | 1,048,576 |
| O_DIRECT | 2.52785110 | `2.98e-8` | 256 | 1,048,576 |

The exact expected traffic is

`256 tokens * 4096 bytes/token = 1,048,576 bytes`,

which both file-backed paths recorded exactly.

This is the first repository result where a **trained LM page** is actually fetched from the file-backed storage abstraction and evaluated with the claimed hard parameter-probe budget.

## Mirrored-store systems sweep

To test whether the process working set and explicit traffic depend on backing-file size, the trained 64-page table was mirrored across larger files. Physical page ids were deterministically spread across the full file. Every mirror contains exactly the same learned page bytes, so operator checksums are identical across conditions.

Each point executes 4,096 one-page probes = exactly 16 MiB of explicit parameter reads.

### `pread`

| store | first pass after `POSIX_FADV_DONTNEED` | second pass | process RSS delta |
|---:|---:|---:|---:|
| 0.25 MiB | 11.5 us/probe | 9.4 us/probe | 0.000 MiB |
| 4 MiB | 196.3 us/probe | 9.6 us/probe | 0.000 MiB |
| 64 MiB | 1284.3 us/probe | 9.9 us/probe | 0.000 MiB |
| 256 MiB | 2005.3 us/probe | 9.9 us/probe | 0.000 MiB |

The dramatic first-vs-second-pass difference is exactly why ordinary buffered/pread measurements cannot be treated as physical-storage evidence: after warming, even a 256 MiB file is served at roughly 10 us/probe in this process while the first pass is around 2 ms/probe.

`POSIX_FADV_DONTNEED` is advisory rather than a proof of perfectly cold media state, so these first-pass values remain diagnostics.

### `O_DIRECT + preadv`

| store | pass 1 | pass 2 | process RSS delta after opening/reads |
|---:|---:|---:|---:|
| 0.25 MiB | 1222 us/probe | 1455 us/probe | about 0.004 MiB then 0 |
| 4 MiB | 1902 us/probe | 1588 us/probe | about 0.004 MiB then 0 |
| 64 MiB | 1172 us/probe | 1116 us/probe | about 0.004 MiB then 0 |
| 256 MiB | 844 us/probe | 2295 us/probe | about 0.004 MiB then 0 |

The direct-I/O values are noisy and non-monotone, as expected on a virtualized hosted runner. They should not be used to argue a device-performance curve. The important correctness observations are:

- the path remains operational from 0.25 MiB through 256 MiB backing files;
- every operation still reads exactly one 4,096-byte block;
- the reusable aligned direct-I/O buffer adds only about 4 KiB of observed process RSS;
- process RSS does not scale with backing-file size;
- repeated direct passes do not show the stable ~10 us page-cache behavior of `pread`.

The checksum `212.201595181` is identical for every store size and backend/pass, confirming that physical mirroring/remapping did not change the learned page function.

## What this result establishes

G4b upgrades the storage claim from an abstract byte counter to an executable trained-model path:

> A trained ParamProbe LM page can be serialized into one 4 KiB external block, fetched with one explicit `pread` or aligned `O_DIRECT` probe per token, and evaluated with numerically equivalent output while explicit parameter traffic remains exactly 4 KiB/token.

It also supplies direct evidence that the **process** resident working set of the storage backend need not grow with backing-file size.

## What it does not establish

This run does not establish publication-grade storage performance because:

- the NVMe device is virtualized by Azure;
- only one hosted-runner instance is measured;
- device/controller caching is uncontrolled;
- queue depth is effectively one;
- the reported probe timing includes block decode + page-MLP execution, not full LM token latency;
- the mirrored 256 MiB store is redundant storage, not 256 MiB of unique learned capacity.

Therefore G4 remains incomplete as a hardware-performance gate.

## Required next physical study

The publication-grade version should run the same trained-page path on named bare-metal devices and report at minimum:

- exact device model/firmware/interface;
- repeated independent trials with dispersion;
- queue depth and concurrency;
- pure I/O latency separately from page-operator and full-token latency;
- cold/warm methodology;
- CPU utilization;
- process RSS and, where possible, OS cache state;
- store sizes large enough to exceed host cache comfortably;
- fixed `q`, `B`, active compute, and routing compute across the capacity sweep.

Artifact: `g4b-filebacked-lm-results`, id `9737021494`, SHA-256 `15a4a00111ec93b682fc9319293af13b18f4abc0f7e2aa5b8ed22994ef120aec`.
