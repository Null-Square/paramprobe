# G4 direct-I/O precheck

A Linux `O_DIRECT + preadv` backend was added so ParamProbe can measure page reads without accidentally benchmarking the OS page cache.

## Local runtime precheck

On the current research container, a 512 MiB allocated backing file was sampled at random offsets. Each point transferred about 32 MiB through aligned direct reads.

| block bytes | reads | probes/s | MiB/s | microseconds/probe |
|---:|---:|---:|---:|---:|
| 4,096 | 8,192 | ~8,278 | ~32.3 | ~120.8 |
| 16,384 | 2,048 | ~15,337 | ~239.6 | ~65.2 |
| 65,536 | 512 | ~13,844 | ~865.2 | ~72.2 |
| 262,144 | 128 | ~9,463 | ~2,365.8 | ~105.7 |

These numbers are **not device claims**. The container's storage stack is not characterized, and `O_DIRECT` bypasses the ordinary page cache but does not eliminate device/controller caching. They are only evidence that the benchmark path is functional and that block granularity materially changes the byte-throughput/latency tradeoff.

## Architectural implication

A 4 KiB page minimizes bytes per probe but pays most of the fixed read latency for very little payload. Larger pages amortize that fixed cost dramatically.

That means `B` should not be treated as a mere filesystem implementation detail. It is an architectural hyperparameter controlling at least:

1. bytes transferred per selected parameter unit;
2. expressivity available inside one page-sized operator;
3. random-read latency amortization;
4. number of independently routable parameter units for a fixed total storage budget.

The eventual model experiments should therefore sweep `B` jointly with `q`, rather than optimizing model quality first and choosing a storage layout afterward.

## Required publication-grade benchmark

Before making hardware claims we need at least:

- named physical devices (NVMe, eMMC, SD, Raspberry Pi-class storage where possible);
- explicit queue depth and concurrency;
- random and sequential access patterns;
- cold/warm distinction;
- repeated trials with dispersion;
- CPU utilization and end-to-end operator time, not only raw read time;
- verification that every model condition obeys the same logical and physical parameter-read budget.
