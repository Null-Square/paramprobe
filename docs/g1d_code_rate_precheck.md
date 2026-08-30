# G1d precheck: redundancy is not free

Before treating error-correcting page addresses as the central architecture, we ran a matched-budget precheck designed to falsify the naive claim that lower code rate automatically improves routing.

## Setup

- 256 semantic items (`k=8` logical address bits).
- The semantic input was generated only from the logical bits plus nuisance/noise; it did **not** expose parity bits directly.
- Systematic binary linear codes of lengths `n in {8,12,16,24}` were compared.
- Code rate therefore ranged from `1.0` down to `1/3`.
- Codebooks were searched for increasing minimum Hamming distance.
- Router hidden width was adjusted so every condition stayed near 9.5k controller parameters and 9.3k routing MACs/query.
- Decoding used nearest-codeword Hamming distance for this small precheck.

## Negative result

At moderate input noise (`0.18`, one seed), the uncoded router achieved about 60.2% exact-address accuracy. The redundant codes were worse:

| code length | rate | min distance | output bit error | exact address accuracy |
|---:|---:|---:|---:|---:|
| 8 | 1.000 | 1 | 0.063 | 0.602 |
| 12 | 0.667 | 3 | 0.147 | 0.453 |
| 16 | 0.500 | 4 | 0.175 | 0.420 |
| 24 | 0.333 | 7 | 0.222 | 0.483 |

The same qualitative pattern held across easier and harder noise levels in this preliminary run.

## Why this matters

The simple channel model in `theory_routing_channel.md` treats symbol error probability as an external property and asks what coding can do **given** that error rate.

A learned router is different: changing the code changes the prediction task itself. Parity-like symbols can be substantially harder to infer from the available representation. Under a fixed controller budget, the raw symbol-error rate may rise enough to erase the coding gain.

Therefore the statement

> more redundant address bits improve routing

is false without additional assumptions.

## Reconciliation with G1c

G1c deliberately gives both raw and coded routers the same input representation generated from a redundant Hamming codeword. In that setting the representation contains evidence for the redundant symbols, and Hamming decoding gives a large, reproducible gain under a matched compute budget.

G1d shows that this gain cannot simply be transplanted to an arbitrary frozen representation.

## Revised hypothesis

The defensible research hypothesis is now:

> If the backbone and router are co-designed or co-trained so that the hidden representation carries redundant address evidence, error-correcting parameter addresses may preserve stable access to a growing external parameter space without increasing external parameter probes.

The next serious experiment must therefore allow the representation to co-adapt with the routing code, rather than fixing semantic features and bolting parity outputs on afterward.

## Consequence for ParamProbe

Error-correcting addressing remains a promising sub-direction, but it is **not yet promoted to the central ParamProbe claim**. The central claim remains the hard parameter-probe resource model and the empirical question of quality scaling at fixed `M`, `C`, and `Q`.
