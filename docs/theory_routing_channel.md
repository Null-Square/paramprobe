# Stable parameter routing as a communication problem

This note is intentionally conditional. It applies when a semantic item/query is expected to recover a stable external parameter address despite perturbation or sampling variation. A fully co-adapted MoE with no notion of a stable target expert is not automatically covered by this model.

## 1. Information requirement for an N-page address

Let `A` be a desired external parameter address, uniformly distributed over `N` pages, and let `Z` be all router observations used by a decoder to produce `A_hat`.

For vanishing address error, the router must communicate essentially

`H(A) = log_2 N`

bits of information about `A`.

Fano's inequality gives

`H(A | Z) <= h_2(P_e) + P_e log_2(N - 1)`,

where `P_e = Pr[A_hat != A]`.

Since

`I(A; Z) = H(A) - H(A | Z)`,

small `P_e` requires `I(A; Z)` to approach `log_2 N`. This is a more general statement than the independent-factor calculation in `theory_address_reliability.md`.

## 2. Binary symmetric routing model

As a tractable model, suppose a logical page address is encoded into `n` binary routing decisions and each predicted code symbol passes through an independent binary symmetric channel with error probability `epsilon`.

The capacity of one such channel use is

`C_BSC = 1 - H_2(epsilon)`

bits, where `H_2` is binary entropy. Therefore

`I(A; Z) <= n [1 - H_2(epsilon)]`.

Combining this with the information requirement above yields the asymptotic necessary condition

`R = log_2(N) / n <= 1 - H_2(epsilon)`

for vanishing address error under this model.

The channel coding theorem supplies the complementary achievability statement: for rates strictly below channel capacity, suitable code families can make error probability vanish as code length grows.

This gives a useful design law:

> The number of reliably addressable external parameter pages is controlled not only by router width, but by router **information capacity**.

## 3. Uncoded product addresses operate at rate one

A raw binary factor address uses one predicted bit for every logical address bit, so

`R = 1`.

For any nonzero BSC error rate, channel capacity is strictly less than one. Thus a raw uncoded address cannot achieve asymptotically vanishing address error in this model.

This recovers the earlier finite calculation

`P_success = (1 - epsilon)^k`

for independent hard bit decisions.

## 4. Coding trades router redundancy for stable capacity

A code of rate `R < 1` emits more router symbols than logical page-address bits. External parameter I/O is unchanged: after decoding, the layer still reads only the selected `q` blocks.

For an external address space of `N` pages,

`k = log_2 N`,

`n = k / R = Theta(log N)`.

So coding preserves the desired asymptotic separation:

- exponentially many external pages in `n`;
- logarithmic resident routing output/decoding state;
- constant `qB` external parameter traffic for fixed `q` and block size `B`.

## 5. The G1c operating points are consistent with the channel picture

Hamming(15,11) has rate

`R = 11/15 ~= 0.7333`.

In the three single-seed noise regimes explored during development, the coded router's raw code-bit error rates were approximately:

- easy: `epsilon ~= 0.030`, BSC capacity `~0.806`;
- moderate: `epsilon ~= 0.047`, BSC capacity `~0.725`;
- hard: `epsilon ~= 0.081`, BSC capacity `~0.594`.

The Hamming rate is below the idealized BSC capacity in the easy regime, approximately at the boundary in the moderate regime, and well above it in the hard regime. Empirically, coded full-address accuracy is strongest in the easy/moderate regimes and degrades in the hard regime.

This is not a proof that the learned router is a BSC: its errors are correlated and non-identical. The value of the calculation is that it gives a falsifiable code-rate design principle for the next experiments.

## 6. Next test implied by the theory

G1d should compare multiple address-code rates under a matched routing-compute budget and several perturbation levels.

The prediction is not merely "more redundancy helps." The sharper prediction is:

1. estimate the router symbol-error process;
2. choose code rates on both sides of the modeled information-capacity boundary;
3. test whether stable-address accuracy and memory quality show a transition near that boundary;
4. measure error correlations to explain deviations from the independent-channel prediction.

A positive result would turn error-correcting parameter routing from a heuristic into an empirically calibrated design law.

## 7. Scope and prior-work caution

Error-Correcting Output Codes, learned hashing, and error-correcting nearest-neighbor schemes already use coding ideas for classification/retrieval robustness. StableMoE and later work also establish that stable expert assignment matters in sparse models.

The working research distinction here is the combination of:

- a hard external parameter-probe budget;
- page addresses that select learned parameters participating in the forward pass;
- coding rate as a resource controlling reliable access to exponentially large external parametric capacity.

That distinction still needs a formal publication-grade novelty audit before any first-work claim.
