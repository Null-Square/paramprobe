# Address reliability under exponentially large parameter spaces

This note records a failure mode discovered in G1b and the corresponding coded-address construction tested in G1c.

## 1. Raw factorized addressing has a reliability wall

Let an external parameter space contain

`N = m^r`

addresses represented by `r` independently interpreted routing factors over radix `m`. Suppose each factor is predicted correctly with probability `p < 1` and, as a first model, factor errors are independent.

The probability of selecting the exact address is

`P_success = p^r`.

Since `r = log_m N`,

`P_success = N^(log_m p)`.

Because `log_m p < 0` whenever `p < 1`, exact-address success tends to zero as the address space grows.

For binary factors (`m=2`), the observed G1b hard-regime bit accuracy was about `p=0.925`. The independence prediction gives

`0.925^12 ~= 0.393`,

which closely matches the measured 12-bit full-address accuracy of about `0.396`.

This means that logarithmic routing metadata by itself is not sufficient for scalable learned addressing. A raw product address can become increasingly unreliable even when per-factor accuracy remains apparently high.

The independence assumption is not required as a claim about all learned routers. It is a diagnostic model. Correlated errors can make the result better or worse; experiments must report full-address accuracy rather than infer it from marginal bit accuracy.

## 2. Error-correcting addresses preserve exponential capacity

Let `k = log_2 N` logical address bits be encoded into an `n`-bit codeword using a binary code family with rate

`R = k / n > 0`.

The router predicts the `n` code symbols, the decoder maps the noisy prediction back to a `k`-bit logical address, and only then is the external parameter page read.

The number of addressable pages remains

`N = 2^k = 2^(R n)`,

so capacity is still exponential in routing-code length. Router output size grows only as

`n = Theta(log N)`.

The external parameter probe budget is unchanged: coding changes the resident routing computation, not the number or size of parameter pages read.

## 3. Reliability proposition under independent symbol errors

Assume a code family can correct any error pattern of at most `tau n` symbol errors for a constant `tau > 0`. Assume router symbol errors are independent Bernoulli events with probability `epsilon < tau`.

Let `X ~ Binomial(n, epsilon)` be the number of symbol errors. Decode failure requires `X > tau n`. A Chernoff bound gives

`P_fail <= exp(-n D(tau || epsilon))`,

where `D(a || b)` is the binary KL divergence.

Using `n = k/R = log_2(N)/R`,

`P_fail <= N^[-D(tau || epsilon)/(R ln 2)]`.

Therefore the decode-failure probability tends to zero polynomially in `N`, while the uncoded exact-address success under constant per-factor error tends to zero.

This yields the asymptotic contrast we care about:

- raw factorized address: `P_success -> 0` for fixed `p < 1` under the independent-factor model;
- positive-rate error-correcting address: `P_success -> 1` when `epsilon` stays below the code's correctable fraction.

This proposition is conditional on an error model. A paper must state that assumption explicitly and then measure learned error correlation empirically.

## 4. Why Hamming(15,11) is only a finite experiment

G1c uses Hamming(15,11) because it has a tiny, auditable encoder/decoder and allows a matched-controller experiment. It corrects one code-bit error and demonstrates that coding the parameter address can materially improve external-memory retrieval.

Hamming codes are not the intended asymptotic construction for the theorem above. For asymptotic claims we need a positive-rate, positive-relative-distance family with an efficient decoder, such as an appropriate expander/LDPC-style construction or another explicitly justified family.

The next theory milestone is to select a concrete code family with:

1. positive asymptotic rate;
2. a non-vanishing correctable error fraction under a stated decoder/model;
3. decoding time compatible with the desired `O(log N)` or near-`O(log N)` routing overhead;
4. a practical differentiable training interface.

## 5. Resource accounting

For a coded ParamProbe router with logical address length `k`, code length `n = k/R`, fixed block size `B`, and `q` external probes:

- external capacity: `P_ext = Theta(2^k B)`;
- router output state/compute: at least `Theta(n) = Theta(log N)`;
- explicit external parameter traffic: at most `qB`;
- external-operator workspace: one block plus accumulator state when pages are streamed sequentially.

Thus error correction does not destroy the central capacity/I/O separation.

## 6. Relation to prior work

Error-Correcting Output Codes (ECOC) are established for converting multiclass prediction into redundant binary decisions and for improving classification robustness. That lineage must be cited and not presented as new.

The ParamProbe use is different: the decoded codeword is not a class label. It is a **physical/logical address selecting which learned external parameter page participates in the forward pass**. The research question is whether such coded addressing allows useful neural capacity to scale under a hard external parameter-I/O budget.

The current literature audit has not identified a direct prior work making error-correcting sparse parameter addresses the mechanism for storage-native neural scaling. That is a working novelty assessment, not yet a publication claim.

## 7. Empirical evidence so far

In the matched-compute G1c experiment at noise `0.08`, three seeds gave approximately:

- raw 11-bit full-address accuracy: `0.5911 +/- 0.0041`;
- Hamming-decoded full-address accuracy: `0.8376 +/- 0.0054`;
- raw full-capacity associative-memory MSE: `0.7471 +/- 0.0048`;
- coded full-capacity MSE: `0.3592 +/- 0.0180`.

The raw and coded routers use nearly identical budgets in the default configuration:

- raw: 9,359 parameters, 9,225 routing MACs/query;
- coded: 9,375 parameters, 9,243 routing MACs/query.

Both issue one 4 KiB parameter-page probe per query.

These results do not establish the asymptotic theorem. They establish that the finite reliability mechanism is measurable and large enough to justify deeper work.
