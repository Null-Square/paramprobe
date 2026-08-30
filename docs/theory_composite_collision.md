# Composite-address collision entropy

This note isolates one routing issue that appears only after we treat a large external parameter store as a **factorized address space** rather than as a flat list of experts.

## 1. Problem

Let a router emit `r` Bernoulli factor probabilities

`p(x) = (p_1(x), ..., p_r(x))`.

The implied external page address is an `r`-bit vector `a in {0,1}^r`, so the number of addressable pages is

`N = 2^r`.

For one input `x`, the factorized soft address distribution is

`P(a | x) = product_j p_j(x)^{a_j} (1-p_j(x))^{1-a_j}`.

The aggregate page distribution is

`Q(a) = E_x P(a | x)`.

A load-balancing objective should care about `Q(a)`, not only the individual factor marginals. Balanced independent-looking bits do **not** imply balanced composite addresses. The even-parity distribution is the simplest counterexample: every bit is balanced and every pair of bits can be independent while exactly half of the complete addresses are unused.

## 2. Collision identity

The collision probability of the aggregate address distribution is

`C2(Q) = sum_a Q(a)^2`.

Take two independent inputs `x` and `x'`. Then

`C2(Q) = E_{x,x'} sum_a P(a|x) P(a|x')`.

Because the address distribution factorizes, the inner product factorizes too:

`sum_a P(a|x) P(a|x')`

`= product_j [p_j(x)p_j(x') + (1-p_j(x))(1-p_j(x'))]`.

Therefore

`C2(Q) = E_{x,x'} product_j [p_j p'_j + (1-p_j)(1-p'_j)]`.

This is the key computational identity. A flat computation of `C2` requires an `N`-way aggregate distribution. The factorized identity requires only the `r = log_2 N` probabilities for two examples.

- exact minibatch diagnostic: `O(B^2 r)`;
- paired stochastic estimate: `O(B r)`;
- no `O(N)` routing logits, table, or histogram are required.

The implementation is in `src/paramprobe/routing_regularizers.py` and is tested against explicit enumeration for small `N`.

## 3. Relation to Renyi-2 entropy

The order-2 Renyi entropy is

`H2(Q) = -log C2(Q)`.

For any distribution over `N` addresses,

`C2(Q) >= 1/N`,

with equality iff `Q` is uniform. This follows immediately from Cauchy-Schwarz:

`1 = (sum_a Q(a))^2 <= N sum_a Q(a)^2`.

So a capacity-normalized objective is

`Delta2(Q) = log N - H2(Q) = log(N C2(Q)) >= 0`.

`Delta2 = 0` iff the complete address distribution is uniform. Unlike raw collision probability, the optimum has the same value for every capacity in an `N` sweep.

## 4. Prior-work boundary

Collision/Renyi entropy is **not** itself a ParamProbe invention. A 2026 MoE theory preprint by Su and Liu (`arXiv:2601.03577`) interprets conventional flat-expert auxiliary load balancing as maximizing Renyi-2 entropy of the aggregate expert posterior. Standard MoE work also has a large literature on balancing flat expert traffic.

The narrower observation relevant to ParamProbe is:

> For an exponentially large **factorized composite parameter address**, full-address collision entropy can be evaluated or estimated directly in factor space, with cost polynomial in `r = log N`, even when enumerating the `N` pages is impossible.

This matters because factor-level balancing is insufficient: it can report perfectly balanced factors while the composite page space remains collapsed.

We do not currently claim this identity is novel enough by itself for publication. Its role is to make the ParamProbe scaling methodology correct and to expose the difference between flat-expert and composite-address load balancing.

## 5. G2e empirical result

G2e replaces G2d's factor-marginal regularizer with the normalized composite-address objective while keeping:

- task-only router supervision;
- the same fixed-size router;
- top-2 training neighborhood;
- hard top-1 inference;
- one 4 KiB parameter page per inference.

On the adversarial 64-item task where every semantic item has an independent random teacher MLP, three seeds at 60 epochs give approximately:

| pages | normalized MSE |
| ---: | ---: |
| 1 | `0.9871 +/- 0.0034` |
| 4 | `1.0301 +/- 0.0086` |
| 16 | `0.9201 +/- 0.0325` |
| 64 | `0.2794 +/- 0.0549` |

The regularizer keeps composite utilization high and largely removes dead pages, but the 4-page condition is still slightly worse than one page.

This is a useful negative result:

**load balancing and task-specializing routing are separate problems.**

Maximizing composite address entropy prevents capacity collapse; it does not tell the router which inputs should share an operator.

## 6. Consequence for the research plan

The next routing objective must combine two properties without creating an `O(N)` router:

1. **global composite diversity** — high `H2(Q)` / low page collision;
2. **task-aware locality** — inputs that benefit from similar page operators should route together.

G2e therefore narrows the remaining bottleneck from `Can exponentially addressed pages be balanced?` to the more specific question `Can useful task specialization be learned with sublinear routing state and bounded inference probes?`
