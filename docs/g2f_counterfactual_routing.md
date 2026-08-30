# G2f — fixed-budget counterfactual page routing

## Question

Can a factorized ParamProbe router learn **task-useful** page specialization without semantic address labels, while keeping the inference contract fixed at one external parameter page?

G2d/G2e showed why this is a separate problem from load balancing:

- factor-level balancing can hide composite-address collapse;
- full-address Renyi-2 balancing fixes collapse much more directly;
- but balanced pages can still be the *wrong* pages for the task.

G2f adds token/example-level counterfactual utility during training.

## Method

Inference remains:

`context -> factorized router -> one page id -> one page MLP`

with

- `q_infer = 1`;
- `B = 4096 bytes`;
- logical external parameter traffic = `4096 bytes / invocation`;
- the same fixed router architecture at every capacity;
- the same active page-MLP shape at every capacity.

Training uses a fixed budget of four candidate pages for every multi-page condition.

For each example:

1. form the current hard page address;
2. generate one-bit neighboring addresses by flipping the least-confident routing factors;
3. fill any remaining candidate slot with an exploratory route;
4. evaluate the page operator for each of the four candidates;
5. compute each candidate's task loss `ell_c`;
6. define a detached utility target

   `u_c = softmax(-ell_c / tau_u)`;

7. train page parameters with the utility-weighted task loss;
8. train the router to match `u` using the factorized log-probabilities of the same candidate addresses;
9. retain composite-address Renyi-2 balancing to prevent global page collapse.

The router therefore receives no explicit page labels. Its additional supervision is only: **among these equal-size page alternatives, which one actually performed the task better?**

At inference none of the alternatives are evaluated.

## Fixed-budget control

An earlier version used the current page, every one-bit neighbor, and one exploratory page, so the training candidate count grew with address width. That could confound a capacity sweep.

The committed G2f experiment caps the candidate budget at exactly four pages for `N >= 4`. Therefore the 4 -> 16 -> 64 page comparison does not receive more counterfactual page evaluations as capacity grows.

Training compute is still larger than ordinary one-route sparse training; G2f is an optimization method with a fixed training exploration budget, not a claim that training uses one page. The **inference** resource theorem remains `q_infer B`.

## Adversarial task

The primary control intentionally removes helpful semantic geometry:

- 64 semantic items have independent random context prototypes;
- each item has an independently sampled nonlinear teacher MLP;
- context prototypes and teacher functions are sampled independently;
- the router receives no item/page address targets.

Thus there is no hand-crafted rule saying which items belong together. The router/page system must discover a useful partition from counterfactual task loss.

## Result

Three seeds, 40 epochs, fixed four-candidate training budget:

| pages | normalized MSE |
| ---: | ---: |
| 1 | `0.9856 +/- 0.0048` |
| 4 | `0.9557 +/- 0.0133` |
| 16 | `0.8258 +/- 0.0088` |
| 64 | `0.2242 +/- 0.0567` |

The curve is monotone in all three seeds.

At 64 pages all three runs had zero dead pages under validation routing. The modal item-collision fraction was approximately `0.078`, `0.141`, and `0.156` for seeds 1-3 respectively; perfect one-item-per-page routing is therefore **not** required to obtain the large quality gain.

The 4/16/64 conditions use the same:

- router parameter count;
- router forward MAC count;
- active page-operator MAC count at inference;
- page size;
- inference page-probe count;
- counterfactual candidate budget during training.

Only total available external page capacity changes.

## What this establishes

G2f is the first ParamProbe synthetic gate in which all of the following hold simultaneously:

1. external pages contain genuine nonlinear neural operators rather than stored values;
2. routing and page operators are learned jointly;
3. no semantic address labels are supplied;
4. the task itself supplies route preference through bounded counterfactual exploration;
5. page count increases while inference remains one fixed-size page;
6. held-out task error improves monotonically across the tested capacity sweep.

This is stronger than G2a/G2b's oracle routing and G2c's address-supervised routing.

## Prior-work boundary

Counterfactual expert utility is not new by itself. Yoon et al., *When Are Experts Misrouted? Counterfactual Routing Analysis in Mixture-of-Experts Language Models* (`arXiv:2605.07260`, 2026), show that standard MoE routing has a counterfactual blind spot: ordinary sparse training evaluates the executed route but not equal-compute alternatives. Their Expert Preference Optimization performs router-only updates using lower-loss alternative routes in already trained MoE models.

G2f should therefore **not** be described as inventing counterfactual routing.

The ParamProbe-specific combination being investigated is:

- exponentially factorized external parameter-page addresses;
- no `O(N)` router state;
- composite-address rather than factor-only balancing;
- a fixed small counterfactual candidate budget during training;
- page-local neural operators designed around physical storage granularity;
- hard `q=1` external-page inference.

Whether that combination is publication-level novel remains subject to a broader literature audit.

## Remaining limitations

G2f is still synthetic.

The important open questions are now:

- does the method work when the page layer sits inside a shared neural backbone rather than receiving an explicit semantic context channel?
- does next-token loss provide a sufficiently clean counterfactual utility signal for page routing in a language model?
- can candidate page operators be evaluated efficiently during training without making training prohibitively expensive?
- does the monotone capacity effect survive matched total training FLOPs, not merely matched candidate count?
- how does page size `B` trade off against number of pages `N` under a fixed total storage and inference-I/O budget?

These are the reasons the next gate should be a **tiny language-model experiment**, not a larger synthetic benchmark.
