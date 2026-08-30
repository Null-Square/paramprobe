import numpy as np

from paramprobe.addressing import FactorizedTopKRouter, brute_force_topk, mixed_radix_id


def test_mixed_radix_is_bijection_on_small_space():
    ids = {
        mixed_radix_id((a, b, c), 3)
        for a in range(3)
        for b in range(3)
        for c in range(3)
    }
    assert ids == set(range(27))


def test_factorized_topk_matches_bruteforce_scores_and_addresses():
    rng = np.random.default_rng(7)
    for r, m, s, k in [(2, 4, 3, 5), (3, 3, 4, 7), (4, 2, 2, 8)]:
        for _ in range(20):
            keys = rng.normal(size=(r, m, s))
            query = rng.normal(size=(r, s))
            router = FactorizedTopKRouter(keys)
            got = router.topk(query, k)
            expected = brute_force_topk(router.factor_scores(query), k)
            assert [x.factors for x in got] == [x.factors for x in expected]
            np.testing.assert_allclose(
                [x.score for x in got], [x.score for x in expected], rtol=0, atol=1e-12
            )
