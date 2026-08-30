import numpy as np

from paramprobe.addressing import FactorizedTopKRouter


def test_address_space_grows_exponentially_while_router_metadata_grows_linearly_in_r():
    m, s = 4, 8
    previous_space = None
    previous_metadata = None
    for r in range(2, 11):
        router = FactorizedTopKRouter(np.zeros((r, m, s)))
        assert router.address_space == m**r
        assert router.metadata_scalars == r * m * s
        if previous_space is not None:
            assert router.address_space == previous_space * m
            assert router.metadata_scalars == previous_metadata + m * s
        previous_space = router.address_space
        previous_metadata = router.metadata_scalars
