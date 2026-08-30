import numpy as np

from experiments.g1c_coded_addressing import hamming1511_decode, hamming1511_encode


def test_hamming1511_corrects_every_single_bit_error() -> None:
    messages = np.array(
        [[int(bit) for bit in f"{value:011b}"] for value in range(64)],
        dtype=np.int8,
    )
    codewords = hamming1511_encode(messages)

    decoded, _ = hamming1511_decode(codewords)
    np.testing.assert_array_equal(decoded, messages)

    for bit in range(15):
        corrupted = codewords.copy()
        corrupted[:, bit] ^= 1
        decoded, _ = hamming1511_decode(corrupted)
        np.testing.assert_array_equal(decoded, messages)
