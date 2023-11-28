import numpy as np
import numpy.typing as npt


def structure_preserving_rearrangement(
    blocks: list[npt.NDArray], nGroups: int
) -> list[npt.NDArray[np.uint8]]:
    """
    Perform structure-preserving feature re-arrangement on biometric template blocks.

    Parameters:
    - blocks (List[numpy.ndarray]): List of 2D arrays representing the blocks of the biometric template.
    - nGroups (int): The number of groups to divide the blocks into.

    Returns:
    - rearranged_blocks (List[numpy.ndarray]): List of rearranged blocks after the structure-preserving process.

    This function rearranges the blocks by grouping, concatenating, and permuting rows within each group.
    """

    nBlocks = len(blocks)
    nBlocksGroup = nBlocks // nGroups

    if nBlocks % nGroups != 0:
        raise ValueError(
            "The total number of blocks must be evenly divisible by nGroups."
        )

    # Random number generator for permutation
    rng = np.random.default_rng()

    # Split blocks into groups
    groups = [blocks[i * nBlocksGroup : (i + 1) * nBlocksGroup] for i in range(nGroups)]

    rearranged_blocks = []

    for group in groups:
        # Vertically concatenate blocks in the group
        concatenated_group = np.vstack(group)

        # Permute the rows of the concatenated group
        rng.shuffle(concatenated_group)

        # Split and add the permuted rows back to rearranged_blocks
        rearranged_blocks.extend(np.array_split(concatenated_group, nBlocksGroup))

    return rearranged_blocks


def compute_bloom_filters(blocks: list[npt.NDArray], nBits: int) -> npt.NDArray:
    """
    Compute Bloom filters from the given blocks, with each filter having a size of 2^nBits.

    Parameters:
    - blocks (List[numpy.ndarray]): List of 2D arrays representing the blocks.
    - nBits (int): The number of bits used to determine the size of the Bloom filter, with actual size being 2^nBits.

    Returns:
    - bloom_filters (numpy.ndarray): An array containing the Bloom filters.
    """

    nBlocks = len(blocks)
    bloom_filter_size = 2**nBits
    bloom_filters = np.zeros((nBlocks, bloom_filter_size), dtype=int)

    for i, block in enumerate(blocks):
        for col in range(block.shape[1]):
            # Translate column to decimal value
            decimal_value = int("".join(map(str, block[:, col])), 2)

            # Set the corresponding bit in the Bloom filter
            bloom_filters[i, decimal_value % bloom_filter_size] = 1

    return bloom_filters
