from typing import Dict, List

import numpy as np


def mat2str(mat):
    return (
        str(mat)
        .replace("'", '"')
        .replace("(", "<")
        .replace(")", ">")
        .replace("[", "{")
        .replace("]", "}")
    )


def dictsum(dic, t):
    return sum([dic[key][t] for key in dic if t in dic[key]])


def moving_average(a, n=3):
    """
    Computes a moving average used for reward trace smoothing.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def fraction_allocation(
    n: int, m: int, p: List[float], labels: List[int] = None
) -> List[int] | Dict[int, int]:
    # labels: List[int],
    """
    allocate n items to m groups according to the fraction p
    return the number of items allocated to each group
    """
    assert len(p) == m
    assert np.isclose(sum(p), 1.0)

    cdf = np.cumsum(p)
    values = np.linspace(0, 1, n, endpoint=False) + 1 / (2 * n)
    indices = np.searchsorted(cdf, values, side="right")
    allocations = np.bincount(indices, minlength=m)
    if labels is not None:
        return {labels[i]: allocations[i] for i in range(m)}
    return allocations
