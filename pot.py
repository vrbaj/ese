import numpy as np
import math


def pot(data, method):
    """
    Peak-Over-Threshold method.
    :param data: input data
    :param method: method identifier
    :return: k highest values
    """
    sorted_data = -np.sort(-data)
    k = 0
    n = len(data)
    if method == 1:
        # 10%
        k = max(int(0.1 * n), 1)
    elif method == 2:
        # sqrt(n)
        k = max(int(math.sqrt(n)), 1)
    elif method == 3:
        k = max(int((n ** (2/3))/math.log10(math.log10(n))), 1)
    elif method == 4:
        k = max(int(math.log10(n)), 1)
    if method == 5:
        # 20%
        k = max(int(0.35 * n), 1)
    return sorted_data[:k]


def pot_min(data, method):
    sorted_data = np.sort(data)
    k = 0
    n = len(data)
    if method == 1:
        # 10%
        k = max(int(0.1 * n), 1)
    elif method == 2:
        # sqrt(n)
        k = max(int(math.sqrt(n)), 1)
    elif method == 3:
        k = max(int((n ** (2/3))/math.log10(math.log10(n))), 1)
    elif method == 4:
        k = max(int(math.log10(n)), 1)
    return sorted_data[:k]
