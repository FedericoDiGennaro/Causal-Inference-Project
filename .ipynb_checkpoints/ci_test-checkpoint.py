
import math
from scipy import stats
from scipy.stats import norm
import numpy as np

# ci_test(D, X, Y, Z, alpha)
# Assume that variables are [0,1,2,...,p-1] and we have n samples
# Input:
# D: Matrix of data (numpy array with size n*p)
# X: index of the first variable
# Y: index of the second variable
# Z: A list of indices for variables of the conditioning set
# alpha: a constant for thresholding
# output = True (independent) or False (dependent)

# Example usage: 1) Z is empty set: ci_test(D, 0, 2, [])
# 2) Z={1,2}: ci_test(D, 0, 4, [1,2])

# Note that D must be a numpy array. (D=np.array([[1,2],[2,3]]))


def ci_test(D, X, Y, Z, alpha):
    n = D.shape[0]
    if len(Z) == 0:
        r = np.corrcoef(D[:, [X, Y]].T)[0][1]
    else:
        sub_index = [X, Y]
        sub_index.extend(Z)
        sub_corr = np.corrcoef(D[:, sub_index].T)
        # inverse matrix
        try:
            PM = np.linalg.inv(sub_corr)
        except np.linalg.LinAlgError:
            PM = np.linalg.pinv(sub_corr)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    cut_at = 0.99999
    r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

    # Fisher’s z-transform
    res = math.sqrt(n - len(Z) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    p_value = 2 * (1 - stats.norm.cdf(abs(res)))

    return p_value >= alpha