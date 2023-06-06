import numpy as np
from scipy.stats import norm


def mean_diff_stat(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    """Compute test-statistic for the difference in means.
    See https://docs.scipy.org/doc/scipy//reference/generated/scipy.stats.permutation_test.html
    """
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def mean_diff_axis(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(x, axis=axis)


def permute_differences(
    differences: np.ndarray, summary=mean_diff_axis, n_perms=1000, rng=None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    null_distribution = np.empty((n_perms,) + differences.shape[1:])
    for p in range(n_perms):
        sign_flipper = rng.choice([-1, 1], differences.shape, replace=True)
        permutation = sign_flipper * differences
        null_distribution[p] = summary(permutation)
    return null_distribution


def cdf_pvalues(
    observed: np.ndarray, null_distrubtion: np.ndarray, alternative: str = "greater"
) -> np.ndarray:
    if observed.ndim == 1:
        observed = observed.reshape(1, -1)
    new = np.append(null_distrubtion, observed, axis=0)
    loc = np.mean(new, axis=0)
    scale = np.std(new, axis=0)
    pvalues = norm.sf(observed, loc=loc, scale=scale).squeeze()
    if alternative == "two-sided":
        pvalues *= 2
    return pvalues


def cdf_pvals2(observed, null_distribution):
    # Optimized version
    N = len(null_distribution) + 1
    mu = (null_distribution.sum(axis=0) + observed) / N
    var = (
        np.square((null_distribution - mu)).sum(axis=0) + np.square(observed - mu)
    ) / N
    std = np.sqrt(var)
    zscore = (observed - mu) / std
    return norm.sf(zscore)


def calculate_pvalues(
    observed: np.ndarray,
    null_distribution: np.ndarray,
    alternative: str = "two-sided",
    adjustment: int = 1,
) -> np.ndarray:
    """Calculate p-value
    See https://github.com/scipy/scipy/blob/v1.10.1/scipy/stats/_resampling.py#L1133-L1602
    """
    n_resamples = len(null_distribution)

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less, "greater": greater, "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return pvalues
