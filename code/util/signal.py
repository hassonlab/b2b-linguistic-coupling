import numpy as np
from scipy.fftpack import fft, ifft
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate, correlation_lags, convolve, windows


def phase_randomize(data: np.ndarray, axis: int = 0, rng=None) -> np.ndarray:
    """Returns a vector of the same size and amplitude spectrum but with shuffled phase information.
    Adapted from https://github.com/brainiak/brainiak/blob/master/brainiak/utils/utils.py
    """

    n_examples, n_samples = data.shape
    if rng is None:
        rng = np.random.default_rng()

    # Get randomized phase shifts
    if n_samples % 2 == 0:
        pos_freq = np.arange(1, n_samples // 2)
        neg_freq = np.arange(n_samples - 1, n_samples // 2, -1)
    else:
        pos_freq = np.arange(1, (n_samples - 1) // 2 + 1)
        neg_freq = np.arange(n_samples - 1, (n_samples - 1) // 2, -1)

    phase_shifts = rng.random(size=(n_examples, len(pos_freq))) * 2 * np.math.pi

    # Fast Fourier transform along time dimension of data
    fft_data = fft(data, axis=axis)

    # Shift pos and neg frequencies symmetrically, to keep signal real
    # TODO: make these two lines respect the axis argument
    fft_data[:, pos_freq] *= np.exp(1j * phase_shifts)
    fft_data[:, neg_freq] *= np.exp(-1j * phase_shifts)

    # Inverse FFT to put data back in time domain
    shifted_data = np.real(ifft(fft_data, axis=axis))

    return shifted_data


def correlate_lags(x: np.ndarray, y: np.ndarray, lags=None, rank=False):
    """Compute correlations for the given lags

    Positive lag: x is ahead
    Negative lag: x is behind

    Note: when talking about lead/lag, uses <y> as a reference.
    Therefore positive lag means <x> lags <y> by <lag>, computation is
    done by shifting <x> to the left hand side by <lag> with respect to
    <y>.
    Similarly negative lag means <x> leads <y> by <lag>, computation is
    done by shifting <x> to the right hand side by <lag> with respect to
    <y>.
    """

    # Prune to same length
    if x.size > y.size:
        x = x[: y.size]
    elif y.size > x.size:
        y = y[: x.size]

    func = spearmanr if rank else pearsonr

    corrs = np.zeros(len(lags))
    for j, i in enumerate(lags):
        if i == 0:
            corrs[j], _ = func(x, y)
        elif i < 0:
            corrs[j], _ = func(x[:i], y[-i:])
        elif i > 0:
            corrs[j], _ = func(x[i:], y[:-i])

    return corrs


def xcorr(
    x: np.ndarray,
    y: np.ndarray,
    smooth_win=None,
    mode="full",
    method="fft",
    norm=True,
    maxlags=None,
    rank=False,
):
    """General function to compute cross correlation using scipy

    This function will center the data and normalize it by default.
    """

    # Rank transform
    if rank:
        # NOTE - equivalent to x = scipy.stats.rankdata(x, method='ordinal')
        x = x.argsort().argsort()
        y = y.argsort().argsort()

    # Center
    x = x - x.mean()
    y = y - y.mean()

    # Correlate
    corr = correlate(x, y, mode=mode, method=method)
    lags = correlation_lags(x.size, y.size, mode=mode)

    if norm:
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    if maxlags is not None:
        middle = (lags == 0).nonzero()[0].item()
        lags = np.arange(-maxlags, maxlags + 1)
        corr = corr[middle - maxlags : middle + maxlags + 1]

    # Smooth the correlations
    # look at https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    # to get rid of edge effects
    if smooth_win is not None:
        win = windows.hamming(smooth_win)
        corr = convolve(corr, win, mode="same") / sum(win)
        print(f"Smoothing xcorr with {smooth_win} window size")

    return corr, lags
