import mne
import os
import argparse
import json
import pickle
import numpy as np
import pandas as pd

# from numba import jit
from numpy.lib.stride_tricks import as_strided
from mne_bids import BIDSPath
from util.path import DerivativeBIDSPath

NONWORDS = {'oh', 'uh', 'um', 'mhm', 'tsk', 'uh-huh', 'mm-hmm', 'ma'}
DCCHS = ['DC9-REF', 'DC10-REF']
DIV_CMAP = 'RdYlBu_r'
SEQ_CMAP = 'Reds'

def save_pickle(filepath: str, *args) -> None:
    if hasattr(filepath, 'mkdir'):
        filepath.mkdir(exist_ok=True)
    with open(filepath, 'wb') as f:
        obj = args[0] if len(args) == 1 else args
        pickle.dump(obj, f)

def load_pickle(filepath: str):
    obj = None
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj

def getcoords(raw, electrodes=None):
    electrodes = raw.ch_names if electrodes is None else electrodes
    ch2loc = {ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}
    coords = np.vstack([ch2loc[ch] for ch in electrodes])
    coords *= 1000  # convert to mm
    return coords, electrodes

def getallcoords(results, Ss, modelname):
    allcoords = {}
    for sub in Ss:
        raw = getraw(sub, root='../dataset/')
        
        elecs = results[(sub, 'prod', modelname)]['electrodes']
        coords, elecs = getcoords(raw, elecs)
        allcoords[sub] = coords
    return allcoords

def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=int, nargs='+', default=[1])
    parser.add_argument('-t', '--task', default='conversation')
    parser.add_argument('-r', '--root', default='dataset')
    parser.add_argument('-w', '--workers', type=int, default=1)
    parser.add_argument('--refalg', default='fastica')
    parser.add_argument('--verbose', action='store_true')
    return parser


def getsubject(raw):
    substr = raw.info['subject_info']['his_id']
    subid = substr.split('-')[1]
    return int(subid)

# > Path/data utils

def getpath(pathorsub, root='dataset'):
    if isinstance(pathorsub, str):
        pathorsub = int(pathorsub)
    if isinstance(pathorsub, BIDSPath):
        return pathorsub
    elif isinstance(pathorsub, int):
        return DerivativeBIDSPath(subject=f'{pathorsub:02d}', task='conversation', root=root)
    else:
        return ValueError

def getparts(string):
    parts = dict(a.split('-', 1) for a in string.split('_') if len(a.split('-')) > 1)
    for key, value in parts.items():
        if value.isnumeric():
            parts[key] = int(value)
    return parts


def suffix(suffix, **kwargs):
    string = '_'.join(f'{key}-{value}' for key, value in kwargs)
    return string + f'_{suffix}'


def getpartner(s):
    return s + (-1 if (s%2 == 0) else 1)


def get_partner(path):
    fname = os.path.join(path.root.parts[0], 'participants.tsv')
    df = pd.read_csv(fname, sep='\t')
    partner = df[df.participant_id == 'sub-' + path.subject].partner
    assert partner.size, "partner not defined in participants.tsv"
    return partner.item()


def getraw(path, refalg='fastica', band='highgamma', trim=False, **kwargs):
    """Get the MNE Raw object for a given subject.
    
    Args:
        path: BIDSPath
        refalg: re-referencing algorithm
        trim: if True, trim data to conversation beginning and end
        kwargs: passed to `gettransctipt`
    
    Returns:
        The MNE Raw object
    """
    path = getpath(path, root=kwargs.get('root', 'dataset'))
    rawpath = path.copy()
    rawpath.update(datatype=refalg,
                   suffix=f'desc-{band}_ieeg',
                   extension='.fif',
                   root=f'{path.root}/derivatives/preprocessed')
    raw = mne.io.read_raw(rawpath.fpath)

    # Trim raw to the transcript start and end
    # The transcript should already be aligned to the raw, so no need to adjust
    # for lags. Just trim.
    if trim:
        df = gettranscript(path, **kwargs)
        convo_start = df.start.values[0]
        convo_end = df.end.values[-1]
        raw = raw.crop(tmin=convo_start, tmax=convo_end)
        df['start'] = df.start - convo_start
        df['end'] = df.end - convo_start
        df['start_id'] = raw.time_as_index(df.start.values)
        df['end_id'] = raw.time_as_index(df.end.values)
        return raw, df

    return raw
    
def gettranscript(path, speaker=None, utterances=False, merge_utts=True,
                  embeddings=None, min_words=2, keep_tokens=False):
    """Get the embedding transcript for the given path.
    
    Args:
        path: BIDS Path object
        embeddings: name of model. If None, returns transcript without embeddings
        utterances: if True, return utterances instead of word-level
        merge_utts: (only if utterances), merge consecutive utterances
        min_words: (only if utterances), remove utterances less than `min_words`
        speaker: filter transcript to only instances where `speaker` produces
        keep_tokens: remove other sub words
    
    Returns:
        pandas DataFrame of the conversation
    """
    eventpath = getpath(path).copy()
    if embeddings is None:
        eventpath.update(datatype='audio', extension='.csv', suffix='audio',
                         root=f'{eventpath.root}/derivatives/preprocessed')
        df = pd.read_csv(eventpath.fpath, index_col=0)
    else:
        df = getembeddings(eventpath, embeddings, keep_tokens=keep_tokens)

    # Remove non-words
    for nonword in ['{lg}', '{}', '{inaudible}', '{inaudbile}', '{gasps}', '{clapping}']:
        df.drop(df[df.word.str.lower() == nonword].index, inplace=True)
        
    # Ensure onsets are good
    assert (df.start >= 0).all(), 'onsets before 0'

    if utterances:

        # Take only utterances
        dfu = df.groupby('utterance_id').agg({'utt_onset': 'first',
                                              'utt_offset': 'last',
                                              'utterance': 'first',
                                              'speaker_id': 'first'})
        dfu.rename(columns={'utt_onset': 'start', 'utt_offset': 'end'},
                   inplace=True)

        dfu['n_words'] = dfu.utterance.apply(lambda x: len(x.split()))
        if min_words is not None:
            dfu = dfu[dfu.n_words >= min_words]

        # Merge consecutive utterances
        if merge_utts:
            shifted = dfu.speaker_id.shift().fillna(method='bfill')
            g = (dfu.speaker_id != shifted).cumsum().rename('group')
            dfu = dfu.groupby(g).agg({'start': 'first',
                                      'end': 'last',
                                      'speaker_id': 'first',
                                      'utterance': lambda s: ' '.join(s.tolist())})
            dfu['n_words'] = dfu.utterance.apply(lambda x: len(x.split()))

        if speaker is not None:
            dfu = dfu[dfu.speaker_id == int(speaker)]

        # breakpoint()
        dfu['overlaps'] = dfu.start.shift(-1) < dfu.end
        dfu = dfu[~dfu.overlaps]
        # if overlap is above 90%...
        # a...b c...d
        #  c d
        # max(0, min(max1, max2) - max(min1, min2)) # normalize?
        # kinds of overlap:
        #   |   S1       |
        #       |S2|
        #     |     S2  |

        #   |   S1       |
        #              |     S2    |

        return dfu
    elif 'utt_onset' in df.columns:
        df.drop(['utt_onset', 'utt_offset'], axis=1, inplace=True)

    if speaker is not None:
        df = df[df.speaker_id == int(speaker)]

    return df


def getembeddings(path, model, keep_tokens=False):
    eventpath = getpath(path).copy()
    parts = getparts(model)
    eventpath.update(datatype=parts['model'], suffix=f'{model}_embeddings',
                     extension='.pkl',
                     root=f'{path.root}/derivatives/embeddings')
    df = pd.read_pickle(eventpath.fpath)
    df = df[df.embedding.notna()]
    df = df[df.word.str.lower().apply(lambda x: x not in NONWORDS)]
    
    # Remove extra tokens
    if not keep_tokens and 'word_idx' in df.columns:

        col_actions = {
            'word': 'first', 'start': 'first', 'end': 'first',
            'speaker_id': 'first', 'word_punc': 'first',
            'utterance_id': 'first', 
        }

        if 'rank' in df.columns:
            col_actions.update({'rank': np.mean, 'true_prob': np.prod, 'entropy': np.sum})

        col_actions['embedding'] = lambda x: [np.mean(x, axis=0)]  # average embeddings of tokens

        df = df.groupby('word_idx').agg(col_actions)

        # This will remove the following tokens:
        #  'token': 'first',  'token_id',  'top_pred': 'last',

    return df


def getbrainlag(path, return_path=False):
    lagpath = path.copy()
    lagpath.update(datatype='audio',
                   suffix='xcorr',
                   extension='.json',
                   root=f'{path.root}/derivatives/preprocessed')

    if return_path:
        return lagpath
    else:
        with open(lagpath.fpath, 'r') as f:
            result = json.load(f)
        return result


def getaudioenv(path, return_path=False):
    audiopath = path.copy()
    audiopath.update(datatype='audio',
                     suffix='desc-envelope_audio',
                     extension='.npy',
                     root=f'{path.root}/derivatives/preprocessed')

    if return_path:
        return audiopath
    else:
        highenv = np.load(audiopath.fpath)
        return highenv


def getplotpath(path):
    plotpath = path.copy()
    plotpath.update(task=None,
                    datatype=None,
                    acquisition='render',
                    suffix=None,
                    extension='.jpg',
                    root=f'{path.root}/derivatives/figures')
    return plotpath

# > Array utils

def onsetslices(x, tmin, tmax):
    slices = np.repeat(x, 2).reshape(len(x), 2)
    slices[:, 0] += tmin
    slices[:, 1] += tmax
    return slices


# @jit(nopython=True)
def epochbin(X, ids, lags, window):
    """Epoch matrix X into the given lags
    
    Args:
        X: a time x n_elecs matrix
        ids: onsets of events
        lags: lags to extract
        window: window size to average
    
    Returns:
        epoch matrix of shape n_words x n_elecs x n_lags
    """
    assert window % 2 == 0
    assert np.all((ids + lags.min() - window //2 ) > 0)
    assert np.all((ids + lags.max() + window //2  + 1) < len(X))
    
    halfWindow = window // 2
    epochs = np.zeros((len(ids), X.shape[1], lags.size), dtype=X.dtype)
    for i, onset in enumerate(ids):
        for j, lag in enumerate(lags):
            epochs[i, :, j] = X[onset+lag-halfWindow:onset+lag+halfWindow+1].sum(0)
    epochs /= window + 1
    return epochs


# @jit(nopython=True)
def epoch(x, window, ids=None, return_ids_only=False):
    '''Take slicese out of an ndarray corresponding to particular ids.

      If input is:
           x = [1, 2, 3, 4, 5, 6, 7, 8, 9], window = 5

       Will create a view with as_strided like so:
                 [[0, 1, 2, 3, 4],
                  [1, 2, 3, 4, 5],
                  [2, 3, 4, 5, 6],
                  [3, 4, 5, 6, 7],
                  [4, 5, 6, 7, 8],
                  [5, 6, 7, 8, 9]]

      Then extract ids = [4, 5, 6] and returns:
                 [[2, 3, 4, 5, 6],
                  [3, 4, 5, 6, 7],
                  [4, 5, 6, 7, 8]]

    '''
    #assert x.base is None, 'Array must not be a view'
    #assert window % 2 == 1, 'Full window size must be odd'

    ncols = window
    nrows = len(x) + 1 - window

    if x.ndim == 1:
        shape = (nrows, ncols)
        strides = x.strides * 2
    if x.ndim == 2:
        shape = (nrows, x.shape[1], ncols)  # n_ids, x_dim2, window
        strides = x.strides + (x.strides[0], )

    xnew = as_strided(x, shape=shape, strides=strides, writeable=False)

    if ids is not None:
        # New array has fewer rows than original, so compensate for it
        # TODO assert nonnegative ids and with signal len
        ids = np.asarray(ids) - (window - 1) // 2
        # creates new array and allocates memory if not None
        # NOTE - this new array is not contiguous
        if not return_ids_only:
            xnew = xnew[ids, ...]
        else:
            xnew = (xnew, ids)

    return xnew


def binsignal(data, window, jump, axis=-1):
    '''Downsample a signal by taking a moving average then skipping points.

       data: shape = (words, channels, time)
       axis defines which axis to downsample (usually time domain).

       See https://stackoverflow.com/a/14314054 for efficient moving average
       implementation.
    '''

    n = window
    ret = np.cumsum(data, axis=axis)
    if axis == -1:
        ret[..., n:] = ret[..., n:] - ret[..., :-n]
        ret = ret[..., n - 1:] / n
        ret = ret[..., ::jump]
    elif axis == 0:
        ret[n:, ...] = ret[n:, ...] - ret[:-n, ...]
        ret = ret[n - 1:, ...] / n
        ret = ret[::jump, ...]
    else:
        raise ValueError

    return ret


# # @jit(nopython=True)
# def epochold(x, slices, normalize=True, reduce=None, axis=0):
#     chunks = []
#     for start, end in slices:
#         chunk = x.take(np.arange(start, end + 1), axis=axis)
#         # if normalize:
#         #     chunk = stats.zscore(chunk, axis=axis)
#         chunks.append(chunk)
#     if reduce == 'concat':
#         if x.ndim == 1:
#             chunks = np.vstack(chunks)
#         else:
#             chunks = np.concatenate(chunks)
#     return chunks


# @jit(nopython=True, parallel=True)
def blockarr(slices, n=None):
    if n is None:
        n = slices[-1][1]
    arr = np.zeros(n, dtype=int)
    for start, stop in slices:
        arr[start:stop] = 1
    return arr


def rescale(X, b, a):
    maxx = X.max()
    minx = X.min()
    return (b - a) * (X - minx) / (maxx - minx) + a


def invertdict(d):
    r = {}
    for key, values in d.items():
        if isinstance(values, list):
            for value in values:
                r[value] = key
    return r


# @jit(nopython=True)
def lcs(x: list[str], y: list[str]):
    '''
    https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
    '''
    m = len(x)
    n = len(y)
    c = np.zeros((m + 1, n + 1))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                c[i, j] = c[i - 1, j - 1] + 1
            else:
                c[i, j] = max(c[i, j - 1], c[i - 1, j])

    mask1, mask2 = [], []
    i = m
    j = n
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            i -= 1
            j -= 1
            mask1.append(i)
            mask2.append(j)
        elif c[i - 1][j] > c[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return np.flip(np.array(mask1)), np.flip(np.array(mask2))
#     return mask1[::-1], mask2[::-1]


def align_dfs(df1, df2, on='', on_right='', strs=True, copy=True):
    vals1 = df1[on]
    vals2 = df2[on if on_right is None else on_right]
    
    if strs:
        vals1 = vals1.str.strip().str.lower()
        vals2 = vals2.str.strip().str.lower()
        
    mask1, mask2 = lcs(vals1.tolist(), vals2.tolist())
    df3 = df1.iloc[mask1]
    df4 = df2.iloc[mask2]
    
    if copy:
        df3 = df3.copy()
        df4 = df4.copy()
    
    return df3, df4

# > Stats

def correlate(A, B, axis=0):
    """Calculate pearson correlation between two matricies.

       axis = 0 correlates columns in A to columns in B
       axis = 1 correlates rows in A to rows in B
    """
    assert A.ndim == B.ndim, 'Matrices must have same number of dimensions'
    assert A.shape == B.shape, 'Matrices must have same shape'

    A_mean = A.mean(axis=axis, keepdims=True)
    B_mean = B.mean(axis=axis, keepdims=True)
    A_stddev = np.sum((A - A_mean)**2, axis=axis)
    B_stddev = np.sum((B - B_mean)**2, axis=axis)

    num = np.sum((A - A_mean) * (B - B_mean), axis=axis)
    den = np.sqrt(A_stddev * B_stddev)

    return num / den


def corrmatrix(x, y, axis=0):
    xv = x - x.mean(axis=axis)
    yv = y - y.mean(axis=axis)
    xvss = (xv * xv).sum(axis=axis)
    yvss = (yv * yv).sum(axis=axis)
    return np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))


def smooth(x, window_len, window='hamming'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        raise ValueError("Window size to small")

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = x
    # s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

def overlap(x1, x2, y1, y2):
    return max(0, min(x2, y2) - max(x1, y1)) / min(x2, y2)

# print(overlap(0, 10, 90, 100))  # no overlap
# print(overlap(0, 55, 45, 100))  # little overlap
# print(overlap(0, 100, 10, 90))  # major overlap
# print(overlap(0, 100, 0, 20))   # little overlap
# print(overlap(0, 100, 80, 100))


def rscorer(estimator, X, y):
    yhat = estimator.predict(X)
    return correlate(yhat, y)


# > ROI utils

def get_rois(sub, root='dataset', atlas='desikan', chs=True):
    path = getpath(sub, root=root).copy()
    # path = BIDSPath(root=os.path.join(root, 'preprocessed'),
    #                 subject=sub,
    #                 task=None,
    #                 datatype='anat',
    #                 suffix=f'atlas-{atlas}_anat', check=False)

    path.update(datatype='anat',
                task=None,
                suffix=f'atlas-{atlas}_anat',
                root=f'{path.root}/derivatives/preprocessed')

    with open(path.fpath, 'r') as f:
        data = json.load(f)

    if isinstance(chs, bool):
        raw = getraw(sub, root=root)
        chs = raw.ch_names
    if len(chs):
        ch_names = set(chs)
        for elecs in data.values():
            bads = set(elecs) - ch_names
            for ch in bads:
                elecs.remove(ch)

    return data


AUDIO_ROIS = {'MOTOR': ['PREC', 'PSTC', 'CMF'],
              'AUD':  ['ST', 'cSTG', 'mSTG', 'rSTG', 'BSTS']}

def get_funcrois(sub, funcs=['MOTOR', 'AUD'], root='dataset/derivatives', task='conversation', n=None):
    if isinstance(sub, int):
        sub = f'{sub:02d}'

    # atlasrois = get_rois(sub, root='dataset/derivatives', atlas='deskin')
    # atlasrois = invertdict(atlasrois)

    modes = {'MOTOR': 'cor-spearman_mode-speaker',
             'AUD': 'cor-spearman_mode-listener'}

    rois = {}
    for label in funcs:
        mode = modes[label]
        path = BIDSPath(root=os.path.join(root, 'audlocalizer'),
                        subject=sub,
                        task=task,
                        datatype=mode,
                        suffix='max',
                        extension='.csv',
                        check=False)
        df = pd.read_csv(path.fpath)
        df.sort_values(by='r', ascending=False, inplace=True)
        df = df[~df.electrode.isin(DCCHS)]
        # df['roi'] = df.electrode.map(atlasrois)

        df = df[df.roi.isin(AUDIO_ROIS[label])]
        rois[label] = df.iloc[:n].electrode.tolist()
        if len(df) < n:
            print(f'[WARN] not enough electrodes found for {sub} in {label}')

    return rois


def get_mni_rois(raw):
    from nilearn import datasets, image
    destrieux = datasets.fetch_atlas_destrieux_2009()
    nimg = image.load_img(destrieux.maps)
    img = nimg.get_fdata().astype(int)
    affine = np.linalg.inv(nimg.affine)

    locs = np.vstack([ch['loc'][:3] for ch in raw.info['chs']]) * 1000

    xyz = np.stack(image.coord_transform(*locs.T, affine)).T
    xyz = xyz.round().astype(int)
    xyz = np.clip(xyz, 0, max(img.shape)-1)
    roi_id = img[xyz[:,0], xyz[:,1], xyz[:,2]]
    rois = [destrieux.labels[rid][1].decode() for rid in roi_id]

    roimap = dict(zip(raw.ch_names, rois))
    return roimap
