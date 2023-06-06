"""Correlate the audio with each electrode separately.
"""

import os
import json

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import audioalign
import utils

from mne_bids import BIDSPath
from scipy.io import wavfile
from tqdm import tqdm


def audioxcorr(rawpath, audiopath, lagpath, **kwargs):
    """Compute xcorr between electrodes and audio envelope.
    """
    raw = mne.io.read_raw(rawpath.fpath, verbose=False)
    highenv = np.load(audiopath.fpath)
    with open(lagpath.fpath, 'r') as f:
        result = json.load(f)
    return audioxcorr_(raw, highenv, result, **kwargs)


def audioxcorr_(raw, highenv, result, method, maxlag, rank=False):
    """Compute xcorr between electrodes and audio envelope.

    highenv: high quality audio envelope
    maxlag: how far from 0 to compute correlations (in seconds)
    rank: whether to rank transform data for spearman correlation
    """

    # Align brain and audio
    # Negative lag means the brain is ahead, positive means audio is ahead.
    lag, lag_s = result['lag_fs'], result['lag_s']
    if lag > 0:
        highenv = highenv[lag:]
        print(f'Trimmed audio by {lag_s}s or {lag} samples')
    elif lag < 0:
        raw = raw.crop(tmin=-lag_s)
        print(f'Cropped brain data by {lag_s}s or {lag} samples')

    if maxlag is not None:
        maxlag = int(maxlag * raw.info['sfreq'])

    # Run xcorr for each electrode
    records = []
    for channel in tqdm(raw.ch_names):
        electrode, _ = raw[channel]
        lowenv = electrode.squeeze()
        if 'DC' in channel:
            lowenv = audioalign.preprocess_lowqa(lowenv)

        if method in ['fft', 'direct']:
            corr, lags = audioalign.xcorr(highenv, lowenv,
                                          method=method,
                                          maxlags=maxlag,
                                          rank=rank)
        elif method == 'lags':
            lags = np.arange(-maxlag, maxlag + 1)
            corr = audioalign.correlate_lags(highenv, lowenv, lags)

        records.append(corr)

    # Return results
    df = pd.DataFrame(np.vstack(records), copy=False)
    df.columns = lags
    df['electrode'] = raw.ch_names
    return df

def main(args):
    # Prepare paths
    path = BIDSPath(subject=args.subject,
                    task=args.task,
                    root=args.root,
                    check=False)

    rawpath = path.copy()
    rawpath.update(datatype=args.refalg,
                   suffix='desc-highgamma_ieeg',
                   extension='.fif',
                   root=f'{path.root}/derivatives/preprocessed')

    lagpath = path.copy()
    lagpath.update(datatype='audio',
                   suffix='xcorr',
                   extension='.json',
                   root=f'{path.root}/derivatives/preprocessed')

    plotpath = path.copy()
    plotpath.update(task=None,
                    datatype=None,
                    acquisition='render',
                    suffix=None,
                    extension='.jpg',
                    root=f'{path.root}/derivatives/figures')

    sub = path.subject
    mode = 'prod'
    if args.partner_audio:
        sub = utils.get_partner(path).split('-')[1]
        mode = 'comp'
    audiopath = path.copy()
    audiopath.update(subject=sub,
                     datatype='audio',
                     suffix='desc-envelope_audio',
                     extension='.npy',
                     root=f'{path.root}/derivatives/preprocessed')

    outpath = path.copy()
    corrtype = 'spearman' if args.spearman else 'pearson'
    outpath.update(suffix='data',
                   extension='.csv',
                   root=f'{path.root}/derivatives/audioxcorr')
    outpath.update(datatype=f'mode-{mode}_cor-{corrtype}')
    outpath.mkdir(exist_ok=True)

    raw = mne.io.read_raw(rawpath.fpath, verbose=False)

    # Run or just load existing resutls
    if args.plot_only:
        df = pd.read_csv(outpath.fpath)
        df.set_index(['subject', 'label', 'electrode'], inplace=True)
    else:
        df = audioxcorr(rawpath, audiopath, lagpath,
                        method=args.method, maxlag=args.maxlag,
                        rank=args.spearman)
        df['subject'] = int(args.subject)
        df['label'] = 'Comprehension' if args.partner_audio else 'Production'
        df.set_index(['subject', 'label', 'electrode'], inplace=True)
        df.to_csv(outpath.fpath)

        # Save summary
        maxdf = df.max(axis=1).to_frame('r')
        x = np.array([int(c) for c in df.columns])
        maxids = df.values.argmax(axis=-1)
        lags = x[maxids] / raw.info['sfreq']
        maxdf['lag_sec'] = lags
        outpath.update(suffix='max', extension='.csv')
        maxdf.sort_values('r', ascending=False, inplace=True)
        maxdf.to_csv(outpath.fpath)

    # Plot
    axargs = dict(xlabel='Lag (s)',
                  ylabel='Correlation (r)',
                  title='audio xcorr')

    fig = utils.plot_results_brain(df, raw, title=axargs['title'])
    outpath.update(extension='.jpg')
    outpath.update(suffix='brain')
    fig.savefig(outpath.fpath)

    fig = utils.plot_results_summary(df, raw.info['sfreq'], **axargs)
    outpath.update(suffix='summary')
    fig.savefig(outpath.fpath)

    outpath.update(suffix='all',
                   extension='.pdf')
    utils.plot_all_results(df, outpath, plotpath, **axargs)


if __name__ == '__main__':
    parser = utils.getparser()
    parser.add_argument('-o', '--outdir', default=None)
    parser.add_argument('--method', default='fft')
    parser.add_argument('--maxlag', type=int, default=2)  # seconds
    parser.add_argument('--partner-audio', action='store_true')
    parser.add_argument('--spearman', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.method == 'lags' and args.maxlag is None:
        raise ValueError('Window must be provided for lags method')

    subjects = args.subject
    for subject in subjects:
        args.subject = f'{subject:02d}'
        try:
            main(args)
        except Exception as e:
            print(f'[ERROR] Subject {subject} failed')
            print(e)
