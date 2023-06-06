"""Trigger average onsets of words for each electrode.
"""

import os
import json
import argparse

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import utils
import audioalign

from mne_bids import BIDSPath
from scipy.stats import zscore


# def take_slices(x, onsets, tmin, tmax, axis=0, standarize=False):
#     """Extract slices (or 'epochs') from an array.

#     x: ndarray
#     onsets: the midpoint of each epochs
#     tmin: how far to go back before onset (should be negative)
#     tmax: how far to go forward after onset
#     axis: which direction to aggregate
#     zscore: whether to zscore the epochs

#     returns ndarray of size (len(onsets), ..., tmax - tmin + 1)
#     """
#     chunks = []
#     for i in onsets:
#         start, end = i + tmin, i + tmax
#         if start >= 0 and end < x.shape[axis]:
#             chunk = x.take(np.arange(start, end + 1), axis=axis)
#             chunks.append(chunk)
#     chunks = np.array(chunks).squeeze()
#     assert chunks.size > 0, 'No events were able to be chunked'

#     if standarize:
#         chunks = zscore(chunks, axis=-1)
#         # NOTE - this operation may introdce nan's

#     return chunks


def plot_chunks(chunks,
                tmin,
                tmax,
                clinfs=512,
                axis=0,
                reduction='mean',
                filename=None,
                title=None,
                ch_name=None,
                labels=[]):
    if not isinstance(chunks, list):
        chunks = [chunks]
    if not len(labels):
        labels = [None] * len(chunks)

    fig, ax = plt.subplots()
    x = np.arange(tmin, tmax + 1) / clinfs * 1000
    for data, label in zip(chunks, labels):
        if reduction == 'mean':
            data = data.mean(axis=axis)
        ax.plot(x, data, label=label)
    ax.axvline(0, ls='--', c='black', alpha=0.3)
    ax.set_xlabel('Onset (ms)')
    ax.set_ylabel('Average signal (z)')
    ax.set_title(title)
    if len(labels):
        ax.legend(loc='lower right')
    if filename is not None:
        fig.savefig(filename)
    plt.close()

    return fig


def triggeraverage_(raw, speaker, df, tmin_sf, tmax_sf, baseline=None):

    chunks = [{
        'label': 'Production',
        'mask': (df.speaker_id == speaker).values
    }, {
        'label': 'Comprehension',
        'mask': (df.speaker_id != speaker).values
    }]

    raw = raw.load_data()
    data = raw._data.T.copy()  # NOTE NOTE NOTE
    # if baseline is None:
    #     data = zscore(raw._data, axis=-1)

    print('Preparing data')
    for chunk in chunks:
        mask = chunk['mask']
        if not mask.any():
            print(f'No events found for {chunk["label"]}')
            continue
        sub_df = df[mask]

        onsets = raw.time_as_index(sub_df.start.values)
        signals = utils.epoch(data, tmax_sf - tmin_sf + 1, onsets)
        signals = zscore(signals, axis=-1)
        # signals = take_slices(raw._data, onsets, tmin_sf, tmax_sf, axis=-1, standarize=True)
        chunk['signals'] = signals
        chunk['n_words'] = len(onsets)

    # Average over words (the evoked response over epochs)
    for i, channel in enumerate(raw.ch_names):
        for chunk in chunks:
            chunk[channel] = chunk['signals'][:, i, :].mean(axis=0)

    # Aggregate results
    dfs = []
    x = np.arange(tmin_sf, tmax_sf + 1)
    exclude = ['signals', 'mask', 'label', 'n_words']
    for chunk in chunks:
        df = pd.DataFrame.from_records(chunk, exclude=exclude).T
        df.columns = x
        df['subject'] = speaker
        df['label'] = chunk['label']
        df['electrode'] = df.index.values
        df.set_index(['subject', 'label', 'electrode'], inplace=True)
        dfs.append(df)
    df = pd.concat(dfs)

    return df


def triggeraverage(path, outpath, tmin, tmax, **kwargs):

    # Load data
    raw = utils.getraw(path)
    df = utils.gettranscript(path)
    tmin_sf, tmax_sf = raw.time_as_index([tmin, tmax])

    # LQ trigger average
    onsets = raw.time_as_index(df.start.values)
    stim_id = mne.pick_types(raw.info, stim=True)
    assert len(stim_id) == 1, f'Found {stim_id.size} LQ channels'
    lq_channel = raw.ch_names[stim_id.item()]
    dc9, _ = raw[lq_channel]
    dc9 = audioalign.preprocess_lowqa(dc9.squeeze())
    chunks = utils.epoch(dc9, tmax_sf - tmin_sf + 1, onsets)
    outpath.update(suffix='desc-lowqa_trigger', extension='.jpg')
    plot_chunks(chunks, tmin_sf, tmax_sf,
                filename=outpath.fpath,
                title='LQ Audio Trigger Average')

    # HQ audio envelope trigger average
    # the onsets from the events file are aligned with the brain, and the
    # audio has not been synnced with it. So, we need to shift them before
    highenv = utils.getaudioenv(path)
    lag_s = utils.getbrainlag(path)['lag_s']

    onsets = raw.time_as_index(df.start.values + lag_s)
    chunks = utils.epoch(highenv, tmax_sf - tmin_sf + 1, onsets)
    outpath.update(suffix='desc-highqa_trigger', extension='.jpg')
    plot_chunks(chunks, tmin_sf, tmax_sf,
                filename=outpath.fpath,
                title='HQ Audio Trigger Average')

    # Per electrode trigger average
    speaker = int(path.subject)
    results_df = triggeraverage_(raw, speaker, df, tmin_sf, tmax_sf, **kwargs)

    return results_df


def main(args):
    # Prepare paths
    path = BIDSPath(subject=args.subject,
                    task=args.task,
                    root=args.root,
                    check=False)

    outpath = path.copy()
    outpath.update(suffix='desc-all_trigger',
                   extension='.csv',
                   root=f'{args.root}/derivatives/trigger')
    outpath.mkdir(exist_ok=True)

    # Load data
    raw = utils.getraw(path)

    if os.path.isfile(outpath.fpath) and not args.ignore_cache:
        df = pd.read_csv(outpath.fpath)
        df.set_index(['subject', 'label', 'electrode'], inplace=True)
    else:
        df = triggeraverage(path, outpath, args.tmin, args.tmax,
                            baseline=args.baseline)
        outpath.update(suffix='desc-all_trigger', extension='.csv')
        df.to_csv(outpath.fpath)

        # Save summary
        maxdf = df.max(axis=1).to_frame('z')
        x = np.array([int(c) for c in df.columns])
        maxids = df.values.argmax(axis=-1)
        lags = x[maxids] / raw.info['sfreq']
        maxdf['lag_sec'] = lags
        outpath.update(suffix='desc-max_trigger', extension='.csv')
        maxdf.sort_values('z', ascending=False, inplace=True)
        maxdf.to_csv(outpath.fpath)

    # Plot
    axargs = dict(xlabel='Onset (s)',
                  ylabel='Average signal (z)',
                  title='trigger average')

    fig = utils.plot_results_brain(df, raw, title=axargs['title'])
    outpath.update(suffix=f'desc-brain_trigger', extension='.jpg')
    fig.savefig(outpath.fpath)

    fig = utils.plot_results_summary(df, raw.info['sfreq'], **axargs)
    outpath.update(suffix=f'desc-summary_trigger')
    fig.savefig(outpath.fpath)

    outpath.update(suffix=f'desc-all_trigger', extension='.pdf')
    utils.plot_all_results(df, outpath, utils.getplotpath(path), **axargs)


if __name__ == '__main__':
    parser = utils.getparser()
    parser.add_argument('--tmin', type=float, default=-1)  # seconds
    parser.add_argument('--tmax', type=float, default=1)
    parser.add_argument('--baseline', type=str, default=None)  # 'zscore' epoch
    parser.add_argument('--ignore-cache', action='store_true')

    args = parser.parse_args()
    subjects = args.subject
    for subject in subjects:
        args.subject = f'{subject:02d}'
        main(args)
