"""Cross-correlate the room audio (HQ) and amp audio (LQ) for alignment.

Saves output in derivatives/preprocessed/sub-*/audio/:
    - xcorr plot
    - xcorr json
    - audio
    - cached room audio envelope

Room audio (HQ) is passed first, so:
    if lag is positive, then the audio (HQ) is ahead the brain in time
      -> and that we need to push the brain by the delay
      -> so trim the audio and subtract delay from onsets
    if lag is negative, then the brain (LQ) is after the audio
      -> so either trim the brain or add delay to onsets

python code/audioalign.py -s $SUB $(jq -r ".[$(($SUB - 1))].sessions[0].ieeg" code/staging.json) $(jq -r ".[$(($SUB - 1))].sessions[0].audio" code/staging.json)
"""

import argparse
import json
import os
import pickle
import shutil

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
import pandas as pd
from scipy import signal
from scipy.io import wavfile

from util.signal import xcorr

SR_SUBS = ['03', '04', '07', '08', '11', '12']

# See https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def preprocess_highqa(x, fs, to_fs, lowcut=200, highcut=5000):
    assert x.ndim == 1
    # x = x[:fs * round(len(x) / fs)]  # trim to nearest second

    # Step 1. Bandpass the high quality audio
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=5)

    # Step 2. Downsample to same freq as clinical system
    # Number of new samples is N = n * (to_fs / fs)
    y = signal.resample(y, num=round(x.size / fs * to_fs))

    # Step 3. Take audio envelope
    envelope = np.abs(signal.hilbert(y - y.mean()))
    return envelope


def preprocess_lowqa(x):
    return np.abs(signal.hilbert(x - x.mean()))


def plot_xcorr(corr, lags, fs, k=None, data=None):
    """General function to plot cross correlations results"""
    n_data = 3 if data is not None else 0
    fig = plt.figure(constrained_layout=True, figsize=(14, 7))
    gs = GridSpec(1 + n_data, 3, figure=fig, height_ratios=[3] + [1]*n_data)
    # fig, axes = plt.subplots(1 + , 3, figsize=(14, 4))

    lagi = np.argmax(np.abs(corr))
    lag = lags[lagi] / fs
    lagfs = lags[lagi]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axvline(lag, ls='--', c='black', alpha=0.3, label=f'{lag:.3f}s')
    ax1.plot(lags / fs, corr)
    ax1.legend()
    ax1.set_xlabel('Lag (s)')
    ax1.set_ylabel('Correlation (r)')
    ax1.set_title('Full xcorr')

    ax2 = fig.add_subplot(gs[0, 1])
    point = slice(int(lagi - 30 * fs), int(lagi + 30 * fs))
    ax2.axvline(lag, ls='--', c='black', alpha=0.3)
    ax2.plot(lags[point] / fs, corr[point])
    ax2.set_xlabel('Lag (s)')
    ax2.set_title('Zoomed in')

    ax3 = fig.add_subplot(gs[0, 2])
    point = slice(int(lagi - .10 * fs), int(lagi + .10 * fs))
    ax3.axvline(lag, ls='--', c='black', alpha=0.3)
    ax3.plot(lags[point] / fs, corr[point])
    ax3.set_xlabel('Lag (s)')
    ax3.set_title('Zoomed in on peak')

    if k is not None:
        topk = np.argsort(np.abs(corr))[-k:]
        for peaklag in topk[::-1]:
            ax2.scatter(lags[peaklag] / fs, corr[peaklag], marker='x', color='r')

    if data is not None and len(data) == 2:

        ax1 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[3, :])

        # Plot HQ
        ax1.plot(np.arange(data[0].size) / fs, data[0], label='HQ')
        ax1.legend(loc='upper right')
        if lag > 0:
            ax1.axvline(abs(lag), ls='--', c='red')

        # Plot alignment
        if lag < 0:
            signal = data[1]
            xaxis = np.arange(signal.size + lagfs) / fs
            ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
            ax2.plot(xaxis, signal[abs(lagfs):], label='Aligned LQ')
        elif lag > 0:
            signal = data[0]
            xaxis = np.arange(signal.size - lagfs) / fs
            ax2 = fig.add_subplot(gs[2, :], sharex=ax3)
            ax2.plot(xaxis, signal[lagfs:], label='Aligned HQ')
        ax2.legend(loc='upper right')

        # Plot HQ
        ax3.plot(np.arange(data[1].size) / fs, data[1], label='LQ')
        ax3.legend(loc='upper right')
        if lag < 0:
            ax3.axvline(abs(lag), ls='--', c='red')

    return fig


def main(args):
    # Prepare paths
    path = BIDSPath(subject=args.subject, task=args.task, root=args.root)
    audio_path = path.copy()
    audio_path.update(root=f'{path.root}/stimuli',
                      datatype='audio',
                      suffix='audio',
                      extension='.wav',
                      check=False)
    outpath = path.copy()
    outpath.update(check=False,
                   root=f'{path.root}/derivatives/preprocessed/',
                   datatype='audio')
    outpath.mkdir(exist_ok=True)

    # Read data
    path.update(datatype='ieeg', suffix='ieeg', extension='.edf')
    raw = read_raw_bids(path, verbose=False)
    clinfs = raw.info['sfreq']

    # Prepare LQ audio
    stim_id = mne.pick_types(raw.info, stim=True)
    assert len(stim_id) == 1
    lq_channel = raw.ch_names[stim_id.item()]
    clin_audio, _ = raw[lq_channel]
    lowqa = clin_audio.squeeze()
    lowenv = preprocess_lowqa(lowqa)

    # Prepare HQ audio
    outpath.update(suffix='desc-envelope_audio', extension='.npy')
    if os.path.isfile(outpath.fpath) and not args.ignore_cache:
        print('Loading cached HQ audio envelope')
        highenv = np.load(outpath.fpath)
    else:
        highfs, highqa = wavfile.read(audio_path.fpath)
        assert highqa.ndim == 1
        if args.subject in SR_SUBS:
            print('NOTE - modifying the audio sampling rate for this subject')
            highfs = 44100 - 441
        highenv = preprocess_highqa(highqa, highfs, clinfs)
        np.save(outpath.fpath, highenv)

    # Align HQ and LQ
    smooth_win = .025  # TODO make an arg
    win_size = int(smooth_win * clinfs)  # s to clinfs
    corr, lags = xcorr(highenv, lowenv, win_size, method='fft')
    fig = plot_xcorr(corr, lags, clinfs, k=10, data=(highenv, lowenv))
    lagi = np.argmax(corr)
    lag = lags[lagi] / clinfs  # in seconds
    lag_sf = lags[lagi]

    # Save results
    result = {
        'lag_s': lag.item(),
        'lag_fs': lag_sf.item(),
    }
    outpath.update(suffix='xcorr', extension='.json')
    with open(outpath.fpath, 'w') as f:
        json.dump(result, f, indent=2)
    outpath.update(extension='.jpg')
    outpath.mkdir(exist_ok=True)
    # fig.suptitle(f'Subject {path.subject} HQLQ xcorr')
    fig.savefig(outpath.fpath)

    # Copy audio to preprocessed
    newhome = f'{path.root}/derivatives/preprocessed/'
    audio_outpath = audio_path.copy().update(root=newhome)
    shutil.copyfile(audio_path, audio_outpath)

    # Adjust event onset times and save
    events_fn = audio_path.update(extension='.csv')
    if os.path.isfile(events_fn):
        events = pd.read_csv(events_fn, index_col=0)

        # First, make sure to fix the SR mistake in onsets before shifting
        if args.subject in SR_SUBS:
            events['start'] *= 1.01
            events['end'] *= 1.01
            events['utt_onset'] *= 1.01
            events['utt_offset'] *= 1.01

        if lag > 0:
            events['start'] -= lag
            events['end'] -= lag
            events['utt_onset'] -= lag
            events['utt_offset'] -= lag
        elif lag < 0:
            events['start'] += -lag
            events['end'] += -lag
            events['utt_onset'] += -lag
            events['utt_offset'] += -lag

        events_fn.update(root=f'{path.root}/derivatives/preprocessed/')
        events.to_csv(events_fn.fpath)


def align_cli(edf_fn, wav_fn, sub, lq_channels=['DC9-REF', 'DC10-REF']):
    """Align non-BIDS edf with wav file
    """
    clinfs = 512

    path = BIDSPath(check=False,
                    subject=sub,
                    task='conversation',
                    suffix='dcs',
                    extension='.pkl',
                    root='staging')
    path.mkdir(exist_ok=True)


    def get_edf_lqs(edf_fn, lq_channels, notch=False, denoise=False):
        raw = mne.io.read_raw_edf(edf_fn, verbose=None)
        dcs = mne.pick_channels_regexp(raw.ch_names, f'DC\d+.REF')
        raw = raw.pick_channels(np.array(raw.ch_names)[dcs])
        raw.set_channel_types({ch: 'misc' for ch in raw.ch_names})
        raw.set_annotations(None)
        raw.load_data()
        print(raw)
        print(raw.info)
        assert clinfs == raw.info['sfreq'], 'Different sampling frequency'

        # print('CROPPING raw')
        # raw.crop(tmax=2500)  # NOTE

        fig = raw.plot_psd(picks='all', show=False)
        path.update(suffix='desc-psd_dcs', extension='.jpg')
        fig.savefig(path.fpath)
        plt.close()

        # Regress out other DC channels from DC9/10
        if denoise:
            dcs = mne.pick_channels_regexp(raw.ch_names, 'DC\d+')
            lqs = mne.pick_channels_regexp(raw.ch_names, 'DC(9|10)')
            for ch in lqs:
                dcs.remove(ch)
            noise = raw[dcs, :][0]
            noise = np.vstack((noise, np.ones(noise.shape[-1]))).T

            for lq in lqs:
                lqc = raw._data[lq]
                b, _, _, _ = np.linalg.lstsq(noise, lqc, rcond=None)
                yhat = noise @ b
                raw._data[lq] = lqc - yhat

            print(f'denoised using {len(dcs)} DC channels')

        # Notch filter
        if notch:
            raw.load_data()
            raw.notch_filter(freqs=[60, 120, 180], notch_widths=5, n_jobs=1, picks='all')

        data, _  = raw[lq_channels]

        # data = data[:, :data.shape[1]//2]  # 1st half
        # data = data[:, data.shape[1]//2:]  # 2nd half

        if not True:  # detrend with moving average
            window = int(512*.5)
            for i in range(len(data)):
                mu = pd.Series(data[i]).rolling(window).mean().values
                mu[np.isnan(mu)] = 0
                data[i,:] -= mu
            print('Detrended LQ with moving average')

        # Plot signal
        print('Plotting signals')
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        for i, ch in enumerate(lq_channels):
            lowqa = data[i]
            axes[i].plot(np.arange(lowqa.size) / raw.info['sfreq'],
                         lowqa, label=ch)
            axes[i].legend(loc='upper left')
        path.update(suffix='dcs', extension='.jpg')
        fig.savefig(path.fpath)
        plt.close()

        # Preprocess
        lqs = [preprocess_lowqa(data[i]) for i in range(len(data))]

        if not True:  # detrend with moving average
            window = int(512*.5)
            for lq in lqs:
                mu = pd.Series(lq).rolling(window).mean().values
                mu[np.isnan(mu)] = 0
                lq -= mu
            print('Detrended envelope with moving average')

        return lqs

    lqs = None
    highfs, highqa = wavfile.read(wav_fn)
    if args.subject in SR_SUBS:
        print('NOTE - modifying the audio sampling rate for this subject')
        highfs = 44100 - 441
    print('Audio length', len(highqa) / highfs, 'seconds', highqa.ndim)
    if edf_fn is not None:
        lqs = get_edf_lqs(edf_fn, lq_channels, notch=not True)

        path.update(suffix='envs', extension='.pickle')
        envfn = path.fpath
        if os.path.isfile(envfn):
            print('Loading env files')
            with open(envfn, 'rb') as f:
                highenv = pickle.load(f)
        else:
            print('Computing envelope')
            if highqa.ndim > 1:
                highenv1 = preprocess_highqa(highqa[:, 0], highfs, clinfs)
                highenv2 = preprocess_highqa(highqa[:, 1], highfs, clinfs)
                highenv = (highenv1, highenv2)
            else:
                highenv1 = preprocess_highqa(highqa, highfs, clinfs)
                highenv = (highenv1,)
            with open(envfn, 'wb') as f:
                pickle.dump(highenv, f)
    else:
        # xcorr between audio files. Doesn't need preprocessing
        assert highqa.ndim > 1
        assert highfs == 16000, 'Resample audio files first'
        clinfs = highfs
        lqs = (highqa[:, 1], )
        highenv = (highqa[:, 0], )
        lq_channels = ['Audio1']

    # highenv = lqs  # NOTE for autocorrelations

    print('Computing xcorrs')
    for i, lqc in enumerate(lq_channels):
        lowenv = lqs[i]
        for j in range(len(highenv)):
            corr, lags = xcorr(highenv[j], lowenv, method='fft', rank=True)
            fig = plot_xcorr(corr, lags, clinfs, k=10, data=(highenv[j], lowenv))
            if False:  # interact
                plt.show()
                # breakpoint()
            if not False:
                audio = np.zeros((max(len(highenv[j]), len(lowenv)), 2))
                audio[:len(highenv[j]), 0] = highenv[j] / highenv[j].max()
                audio[:len(lowenv), 1] = lowenv / lowenv.max()
                path.update(suffix=f'lq-{lqc}_hq-C{j}_hqlq', extension='.wav')
                wavfile.write(path.fpath, 512, audio)
            path.update(suffix=f'desc-xcorr_lq-{lqc}_hq-C{j}_xcorr',
                        extension='.jpg')
            fig.savefig(path.fpath)
            plt.close()
            lagi = np.argmax(np.abs(corr))
            lag = lags[lagi] / clinfs  # in seconds
            lag_sf = lags[lagi]
            print(lqc, j, lag, lag_sf, np.abs(corr).max())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=int, default=1)
    parser.add_argument('-t', '--task', default='conversation')
    parser.add_argument('-r', '--root', default='dataset')
    parser.add_argument('--ignore-cache', action='store_true')
    parser.add_argument('edf', nargs='?')
    parser.add_argument('wav', nargs='?')

    args = parser.parse_args()
    args.subject = f'{args.subject:02d}'
    if args.edf is not None and args.wav is not None:
        align_cli(args.edf, args.wav, args.subject)
    elif args.edf is not None and args.wav is None:
        if args.edf.endswith('.wav'):
            align_cli(None, args.edf, args.subject)
        else:
            print('Need additional input')
    else:
        main(args)
