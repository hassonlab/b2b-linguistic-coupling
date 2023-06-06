"""Preprocess raw data by despiking, detrending, and rereferencing.
"""
import argparse
import json
import os
import pickle

import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from mne_bids import BIDSPath, read_raw_bids

from scipy import ndimage, signal, stats, interpolate, linalg


def plot(raw, events=None, title=None, block=True, proj=False):
    """Common function to plot the the signals
    """
    raw.plot(events=events,
             duration=10,
             n_channels=len(raw.ch_names),
             clipping=None,
             proj=False,
             block=True,
             title=title)


def find_spike_events(raw, iqr_mult, dil_len):
    """Find spikes in the data.
    """
    raw.load_data()

    # Find spike artifacts
    picks = mne.pick_types(raw.info, ecog=True)
    data = raw._data[picks, :]
    meds = np.median(data, axis=-1, keepdims=True)
    iqrs = stats.iqr(data, axis=-1)
    spike_mask = np.abs(data - meds) > (iqrs.reshape(-1, 1) * iqr_mult)

    # Dilate
    if dil_len > 0:
        f = np.ones(int(dil_len * raw.info['sfreq']))
        spike_mask = signal.filtfilt(f, 1, spike_mask) > 0

    return spike_mask


def despike(raw, iqr_mult, dil_len, auto=False):
    # TODO add permission range
    spike_mask = find_spike_events(raw, iqr_mult, dil_len)

    # Plot and draw spikes
    onsets = np.nonzero(spike_mask.sum(axis=0) > 3)[0]
    events = np.zeros((onsets.size, 3), dtype=int)
    events[:, 0] = onsets

    # if not auto:
    #     plot(raw, events, title='Spikes before interpolation')

    despiked_raw = interpolate_spikes(raw, spike_mask)
    # if not args.auto:
    #     plot(despiked_raw, events, title='Despiked data')

    return despiked_raw


def interpolate_spikes(raw, spike_mask):
    """Find and interpolate spikes
    """

    picks = mne.pick_types(raw.info, ecog=True)
    for i in picks:
        spikes = spike_mask[i]
        x = np.nonzero(~ spikes)[0]  # good indices
        y = raw._data[i, ~ spikes]        # good values
        x_spikes = spikes.nonzero()[0]

        # Pchip
        xnew = interpolate.pchip_interpolate(x, y, x_spikes)
        raw._data[i, spikes] = xnew

        # General function
        # f = interpolate.interp1d(x, y, kind='cubic', axis=-1)
        # xnew = f(x_spikes)
        # raw._data[i, spikes] = xnew

    return raw


def find_artifacts(raw, iqr_mult, dil_len, n_bads, cls_th):
    """Find spikes in the data and set them as annotations.
    """
    # Find spikes
    spike_mask = find_spike_events(raw, iqr_mult, dil_len)

    # Keep only spikes that are in n_bads channels
    spike_mask = spike_mask.sum(axis=0) >= n_bads

    # Label each cluster with a unique number
    components, n_arts = ndimage.label(spike_mask)
    art_keep = np.arange(1, 1 + n_arts)

    # Compute the "size" of each artifact
    if cls_th is not None:
        data = raw._data
        art_sizes = np.zeros(n_arts)
        for i in range(1, n_arts + 1):
            art_sizes[i-1] = stats.iqr(data[:, components == i], axis=-1).sum()**2
        art_sizes = stats.zscore(art_sizes)

        # Only keep artifacts that pass the cluster threshold
        art_keep = np.arange(1, 1 + n_arts)[art_sizes > cls_th]
    print(f'Found {len(art_keep)} artifacts')

    # Translate artifact blobs into index coordinates
    fs = raw.info['sfreq']
    onsets = np.zeros(art_keep.size)
    durations = np.zeros(art_keep.size)
    for i, c in enumerate(art_keep):
        comp = (components == c).nonzero()[0]
        onsets[i] = comp[0] / fs
        durations[i] = (comp[-1] - comp[0]) / fs

    descriptions = ['bad spike'] * onsets.shape[0]
    spike_annot = mne.Annotations(onsets, durations, descriptions)
    raw.set_annotations(spike_annot)

    print(f'Average artifact dur is {raw.annotations.duration.mean():.3f}s')
    return raw


def do_ica_sk(clean_raw, path, orig, verify_recover=False, seed=42):
    """
    clean_raw means despiked
    orig is the untouched raw signal
    """
    raw = orig
    outpath = path.copy()

    outpath.update(suffix='desc-ica_object', extension='.pkl')
    if os.path.isfile(outpath.fpath):
        print('Loading prefit ICA object')
        with open(outpath.fpath, 'rb') as f:
            ica = pickle.load(f)
    else:
        # 1. Clean the data
        freqs = [raw.info['line_freq'] * m for m in range(1, 4)]
        raw_notch = raw.copy()
        raw_notch.load_data()
        raw_notch.notch_filter(freqs=freqs,
                               picks='ecog',
                               notch_widths=2,  # in Hz
                               n_jobs=1)

        # Detrend with highpass filter as recommended by MNE
        raw_filt = raw_notch.filter(l_freq=1.0, h_freq=None)

        # Remove bad segments
        data = raw_filt.get_data(picks='data',
                                 reject_by_annotation='omit')

        # 2. Fit ICA
        print('Fitting FastICA')
        ica = decomposition.FastICA(whiten=True,
                                    max_iter=200,
                                    random_state=seed)
        ica = ica.fit(data.T)  #  requires (n_samples, n_features), so flip
        print(f'Fitting took {ica.n_iter_} iterations out of {ica.max_iter}')
        with open(outpath.fpath, 'wb') as f:
            pickle.dump(ica, f)

    mixing_matrix = ica.mixing_
    unmixing_matrix = ica.components_

    # Verify mixing matrix recovers original data
    if verify_recover:
        X = data.copy().T
        mixed = (X - ica.mean_) @ unmixing_matrix
        unmixed = mixed @ mixing_matrix + ica.mean_
        assert np.allclose(X, unmixed)
        # np.allclose(data.T, ica.inverse_transform(ica.transform(data.T)))

    # Plot mixing matrix
    outpath.update(suffix='desc-mixing_matrix', extension='.jpg')
    x = np.arange(len(mixing_matrix), dtype=int)
    order = x
    # chis = stats.chisquare(mixing_matrix, axis=1).statistic
    # order = np.argsort(chis)
    sorted_mixing = mixing_matrix[:, order]
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.matshow(sorted_mixing, cmap='seismic')
    absmax = np.abs(mixing_matrix).max()
    im.set_clim(-absmax, absmax)
    # ax.locator_params(nbins=len(mixing_matrix) // 2)
    # ax.set_xticks(order[::2])
    # ax.set_xticklabels(map(str, ax.get_xticks()))
    # ax2 = ax.twinx()
    # ax2.set_xticks(range(1, x.size, 2))
    # ax2.set_xticklabels(map(str, ax2.get_xticks()))
    # ax2.tick_params(labeltop='on', labelright='off')
    fig.colorbar(im, ax=ax)
    fig.savefig(outpath.fpath)
    plt.show()
    plt.close()

    # Choose components to reject
    reject_ids = input('Components to reject: ')
    if reject_ids.strip() != '':
        reject_ids = [int(i.strip()) for i in reject_ids.split(',')]
    else:
        reject_ids = []
    outpath.update(suffix='desc-rejected_list', extension='.json')
    with open(outpath.fpath, 'w') as f:
        json.dump(reject_ids, f)

    # TODO - plot source time course
    # info = mne.create_info(len(mixing_matrix), len(mixing_matrix))
    # mixing_raw = mne.io.RawArray(mixing_matrix.T, info)
    # mixing_raw.plot(n_channels=len(mixing_matrix), clipping=None, block=True)

    # 3. Apply to clean signal and return raw
    if len(reject_ids):
        for i in reject_ids:
            mixing_matrix[:, i] = 0
        clean_data = clean_raw.get_data(picks='data')
        t = ica.transform(clean_data.T, copy=False)
        reref_data = np.dot(t, mixing_matrix.T)  # inverse transform
        reref_data += ica.mean_  # unwhiten

        picks = mne.pick_types(raw.info, ecog=True)
        clean_raw[picks, :] = reref_data.T

    clean_raw.set_annotations(None)

    return clean_raw


def do_ica(raw, path, alg='fastica', verify_mixing=True, auto=False):

    # Load ICA if already trained
    fname = path.update(suffix=f'{alg}_ica', extension='.fif').fpath
    if os.path.isfile(fname):
        print('Loading previous ICA run')
        ica = mne.preprocessing.read_ica(fname, verbose=True)
    else:
        # Notch filter
        freqs = [raw.info['line_freq'] * m for m in range(1, 4)]
        raw_notch = raw.copy().notch_filter(freqs=freqs,
                                            picks='ecog',
                                            notch_widths=2,  # in Hz
                                            n_jobs=1)
        raw_notch = raw_notch.pick_types(ecog=True)

        # Detrend with highpass filter as recommended by MNE
        raw_filt = raw_notch.filter(l_freq=1.0, h_freq=None)

        # EEGLab whitininng sphere
        sphere = 2 * np.linalg.pinv(linalg.sqrtm(np.cov(raw_filt._data)))
        noise_cov = mne.Covariance(sphere, raw_filt.ch_names,
                                   raw_filt.info['bads'], [],
                                   len(raw_filt.ch_names))

        n_channels = len(raw_filt.ch_names)
        ica = mne.preprocessing.ICA(
                method=alg,
                n_components=n_channels,
                max_iter='auto',
                noise_cov=noise_cov,
                fit_params=dict(extended=True) if alg == 'infomax' else {},
                verbose=True,
                random_state=42)

        ica.fit(raw_filt, reject_by_annotation=True)
        ica.save(fname)

    if verify_mixing:

        rawc = raw.copy()
        data = rawc.pick_types(ecog=True).get_data()
        # data /= ica.pre_whitener_  # standarize if not noise_cov
        data = ica.pre_whitener_ @ data  # cov
        data -= ica.pca_mean_.reshape(-1, 1)  # center
        unmixing = ica.pca_components_ @ ica.unmixing_matrix_
        data = unmixing @ data
        mixing = ica.mixing_matrix_ @ ica.pca_components_.T
        data = mixing @ data  # (dxd * nxd), components on rows
        data += ica.pca_mean_.reshape(-1,1)
        winv = np.linalg.pinv(ica.pre_whitener_, rcond=1e-14)
        data = winv @ data
        # data = data * ica.pre_whitener_ if not noise_cov
        assert np.allclose(rawc._data, data)

        # componets zero mean?

        mixed = (ica.pca_components_ @ ica.unmixing_matrix_ @ ica.pre_whitener_) @ rawc.get_data() - unmixing @ ica.pca_mean_[:, None]
        unmixed = np.linalg.inv(ica.pre_whitener_) @ ica.mixing_matrix_ @ ica.pca_components_.T @ mixed + winv @ ica.pca_mean_[:, None]
        print(np.allclose(rawc._data, unmixed))

    # Get mixing matrix
    # mixing = ica.mixing_matrix_ @ ica.pca_components_.T
    mixing = np.linalg.inv(ica.pre_whitener_) @ ica.mixing_matrix_ @ ica.pca_components_.T
    mixing = mixing.T
    fname = path.update(suffix=f'{alg}_mixing', extension='.npy').fpath
    np.save(fname, mixing)

    # Plot mixing matrix
    chis = stats.chisquare(mixing, axis=1).statistic
    x = np.argsort(chis)
    sorted_mixing = mixing[:, x]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sorted_mixing, cmap='Spectral')
    ax.set_xticks(range(x.size))
    ax.set_xticklabels(map(str, x))
    plt.xticks(range(x.size), rotation=90)
    # ax.set(xticks=range(x.size), xticklabels=map(str, x), rotation= 5)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fname = path.update(suffix=f'{alg}_sorted_mixing', extension='.jpg').fpath
    plt.savefig(fname)
    # plt.show()
    plt.close()

    exit()  # NOTE

    if not args.auto:
        ica.plot_sources(raw_filt, block=True)
        print('Chose to exclude ICs', ica.exclude)
        # fname = path.update(suffix='ica', extension='.fif').fpath
        # ica.save(fname)
    reconst_raw = ica.apply(raw)  # despiked data

    return reconst_raw


def do_car(raw, **kwargs):
    referenced = raw.set_eeg_reference(ref_channels='average', ch_type='ecog')
    return referenced


def ica_exclusions(raw, path):
    """
    TODO: do a high pass filter first
    """
    fname = path.update(suffix='desc-artifacts_spikes', extension='.csv').fpath
    if args.ignore_cache or not os.path.isfile(fname):
        raw = find_artifacts(raw, iqr_mult=5, dil_len=0.8, n_bads=5, cls_th=-1)
        raw.annotations.save(fname, overwrite=True)
    else:
        print(f'Loading annotations from {fname}')
        try:
            annots = mne.read_annotations(fname)
            raw.set_annotations(annots)
        except IndexError:
            print('[WARNING] loaded annotations with 0 segments')

    # Plot them for manual inspection
    if not args.auto:
        plot(raw, title='Auto-detected artifacts')
        path.update(suffix='desc-artifacts_manual', extension='.csv')
        raw.annotations.save(path.fpath, overwrite=True)

    return raw.annotations


def rereference(raw, method, **kwargs):
    if method == 'car':
        return do_car(raw, **kwargs)

    # Automatically find artifacts to ignore when doing ICA
    path = kwargs['path']
    orig = kwargs['orig']
    annots = ica_exclusions(orig, path)
    raw.set_annotations(annots)

    if method in ['infomax', 'picard']:
        return do_ica(raw, alg=method, **kwargs)
    elif method == 'fastica':
        return do_ica_sk(raw, **kwargs)
    else:
        raise ValueError('Re-referencing method must be ica or car, got %s',
                         method)


def main(args):
    path = BIDSPath(subject=args.subject,
                    task=args.task,
                    root=args.root)

    # Load data
    path.update(datatype='ieeg', suffix='ieeg', extension='.edf')
    raw = read_raw_bids(path, verbose=False)
    raw.set_annotations(None)

    # raw.crop(0, 1000)  # NOTE

    # Prepare path
    path.update(root=f'{path.root}/derivatives/preprocessed',
                datatype=args.refalg,
                check=False)
    outpath = path.copy()
    outpath.mkdir()

    # Select and remove bad channels
    outpath.update(suffix='desc-bads_list', extension='.json')
    if os.path.isfile(outpath.fpath):
        with open(outpath.fpath, 'r') as f:
            bads = json.load(f)
            raw.info['bads'].extend(bads)
            print(f'Removed {len(bads)} pre-selected bad channels')
    elif args.auto:
        print('[WARN] Skipping bad channel manual selection proccess')
    else:
        freqs = [raw.info['line_freq'] * m for m in range(1, 4)]
        tmp_raw = raw.copy()
        tmp_raw.pick_types(ecog=True)
        tmp_raw.load_data()
        tmp_raw.notch_filter(freqs=freqs, notch_widths=5)
        plot(tmp_raw, title='Select bad channels')
        bads = tmp_raw.info['bads']
        with open(outpath.fpath, 'w') as f:
            json.dump(bads, f)
        raw.info['bads'].extend(bads)

    # Keep only good channels
    raw = raw.pick_types(ecog=True, stim=True, exclude='bads')

    # Interpolate spikes
    despike_raw = despike(raw.copy(), iqr_mult=4, dil_len=0.10, auto=args.auto)
    # despike_raw = raw.copy()
    # despike_raw.load_data()

    # Re-reference data
    reref_raw = rereference(despike_raw, method=args.refalg, path=path, orig=raw)

    # Notch it
    freqs = [raw.info['line_freq'] * m for m in range(1, 4)]
    reref_raw.notch_filter(freqs=freqs, notch_widths=2, n_jobs=1)
    if not args.auto:
        plot(reref_raw, title='Re-referenced and notched')

    # Save it
    reref_raw.set_annotations(None)  # Note, Remove annotation before saving
    fname = path.update(suffix='desc-preproc_ieeg', extension='.fif').fpath
    reref_raw.save(fname, overwrite=True)

    # Extract frequency bands
    bands = {
        "alpha":     (8, 13),
        "beta":      (13, 30),
        "theta":     (4, 8),
        "gamma":     (30, 55),
        "highgamma": (70, 200)
    }
    iir_params = dict(order=4, ftype='butter')
    for band, freqs in bands.items():
        band_raw = reref_raw.copy()
        band_raw = band_raw.filter(*freqs,
                                   picks='data',
                                   method='iir',
                                   iir_params=iir_params)
        band_raw = band_raw.apply_hilbert(envelope=True)
        path.update(suffix=f'desc-{band}_ieeg')
        band_raw.save(path.fpath, overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=int, default=1)
    parser.add_argument('-t', '--task', default='conversation')
    parser.add_argument('-r', '--root', default='dataset')
    parser.add_argument('--ignore-cache', action='store_true', default=False,
                        help='Ignore cached stages.')
    parser.add_argument('--auto', action='store_true',
                        help='Skip any user interaction steps.')
    parser.add_argument('--refalg', default='car')
    parser.add_argument('--seed', type=int, default=42)  # NOTE unused
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    args.subject = f'{args.subject:02d}'
    main(args)
