"""Plot the electrodes on the brain and create thumbnails for plotting.

See:
https://nilearn.github.io/modules/generated/nilearn.plotting.plot_markers.html
https://matplotlib.org/stable/tutorials/colors/colormaps.html
https://nilearn.github.io/auto_examples/01_plotting/plot_colormaps.html

TODO
- try plotting a surface plot then add_markers:
https://nilearn.github.io/auto_examples/01_plotting/plot_demo_more_plotting.html
"""
import argparse
import os

import utils
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from nilearn import plotting
from mne_bids import BIDSPath, read_raw_bids

def plot_values(raw, electrodes, values,
                vmin=None, vmax=None, threshold=None,
                figure=None, axis=None, title=None,
                cmap=plt.cm.viridis_r,
                mode='lzry', colorbar=True):
    """Plot a brain map of the given electrodes and their values
        TODO: use kwargs instead?
    """
    if isinstance(raw, BIDSPath):
        raw = utils.getraw(raw)
    ch2loc = {ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}
    coords = np.vstack([ch2loc[ch] for ch in electrodes])
    coords *= 1000  # convert to mm

    fig = plt.figure(figsize=(12, 6)) if figure is None else figure
    plotting.plot_markers(values,
                          coords,
                          node_size=50,
                          node_vmin=vmin,
                          node_vmax=vmax,
                          node_cmap=cmap,
                          node_threshold=threshold,
                          display_mode=mode,
                          figure=fig,
                          axes=axis,
                          title=title,
                          annotate=True,
                          colorbar=colorbar)

    return fig


def plot_brain(raw, path, ignore_cache=False):
    """Unusued. Plot using MNE."""
    import matplotlib.pyplot as plt
    sub_dir = '/Applications/freesurfer/subjects'  # NOTE hardcoded

    # TODO - try to only plot once, replace axis?
    for channel in raw.ch_names:
        path.update(suffix=f'desc-thumbnail_{channel}')
        if os.path.isfile(path) and ignore_cache:
            continue

        info = raw.info.copy().pick_channels([channel])
        locs = info['chs'][0]['loc'][:3]
        if np.all(locs == 0) or np.isnan(locs).any():
            print(f'{channel} has no location, will not plot')
            continue

        fig = mne.viz.plot_alignment(info,
                                     subject='fsaverage',
                                     coord_frame='mri',
                                     surfaces=['pial'],
                                     subjects_dir=sub_dir)
        mne.viz.set_3d_view(fig, azimuth=180)
        xy, im = mne.viz.snapshot_brain_montage(fig, info)
        xy_pts = np.vstack([xy[ch] for ch in info['ch_names']])
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im)
        ax.scatter(*xy_pts.T, s=1, c='white')
        plt.savefig(path.fpath, dpi=100)
        plt.close()

    # Plots all electrodes
    path.update(suffix='desc-photo_all')
    if not os.path.isfile(path) or not ignore_cache:
        fig = mne.viz.plot_alignment(raw.info,
                                     subject='fsaverage',
                                     coord_frame='mri',
                                     surfaces=['pial'],
                                     subjects_dir=sub_dir)
        mne.viz.set_3d_view(fig, azimuth=180)
        _, im = mne.viz.snapshot_brain_montage(fig, raw.info)
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im)
        plt.savefig(path.fpath, dpi=im.shape[0])
        plt.close()


def plot_coords(names, coords, path, ignore_cache=False):
    """Plot electrodes together and separately on a glass brain using nilearn.

    NOTE - can choose the display_mode based on type of electrode instead
    """

    path.update(suffix='desc-photo_all')
    plotting.plot_markers(np.ones(len(coords)),
                          coords,
                          node_size=50,
                          display_mode='lzry',
                          colorbar=False,
                          output_file=path.fpath)

    for i, channel in enumerate(names):
        path.update(suffix=f'desc-thumbnail_{channel}')
        if os.path.isfile(path) and ignore_cache:
            continue

        if not coords[i].any():
            print(f'Skipping {channel}, coords are 0.')
            continue

        # Create a 100x100 image
        fig = plt.figure(figsize=(2.0, 1.0))
        # Can't seem to plot just one location, so workaround is to plot
        # the same one twice
        plotting.plot_markers(np.ones(2),
                              np.repeat(coords[i].reshape(1, 3), 2, axis=0),
                              node_size=30,
                              node_cmap='Reds',
                              display_mode='lr',
                              figure=fig,
                              annotate=False,
                              colorbar=False)
        fig.savefig(path.fpath, dpi=100)
        plt.close()


def main(args):
    # matplotlib.use("Agg")

    path = BIDSPath(subject=args.subject, task=args.task, root=args.root)

    path.update(datatype='ieeg', suffix='ieeg', extension='.edf')
    raw = read_raw_bids(path)
    raw = raw.pick_types(ecog=True, stim=True, exclude='bads')

    path.update(root=f'{path.root}/derivatives/figures',
                task=None,
                check=False,
                datatype=None,
                suffix=None,
                acquisition='render',
                extension='.jpg')
    path.mkdir(exist_ok=True)

    # Get coordinates
    coords = np.vstack([ch['loc'][:3] for ch in raw.info['chs']]) * 1000
    channels = np.array(raw.ch_names)
    mask = np.any(coords, axis=1)
    channels, coords = channels[mask], coords[mask]

    # Plot
    plot_coords(channels, coords, path)
    # plot_brain(raw, path, args.ignore_cache)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subject', type=int, default=1)
    parser.add_argument('-t', '--task', default='conversation')
    parser.add_argument('-r', '--root', default='dataset')
    parser.add_argument('--ignore-cache',
                        action='store_true',
                        default=False,
                        help='Ignore cached stages.')

    args = parser.parse_args()
    args.subject = f'{args.subject:02d}'
    main(args)
