import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from plotbrain import plot_values
from utils import DCCHS, DIV_CMAP, SEQ_CMAP, get_rois

# > Plotting helpers


def formatim(
    fig,
    ax,
    im,
    lags,
    xl="listener lag (s)",
    yl="speaker lag (s)",
    cl="(r)",
    title=None,
    cbar_loc="v",
    cbar_sym=not True,
):
    """Format a matrix plotted with imshow."""
    if lags is not None:
        zeroi = (lags == 0).nonzero()[0].item()
        ax.axvline(zeroi, c="k", alpha=0.1)
        ax.axhline(zeroi, c="k", alpha=0.1)
        ax.axline((1, 1), slope=1, c="k", alpha=0.1)
        tickids = list(range(0, lags.size, lags.size // 4))
        ticklbs = lags[:: lags.size // 4]
        ax.set_yticks(tickids)
        ax.set_yticklabels(ticklbs)
        ax.set_xticks(tickids)
        ax.set_xticklabels(ticklbs)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.spines.right.set_visible(True)
    ax.spines.top.set_visible(True)
    # cax = fig.add_axes([ax.get_position().x1+0.01,
    #                     ax.get_position().y0,
    #                     0.03,
    #                     ax.get_position().height])
    cbar = None
    if cbar_loc == "v":
        cbar = fig.colorbar(im, ax=ax, orientation="vertical")  # , shrink=0.85)
        cbar.ax.set_ylabel(cl)
        clims = im.get_clim()
        cbar.set_ticks((clims[0], 0, clims[-1]))
        # cbar.set_ticks([cbar.get_ticks()[0], 0, cbar.get_ticks()[-1]])
    elif cbar_loc == "h":
        cbar = fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal"
        )  # , shrink=0.6)
        cbar.ax.set_xlabel(cl, rotation=0)
        cbar.set_ticks([cbar.get_ticks()[0], 0, cbar.get_ticks()[-1]])
    lims = im.get_clim()
    if cbar_loc is not None and cbar_sym:
        maxlim = np.abs(lims).max()
        im.set_clim((-maxlim, maxlim))
        cbar.set_ticks((-maxlim, maxlim))
    return cbar


def formatenc(ax):
    ax.axvline(0, c="k", ls="--", alpha=0.1)


# For drawing contours
# from: https://stackoverflow.com/a/60098944
def get_all_edges(bool_img):
    """
    Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
    The returned array edges has he dimension (n, 2, 2).
    Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
    Note that the indices of a pixel also denote the coordinates of its lower left corner.
    """
    edges = []
    ii, jj = np.nonzero(bool_img)
    for i, j in zip(ii, jj):
        # North
        if j == bool_img.shape[1] - 1 or not bool_img[i, j + 1]:
            edges.append(np.array([[i, j + 1], [i + 1, j + 1]]))
        # East
        if i == bool_img.shape[0] - 1 or not bool_img[i + 1, j]:
            edges.append(np.array([[i + 1, j], [i + 1, j + 1]]))
        # South
        if j == 0 or not bool_img[i, j - 1]:
            edges.append(np.array([[i, j], [i + 1, j]]))
        # West
        if i == 0 or not bool_img[i - 1, j]:
            edges.append(np.array([[i, j], [i, j + 1]]))

    if not edges:
        return np.zeros((0, 2, 2))
    else:
        return np.array(edges)


def close_loop_edges(edges):
    """
    Combine thee edges defined by 'get_all_edges' to closed loops around objects.
    If there are multiple disconnected objects a list of closed loops is returned.
    Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
    """

    loop_list = []
    while edges.size != 0:
        loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
        edges = np.delete(edges, 0, axis=0)

        while edges.size != 0:
            # Get next edge (=edge with common node)
            ij = np.nonzero((edges == loop[-1]).all(axis=2))
            if ij[0].size > 0:
                i = ij[0][0]
                j = ij[1][0]
            else:
                loop.append(loop[0])
                # Uncomment to to make the start of the loop invisible when plotting
                # loop.append(loop[1])
                break

            loop.append(edges[i, (j + 1) % 2, :])
            edges = np.delete(edges, i, axis=0)

        loop_list.append(np.array(loop))

    return loop_list


def plot_outlines(bool_img, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    edges = get_all_edges(bool_img=bool_img)
    # convert indices to coordinates; TODO adjust according to image extent
    edges = edges - 0.5
    outlines = close_loop_edges(edges=edges)
    cl = LineCollection(outlines, **kwargs)
    ax.add_collection(cl)


# > Plot utils


def plot_results_brain(df, raws, title="", cmap=DIV_CMAP, **kwargs):
    df = df.drop(DCCHS, level=2, errors="ignore")

    subid = df.index.get_level_values("subject")[0]
    labels = df.index.get_level_values("label").unique()
    channels = df.index.get_level_values("electrode").unique()

    n = len(labels)
    if not isinstance(raws, list):
        raws = [raws] * n

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    for raw, label, ax in zip(raws, labels, axes if n > 1 else [axes]):
        labeldf = df.loc[subid, label, :]

        electrodes = labeldf.index.get_level_values("electrode").tolist()
        values = labeldf.max(axis=1).values

        vmax = None if cmap == SEQ_CMAP else values.max()
        vmin = -vmax if vmax is not None else None

        plot_values(
            raw,
            electrodes,
            values,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figure=fig,
            axis=ax,
        )
        ax.set_title(f"{label} {title}")
        ax.set(**kwargs)

    return fig


def plot_results_summary(df, clinfs=512, title="", **kwargs):
    df = df.drop(DCCHS, level=2, errors="ignore")

    subid = df.index.get_level_values("subject")[0]
    labels = df.index.get_level_values("label").unique()
    channels = df.index.get_level_values("electrode").unique()
    x = np.array([int(c) for c in df.columns])
    rois = get_rois(subid.item())
    e2r = {elec: roi for roi, elecs in rois.items() for elec in elecs}

    minval, maxval = df.max(axis=1).min(), df.max().max()
    values = df.max(axis=1).values
    lags = x[df.values.argmax(axis=1)] / clinfs

    colors = plt.cm.tab20(range(20))
    markers = ["o", "v", "s", "p", "P", "*", "x", "D", "+", "1"]
    # markers = matplotlib.lines.Line2D.markers.keys()

    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(5, n * 3.5))
    if n == 1:
        axes = [axes]
    for label, ax in zip(labels, axes):
        labeldf = df.loc[subid, label, :]
        values = labeldf.max(axis=1).values
        lags = x[labeldf.values.argmax(axis=1)] / clinfs
        ax.scatter(lags, values, s=5, c="k", marker=".")
        ax.axvline(0, ls="--", c="black", alpha=0.3)
        # ax.set_ylim(minval - .05, maxval + .05)
        ax.set_title(f"{label} {title}")
        ax.set(**kwargs)
        if rois is not None:
            best_elecs = labeldf.max(axis=1)
            best_elecs = best_elecs[best_elecs > best_elecs.quantile(0.90)]
            roidf = best_elecs.sort_values().to_frame()
            roidf["roi"] = [e2r.get(elec, "zUNK") for elec in roidf.index]
            dfg = roidf.groupby("roi")
            for (roi, subdf), col, mark in zip(dfg, colors, markers):
                elecs = subdf.index.values
                roidf = labeldf.loc[elecs]
                values = roidf.max(axis=1).values
                lags = x[roidf.values.argmax(axis=1)] / clinfs
                ax.scatter(lags, values, s=25, label=roi, color=col, marker=mark)
    handles, pltlabels = ax.get_legend_handles_labels()
    fig.legend(handles, pltlabels, loc="upper right")
    fig.tight_layout()
    return fig


def plot_all_results(df, outpath, plotpath, title=None, **kwargs):
    """Plot all electrodes: global average, ROIs, then individual"""
    subid = df.index.get_level_values("subject")[0]
    labels = df.index.get_level_values("label").unique()
    channels = df.index.get_level_values("electrode").unique()
    x = np.array([int(c) for c in df.columns]) / 512  # NOTE - hardcoded
    rois = get_rois(subid.item())

    # minval, maxval = df.min().min(), df.max().max()
    pdf = PdfPages(outpath.fpath)
    fig, ax = plt.subplots()
    datadf = df.drop(DCCHS, level=2, errors="ignore")
    for label in labels:
        mean = datadf.loc[subid, label, :].mean()
        err = datadf.loc[subid, label, :].sem()
        ax.fill_between(x, mean - err, mean + err, alpha=0.1)
        ax.plot(x, mean, label=label)
        ax.axvline(0, ls="--", c="black", alpha=0.3)
        # ax.set_ylim(minval - .05, maxval + .05)
        ax.set_title(f"Global {title} (N={len(datadf)})")
        ax.legend(loc="lower right")
        ax.set(**kwargs)
    # ax.text(x[0], maxval, f'N={len(datadf)//2}')
    pdf.savefig(fig)
    plt.close()

    for roi, elecs in rois.items():
        fig, ax = plt.subplots()
        for label in labels:
            roidf = df.loc[subid, label, :]
            roidf = roidf[roidf.index.isin(elecs)]
            mean, err = roidf.mean(), roidf.sem()
            ax.fill_between(x, mean - err, mean + err, alpha=0.1)
            ax.plot(x, mean, label=label)
            ax.axvline(0, ls="--", c="black", alpha=0.3)
            # ax.set_ylim(minval - .05, maxval + .05)
            ax.set_title(f"{roi} {title} (N={len(roidf)})")
            ax.legend(loc="lower right")
            ax.set(**kwargs)
        pdf.savefig(fig)
        plt.close()

    # minval, maxval = df.min().min(), df.max().max()
    pdf = PdfPages(outpath.fpath)
    fig, ax = plt.subplots()
    datadf = df.drop(DCCHS, level=2, errors="ignore")
    for label in labels:
        mean = datadf.loc[subid, label, :].mean()
        err = datadf.loc[subid, label, :].sem()
        ax.fill_between(x, mean - err, mean + err, alpha=0.1)
        ax.plot(x, mean, label=label)
        ax.axvline(0, ls="--", c="black", alpha=0.3)
        # ax.set_ylim(minval - .05, maxval + .05)
        ax.set_title(f"Global {title}")
        ax.legend(loc="lower right")
        ax.set(**kwargs)
    # ax.text(x[0], maxval, f'N={len(datadf)//2}')
    pdf.savefig(fig)

    for roi, elecs in rois.items():
        fig, ax = plt.subplots()
        for label in labels:
            roidf = df.loc[subid, label, :]
            roidf = roidf[roidf.index.isin(elecs)]
            mean, err = roidf.mean(), roidf.sem()
            ax.fill_between(x, mean - err, mean + err, alpha=0.1)
            ax.plot(x, mean, label=label)
            ax.axvline(0, ls="--", c="black", alpha=0.3)
            # ax.set_ylim(minval - .05, maxval + .05)
            ax.set_title(f"{roi} {title} (N={len(roidf)})")
            ax.legend(loc="lower right")
            ax.set(**kwargs)
        pdf.savefig(fig)
        plt.close()

    minval, maxval = datadf.min().min(), datadf.max().max()
    for channel in channels:
        fig, ax = plt.subplots()
        for i, label in enumerate(labels):
            data = df.loc[(subid, label, channel)]
            ax.plot(x, data, label=label)
            # ax.text(x[0], maxval - i * .05, f'{label}-N: {df.attrs[label + "-N"]}')
        ax.axvline(0, ls="--", c="black", alpha=0.3)
        ax.set_title(f"{channel} {title}")
        if channel not in DCCHS:
            ax.set_ylim(minval - 0.05, maxval + 0.05)
        ax.set(**kwargs)
        ax.legend(loc="lower right")
        plotpath.update(suffix=f"desc-thumbnail_{channel}")
        import os

        if os.path.isfile(plotpath.fpath):
            arr_image = plt.imread(plotpath.fpath, format="jpg")
            fig.figimage(
                arr_image,
                fig.bbox.xmax - arr_image.shape[1],
                fig.bbox.ymax - arr_image.shape[0],
                zorder=5,
            )
        pdf.savefig(fig)
        plt.close()
    pdf.close()
