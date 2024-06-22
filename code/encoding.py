"""Predict brain activity from word embeddings (encoding model).
"""

import pickle
import uuid
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import utils
from himalaya.lasso import SparseGroupLassoCV
from himalaya.ridge import Ridge, RidgeCV
from himalaya.scoring import correlation_score, correlation_score_split
from scipy import signal
from sklearn import set_config
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from util.path import DerivativeBIDSPath
from util.signal import phase_randomize

set_config(assume_finite=True)  # turn off check array of nans or inf


def verify_args(args, sf):
    assert args.tmax * sf == int(args.tmax * sf), "tmax must be multiple of sf"
    assert args.jump * sf == int(args.jump * sf), "jump must me mutliple of sf"
    assert args.window * sf == int(args.window * sf), "window must be multiple of sf"


def decoding(S, E, kfold, sigmask=None, pca=0):
    ndim = E.shape[1] if not pca else pca
    nwords, _, nlags = S.shape
    predictions = np.zeros((nlags, nwords, ndim), dtype=np.float32)
    actuals = np.zeros((nwords, ndim), dtype=np.float32)
    corrs = np.zeros((kfold.n_splits, nlags, ndim), dtype=np.float32)
    corrs2 = np.zeros((nwords, nlags), dtype=np.float32)
    folds = np.zeros(nwords, dtype=int)
    coefs = []

    if sigmask is not None:
        if not sigmask.any():
            return {"corrs": corrs.mean(0), "preds": predictions}
        S = S[:, sigmask, :]

    for k, (train_index, test_index) in enumerate(kfold.split(S)):
        folds[test_index] = k
        S_train, S_test = S[train_index], S[test_index]
        E_train, E_test = E[train_index], E[test_index]

        if pca > 0:
            pca_model = PCA(pca, whiten=True)
            E_train = pca_model.fit_transform(E_train)
            E_test = pca_model.transform(E_test)
        actuals[test_index] = E_test

        for i in range(nlags):
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(S_train[..., i], E_train)
            preds = model.predict(S_test[..., i]).astype(np.float32)
            predictions[i, test_index] = preds
            corrs[k, i] = utils.correlate(preds, E_test).astype(np.float32)
            corrs2[test_index, i] = utils.correlate(preds, E_test, axis=1).astype(
                np.float32
            )
            coefs.append(model.steps[1][1].coef_)

    return {
        "corrs": corrs.mean(0),
        "corrsW": corrs2,
        "preds": predictions,
        "true": actuals,
        # 'regs': S,
        "folds": folds,
        "coefs": coefs,
    }


def pls(S, E, kfold, sigmask=None, n_components=10):
    nwords, _, _ = S.shape
    n_components = min(len(S), sigmask.sum(), E.shape[1])

    S_latent = np.zeros((nwords, n_components), dtype=np.float32)
    E_latent = np.zeros((nwords, n_components), dtype=np.float32)
    S_latent_train = np.zeros((nwords, n_components), dtype=np.float32)
    E_latent_train = np.zeros((nwords, n_components), dtype=np.float32)
    corrs = np.zeros((kfold.n_splits, n_components), dtype=np.float32)
    corrs2 = np.zeros(nwords, dtype=np.float32)
    folds = np.zeros(nwords, dtype=int)

    if sigmask is not None:
        if not sigmask.any():
            return {}  # 'corrs': corrs.mean(0), 'preds': predictions}
        S = S[:, sigmask, :]

    S = S.reshape(nwords, -1)

    for k, (train_index, test_index) in enumerate(kfold.split(S)):
        folds[test_index] = k
        S_train, S_test = S[train_index], S[test_index]
        E_train, E_test = E[train_index], E[test_index]

        scaler = StandardScaler()
        S_train = scaler.fit_transform(S_train)
        S_test = scaler.transform(S_test)
        scaler = StandardScaler()
        E_train = scaler.fit_transform(E_train)
        E_test = scaler.transform(E_test)

        model = PLSCanonical(n_components)
        S_latent_train[train_index], E_latent_train[train_index] = model.fit_transform(
            S_train, E_train
        )
        latentS, latentE = model.transform(S_test, E_test)
        corrs[k] = correlation_score(latentS, latentE)
        corrs2[test_index] = correlation_score(latentS.T, latentE.T)
        S_latent[test_index] = latentS
        E_latent[test_index] = latentE

    return {
        "corrs": corrs,
        "corrsW": corrs2,
        "folds": folds,
        # 'ecog': S,
        # 'emb': E,
        "latentS": S_latent,
        "latentE": E_latent,
        "latentS_train": S_latent_train,
        "latentE_train": E_latent_train,
    }


def encoding_bands(
    sub,
    speaker,
    models,
    tmax,
    window,
    jump,
    k=10,
    alphas=None,
    band="highgamma",
    **kwargs,
):
    from himalaya.kernel_ridge import (
        ColumnKernelizer,
        Kernelizer,
        MultipleKernelRidgeCV,
    )

    raw, df = utils.getraw(
        sub, trim=True, embeddings=models[0], speaker=speaker, band=band
    )
    raw.pick_types(ecog=True)
    # Get electrodes and ROIs, remove unknown locations
    rois = utils.get_rois(sub, atlas="desikan", chs=raw.ch_names)
    del rois["zUNK"]
    elec2roi = utils.invertdict(rois)
    electrodes = np.array([ch for ch in raw.ch_names if ch in elec2roi])
    raw.pick_channels(electrodes)
    elecrois = np.array([elec2roi[e] for e in electrodes])

    raw.load_data()
    data = raw._data.T  # becomes F-contiguous with the transpose
    data = signal.detrend(data, axis=0, type="linear")

    tmax, window, jump = raw.time_as_index([tmax, window, jump])
    tmax_wide = int(tmax + window / 2)

    onsets = raw.time_as_index(df.start.values)
    mask2 = (onsets - tmax_wide > 0) & (onsets + tmax_wide < data.shape[0])
    df = df[mask2]

    onsets = raw.time_as_index(df.start.values)
    E = np.vstack(df.embedding)

    if len(df) != len(E):
        print("Warning: words not aligned with embeddings in results", len(df), len(E))

    lags = np.arange(-tmax, tmax + jump, jump)
    S = utils.epochbin(data, onsets, lags, window)
    nwords, nelecs, nlags = S.shape
    S = S.reshape(nwords, -1)

    # detrend the epochs
    S = signal.detrend(S, axis=-1, type="linear")

    # Create design matrix
    X = []
    feature_names = ["llm", "symbolic", "phonetic"]
    feature_names = models
    for model in models:
        _, dft = utils.getraw(
            sub, trim=True, embeddings=model, speaker=speaker, band=band
        )
        X.append(np.vstack(dft[mask2].embedding))
    n_dims = np.cumsum([0] + [x.shape[1] for x in X])
    slices = [slice(start, end) for start, end in zip(n_dims[:-1], n_dims[1:])]
    E = np.hstack(X)

    # define model
    kernelizers = [
        (name, Kernelizer(), slice_) for name, slice_ in zip(feature_names, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers)
    solver_params = dict(alphas=alphas, progress_bar=False)
    ridge = MultipleKernelRidgeCV(
        kernels="precomputed", solver="random_search", solver_params=solver_params
    )
    model = make_pipeline(StandardScaler(), column_kernelizer, ridge)

    # Run encoding
    kfold = KFold(k, shuffle=False)

    print(E.shape, S.shape)

    results = defaultdict(list)
    for train_index, test_index in kfold.split(S):
        S_train, S_test = S[train_index], S[test_index]
        E_train, E_test = E[train_index], E[test_index]

        scaler = StandardScaler()
        S_train = scaler.fit_transform(S_train)
        S_test = scaler.transform(S_test)

        # Train and evaluate model
        model.fit(E_train, S_train)
        preds = model.predict(E_test, split=True)

        corrs = correlation_score_split(S_test, preds).get().reshape(-1, nelecs, nlags)
        pred = preds[0].get().reshape(-1, nelecs, nlags)

        results["corrs"].append(corrs.astype(np.float32))
        results["preds"].append(pred.astype(np.float32))

    results["electrodes"] = electrodes
    results["rois"] = elecrois
    return results


def encoding(
    sub,
    speaker,
    model,
    tmax,
    window,
    jump,
    k=10,
    pca=50,
    alphas=None,
    reg=0,
    subset=None,
    band="highgamma",
    shuffle=False,
    shift=0,
    slim_results=False,
    decode=None,
):
    """Fit embeddings of model to subject's activity when speaker is producing.

    If speaker == sub, then it's a production model.
    If speaker == partner of sub, then it's a comprehension model.
    """

    raw, df = utils.getraw(sub, trim=True, embeddings=model, speaker=speaker, band=band)
    raw.pick_types(ecog=True)
    # Get electrodes and ROIs, remove unknown locations
    rois = utils.get_rois(sub, atlas="desikan", chs=raw.ch_names)
    del rois["zUNK"]
    elec2roi = utils.invertdict(rois)
    electrodes = np.array([ch for ch in raw.ch_names if ch in elec2roi])
    raw.pick_channels(electrodes)
    elecrois = np.array([elec2roi[e] for e in electrodes])

    raw.load_data()
    data = raw._data.T  # becomes F-contiguous with the transpose
    data = signal.detrend(data, axis=0, type="linear")

    if shuffle == "phase":
        data = phase_randomize(data)
    elif shuffle == "random":
        data = np.random.randn(*data.shape)

    tmax, window, jump = raw.time_as_index([tmax, window, jump])
    tmax_wide = int(tmax + window / 2)

    onsets = raw.time_as_index(df.start.values)
    mask2 = (onsets - tmax_wide > 0) & (onsets + tmax_wide < data.shape[0])
    df = df[mask2]

    # Shift
    if shift != 0:
        df["embedding"] = df.embedding.shift(shift)
        df.dropna(axis=0, subset=["embedding"], inplace=True)

    # Select only a portion of the dataset to do encoding on
    if subset == "pred":
        df = df[df["true_prob"] >= df["true_prob"].median()].reset_index()
    elif subset == "unpred":
        df = df[df["true_prob"] < df["true_prob"].median()].reset_index()
    elif subset == "correct":
        df = df[df["rank"] < 5]
    elif subset == "incorrect":
        df = df[df["rank"] >= 5]

    onsets = raw.time_as_index(df.start.values)
    E = np.vstack(df.embedding)

    if len(df) != len(E):
        print("Warning: words not aligned with embeddings in results", len(df), len(E))

    if shuffle == "label":
        np.random.shuffle(E)

    lags = np.arange(-tmax, tmax + jump, jump)
    S = utils.epochbin(data, onsets, lags, window)
    nwords, nelecs, nlags = S.shape
    S = S.reshape(nwords, -1)

    # detrend the epochs
    S = signal.detrend(S, axis=-1, type="linear")
    print("ZZ", S.shape, E.shape)

    # Run encoding
    kfold = KFold(k, shuffle=False)

    if decode is not None:
        sigmask = None
        if decode == "sig":
            with open(
                "dataset/derivatives/encoding/sigmasks_shuffle-phase_model-gpt2-xl_layer-24_shift.pkl",
                "rb",
            ) as f:
                sigmasks = pickle.load(f)
            sigmask = sigmasks[
                (sub, "prod" if sub == speaker else "comp", "gpt2-xl_layer-24_shift")
            ]

        results = pls(S.reshape(-1, nelecs, nlags), E, kfold, sigmask)
        results["electrodes"] = electrodes
        results["rois"] = elecrois
        results["df"] = df.drop("embedding", axis=1)
        return results

        results = decoding(S.reshape(-1, nelecs, nlags), E, kfold, sigmask, pca)
        save_cols = [
            "word_idx",
            "word",
            "token",
            "top_pred",
            "rank",
            "true_prob",
            "entropy",
        ]
        results["df"] = df[save_cols]
        results["args"] = {
            "model": model,
            "pca": pca,
            "shift": shift,
            "decode": decode,
            "tmax": tmax,
            "window": window,
            "jump": jump,
            "shuffle": slim_results,
        }
        return results

    results = defaultdict(list)
    for train_index, test_index in kfold.split(S):
        S_train, S_test = S[train_index], S[test_index]
        E_train, E_test = E[train_index], E[test_index]

        scaler = StandardScaler()
        S_train = scaler.fit_transform(S_train)
        S_test = scaler.transform(S_test)

        # Create model
        steps = []
        if pca > 0:
            pca = min(pca, len(E_train))
            steps.append(PCA(pca, whiten=not True))
        steps.append(StandardScaler())
        if reg == 1:
            steps.append(
                SparseGroupLassoCV(
                    l1_regs=alphas,
                    l21_regs=[0],
                    groups=None,
                    solver_params=dict(max_iter=100),
                )
            )
        elif reg == 2:
            if len(alphas) == 1:
                steps.append(Ridge(alpha=alphas[0], fit_intercept=True))
            else:
                steps.append(RidgeCV(alphas, fit_intercept=True))
        else:
            steps.append(LinearRegression())
        model = make_pipeline(*steps)

        # Train and evaluate model
        model.fit(E_train, S_train)
        preds = model.predict(E_test)
        corr = correlation_score(S_test, preds).reshape(nelecs, nlags)
        preds = preds.reshape(-1, nelecs, nlags)
        if reg > 0:
            if isinstance(corr, cp.ndarray):
                corr = corr.get()
            if isinstance(preds, cp.ndarray):
                preds = preds.get()
            # corr = corr.detach().cpu().numpy()
            # preds = preds.detach().cpu().numpy()
        results["corrs"].append(corr.astype(np.float32))
        results["preds"].append(preds.astype(np.float32))

        if not slim_results:
            # Prepare data to save
            linmodel = model[-1]
            embs = model[:-1].transform(E_test)
            # coefs = linmodel.coef_.T.reshape(-1, nelecs, nlags)  # TODO scikit (n_targets, n_features) i.e. n x 50
            coefs = linmodel.coef_.reshape(
                -1, nelecs, nlags
            )  # TODO himalaya (n_features, n_targets)  e.g 50 x n
            intercept = linmodel.intercept_.reshape(
                nelecs, nlags
            )  # TODO himalaya (n_features, n_targets)  e.g 50 x n
            ecog = S_test.reshape(-1, nelecs, nlags)

            if reg > 0:
                if isinstance(coefs, cp.ndarray):
                    coefs = coefs.get()
                    intercept = intercept.get()
                if len(alphas) > 1:
                    best_alphas = linmodel.best_alphas_
                    if isinstance(best_alphas, cp.ndarray):
                        best_alphas = best_alphas.get()
                    # coefs = coefs.detach().cpu().numpy()
                    # best_alphas = best_alphas.detach().cpu().numpy()
                    results["alphas"].append(best_alphas)

            results["coefs"].append(coefs.astype(np.float32))
            results["intercept"].append(intercept.astype(np.float32))
            results["embs"].append(embs.astype(np.float32))
            results["true"].append(ecog.astype(np.float32))

            if pca:
                results["pca_xv"].append(
                    model[0].explained_variance_ratio_.astype(np.float32)
                )

    if slim_results:
        return results

    results["electrodes"] = electrodes
    results["rois"] = elecrois
    results["df"] = df.drop("embedding", axis=1)
    return results


def main(args, subject, speaker):
    # Prepare paths
    path = DerivativeBIDSPath(subject=f"{subject:02d}", task=args.task, root=args.root)

    mode = "prod" if subject == speaker else "comp"
    suffix_ext = f"_mode-{mode}_band-{args.band}_nfolds-{args.nfolds}"

    dtypename = args.model
    if args.shift > 0:
        dtypename += f"_shift-p{args.shift}"
    elif args.shift < 0:
        dtypename += f"_shift-n{args.shift}"
    if args.subset:
        dtypename += f"_subset-{args.subset}"

    if args.shuffle:
        uid = uuid.uuid4()
        suffix_ext += f"_shuffle-{args.shuffle}_{uid.hex}"

    dname = "encoding"
    dname = "encoding_bands"
    if args.decode is not None:
        dname = "decoding"
        dname = "xdecomp"
        dtypename += f"_elecs-{args.decode}"

    if args.verbose_dirname:
        dtypename += f"_tmax-{args.tmax}_win-{args.window}_jump-{args.jump}"

    if ncomp := args.pca:
        if ncomp > 1:
            ncomp = int(ncomp)
        dtypename += f"_pca-{ncomp}"

    alphas = None
    if alphas := args.reg:
        dtypename += f"_reg-l{args.reg}"
        alphas = np.logspace(6, 0, 20)
        if args.alpha is not None:
            alphas = [args.alpha]
            dtypename += f"_a-{args.alpha}"

    outpath = path.copy()
    outpath.update(
        datatype=dtypename,
        suffix=f"{dname}{suffix_ext}",
        extension=".pkl",
        root=f"{path.root}/derivatives/{dname}",
    )

    # Run encoding
    models = args.model
    # models = [args.model, "model-symbolic", "model-phonetic"]  # for encoding_bands
    # models = ["model-symbolic", "model-phonetic"]  # for encoding_bands
    # print("encoding bands", models)
    result = encoding(
        subject,
        speaker,
        models,
        args.tmax,
        args.window,
        args.jump,
        alphas=alphas,
        reg=args.reg,
        k=args.nfolds,
        subset=args.subset,
        band=args.band,
        shuffle=args.shuffle,
        shift=args.shift,
        pca=ncomp,
        decode=args.decode,
        slim_results=args.shuffle is not None,
    )
    result["args"] = vars(args)

    outpath.mkdir(exist_ok=True)
    print(outpath.fpath)
    with open(outpath.fpath, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = utils.getparser()
    parser.add_argument("-m", "--model", default="glove50")
    parser.add_argument("--tmax", type=float, default=4)  # s
    parser.add_argument("--window", type=float, default=0.250)  # s
    parser.add_argument("--jump", type=float, default=2 * 0.03125)  # s
    parser.add_argument("--shift", type=int, default=0)
    parser.add_argument("--pca", type=float, default=0)
    parser.add_argument("--nfolds", type=int, default=10)
    parser.add_argument("--reg", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--shuffle", type=str, default=None)
    parser.add_argument("--decode", type=str, default=None)
    parser.add_argument("--band", type=str, default="highgamma")
    parser.add_argument("--verbose-dirname", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.gpu:
        if args.reg:
            import cupy as cp
            from himalaya.backend import set_backend  # , get_backend

            backend = set_backend("cupy")
        else:
            print("[WARNING] using a gpu but without regularization")

    pargs = []
    pargs += [(args, sub, utils.getpartner(sub)) for sub in args.subject]
    pargs += [(args, sub, sub) for sub in args.subject]
    if len(pargs) == 1 or args.workers == 1:
        for param in pargs:
            main(*param)
    else:
        with Pool(min(args.workers, len(pargs))) as p:
            p.starmap(main, pargs)
