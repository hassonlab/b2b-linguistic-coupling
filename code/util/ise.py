import pickle

import numpy as np
from scipy.spatial.distance import cdist
from utils import getpartner

ROIS = {
    'SM': ('PREC', 'PSTC'),
    'STG': ('ST', 'mSTG', 'cSTG', 'rSTG', 'BSTS'),
    'IFG': ('PORB', 'PTRI', 'POPE'),
    'MTG': ('MT', 'mMTG', 'rMTG', 'cMTG'),
    'SMG': ('SMAR', ),
    'ATL': ('TP', 'IT'),
    'MFG': ('RMF', 'CMF'),
    'SFG': ('SF', ),
}

ALL_ROIs = sum([list(roi) for roi in ROIS.values()], [])
rois = list(ROIS)

# > Encoding functions

def getSubjectMap(results, sub, part, modelname, speakerROI=None, partnerROI=None, mode='s2l-average', significant=None, shuffle=False, subPermRun=None, partPermRun=None, reduce_folds=True, reduce_elecs=True):

    subResult = results[(sub, 'prod', modelname)]
    partResult = results[(part, 'comp', modelname)]
    
    # Filter to only electrodes we're interested in
    subElecs = np.ones(len(subResult['rois']), dtype=bool)
    partElecs = np.ones(len(partResult['rois']), dtype=bool)
    if speakerROI is not None:
        subElecs &= np.in1d(subResult['rois'], speakerROI)
    if partnerROI is not None:
        partElecs &= np.in1d(partResult['rois'], partnerROI)
    
    # Choose significant electrodes
    if significant:
        subElecs &= significant[(sub, 'prod')]
        partElecs &= significant[(part, 'comp')]

    # Ensure we have enough left
    nElecsSub, nElecsPart = subElecs.sum(), partElecs.sum()

    nlags = subResult['corrs'][0].shape[1]
    if shuffle:
        rng = np.random.default_rng()

    # If just looking within subjects, use the correlations per electrode instead
    if mode == 'within-prod':
        axes = tuple()
        if reduce_folds: axes += (0, )
        if reduce_elecs: axes += (1, )
        if nElecsSub == 0:
            empty = np.ma.zeros(nlags)
            empty.mask = True
            return empty, 0, 0
        else:
            result = np.stack(subResult['corrs'])[:, subElecs, :]
            if len(axes): result = result.mean(axes)
            return result, nElecsSub, 0
    elif mode == 'within-comp':
        axes = tuple()
        if reduce_folds: axes += (0, )
        if reduce_elecs: axes += (1, )
        if nElecsPart == 0:
            empty = np.ma.zeros(nlags)
            empty.mask = True
            return empty, 0, 0
        else:
            result = np.stack(partResult['corrs'])[:, partElecs, :]
            if len(axes): result = result.mean(axes)
            return result, 0, nElecsPart
    # Or return all nan (maybe replace with masked array?)
    elif nElecsSub == 0 or nElecsPart == 0:
        empty = np.ma.zeros((nlags, nlags))
        empty.mask = True
        return empty, 0, 0
    
    if mode == 's2l-brain':
        amax = np.stack(subResult['corrs'])[:, subElecs, :].mean((0,1)).argmax()
        bmax = np.stack(partResult['corrs'])[:, partElecs, :].mean((0,1)).argmax()
        M = np.ma.zeros((10, nElecsPart))
        for k in range(10):
            subPreds = subResult['preds'][k][:, subElecs, :]
            subPreds = subPreds.mean(1)
            partTrues = partResult['true'][k][:, partElecs, :]
            M[k] = 1 - cdist(subPreds[:, amax:amax+1].T, partTrues[..., bmax].T, metric='correlation')
        return M.mean(0), nElecsSub, nElecsPart

    # Load permutations if applicable
    if subPermRun is not None:
        with open(subPermRun, 'rb') as f:
            subPermRun = pickle.load(f)
    if partPermRun is not None:
        with open(partPermRun, 'rb') as f:
            partPermRun = pickle.load(f)
    
    # Build inter-subject encoding result
    K = 10
    M = np.ma.zeros((K, nlags, nlags))
    for k in range(K):
        subTrues = subResult['true'][k][:, subElecs, :]
        partTrues = partResult['true'][k][:, partElecs, :]

        if subPermRun is not None:
            subPreds = subPermRun['preds'][k][:, subElecs, :]
        else:
            subPreds = subResult['preds'][k][:, subElecs, :]
        
        if partPermRun is not None:
            partPreds = partPermRun['preds'][k][:, partElecs, :]
        else:
            partPreds = partResult['preds'][k][:, partElecs, :]
        
        # Average over electrode predictions
        subPreds = subPreds.mean(1)
        subTrues = subTrues.mean(1)
        partPreds = partPreds.mean(1)
        partTrues = partTrues.mean(1)
        
        if shuffle:
            rng.shuffle(subPreds)
            rng.shuffle(partPreds)

        if mode.startswith('preds'):
            a, b = subPreds, partPreds
        elif mode.startswith('s2l'):
            a, b = subPreds, partTrues
        elif mode.startswith('l2s'):
            a, b = subTrues, partPreds
        elif mode.startswith('l2l'):
            a, b = partTrues, partPreds
        elif mode.startswith('direct'):
            a, b = subTrues, partTrues
        elif mode.startswith('residuals'):
            a = subTrues - subPreds - partPreds
            b = partTrues - partPreds - subPreds
        elif mode.startswith('weights'):
            a = subResult['coefs'][k][:, subElecs, :].mean(1)
            b = partResult['coefs'][k][:, partElecs, :].mean(1)
#         elif mode.startswith('average'):
#             a = (subTrues + subPreds) / 2
#             b = (partTrues + partPreds) / 2
        else:
            raise ValueError(f'Unknown mode: {mode}')

        if mode.endswith('bestlag'):
            amax =  subResult['corrs'][subElecs, :].mean(0).argmax()
            bmax = partResult['corrs'][partElecs, :].mean(0).argmax()
            a = a[..., [amax]]
            b = b[..., [bmax]]
            
        if mode.endswith('average'):
            a, b = subPreds, partTrues
            S2L = 1 - cdist(a.T, b.T, metric='correlation')
            a, b = subTrues, partPreds
            L2S = 1 - cdist(a.T, b.T, metric='correlation')
            M[k] = (S2L + L2S) / 2
        else:
            M[k] = 1 - cdist(a.T, b.T, metric='correlation')
        
    if reduce_folds: M = M.mean(0)
    return M, nElecsSub, nElecsPart


# Optimize for permutations
def getSubjectROIMaps(results, sub, part, modelname, rois, symmetric=True, subPermRun=None, partPermRun=None, significant=None, subMode='prod', partMode='comp', mode=None, shuffle=False):

    # Load data
    subResult = results[(sub, subMode, modelname)]
    subTrue = subResult['true']
    subPred = subResult['preds']
    if subPermRun is not None:
        with open(subPermRun, 'rb') as f:
            subPermRun = pickle.load(f)
            subPred = subPermRun['preds']

    if part == sub:
        partResult = subResult
        partTrue = subTrue
        partPred = subPred
    else:
        partResult = results[(part, partMode, modelname)]
        partTrue = partResult['true']
        partPred = partResult['preds']
        if partPermRun is not None:
            with open(partPermRun, 'rb') as f:
                partPermRun = pickle.load(f)
                partPred = partPermRun['preds']
    
    # Choose significant electrodes
    subElecs0, partElecs0 = True, True
    if significant is not None:
        subElecs0 = significant[(sub, subMode)]
        partElecs0 = significant[(part, partMode)]

    nlags = subResult['corrs'][0].shape[1]
    if shuffle:
        rng = np.random.default_rng()

    K = 10
    M = np.ma.zeros((K, len(rois), len(rois), nlags, nlags))
    mask = np.zeros((K, len(rois), len(rois), nlags, nlags), dtype=bool)
    NS = np.zeros((len(rois), len(rois)), dtype=int)
    NP = np.zeros((len(rois), len(rois)), dtype=int)
    for i, roi1 in enumerate(rois):
        for j, roi2 in enumerate(rois):
            # Filter to only electrodes we're interested in
            subElecs = subElecs0 & np.in1d(subResult['rois'], ROIS[roi1])
            partElecs = partElecs0 & np.in1d(partResult['rois'], ROIS[roi2])
            
            # Ensure we have enough left
            nElecsSub, nElecsPart = subElecs.sum(), partElecs.sum()
            if nElecsSub == 0 or nElecsPart == 0:
                # M[:, i, j] = np.nan
                mask[:, i, j] = True
                continue
            
            NS[i, j] = nElecsSub
            NP[i, j] = nElecsPart

            # Build inter-subject encoding result
            for k in range(K):
                subTrues = subTrue[k][:, subElecs, :]
                subPreds = subPred[k][:, subElecs, :]
                partTrues = partTrue[k][:, partElecs, :]
                partPreds = partPred[k][:, partElecs, :]
                
                # Average over electrode predictions
                subPreds = subPreds.mean(1)
                partPreds = partPreds.mean(1)
                subTrues = subTrues.mean(1)
                partTrues = partTrues.mean(1)

                if shuffle:
                    rng.shuffle(subPreds)
                    rng.shuffle(partPreds)

                # Direct ISFC
                if mode == 'isfc':
                    M[k, i, j] = 1 - cdist(subTrues.T, partTrues.T, metric='correlation')
                else:
                    # ISE average both directions
                    S2L = 1 - cdist(subPreds.T, partTrues.T, metric='correlation')
                    if symmetric:
                        L2S = 1 - cdist(subTrues.T, partPreds.T, metric='correlation')
                        M[k, i, j] = (S2L + L2S) / 2
                    else:
                        M[k, i, j] = S2L
                
    M.mask = mask
    return M.mean(0), NS, NP


def getMaps(results, Ss, Ps=None, modelname=None, weight=True, reduce=None, prod_perms=None, comp_perms=None, rand_part=False, **kwargs):
    # Collect maps
    if prod_perms is None:
        prod_perms = [None] * len(Ss)
    if comp_perms is None:
        comp_perms = [None] * len(Ss)
    if Ps is None:
        Ps = [getpartner(sub) for sub in Ss]

    maps, nsub, npart = [], [], []
    for sub, part, prod_perm_file, comp_perm_file in zip(Ss, Ps, prod_perms, comp_perms):
        if rand_part:
            part = np.random.choice([s for s in Ss if s != sub and s != part], 1).item()
        if 'rois' in kwargs:
            M, NS, NP = getSubjectROIMaps(results, sub, part, modelname, subPermRun=prod_perm_file, partPermRun=comp_perm_file, **kwargs)
        else:
            M, NS, NP = getSubjectMap(results, sub, part, modelname, subPermRun=prod_perm_file, partPermRun=comp_perm_file, **kwargs)
        maps.append(M)
        nsub.append(NS)
        npart.append(NP)
        
    # Convert
    addaxis = all(len(m) == len(maps[0]) for m in maps)
    M = np.ma.stack(maps) if addaxis else np.ma.vstack(maps)
    nsub = np.array(nsub)
    npart = np.array(npart)
    
    # Reduce
    if reduce and not weight:
        M = np.nanmean(M, axis=0)
    if reduce and weight and len(M) > 1:
        # Compute weights for weighted average based on number of words, speakers, and listeners
        # Load weights
        nwords_prod = np.array([np.mean([len(results[(s, 'prod', modelname)]['true'][i]) for i in range(10)]) for s in Ss])
        # nwords_comp = np.array([np.mean([len(results[(s, 'comp', modelname)]['true'][i]) for i in range(10)]) for s in Ss])
        weightsWords = nwords_prod # / nwords_prod.sum()
        M = np.ma.average(M, weights=weightsWords, axis=0)
    
    # Return
    return M, nsub, npart

