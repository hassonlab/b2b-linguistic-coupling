# Runs permutations to build a null distribution

import gc
import pickle
import numpy as np

from glob import glob
from tqdm import tqdm, trange
from util.path import derivpath
from util.ise import getMaps, ROIS
from utils import load_pickle, getpartner

# Options
root = 'dataset/derivatives/encoding/'
Ss = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
Ps = [Ss.index(getpartner(s)) for s in Ss]
modes = ['prod', 'comp']
tmax, window, jump = 4, .250, .03125*2
lags = np.arange(-tmax, tmax+jump, jump)
rois = list(ROIS.keys())

modelname = 'model-gpt2-xl_maxlen-1024_layer-24_reg-l2'

method, alpha = 'fdr_bh', .01
datatype = f'method-{method}_alpha-{alpha}_lags-1'
sigmodelname = 'model-gpt2-xl_maxlen-0_reg-l2_perm-phase'
p = derivpath(f'sub-all_model-{sigmodelname}.pkl', derivative='electrode-selection', datatype=datatype, root='dataset/derivatives')
sigmasks = load_pickle(p)
sigmasks = {key[:2]: value for key, value in sigmasks.items()}

# Load actual results
results = {}
for sub in Ss:
    for mode in modes:
            actual_pickle = glob(f'{root}sub-{sub:02d}/{modelname}/sub-{sub:02d}_task-conversation_encoding_mode-{mode}.pkl')
            if len(actual_pickle):
                with open(actual_pickle[0], 'rb') as f:
                    result = pickle.load(f)
                    # For encoding
                    del result['embs']
                    del result['df']
                    del result['coefs']
                    del result['args']
                    results[(sub, mode, modelname)] = result
            gc.collect()

n_lags = lags.size
n_rois = len(rois)

# Get available control runs
prod_shuffle_files = [glob(
    f'{root}sub-{sub:02d}/{modelname}/sub-{sub:02d}_task-conversation_encoding_mode-prod_shuffle-phase_*.pkl') for sub in Ss]
comp_shuffle_files = [glob(
    f'{root}sub-{sub:02d}/{modelname}/sub-{sub:02d}_task-conversation_encoding_mode-comp_shuffle-phase_*.pkl') for sub in Ss]
n_perms = min(min([len(f) for f in prod_shuffle_files]),
              min([len(f) for f in comp_shuffle_files]))
print('Number of permutation files:', n_perms)

# # INTERSUBJECT ALL
# # note - set shuffle to true to do label shuffle instead
# n = 0
# null_distribution = np.empty((n_perms, n_lags, n_lags), dtype=np.float32)
# for i in trange(n_perms):
#     prod_files_i = [prod_shuffle_files[j][i] for j in range(len(Ss))]
#     comp_files_i = [comp_shuffle_files[j][i] for j in Ps]
#     try:
#         M, _, _ = getMaps(results, Ss, modelname=modelname, reduce=True, weight=True, significant=sigmasks,
#                         prod_perms=prod_files_i, comp_perms=comp_files_i)
#         null_distribution[i] = M
#         n += 1
#     except (pickle.UnpicklingError, EOFError):
#         print('unpickle error', i)
# print(n)
# null_distribution = null_distribution[:n]
# observed, _, _ = getMaps(results, Ss, modelname=modelname, significant=sigmasks, reduce=True, weight=True)
# path = derivpath(f'sub-all_model-{modelname}_perm-phase.npz',
#                  derivative='ise',
#                  root='dataset/derivatives')
# path.mkdir(exist_ok=True)
# print(path.fpath)
# np.savez(path.fpath, observed=observed, null_distribution=null_distribution)


# # INTERSUBJECT ROIS
# n = 0
# null_distribution = np.empty((n_perms, n_rois, n_rois, n_lags, n_lags), dtype=np.float32)
# for i in trange(n_perms):
#     try:
#         prod_files_i = [prod_shuffle_files[j][i] for j in range(len(Ss))]
#         comp_files_i = [comp_shuffle_files[j][i] for j in Ps]
#         M, _, _ = getMaps(results, Ss, modelname=modelname, significant=sigmasks, reduce=True, weight=True, rois=rois,
#                         prod_perms=prod_files_i, comp_perms=comp_files_i)
#         null_distribution[i] = M
#         n += 1
#     except pickle.UnpicklingError:
#         continue
# print(n)
# null_distribution = null_distribution[:n]
# observed, _, _ = getMaps(results, Ss, modelname=modelname, significant=sigmasks, reduce=True, weight=True, rois=rois)
# path = derivpath(f'sub-all_mode-spklst_model-{modelname}_perm-phase.npz',
#                  derivative='ise',
#                  root='dataset/derivatives')
# path.mkdir(exist_ok=True)
# np.savez(path.fpath, observed=observed, null_distribution=null_distribution)

# WITHIN-SPEAKER
n = 0
null_distribution = np.empty((n_perms, n_rois, n_rois, n_lags, n_lags), dtype=np.float32)
for i in trange(n_perms):
    try:
        prod_files_i = [prod_shuffle_files[j][i] for j in range(len(Ss))]
        M, _, _ = getMaps(results, Ss, Ss, modelname=modelname, reduce=True, weight=True, rois=rois, significant=sigmasks, partMode='prod',
                          symmetric=False,
                          prod_perms=prod_files_i, comp_perms=prod_files_i)
        null_distribution[i] = M
        n += 1
    except pickle.UnpicklingError:
        continue
print(n)
null_distribution = null_distribution[:n]
# observed, _, _ = getMaps(results, Ss, Ss, modelname=modelname, reduce=True, weight=True, rois=rois, significant=sigmasks, partMode='prod')
path = derivpath(f'sub-all_mode-speaker_model-{modelname}_perm-phase_sym-False.npz',
                 derivative='ise',
                 root='dataset/derivatives')
path.mkdir(exist_ok=True)
np.savez(path.fpath, null_distribution=null_distribution)
print(path.fpath)


# WITHIN_LISTEER
n = 0
null_distribution = np.empty((n_perms, n_rois, n_rois, n_lags, n_lags), dtype=np.float32)
for i in trange(n_perms):
    try:
        comp_files_i = [comp_shuffle_files[j][i] for j in range(len(Ss))]
        M, _, _ = getMaps(results, Ss, Ss, modelname=modelname, reduce=True, weight=True, rois=rois, significant=sigmasks, subMode='comp',
                          symmetric=False,
                          prod_perms=comp_files_i, comp_perms=comp_files_i)
        null_distribution[i] = M
        n += 1
    except pickle.UnpicklingError:
        continue
print(n)
null_distribution = null_distribution[:n]
# observed, _, _ = getMaps(results, Ss, Ss, modelname=modelname, reduce=True, weight=True, rois=rois, significant=sigmasks, subMode='comp')
path = derivpath(f'sub-all_mode-listener_model-{modelname}_perm-phase_sym-False.npz',
                 derivative='ise',
                 root='dataset/derivatives')
path.mkdir(exist_ok=True)
np.savez(path.fpath, null_distribution=null_distribution)
print(path.fpath)