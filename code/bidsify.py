"""Transforms source data into a BIDS-compatible structure.

All configuration is specified in a json file passed in.
"""
import argparse
import collections
import json
import os
import re
import shutil
import subprocess
from itertools import zip_longest

import mne
import numpy as np
import pandas as pd
from scipy.io import wavfile
from mne_bids import BIDSPath, write_raw_bids, write_anat

import asr

# https://doi.org/10.1523/JNEUROSCI.1091-13.2013
# https://upload.wikimedia.org/wikipedia/commons/c/c2/Desikan-Killiany_atlas_regions.pdf
ROI2ABRV = {
        'rostralmiddlefrontal': 'RMF',
        'caudalmiddlefrontal': 'CMF',
        'precentral': 'PREC',
        'postcentral': 'PSTC',
        'parsorbitalis': 'PORB',
        'parstriangularis': 'PTRI',
        'parsopercularis': 'POPE',
        'supramarginal': 'SMAR',
        'rSTG': 'rSTG',
        'mSTG': 'mSTG',
        'cSTG': 'cSTG',
        'rMTG': 'rMTG',
        'mMTG': 'mMTG',
        'cMTG': 'cMTG',
        'middletemporal': 'MT',
        'superiortemporal': 'ST',
        'inferiortemporal': 'IT',
        'superiorparietal': 'SP',
        'inferiorparietal': 'IP',
        'frontalpole': 'FP',
        'lateraloccipital': 'LO',
        'lateralorbitofrontal': 'LOF',
        'posteriorcingulate': 'PC',
        'superiorfrontal': 'SF',
        'caudalanteriorcingulate': 'CAC',
        'medialorbitofrontal': 'MOF',
        'entorhinal': 'ENT',
        'temporalpole': 'TP',
        'parahippocampal': 'PARH',
        'Hippocampus': 'HIP',  # custom
        'insula': 'INS',  # custom
        'fusiform': 'FUS',
        'bankssts': 'BSTS',
        'cuneus': 'CUN',
        'paracentral': 'PARC',
        'lingual': 'LING',
        'superiortemporal_div1': 'cSTG',
        'superiortemporal_div2': 'mSTG',
        'superiortemporal_div3': 'rSTG',
        'middletemporal_div1': 'cMTG',
        'middletemporal_div2': 'mMTG',
        'middletemporal_div3': 'rMTG'
}



def read_mni(filename, rename=False):
    # Read in MNI file
    column_names = ['name', 'x', 'y', 'z', 'g']
    df = pd.read_csv(filename, sep='\t', header=None, names=column_names)
    # Sometime separator is a space and not tab
    if df.isna().mean().mean() > 0.25:
        print('Reading MNI with space delimiter')
        column_names = ['name', 'x', 'y', 'z', 'g', 'na']
        df = pd.read_csv(filename, sep=' ', header=None, names=column_names)
    if df.isna().mean().mean() > 0.25:
        print('MNI not read properly', filename)
        exit(1)
    locs = df.iloc[:, 1:4].values
    locs = locs / 1000  # convert to meters, mne will reconvert to mm

    # Channel names in MNI file is not same as EDF, so rename.
    # MNI file is `GA_46` and EDF is `EEG GA_46-REF`
    ch_names = df.name.tolist()
    if rename:
        renamed = []
        for name in df.name:
            i = re.search(r'\d', name).span()[0]
            renamed.append(f'EEG {name[:i]}_{name[i:]}-REF')
        ch_names = renamed

    # renamed = []
    # for name in df.name:
    #     i = re.search(r'\d', name)
    #     if i is not None:
    #         i = i.span()[0]
    #         renamed.append(name[:i] + str(int(name[i:])))
    #     else:
    #         renamed.append(name)
    # ch_names = renamed

    montage = mne.channels.make_dig_montage(
            ch_pos=dict(zip(ch_names, locs)),
            coord_frame='mni_tal'
    )
    print(f'Created {len(ch_names)} channel positions')
    return montage


def read_edf(filename, line_freq=60, stims=[]):
    raw = mne.io.read_raw_edf(filename)
    raw.info['line_freq'] = line_freq

    # Set channel types
    types = {}
    for name in raw.ch_names:
        if 'EEG' in name:
            types[name] = 'ecog'
        elif 'ECG' in name:
            types[name] = 'ecg'
        elif name in stims:
            types[name] = 'stim'
        else:
            types[name] = 'misc'
    raw.set_channel_types(types)

    return raw


def handle_audio(audio_fn, path, events_fn=None, channel=None,
                 transcribe=False):

    # Save relevant audio to stimuli folder
    audio_path = path.copy().update(root=f'{path.root}/stimuli',
                                    datatype='audio',
                                    suffix='audio',
                                    extension='.wav',
                                    check=False)
    audio_path.mkdir(exist_ok=True)

    outfn = audio_path.fpath
    if channel is not None:
        fs, audio = wavfile.read(audio_fn)
        audio = audio[:, channel-1]
        wavfile.write(outfn, fs, audio)
    else:
        shutil.copyfile(audio_fn, outfn)

    if events_fn is not None:
        outfn = audio_path.update(extension='.csv')
        shutil.copyfile(events_fn, outfn)

    # Transcribe the audio file automaticcally
    if transcribe:
        raise NotImplementedError

        # This is only partially implemented.
        dsfn = audio_path.update(root='staging-data', suffix='audio16khz')
        dsfn.mkdir(exist_ok=True)
        asr.downsample(outfn, dsfn.fpath, tofs=16000)

        chunks = asr.transcribe(str(dsfn.fpath),
                                'code/vosk-model-small-en-us-0.15')

        df_word, df_utterance = asr.chunks2dfs(chunks)
        fn = audio_path.update(suffix='words', extension='.tsv')
        df_word.to_csv(fn.fpath, sep='\t', index=False)

        fn = audio_path.update(suffix='utterances', extension='.tsv')
        df_utterance.to_csv(fn.fpath, sep='\t', index=False)


def anat2roi(anatfiles, ch_names):
    loc2elec = collections.defaultdict(list)
    for filename in anatfiles:
        with open(filename, 'r') as f:
            for line in f:
                # Skip comments
                if line.startswith('%') or not len(line.strip()):
                    continue

                parts = line.split()

                # Parse name
                name = parts[0]
                i = re.search(r'\d', name).span()[0]
                elec = f'EEG {name[:i]}_{name[i:]}-REF'

                # Skip this electrode if not in ch_names
                if elec not in ch_names:
                    # print(f'Skipping {elec} - not in MNI/EDF')
                    continue

                # Parse location
                locations = parts[4:]
                if len(locations) == 2:
                    loc = locations[1]
                elif len(locations) > 2:
                    if len(locations) == 3:
                        loc = locations[-1]
                    # Take the largest percentage
                    else:
                        if len(locations) % 2 == 1:
                            locations = locations[1:]
                        percents = [float(loc[:-1]) for loc in locations[::2]]
                        best_id = percents.index(max(percents))
                        loc = locations[best_id * 2 + 1]
                elif len(parts) == 4:
                    loc = 'zUNK'
                else:
                    print('Unhandled line format!')
                    print(line)
                    exit(1)

                # Clean up location
                loc = loc.removeprefix('ctx-lh-')
                loc = loc.removeprefix('Left-')

                # Look up
                acr = ROI2ABRV.get(loc, 'zUNK')
                loc2elec[acr].append(elec)

        return loc2elec


def write_anat(t1fname):
    # Write anat data
# https://mne.tools/mne-bids/stable/auto_examples/convert_mri_and_trans.html
    raise NotImplementedError
    #     dig = None
    # Use electrodes as landmarks?
    if lmfname := entry.get('t1-coords'):
        df = pd.read_csv(lmfname, sep=' ', header=None)
        ch_names = df.iloc[:, 0].tolist()
        locs = df.iloc[:, 1:4].values
        dig = mne.channels.make_dig_montage(
                ch_pos=dict(zip(ch_names, locs)),
                coord_frame='mri'  # right?
        )
    anatfn = write_anat(t1fname, bids_path,
                        landmarks=dig, overwrite=True, deface=False)
    subprocess.call(f'pydeface --outfile {anatfn} --force {anatfn}',
                    shell=True)


def main(args):

    with open(args.file) as f:
        subjects = json.load(f)
        entry = [e for e in subjects if e['subject'] == args.subject]
        assert len(entry) != 0
        entry = entry[0]

    sub = entry['subject']
    print('--------------------------------------------------------------')
    print(f'Subject {sub} ({entry["ny-id"]})\n')

    bids_path = BIDSPath(root=args.root, subject=sub)

    # Read electrode location data
    montage = None
    if mni_fn := entry.get('mni-file'):
        montage = read_mni(mni_fn)

    # Add sessions
    for i, session in enumerate(entry['sessions'][:1], 1):  # NOTE 1st session
        bids_path.update(datatype='ieeg',
                         task=session['task'],
                         suffix='ieeg',
                         extension='.edf')

        # Read edf
        raw = read_edf(session['ieeg'], stims=session.get('lq-channel'))
        raw.info['subject_info'] = {
            'sex': {'male': 1, 'female': 2}.get(entry.get('sex'), 0),
            'hand': {'right': 1, 'left': 2}.get(entry.get('hand'), 0),
            'age': entry.get('age'),
            'id': sub
        }

        # Align EDF and MNI channels, make sure there's no typos
        print(f'There are {len(raw.ch_names)} channels in the EDF.\n'
              f'There are {len(montage.ch_names)} channels in MNI file.')
        only_edf = sorted(set(raw.ch_names) - set(montage.ch_names))
        print(f'{len(only_edf)} channels only in the EDF:')
        print(only_edf)
        only_mni = sorted(set(montage.ch_names) - set(raw.ch_names))
        print(f'{len(only_mni)} channels only in the MNI file:')
        print(only_mni)
        df = pd.DataFrame(zip_longest(only_edf, only_mni),
                          columns=['only_edf', 'only_mni'])
        print(df)
        # df.to_csv(f'staging/{sub}-elecs.csv')  # TODO
        raw.set_montage(montage, on_missing='warn')
        # missing = [c['ch_name'] for c in raw.info['chs']
        #            if np.isnan(c['loc']).any()]
        stims = session.get('lq-channel')
        if len(only_edf):
            for stim in stims:
                if stim in only_edf:
                    only_edf.remove(stim)
            raw.info['bads'].extend(only_edf)

        # If all channels are of type MISC ...
        # MINUS one
        if all([ch['kind'].numerator == 502 for ch in raw.info['chs']]):
            types = {ch: 'ecog' for ch in montage.ch_names if ch in raw.ch_names}
            types.update({ch: 'misc' for ch in session.get('lq-channel')})
            raw.set_channel_types(types)

        # Translate ROIs and save
        # NOTE - if there are multiple sessions, this assumes that all EDFs
        # share the same electrodes
        if roi_fn := entry.get('roi-file'):
            if isinstance(roi_fn, str):
                roi_fn = [roi_fn]
            rois = anat2roi(roi_fn, raw.ch_names)
            outpath = bids_path.copy()
            outpath.update(root='dataset/derivatives/preprocessed',
                           task=None,
                           datatype='anat',
                           suffix='atlas-desikan_anat',
                           extension='.json', check=False)
            outpath.mkdir(exist_ok=True)
            with open(outpath.fpath, 'w') as f:
                json.dump(rois, f, indent=2)
            print(roi_fn[0], outpath.fpath)
            exit()

        # Handle audio
        if audio_fn := session.get('audio'):
            print(f'Reading audio from {audio_fn}')
            # Put in stimuli/ and transcribe
            channel = session.get('audio-channel')
            handle_audio(audio_fn, bids_path, session.get('events'), channel)

        # Save raw to BIDS format
        raw.set_annotations(None)
        write_raw_bids(raw,
                       bids_path=bids_path,
                       overwrite=True,
                       verbose=False)

    # Add partner columns to participants.tsv
    if partner := entry.get('partner'):
        fname = os.path.join(args.root, 'participants.tsv')
        df = pd.read_csv(fname, sep='\t', index_col=0)
        df.loc['sub-' + sub, 'partner'] = 'sub-' + partner
        df.to_csv(fname, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', default='dataset')
    parser.add_argument('-s', '--subject', type=int, nargs='+', default=[1])
    parser.add_argument('-f', '--file', default='code/staging.json')

    args = parser.parse_args()
    subjects = args.subject
    for subject in subjects:
        args.subject = f'{subject:02d}'
        main(args)
