# Copyright 2021 InterDigital R&D and Télécom Paris.
# Author: Giorgia Cantisani
# License: Apache 2.0

"""Utils for the evaluation
"""
import torch
import museval
import numpy as np
import pandas as pd
from utils.silent_frames_evaluation import eval_silent_frames


def evaluate_mia(ref, est, track_name, source_names, eval_silence, conf):
    references = ref.copy()
    estimates = est.copy()

    # If evaluate silence, skip examples with a silent source
    skip = False
    silence_frames = pd.DataFrame({'target': [], 'PES': [], 'EPS': [], 'track': []})
    if eval_silence:
        PES, EPS, _, __ = eval_silent_frames(true_source=references,
                                            predicted_source=estimates,
                                            window_size=int(conf['win']*conf['sample_rate']),
                                            hop_size=int(conf['hop']*conf['sample_rate']))   

        for i, target in enumerate(source_names):
            reference_energy = np.sum(references[i, :, :]**2)
            # estimate_energy = np.sum(estimates[i, :, :]**2)
            if reference_energy == 0: # or estimate_energy == 0:
                skip = True
                sdr = isr = sir = sar = (np.ones((1,)) * (-np.inf), np.ones((1,)) * (-np.inf))
                print("skip {}, {} source is all zero".format(track_name, target))
        
        print("mean over evaluation frames, mean over channels")
        for target in source_names:
            silence_frames = silence_frames.append({'target': target, 'PES': PES[i], 'EPS': EPS[i], 'track': track_name}, ignore_index=True)
            print(target + ' ==>', silence_frames.loc[silence_frames['target'] == target].mean(axis=0, skipna=True))

    # Compute metrics for a given song using window and ho size
    if not skip: 
        sdr, isr, sir, sar = museval.evaluate(references, 
                                                estimates, 
                                                win=int(conf['win']*conf['sample_rate']), 
                                                hop=int(conf['hop']*conf['sample_rate']))
        
    # Save results over the track
    track_store = museval.TrackStore(win=conf['win'], hop=conf['hop'], track_name=track_name)
    for index, target in enumerate(source_names):
        values = {
            "SDR": sdr[index].tolist(),
            "SIR": sir[index].tolist(),
            "ISR": isr[index].tolist(),
            "SAR": sar[index].tolist()}
        track_store.add_target(target_name=target, values=values) 
    track_store.validate()    

    return track_store, silence_frames

