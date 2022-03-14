# Copyright 2021 InterDigital R&D and Télécom Paris.
# Author: Giorgia Cantisani
# License: Apache 2.0

"""Code to generate the dataset and set sources to zero manually.
"""
import os
import argparse
import random
import numpy as np
import librosa
import musdb
import soundfile as sf
from copy import deepcopy as cp
from utils.utils_adaptation import *


def main():
    source_names = ["drums", "bass", "other", "vocals"]
    random_order = True
    np.random.seed(42)
    channels = [0, 1]

    path = '/tsi/doctorants/gcantisani/Datasets/MUSDB18/'
    new_path = '/tsi/doctorants/gcantisani/Datasets/MUSDB18_UGOSA/'
    os.makedirs(new_path, exist_ok=True)

    # Iterate over all the tracks in the test set
    test_set = musdb.DB(root=path, subsets=["test"], is_wav=False)
    for idx in range(len(test_set)):
        track = test_set.tracks[idx]
        print('-------------------')
        print(idx, str(track.name))  

        # copy the track object and associate the new path
        new_track = cp(track)
        new_track.path = os.path.join(new_path, track.subset, track.name)
        os.makedirs(os.path.join(new_path, track.subset, track.name), exist_ok=True)

        # generate a random order of sources
        if random_order:
            sources = random.sample(source_names, 4)    
            print(sources)    

        # Load the mixture, make STFT, divide it into a number of 
        # segments equal to the number of sources and make ISTFT
        # Transoform to STFT and then back to have smoothing at boarders
        linear_mixture = track.targets['linear_mixture'].audio
        stft_mixture = librosa.stft(linear_mixture[:, 0])
        segment_len = stft_mixture.shape[1]//len(source_names)        

        new_references = []
        for t, name in enumerate(sources):
            audio = track.targets[name].audio
            audio_new = np.zeros_like(audio)  
            win = slice(t*segment_len, (t+1)*segment_len)  
            if t == len(source_names)-1:
                win = slice(t*segment_len, stft_mixture.shape[1] )      
            
            for ch in channels:
                stft = librosa.stft(audio[:, ch])
                stft[:, win] = 0           
                istft = librosa.istft(stft)
                audio_new[:, ch] = istft

            new_track.sources[name].audio = audio_new
            sf.write(os.path.join(new_track.path, name + '.wav'), audio_new, track.rate)

        new_references = np.stack([new_track.sources[name].audio for name in source_names])
        audio_mix = new_references.sum(0)
        sf.write(os.path.join(new_track.path, 'mixture.wav'), audio_mix, track.rate)


if __name__ == "__main__":
    main()
