# Copyright 2021 InterDigital R&D and Télécom Paris.
# Author: Giorgia Cantisani
# License: Apache 2.0

"""Utils for framing and data augmentation
"""
import torch
import numpy as np
import librosa
import random


class Framing:
    """helper iterator class to do overlapped windowing - Code snipped taken from museval"""
    def __init__(self, window, hop, length):
        self.current = 0
        self.window = window
        self.hop = hop
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.nwin:
            raise StopIteration
        else:
            start = self.current * self.hop
            if np.isnan(start) or np.isinf(start):
                start = 0
            stop = min(self.current * self.hop + self.window, self.length)
            if np.isnan(stop) or np.isinf(stop):
                stop = self.length
            start = int(np.floor(start))
            stop = int(np.floor(stop))
            result = slice(start, stop)
            self.current += 1
            return result

    @property
    def nwin(self):
        if self.window < self.length:
            return int(
                np.floor((self.length - self.window + self.hop) / self.hop)
            )
        else:
            return 1

    next = __next__


class DAdataloader(torch.utils.data.Dataset):
  'Datata loader + Data augmentation'
  def __init__(self, mix, win, hop, sample_rate, n_observations=1, pitch_list=None, min_semitones=12, max_semitones=12, same_pitch_list_all_chunks=False):
        'Initialization'
        self.mix = mix
        self.sample_rate = sample_rate
        
        self.framer = Framing(window=win*self.sample_rate, hop=hop*self.sample_rate, length=self.mix.shape[1])
        self.frames = list(self.framer)
        self.n_frames = len(self.frames)

        self.same_pitch_list_all_chunks = same_pitch_list_all_chunks
        self.n_observations = n_observations
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones

        # Create random pitches. If no augmentation, skip it
        if self.n_observations > 1:
            # Create the matrix containing random pitches for each chunk
            self.matrix = np.ndarray((self.n_frames, self.n_observations))

            # if all the chunks should be pitched in the same way
            if self.same_pitch_list_all_chunks:
                # if the user gave his own list
                if pitch_list is not None:
                    self.pitch_list = pitch_list
                # else generate it random
                else:
                    self.pitch_list = np.random.random_integers(low=self.min_semitones, 
                                                                high=self.max_semitones, 
                                                                size=self.n_observations)
                    self.pitch_list[0] = 0

                for t in range(self.n_frames):
                    self.matrix[t, :] = np.asarray(self.pitch_list)
            
            # if the chunks should be pitched each one differently
            else:
                for t in range(self.n_frames):
                    self.pitch_list = np.random.random_integers(low=self.min_semitones, 
                                                                high=self.max_semitones, 
                                                                size=self.n_observations)
                    self.pitch_list[0] = 0                    
                    self.matrix[t, :] = np.asarray(self.pitch_list)


  def __len__(self):
        'Denotes the total number of samples'
        return self.n_observations * self.n_frames

  def __getitem__(self, idx):
        'Generates one pitched mix chunck'

        # If data augmentation: extract a chunk from the mix and pitch it
        if self.n_observations > 1:
            frame_idx = idx // self.n_observations
            augmn_idx = idx % self.n_observations

            win_slice = self.frames[frame_idx]
            num_semitones = self.matrix[frame_idx, augmn_idx]

            pitched_chunk = np.zeros_like(self.mix[:, win_slice])
            pitched_chunk[0, :] = librosa.effects.pitch_shift(self.mix[0, win_slice], self.sample_rate, n_steps=num_semitones)
            pitched_chunk[1, :] = librosa.effects.pitch_shift(self.mix[1, win_slice], self.sample_rate, n_steps=num_semitones)

        # If not data augmentation: just extract a chunk from the mix
        else:
            win_slice = self.frames[idx]
            num_semitones = 0

            pitched_chunk = self.mix[:, win_slice]
 
        return pitched_chunk, win_slice, num_semitones
