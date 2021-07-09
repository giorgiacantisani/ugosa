# Copyright 2021 InterDigital R&D and Télécom Paris.
# Author: Giorgia Cantisani
# License: Apache 2.0

import pandas as pd
import museval
import os
import json
import numpy as np

# Choose if generate latex table
latex = False

# choose which target to visualize. Options are: vocals, drums, bass, other, all 
target = 'drums'  

# choose which tags to visualize or which to excluse
exclude_tags = []
tags = [directory for directory in os.listdir('./exp/')]
tags = sorted(tags)

# prepare basic dataframes to append to
median_over_tracks = pd.DataFrame({'tag': [], 'target': [], 'SDR': [], 'SIR': [], 'SAR': [], 'ISR': []})
mean_over_tracks = pd.DataFrame({'tag': [], 'target': [], 'SDR': [], 'SIR': [], 'SAR': [], 'ISR': []})

mean_over_frames = pd.DataFrame({'tag': [], 'target': [], 'SDR': [], 'SIR': [], 'SAR': [], 'ISR': []})
median_over_frames = pd.DataFrame({'tag': [], 'target': [], 'SDR': [], 'SIR': [], 'SAR': [], 'ISR': []})


print('################')
print(target)
print('################')
for tag in tags:

    if tag in exclude_tags:
        continue

    museval_path = os.path.join('exp', tag, 'bss_eval.pkl')   
    if not os.path.exists(museval_path):
        continue

    museval_data = pd.read_pickle(museval_path)

    # ------------------------------------------------------------------------------------------------------------------
    # median over frames, median over tracks
    method_median_median = museval.MethodStore(frames_agg='median', tracks_agg='median')
    method_median_median.df = museval_data

    agg_median_median = method_median_median.agg_frames_tracks_scores()
    agg_median_median = pd.DataFrame(agg_median_median, index=None)

    median_median = {'tag': [tag], 'target': [target]}

    # add museval metrics to method results dicts
    for row in agg_median_median.itertuples():
        if target == 'all':
            metric = row.Index[2]
            median_median[metric] = [row.score]            
        elif row.Index[1] == target:
            metric = row.Index[2]
            median_median[metric] = [row.score] 

    median_over_tracks = median_over_tracks.append(pd.DataFrame(median_median), ignore_index=True, sort=False)

    # ------------------------------------------------------------------------------------------------------------------
    # median over frames, mean over tracks
    method_median_mean = museval.MethodStore(frames_agg='median', tracks_agg='mean')
    method_median_mean.df = museval_data

    agg_median_mean = method_median_mean.agg_frames_tracks_scores()
    agg_median_mean = pd.DataFrame(agg_median_mean, index=None)

    median_mean = {'tag': [tag], 'target': [target]}

    # add museval metrics to method results dicts
    for row in agg_median_mean.itertuples():
        if target == 'all':
            metric = row.Index[2]
            median_median[metric] = [row.score] 
        elif row.Index[1] == target:
            metric = row.Index[2]
            median_mean[metric] = [row.score]

    mean_over_tracks = mean_over_tracks.append(pd.DataFrame(median_mean), ignore_index=True, sort=False)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    # make option to show vocals, acc, or all
    is_target = mean_over_tracks['target'] == target

    if target == target:
        print('MEDIAN OVER FRAMES, MEDIAN OVER SAMPLES')
        print(median_over_tracks[is_target].round(1))
        print('MEDIAN OVER FRAMES, MEAN OVER SAMPLES')
        print(mean_over_tracks[is_target].round(1))
    else:
        print('MEDIAN OVER FRAMES, MEDIAN OVER SAMPLES')
        print(median_over_tracks.round(1))
        print('MEDIAN OVER FRAMES, MEAN OVER SAMPLES') 
        print(mean_over_tracks.round(1))

    if latex:
        print(median_over_tracks[is_target][['tag', 'SDR', 'SIR', 'SAR']].to_latex(float_format="{:0.1f}".format, index=False))
        print(mean_over_tracks[is_target][['tag', 'SDR', 'SIR', 'SAR']].to_latex(float_format="{:0.1f}".format, index=False))