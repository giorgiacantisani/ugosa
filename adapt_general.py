# Copyright 2021 InterDigital R&D and Télécom Paris.
# Author: Giorgia Cantisani
# License: Apache 2.0

"""Main script for adaptation
"""

import os
import random
import soundfile as sf
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy as cp

import torch
from torch import optim
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from ranger import Ranger  

from demucs.utils import *
from demucs.tasnet import *
from demucs.test import *

from utils.utils_adaptation import *
from utils.utils_activation import *
from utils.utils_evaluation import *
from utils.utils_data import *


# Input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', type=int, default=1,
                    help='Whether to use the GPU for model execution')
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Experiment root')
parser.add_argument('--n_save_ex', type=int, default=50,
                    help='Number of audio examples to save, -1 means all')
parser.add_argument('--eval_silence', type=bool, default=True,
                    help='Evaluate silent frames differently')
parser.add_argument('--stop_index', type=int, default=10,
                    help='Nr of songs to evaluate')
parser.add_argument('--monitor_metrics', type=bool, default=True,
                    help='Monitor sdr, sir, sar and isr during training')

parser.add_argument('--method', type=str, default='LmixLact',
                    help='available methods: baseline, LmixLact, Lmix, mixIT')
parser.add_argument('--gamma', type=float, default=1,
                    help='weight of the activation loss')
parser.add_argument('--frozen_layers', type=int, default=338,
                    help='Nr of layer to be freezed during fine tuning')
parser.add_argument('--freeze_decoder', type=bool, default=True,
                    help='Freeze decoder during fine tuning')
parser.add_argument('--win_fine', type=float, default=10.,
                    help='Window in seconds for fine tuning')
parser.add_argument('--hop_fine', type=float, default=1.,
                    help='Hope size in seconds for fine tuning')
parser.add_argument('--lr_fine', type=float, default=1e-5,
                    help='Learning rate for fine tuning')
parser.add_argument('--epochs_fine', type=int, default=10,
                    help='Number of epochs for fine tuning')

parser.add_argument('--n_observations', type=int, default=3,
                    help='Number of pitch shifting to do on each chunck')
parser.add_argument('--pitch_list', type=list, default=None,
                    help='Pitch shifting to do on each chunck')
parser.add_argument('--min_semitones', type=int, default=-12,
                    help='Min semitone for pitch shifting')
parser.add_argument('--max_semitones', type=int, default=12,
                    help='Max semitone for pitch shifting')
parser.add_argument('--same_pitch_list_all_chunks', type=bool, default=False,
                    help='Augment all chunks of the same mixture with the same pitch shifting')

parser.add_argument('--apply_act_output', type=bool, default=True,
                    help='Apply the binary activations to the output during evaluation')
parser.add_argument('--th', type=float, default=0.01,
                    help='Threshold to compute the binary activations')

parser.add_argument('--win', type=float, default=2.0,
                    help='Window in seconds over we use to compute the metrics')
parser.add_argument('--hop', type=float, default=1.5,
                    help='Hop size in seconds we use to compute the metrics')
parser.add_argument('--shifts', type=int, default=10, 
                    help='Shifts')
parser.add_argument('--split', type=bool, default=True,
                    help='Split test song into segments')
                    
parser.add_argument('--seed', type=int, default=42,
                    help='Seed') 

parser.add_argument('--musdb_path', type=str, default='/tsi/doctorants/gcantisani/Datasets/MUSDB18_UGOSA/',
                    help='Path to the data')
parser.add_argument('--sample_rate', type=int, default=44100,
                    help='Sample rate')
parser.add_argument('--model_path', type=str, default='demucs/tasnet.th',
                    help='Path to the pretrained model')

source_names = ["drums", "bass", "other", "vocals"]


def main(conf):
    if conf['method'] not in ['baseline', 'LmixLact', 'Lmix']:
        raise ValueError("method must be baseline, LmixLact or Lmix")

    # Set random seeds both for pytorch and numpy
    th.manual_seed(conf['seed'])
    np.random.seed(conf['seed'])

    # Create experiment folder and save conf file with the final configuration
    os.makedirs(conf['exp_dir'], exist_ok=True)    
    conf_path = os.path.join(conf['exp_dir'], 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)

    # Load test set. Be careful about is_wav!
    test_set = musdb.DB(root= conf['musdb_path'], subsets=["test"], is_wav=True)

    # Randomly choose the indexes of sentences to save.
    ex_save_dir = os.path.join(conf['exp_dir'], 'examples/')
    if conf['n_save_ex'] == -1:
        conf['n_save_ex'] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf['n_save_ex'])

    # If stop_index==-1, evaluate the whole test set
    if conf['stop_index'] == -1:
        conf['stop_index'] = len(test_set)

    # prepare data frames
    results_applyact = museval.EvalStore()    
    results_adapt = museval.EvalStore()
    silence_adapt = pd.DataFrame({'target': [], 'PES': [], 'EPS': [], 'track': []})  

    # Loop over test examples
    for idx in range(len(test_set)):
        torch.set_grad_enabled(False)
        track = test_set.tracks[idx]
        print(idx, str(track.name))  

        # Create local directory
        local_save_dir = os.path.join(ex_save_dir, str(track.name))
        os.makedirs(local_save_dir, exist_ok=True)      

        # Load mixture
        mix = th.from_numpy(track.audio).t().float()
        ref = mix.mean(dim=0)  # mono mixture
        mix = (mix - ref.mean()) / ref.std()
        
        # Load pretrained model
        klass, args, kwargs, state = torch.load(conf['model_path'], 'cpu')
        model = klass(*args, **kwargs)
        model.load_state_dict(state)

        # Handle device placement
        if conf['use_gpu']:
            model.cuda() 
        device = next(model.parameters()).device  

        # Create references matrix
        references = th.stack([th.from_numpy(track.targets[name].audio) for name in source_names])
        references = references.numpy()        

        # Get activations
        H = []
        for name in source_names:
            audio = track.targets[name].audio
            H.append(audio)
        H = np.array(H)
        _, bn_ch1, _ = compute_activation_confidence(H[:, :, 0], theta=conf['th'], hilb=False)
        _, bn_ch2, _ = compute_activation_confidence(H[:, :, 1], theta=conf['th'], hilb=False)
        activations = th.from_numpy(np.stack((bn_ch1, bn_ch2), axis=2))

        # FINE TUNING
        if conf['method'] != 'baseline':
            print('ADAPTATION')
            torch.set_grad_enabled(True)

            # Freeze layers
            freeze(model.encoder)
            freeze(model.separator, n=conf['frozen_layers'])
            if conf['freeze_decoder']:
                freeze(model.decoder)

            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['lr_fine'])
            optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['lr_fine'])
            loss_func = nn.L1Loss() 
                
            # Initialize writer for Tensorboard
            writer = SummaryWriter(log_dir=local_save_dir)
            for epoch in range(conf['epochs_fine']):
                total_loss = 0
                epoch_loss = 0

                total_rec = 0
                epoch_rec = 0
                total_act = 0
                epoch_act = 0

                if conf['monitor_metrics']:
                    total_sdr = dict([(key, 0) for key in source_names])
                    epoch_sdr = dict([(key, 0) for key in source_names])

                    total_sir = dict([(key, 0) for key in source_names])
                    epoch_sir = dict([(key, 0) for key in source_names])

                    total_sar = dict([(key, 0) for key in source_names])
                    epoch_sar = dict([(key, 0) for key in source_names])

                    total_isr = dict([(key, 0) for key in source_names])
                    epoch_isr = dict([(key, 0) for key in source_names])              

                # Data loader with eventually data augmentation
                mix_set = DAdataloader(mix.numpy(), 
                                        win=conf['win_fine'], 
                                        hop=conf['hop_fine'], 
                                        sample_rate=conf['sample_rate'], 
                                        n_observations=conf['n_observations'],
                                        pitch_list=conf['pitch_list'], 
                                        min_semitones=conf['min_semitones'],
                                        max_semitones=conf['max_semitones'],
                                        same_pitch_list_all_chunks=conf['same_pitch_list_all_chunks'])

                # Iterate over chuncks 
                for t, item in enumerate(mix_set):
                    sample, win, _ = item
                    mix_chunk = th.from_numpy(sample[None, :, :]).to(device) 
                    est_chunk = model(cp(mix_chunk)) 

                    act_chunk = activations[None, :, win, :].transpose(3, 2).to(device)
                    loss_act = loss_func(est_chunk * (1-act_chunk), torch.zeros_like(est_chunk))
                    
                    if conf['method'] == 'LmixLact':
                        loss_rec = loss_func(mix_chunk, torch.sum(est_chunk * act_chunk, dim=1))
                        loss = loss_rec + conf['gamma'] * loss_act

                    if conf['method'] == 'Lmix':
                        loss_rec = loss_func(mix_chunk, torch.sum(est_chunk, dim=1))
                        loss = loss_rec

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()


                    total_loss += loss.item()
                    epoch_loss = total_loss / (1 + t)

                    total_rec += loss_rec.item()
                    total_act += loss_act.item()

                    epoch_rec = total_rec / (1 + t)
                    epoch_act= total_act/ (1 + t)

                    # Monitor sdr, sir, and sar over epochs
                    if conf['monitor_metrics']:
                        ref_chunk = references[:, win, :]
                        skip = False
                        for i, target in enumerate(source_names):
                            if np.sum(ref_chunk[i, :, :]**2) == 0:
                                skip = True
                        if not skip:
                            sdr, isr, sir, sar = museval.evaluate(ref_chunk, est_chunk.squeeze().transpose(1, 2).detach().cpu().numpy(), win=np.inf)

                            sdr = np.array(sdr)
                            sir = np.array(sir)
                            sar = np.array(sar)
                            isr = np.array(isr)

                            for i, target in enumerate(source_names):
                                total_sdr[target] += sdr[i]
                                epoch_sdr[target] = total_sdr[target]/ (1 + t)

                                total_sir[target] += sir[i]
                                epoch_sir[target] = total_sir[target]/ (1 + t)

                                total_sar[target] += sar[i]
                                epoch_sar[target] = total_sar[target]/ (1 + t)

                                total_isr[target] += isr[i]
                                epoch_isr[target] = total_isr[target]/ (1 + t)

                if conf['monitor_metrics']:
                    for i, target in enumerate(source_names):
                        writer.add_scalar("SDR/" + target, epoch_sdr[target], epoch)           
                        writer.add_scalar("SIR/" + target, epoch_sir[target], epoch)  
                        writer.add_scalar("SAR/" + target, epoch_sar[target], epoch)  
                        writer.add_scalar("ISR/" + target, epoch_isr[target], epoch)  

                writer.add_scalar("Loss/total", epoch_loss, epoch)
                writer.add_scalar("Loss/rec", epoch_rec, epoch)
                writer.add_scalar("Loss/act", epoch_act, epoch)
                print('epoch, nr of training examples and loss: ', epoch, t, epoch_loss, epoch_rec, epoch_act, epoch_sdr['other'])

            writer.flush()
            writer.close()
          
        # apply model
        print('Apply model')
        estimates = apply_model(model, mix.to(device), shifts=conf['shifts'], split=conf['split'])
        estimates = estimates * ref.std() + ref.mean()
        estimates = estimates.transpose(1, 2).cpu().numpy()     

        # get results of this track
        print('Evaluate model')
        assert references.shape == estimates.shape
        track_store, silence_frames = evaluate_mia(ref=references, 
                                                   est=estimates, 
                                                   track_name=track.name, 
                                                   source_names=source_names, 
                                                   eval_silence=True,
                                                   conf=conf)
        
        # aggregate results over the track and save the partials
        silence_adapt = silence_adapt.append(silence_frames, ignore_index=True)
        silence_adapt.to_json(os.path.join(conf['exp_dir'], 'silence.json'), orient='records')        

        results_adapt.add_track(track_store)
        results_adapt.save(os.path.join(conf['exp_dir'], 'bss_eval_tracks.pkl'))        
        print(results_adapt)              

        # Save some examples with corresponding metrics in a folder
        if idx in save_idx: 
            silence_frames.to_json(os.path.join(local_save_dir, 'silence_frames.json'), orient='records')
            with open(os.path.join(local_save_dir, 'metrics_museval.json'), 'w+') as f:
                f.write(track_store.json)
            sf.write(os.path.join(local_save_dir, "mixture.wav"), mix.transpose(0, 1).cpu().numpy(), conf['sample_rate'])
            for name, estimate, reference, activation in zip(source_names, estimates, references, activations):
                print(name)

                unique, counts = np.unique(activation, return_counts=True)
                print(dict(zip(unique, counts/(len(activation)*2)*100)))

                assert estimate.shape == reference.shape
                sf.write(os.path.join(local_save_dir, name + "_est.wav"), estimate, conf['sample_rate'])
                sf.write(os.path.join(local_save_dir, name + "_ref.wav"), reference, conf['sample_rate'])
                sf.write(os.path.join(local_save_dir, name + "_act.wav"), activation.cpu().numpy(), conf['sample_rate'])

        # Evaluate results when applying the activations to the output
        if conf['apply_act_output']:
            track_store_applyact, _ = evaluate_mia(ref=references, 
                                                    est=estimates * activations.cpu().numpy() , 
                                                    track_name=track.name, 
                                                    source_names=source_names, 
                                                    eval_silence=False,
                                                    conf=conf)  

            # aggregate results over the track and save the partials
            results_applyact.add_track(track_store_applyact)
            print('after applying activations')
            print(results_applyact)    

            results_applyact.save(os.path.join(conf['exp_dir'], 'bss_eval_tracks_applyact.pkl'))  

            # Save some examples with corresponding metrics in a folder
            if idx in save_idx:   
                with open(os.path.join(local_save_dir, 'metrics_museval_applyact.json'), 'w+') as f:
                    f.write(track_store_applyact.json)  

            del track_store_applyact           

        # Delete some variables
        del references, mix, estimates, track, track_store, silence_frames,  model

        # Stop if reached the limit
        if idx == conf['stop_index']:
            break

        print('------------------')


    # Print and save aggregated results
    print('Final results')
    print(results_adapt) 
    method = museval.MethodStore()
    method.add_evalstore(results_adapt, conf['exp_dir'])
    method.save(os.path.join(conf['exp_dir'], 'bss_eval.pkl'))   

    if conf['eval_silence']:
        print("mean over evaluation frames, mean over channels, mean over tracks")
        for target in source_names:
            print(target + ' ==>', silence_adapt.loc[silence_adapt['target'] == target].mean(axis=0, skipna=True))
        silence_adapt.to_json(os.path.join(conf['exp_dir'], 'silence.json'), orient='records')

    print('Final results apply act')
    print(results_applyact) 
    method = museval.MethodStore()
    method.add_evalstore(results_applyact, conf['exp_dir'])
    method.save(os.path.join(conf['exp_dir'], 'bss_eval_applyact.pkl')) 
 


if __name__ == '__main__':
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    print(arg_dic)
    main(arg_dic)
