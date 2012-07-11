#!/usr/bin/env python
# encoding: utf-8
"""
I think this file was used to determine a reasonable number of modes. It plots
the eigenvalue spectrum of the ICA to investigate a good number of components.

It does not work anymore with the other components of the code but should be
easy to fix.

Created by Stephan on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import logging
import os
import json
import glob
import basic_functions as bf

import numpy as np
import pylab as plt
from scipy.ndimage import filters as filters
from scipy.spatial.distance import pdist


# logger setup
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger()

n_modes_list = range(3, 50, 3)
frames_per_trial = 40
lowpass = 0.5
similarity_threshold = 0.2
modesim_threshold = 0.3
medianfilter = 8
data_path='/Users/dedan/projects/fu/data/dros_calcium/'
save_path = os.path.join(data_path, 'mode_num_out')
if not os.path.exists(save_path):
    os.mkdir(save_path)
prefix = ''

filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
# filelist = [os.path.join(data_path, 'LIN_111026a.json')]

#####################################################
#       initialize the processing functions
#####################################################


# temporal downsampling by factor 2 (originally 40 frames)
temporal_downsampling = bf.TrialMean(20)
# cut baseline signal (odor starts at frame 4 (original frame8))
baseline_cut = bf.CutOut((0, 3))
# signal cut
signal_cut = bf.CutOut((6, 13))
# calculate trial mean
trial_mean = bf.TrialMean()
# calculate (delta F) / F
rel_change = bf.RelativeChange()
# MedianFilter
pixel_filter = bf.Filter(filters.median_filter, {'size':medianfilter}, downscale=2)
#sorting
sorted_trials = bf.SortBySamplename()
# select stimuli such that their mean correlation is below similarity_threshold
stimuli_mask = bf.SampleSimilarity(similarity_threshold)
# select stimuli bases on stimuli mask
stimuli_filter = bf.SelectTrials()
# create mode filter
modefilter = bf.CalcStimulusDrive()
# select modes based on where mask are below threshold
select_modes = bf.SelectModes(modesim_threshold)


for filename in filelist:

    logger.info(filename)
    # load timeseries, shape and labels
    meas_path = os.path.splitext(filename)[0]
    timeseries = np.load(meas_path + '.npy')
    info = json.load(open(meas_path + '.json'))
    label = info['labels']
    shape = info['shape']
    # create trial labels
    label = [i.strip('.png') for i in label[::frames_per_trial]]
    # create timeseries
    ts = bf.TimeSeries(shape=shape, series=timeseries, 
                       name=os.path.basename(meas_path),
                       label_sample=label)
    ts = temporal_downsampling(ts)
    baseline = trial_mean(baseline_cut(ts))
    preprocessed = sorted_trials(pixel_filter(rel_change(ts, baseline)))

    res = {'eigen': [], 'stim_driven': [], 'dist_dists': []}
    for n_modes in n_modes_list:

        # ICA
        ica = bf.sICA(variance=n_modes)
        raw_ica = ica(preprocessed)
        res['eigen'].append(np.sum(raw_ica.eigen))
        mean_resp = trial_mean(signal_cut(preprocessed))
        stimuli_selection = stimuli_mask(mean_resp)
        mode_cor = modefilter(stimuli_filter(raw_ica, stimuli_selection))
        res['stim_driven'].append(np.sum(mode_cor.timecourses < modesim_threshold))
        selected_ica = select_modes(raw_ica, mode_cor)
        selected_ica_and_trial = stimuli_filter(selected_ica, stimuli_selection)
        full_selection_condensed = trial_mean(signal_cut(selected_ica_and_trial))
        res['dist_dists'].append(pdist(full_selection_condensed.timecourses.T, 'correlation'))

    ####################################################################
    # cluster modes
    ####################################################################

    # save plot and data
    tmp_save = os.path.join(save_path, os.path.basename(meas_path))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(n_modes_list, res['eigen'])
    plt.title('explained variance')
    plt.subplot(2, 1, 2)
    plt.plot(n_modes_list, res['stim_driven'])
    plt.title('number of stimulus driven')
    plt.xticks(n_modes_list)
    plt.grid()
    plt.savefig(tmp_save + '_p1.png')

    plt.figure()
    for i, dist in enumerate(res['dist_dists']):
        plt.subplot(len(n_modes_list), 1, i+1)
        if len(dist) > 0:
            plt.hist(dist, 100)
        plt.ylabel(n_modes_list[i])
    plt.savefig(tmp_save + '_p2.png')
    plt.close('all')

    #modedist = -(np.abs(pdist(selected_ica_and_trial.timecourses.T, 'correlation') - 1) - 1)

