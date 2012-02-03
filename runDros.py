'''
Created on Aug 11, 2011

@author: jan
'''

import os
import json
import glob

import numpy as np
import basic_functions as bf
import illustrate_decomposition as vis
import pylab as plt

from scipy.ndimage import filters as filters
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

reload(bf)
reload(vis)

frames_per_trial = 40
variance = 0.9
lowpass = 0.5
similarity_threshold = 0.2
modesim_threshold = 0.3
medianfilter = 8
data_path = '/home/jan/Documents/dros/new_data/numpyfiles/'
#data_path='/Users/dedan/projects/fu/data/dros_calcium/test_data/'
savefolder = 'med' + str(medianfilter) + 'simil' + str(similarity_threshold * 10) + 'mode' + str(modesim_threshold * 10)
save_path = os.path.join(data_path, savefolder)
if not os.path.exists(save_path):
    os.mkdir(save_path)
prefix = 'LIN'

#filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
filelist = [os.path.join(data_path, 'LIN_111026a.json')]

#####################################################
#        initialize the processing functions
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
# ICA
ica = bf.sICA(variance=variance)
# select stimuli such that their mean correlation is below similarity_threshold
stimuli_mask = bf.SampleSimilarity(similarity_threshold)
# select stimuli bases on stimuli mask
stimuli_filter = bf.SelectTrials()
# create mode filter
modefilter = bf.CalcStimulusDrive()
# select modes based on where mask are below threshold
select_modes = bf.SelectModes(modesim_threshold)
#create mean stimuli response
standard_response = bf.MeanSampleResponse()


for filename in filelist:  
    
    # load timeseries, shape and labels
    meas_path = os.path.splitext(filename)[0]
    timeseries = np.load(meas_path + '.npy')
    info = json.load(open(meas_path + '.json')) 
    label = info['labels']
    shape = info['shape']

    # create trial labels 
    label = [i.strip('.png') for i in label[::frames_per_trial]]
    
    # create timeseries
    ts = bf.TimeSeries(shape=shape, series=timeseries, name=os.path.basename(meas_path),
                       label_sample=label)
    
    ts = temporal_downsampling(ts)
    baseline = trial_mean(baseline_cut(ts))
    preprocessed = sorted_trials(pixel_filter(rel_change(ts, baseline)))
    raw_ica = ica(preprocessed)
    mean_resp = trial_mean(signal_cut(preprocessed))
    stimuli_selection = stimuli_mask(mean_resp)
    mode_cor = modefilter(stimuli_filter(raw_ica, stimuli_selection))
    selected_ica = select_modes(raw_ica, mode_cor)
    selected_ica_and_trial = stimuli_filter(selected_ica, stimuli_selection)
    full_selection_condensed = trial_mean(selected_ica_and_trial)
    
    ####################################################################
    # cluster modes
    ####################################################################    
    
    #modedist = -(np.abs(pdist(selected_ica_and_trial.timecourses.T, 'correlation') - 1) - 1)
    modedist = pdist(full_selection_condensed.timecourses.T, 'correlation')
    d = dendrogram(linkage(modedist + 1E-10, 'single'), labels=selected_ica_and_trial.label_objects, leaf_font_size=12)


    ####################################################################
    # plot and save results
    ####################################################################    
    
    # save plot and data
    tmp_save = os.path.join(save_path, os.path.basename(meas_path))
    
    
    # draw signal overview
    resp_overview = vis.VisualizeTimeseries()
    resp_overview.subplot(mean_resp.samplepoints)
    resp_overview.imshow('base', 'onetoone', mean_resp, title=True, colorbar=True)
    
    # draw ica overview
    toplot = raw_ica
    ica_overview = vis.VisualizeTimeseries()
    ica_overview.base_and_time(toplot.num_objects)
    ica_overview.imshow('base', 'onetoone', toplot.base)
    ica_overview.plot('time', 'onetoone', toplot)
    ica_overview.add_labelshade('time', 'onetoall', toplot)
    ica_overview.add_samplelabel([-1], toplot, rotation='45', toppos=True)
    
    preprocessed.save(tmp_save + 'prepocess')
    raw_ica.save(tmp_save + 'rawica')
    
    
    plt.show()

    
    
