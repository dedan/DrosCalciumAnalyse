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
variance = 25
lowpass = 0.5
similarity_threshold = 0.4
modesim_threshold = 0.5
medianfilter = 8
data_path = '/home/jan/Documents/dros/new_data/numpyfiles/'
#data_path='/Users/dedan/projects/fu/data/dros_calcium/test_data/'
savefolder = 'med' + str(medianfilter) + 'simil' + str(similarity_threshold * 10) + 'mode' + str(modesim_threshold * 10)
save_path = os.path.join(data_path, savefolder)
if not os.path.exists(save_path):
    os.mkdir(save_path)
prefix = 'OCO'

filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
#filelist = [os.path.join(data_path, 'LIN_111026a.json')]

colorlist = {}

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
pixel_filter = bf.Filter('median', medianfilter, downscale=2)
#sorting
sorted_trials = bf.SortBySamplename()
# ICA
ica = bf.stICA(variance=variance, param={'alpha':1E-10})
# select stimuli such that their mean correlation is below similarity_threshold
stimuli_mask = bf.SampleSimilarity(similarity_threshold)
# select stimuli bases on stimuli mask
stimuli_filter = bf.SelectTrials()
# create mode filter
modefilter = bf.CalcStimulusDrive()
# select modes based on where mask are below threshold
select_modes = bf.SelectModes(modesim_threshold)
#create mean stimuli response
standard_response = bf.SingleSampleResponse()
# and calculate distance between modes
combine = bf.ObjectConcat()
cor_dist = bf.Distance()

#create lists to collect results
all_sel_modes, all_sel_modes_condensed = [], []
all_stimulifilter = []

for file_ind, filename in enumerate(filelist):  
    
    # load timeseries, shape and labels
    meas_path = os.path.splitext(filename)[0]
    timeseries = np.load(meas_path + '.npy')
    info = json.load(open(meas_path + '.json')) 
    label = info['labels']
    shape = info['shape']
    
    #assign each file a color:
    colorlist[os.path.basename(meas_path)] = plt.cm.jet(file_ind / (len(filelist) - 1.))
        
    # create trial labels 
    label = [i.strip('.png') for i in label[::frames_per_trial]]
    
    # create timeseries
    ts = bf.TimeSeries(shape=shape, series=timeseries, name=os.path.basename(meas_path),
                       label_sample=label)
    
    ts = temporal_downsampling(ts)
    baseline = trial_mean(baseline_cut(ts))
    preprocessed = sorted_trials(pixel_filter(rel_change(ts, baseline)))
    preprocessed.timecourses[np.isnan(preprocessed.timecourses)] = 0
    raw_ica = ica(preprocessed)
    mean_resp = trial_mean(signal_cut(preprocessed))
    stimuli_selection = stimuli_mask(mean_resp)
    mode_cor = modefilter(stimuli_filter(raw_ica, stimuli_selection))
    selected_ica = select_modes(raw_ica, mode_cor)
    selected_ica_and_trial = stimuli_filter(selected_ica, stimuli_selection)
    final_modes = sorted_trials(standard_response(selected_ica_and_trial))
    final_modes_condensed = trial_mean(signal_cut(final_modes))
    
   
    all_sel_modes.append(final_modes)
    all_sel_modes_condensed.append(final_modes_condensed)
    
    ####################################################################
    # plot and save results
    ####################################################################    
    
    # save plot and data
    tmp_save = os.path.join(save_path, os.path.basename(meas_path))
    
    '''
    # draw signal overview
    resp_overview = vis.VisualizeTimeseries()
    resp_overview.subplot(mean_resp.samplepoints)
    resp_overview.imshow('base', 'onetoone', mean_resp, title=True, colorbar=True)
    '''
    
    # draw ica overview
    toplot = selected_ica
    ica_overview = vis.VisualizeTimeseries()
    ica_overview.base_and_time(toplot.num_objects)
    ica_overview.imshow('base', 'onetoone', toplot.base)
    ica_overview.plot('time', 'onetoone', toplot)
    ica_overview.add_labelshade('time', 'onetoall', toplot)
    ica_overview.add_samplelabel([-1], toplot, rotation='45', toppos=True)
    [ax.set_title(toplot.label_objects[i]) for i, ax in enumerate(ica_overview.axes['base'])]
    ica_overview.fig.savefig(tmp_save + '_goodmodes')
    
    preprocessed.save(tmp_save + '_prepocess')
    raw_ica.save(tmp_save + '_rawica')

####################################################################
# cluster modes
####################################################################    

#create dictionary with mode distances      
def cordist(modelist):
    #modelist.pop(4)
    #modelist.pop(0)
    alldist_dic = {}
    for i in range(len(modelist)):
        for j in range(i + 1, len(modelist)):
            stimuli_i = modelist[i].label_sample
            stimuli_j = modelist[j].label_sample
            common = set(stimuli_i).intersection(stimuli_j)
            #print i, modelist[i].name, 'good simuli: ', len(stimuli_i)
            #print j, modelist[j].name, 'good simuli: ', len(stimuli_j)
            print 'overlap: ', len(common)
            mask_i, mask_j = np.zeros(len(stimuli_i), dtype='bool'), np.zeros(len(stimuli_j), dtype='bool')
            for stim in common:
                mask_i[stimuli_i.index(stim)] = True
                mask_j[stimuli_j.index(stim)] = True
            ts1 = stimuli_filter(modelist[i], bf.TimeSeries(mask_i))
            ts2 = stimuli_filter(modelist[j], bf.TimeSeries(mask_j))
            cor = cor_dist(combine([ts1, ts2]))
            alldist_dic.update(cor.as_dict('objects'))
    return alldist_dic


# helper function to convert dictionary to pdist format
def dict2pdist(dic):
    key_parts = (reduce(lambda x, y: x + y, [i.split(':') for i in dic.keys()]))
    key_parts = list(set(key_parts))
    new_pdist, new_labels = [], []
    for i in range(len(key_parts)):
        new_labels.append(key_parts[i]) 
        for j in range(i + 1, len(key_parts)):
            try:
                new_pdist.append(dic[':'.join([key_parts[i], key_parts[j]])])
            except KeyError:
                new_pdist.append(dic[':'.join([key_parts[j], key_parts[i]])])
    return new_pdist, new_labels

#creates dendrogram
def dendro(modelist, titleextra=''):
    modedist_dic = cordist(modelist)
    modedist, lables = dict2pdist(modedist_dic) 
    lables = ['_'.join(lab[4:].split('_stica_mode')) for lab in lables]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(prefix + titleextra)
    
    d = dendrogram(linkage(np.array(modedist).squeeze() + 1E-10, 'single'), labels=lables, leaf_font_size=12)
    
    group_colors = []
    for i in d['ivl']:
        group_colors.append(colorlist[prefix + '_' + '_'.join(i.split('_')[:-1])])
     
    labelsn = ax.get_xticklabels()
    for j, i in enumerate(labelsn):
        i.set_color(group_colors[j])

    
dendro(all_sel_modes_condensed, 'condensed')
dendro(all_sel_modes)
