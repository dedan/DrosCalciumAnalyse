'''

create the plots which show the matrix of channels available in common
for n animals for a given threshold.

@author: stephan.gabler@gmail.com
'''

import os, glob
import numpy as np
import pylab as plt
from NeuralImageProcessing import basic_functions as bf
import utils
reload(bf)

lowpass = 2
similarity_threshold = 0.3
medianfilter = 5
base_path = '/Users/dedan/projects/fu/'
data_path = os.path.join(base_path, 'data', 'dros_calcium_new')
save_path = os.path.join(base_path, 'results', 'common_channels')
if not os.path.exists(save_path):
    os.mkdir(save_path)

prefixes = ['CVA', 'LIN', '2PA', 'OCO']

#####################################################
#        initialize the processing functions
#####################################################

# temporal downsampling by factor 2 (originally 40 frames)
temporal_downsampling = bf.TrialMean(20)
# cut baseline signal (odor starts at frame 4 (original frame8))
baseline_cut = bf.CutOut((0, 3))
# signal cut
signal_cut = bf.CutOut((6, 12))
# calculate trial mean
trial_mean = bf.TrialMean()
# calculate (delta F) / F
rel_change = bf.RelativeChange()
# MedianFilter
pixel_filter = bf.Filter('median', medianfilter)
gauss_filter = bf.Filter('gauss', lowpass, downscale=3)
#sorting
sorted_trials = bf.SortBySamplename()
# select stimuli such that their mean correlation is below similarity_threshold
stimuli_mask = bf.SampleSimilarity(similarity_threshold)
stimulirep = bf.SampleSimilarityPure()
# select stimuli bases on stimuli mask
stimuli_filter = bf.SelectTrials()

for prefix in prefixes:

    filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
    all_raw = []

    for file_ind, filename in enumerate(filelist):

        # create timeseries
        meas_path = os.path.splitext(filename)[0]
        ts = bf.TimeSeries()
        ts.load(meas_path)
        ts.shape = tuple(ts.shape)

        # apply the processing pipeline
        ts = temporal_downsampling(ts)
        baseline = trial_mean(baseline_cut(ts))
        preprocessed = gauss_filter(pixel_filter(rel_change(ts, baseline)))
        preprocessed.timecourses[np.isnan(preprocessed.timecourses)] = 0
        preprocessed.timecourses[np.isinf(preprocessed.timecourses)] = 0
        mean_resp_unsort = trial_mean(signal_cut(preprocessed))
        mean_resp = sorted_trials(mean_resp_unsort)
        preprocessed = sorted_trials(preprocessed)
        stimuli_selection = stimuli_mask(mean_resp)

        all_raw.append(stimuli_filter(preprocessed, stimuli_selection))

    # create matrix that contains which odors are available for which animal
    # when filtered for the given threshold
    allodors = list(set(ts.label_sample + np.sum([t.label_sample for t in all_raw])))
    allodors.sort()
    quality_mx = np.zeros((len(all_raw), len(allodors)))
    for t_ind, t in enumerate(all_raw):
        for od in set(t.label_sample):
            quality_mx[t_ind, allodors.index(od)] = 1
    x, y = np.shape(quality_mx)
    aspect_ratio = y/float(x)

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(6, 1, 1)
    ax.imshow(quality_mx, interpolation='nearest', cmap=plt.cm.bone)
    ax.set_xticks(range(len(allodors)))
    ax.set_xticklabels([])
    ax.set_yticks(range(len(all_raw)))
    ax.set_yticklabels([t.name for t in all_raw])
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    utils.force_aspect(ax, aspect_ratio)

    # plot only i best animals (best --> most common channels)
    for i in range(2, 7):

        ax = fig.add_subplot(6, 1, i)
        best = utils.select_n_channels(quality_mx, i)
        ax.imshow(quality_mx[best, :], interpolation='nearest', cmap=plt.cm.bone)
        ax.set_yticks(range(i))
        ax.set_yticklabels([t.name for t in [all_raw[b] for b in best]])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10)
        ax.set_xticks(range(len(allodors)))
        ax.set_xticklabels([], rotation='45')
        utils.force_aspect(ax, aspect_ratio)
    ax.set_xticks(range(len(allodors)))
    ax.set_xticklabels(allodors, rotation='45')
    fig.savefig(os.path.join(save_path, prefix + '_common_channels.png'))
