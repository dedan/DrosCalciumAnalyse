'''
investigate the influence of the sample similarity threshold on how many common
channels are available for a group of n animals. The sample similarity is a
value that shows how consistent the response is over repetitions. All over this
script the term *channels* refers to what we usually called *stimuli* or *odors*

requirements: this script assumes the data to be stored in the following structure
                * basepath/data/data_folder
                * basepath/results/common_channels

input: raw_data (the raw data as saved TimeSeries objects)
output: * plots of # of usable channels for certain thresholds (depending on the # of animals)
        * a pickle containing this information for all thresholds
        (this can be used in other scripts to determine the N animals wich have a maximum
        number of channels in common)
        * this pickle is used in run_crossvalidation and runDros

@author: stephan.gabler@gmail.com
'''

import os, glob, pickle, json, sys
from collections import defaultdict
import numpy as np
import pylab as plt
import utils
from NeuralImageProcessing import basic_functions as bf
import matplotlib.gridspec as gridspec

base_path = '/Users/dedan/projects/fu/'
prefixes = ['LIN']#, '2PA', 'CVA', 'OCO']
thresholds = [round(t, 1) for t in np.linspace(0.1, 1, 10)]
lowpass = 2
medianfilter = 5
data_path = os.path.join(base_path, 'data', 'dros_calcium_new')
save_path = os.path.join(base_path, 'results', 'common_channels')
if not os.path.exists(save_path):
    os.mkdir(save_path)

res = {}

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
# select stimuli bases on stimuli mask
stimuli_filter = bf.SelectTrials()

for prefix in prefixes:

    print 'working on: %s' % prefix
    filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')

    collect = defaultdict(list)
    res[prefix] = {}

    for file_ind, filename in enumerate(filelist):

        print "loading file: ", filename
        info = json.load(open(filename))
        if 'bad_data' in info:
            print 'skip this file: bad_data flag found'
            continue
        # create timeseries and change shape from list to tuple
        meas_path = os.path.splitext(filename)[0]
        ts = bf.TimeSeries()
        ts.load(meas_path)
        ts.shape = tuple(ts.shape)

        ts = temporal_downsampling(ts)
        baseline = trial_mean(baseline_cut(ts))
        preprocessed = gauss_filter(pixel_filter(rel_change(ts, baseline)))
        preprocessed.timecourses[np.isnan(preprocessed.timecourses)] = 0
        preprocessed.timecourses[np.isinf(preprocessed.timecourses)] = 0
        mean_resp_unsort = trial_mean(signal_cut(preprocessed))
        mean_resp = sorted_trials(mean_resp_unsort)
        preprocessed = sorted_trials(preprocessed)

        for thres in thresholds:
            stimuli_mask = bf.SampleSimilarity(thres)
            stimuli_selection = stimuli_mask(mean_resp)
            collect[thres].append(stimuli_filter(preprocessed, stimuli_selection))

    for thres in thresholds:
        allodors = list(set(ts.label_sample + sum([t.label_sample for t in collect[thres]], [])))
        allodors.sort()
        quality_mx = np.zeros((len(collect[thres]), len(allodors)))
        for t_ind, t in enumerate(collect[thres]):
            for od in set(t.label_sample):
                quality_mx[t_ind, allodors.index(od)] = 1
        res[prefix][thres] = quality_mx

pickle.dump(res, open(os.path.join(save_path, 'thres_res.pckl'), 'w'))

# plotting
for prefix in res.keys():

    fig = plt.figure()
    fig.suptitle(prefix)
    gs = gridspec.GridSpec(len(range(2, 7)), 1)
    gs.update(hspace=0.7)

    thresholds = sorted(res[prefix].keys())
    for n in range(2, 7):
        p = fig.add_subplot(gs[n - 2, 0])
        channels = np.zeros(len(thresholds))

        for i, thres in enumerate(thresholds):
            quality_mx = res[prefix][thres]
            best = utils.select_n_channels(quality_mx, n)
            channels[i] = sum((np.sum(quality_mx[best, :], 0) == n).astype('int'))
        p.grid()
        p.plot(thresholds, channels.T)
        p.set_title('# of animals: %d' % int(n))
        p.set_xticklabels([])
        p.set_yticks(range(0, np.size(quality_mx, 1), 5))
    p.set_xlabel('threshold')
    p.set_xticklabels(thresholds)
    fig.savefig(os.path.join(save_path, 'groupsize_' + prefix + '.png'))

