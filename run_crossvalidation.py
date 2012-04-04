'''
    This file computes a kind of 'leave one out cross-validation'.

    The matrix factorization (spatio-temporal ICA) is computed on data
    from all but one animal. Each animal is left out once. This data
    can then be used to investigate how strong a single animal dominates
    the resulting factorization.

    The whole analysis is divided in two steps. First, and computationally more
    expensive part is done in this file. Results are written to a folder and
    can later be visualized by the plot_crossval.py script.

    @author: stephan
'''

import os, glob, sys
import pickle, json
import itertools as it
import numpy as np
import pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as vis
from NeuralImageProcessing.pipeline import TimeSeries
import utils
reload(bf)
reload(vis)

n_best = 5
frames_per_trial = 40
variance = 5
lowpass = 2
similarity_threshold = 0.3
selection_thres = 0.3
normalize = True
modesim_threshold = 0.5
medianfilter = 5
alpha = 0.9
prefixes = ['OCO', '2PA', 'LIN', 'CVA']
format = 'svg'

add = ''
if normalize:
    add = '_maxnorm'

' +++ jan specific +++'
# base_path = '/home/jan/Documents/dros/new_data/'
# data_path = os.path.join(base_path, 'aligned')
# loadfolder = os.path.join(base_path, 'aligned', 'common_channels')
# savefolder = 'simil' + str(int(similarity_threshold * 100)) + 'n_best' +
# str(n_best) + add + '_' + format
# save_path = os.path.join(base_path, savefolder)

' +++ dedan specific +++'
base_path = '/Users/dedan/projects/fu'
data_path = os.path.join(base_path, 'data', 'dros_calcium_new')
loadfolder = os.path.join(base_path, 'results', 'common_channels')
savefolder = 'nbest-' + str(n_best) + '_thresh-' + str(int(similarity_threshold * 100))
save_path = os.path.join(base_path, 'results', 'cross_val', savefolder)

if not os.path.exists(save_path):
    os.mkdir(save_path)


#####################################################
#        initialize the processing functions
#####################################################

# temporal downsampling by factor 4 (originally 40 frames)
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
# ICA
icaend = bf.stICA(variance, {'alpha':alpha})

# select stimuli such that their mean correlation is below similarity_threshold
stimuli_mask = bf.SampleSimilarity(similarity_threshold)
stimulirep = bf.SampleSimilarityPure()
# select stimuli bases on stimuli mask
stimuli_filter = bf.SelectTrials()
# create mode filter
modefilter = bf.CalcStimulusDrive()
# select modes based on where mask are below threshold
select_modes = bf.SelectModes(modesim_threshold)
#create mean stimuli response
standard_response = bf.SingleSampleResponse(method='mean')
# and calculate distance between modes
combine = bf.ObjectConcat()
combine_common = bf.ObjectScrambledConcat(n_best)
cor_dist = bf.Distance()

for prefix in prefixes:

    filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
    tmp_filelist = []
    for filename in filelist:
        info = json.load(open(filename))
        print "checking file: ", filename
        if 'bad_data' in info:
            print 'skip this file: bad_data flag found'
            continue
        else:
            tmp_filelist.append(filename)
    filelist = tmp_filelist

    colorlist, allodors = {}, []

    # use only the n_best animals --> most stable odors in common
    res = pickle.load(open(os.path.join(data_path, loadfolder, 'thres_res.pckl')))
    best = utils.select_n_channels(res[prefix][selection_thres], n_best)
    filelist = [filelist[i] for i in best]

    for i in range(-1, len(filelist)):

        if i == -1:
            filelist_fold = filelist
            save_name = prefix + '_all'
        else:
            filelist_fold = [filelist[j] for j in range(len(filelist)) if j != i]
            name_list = [os.path.splitext(os.path.basename(f))[0] for f in filelist_fold]
            name_list = [f.split('_')[1] for f in name_list]
            save_name = prefix + "_" + "_".join(sorted(name_list))
            save_name = save_name + "-" + os.path.splitext(os.path.basename(filelist[i]))[0]
        print save_name

        all_raw, baselines = [], []
        for file_ind, filename in enumerate(filelist_fold):

            # load timeseries, shape and labels
            meas_path = os.path.splitext(filename)[0]

            #assign each file a color:
            colorlist[os.path.basename(meas_path)] = plt.cm.jet(file_ind / (len(filelist) - 1.))

            # create timeseries
            ts = bf.TimeSeries()
            ts.load(meas_path)

            # change shape from list to tuple to treat timecourse as one image
            ts.shape = tuple(ts.shape)

            ts = temporal_downsampling(ts)
            baseline = trial_mean(baseline_cut(ts))
            baselines.append(np.mean(baseline.shaped2D(), 0))
            preprocessed = gauss_filter(pixel_filter(rel_change(ts, baseline)))
            preprocessed.timecourses[np.isnan(preprocessed.timecourses)] = 0
            preprocessed.timecourses[np.isinf(preprocessed.timecourses)] = 0

            if normalize:
                preprocessed.timecourses = preprocessed.timecourses / np.max(preprocessed.timecourses)
            mean_resp_unsort = trial_mean(signal_cut(preprocessed))
            mean_resp = sorted_trials(mean_resp_unsort)
            preprocessed = sorted_trials(preprocessed)
            stimuli_selection = stimuli_mask(mean_resp)
            filtered = stimuli_filter(preprocessed, stimuli_selection)

            # remember all available odors from the largest group (size N)
            if i == -1:
                if not allodors:
                    allodors = set(filtered.label_sample)
                else:
                    allodors = allodors.intersection(set(filtered.label_sample))
                all_raw.append(filtered)
            # filter all N-1 folds with allodors from the N fold
            else:
                all_mask = np.zeros(len(filtered.label_sample), dtype=np.bool)
                for i, label in enumerate(filtered.label_sample):
                    if label in allodors:
                        all_mask[i] = 1
                all_filtered = stimuli_filter(filtered, TimeSeries(series=all_mask))
                all_raw.append(all_filtered)


        # stimultanieous ICA
        intersection = sorted_trials(combine_common(all_raw))
        mo2 = icaend(intersection)
        mo2.name = save_name
        mo2 = sorted_trials(standard_response(mo2))
        res = {'base': mo2, 'baselines': baselines, 'names': filelist_fold}
        pickle.dump(res, open(os.path.join(save_path, save_name + '.pckl'), 'w'))

