'''
Created on Aug 11, 2011

@author: jan
'''

import os, glob, sys
import pickle
import itertools as it
import numpy as np
import pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as vis
import utils
reload(bf)
reload(vis)

n_best = 6
frames_per_trial = 40
variance = 10
lowpass = 2
similarity_threshold = 0.8
normalize = False
modesim_threshold = 0.5
medianfilter = 5
alpha = 0.9

format = 'svg'

add = ''
if normalize:
    add = '_maxnorm'

' +++ jan specific +++'
base_path = '/home/jan/Documents/dros/new_data/'
data_path = os.path.join(base_path, 'aligned')
loadfolder = os.path.join(base_path, 'aligned', 'common_channels')
savefolder = 'simil' + str(int(similarity_threshold * 100)) + 'n_best' + str(n_best) + add + '_' + format
save_path = os.path.join(base_path, savefolder)

#' +++ dedan specific +++'
#base_path = '/Users/dedan/projects/fu'
#data_path = os.path.join(base_path, 'data', 'dros_calcium_new')
#loadfolder = os.path.join(base_path, 'results', 'common_channels')
#savefolder = 'simil' + str(int(similarity_threshold * 100)) + 'n_best' + str(n_best) + add + '_' + format
#save_path = os.path.join(base_path, 'results', savefolder)

if not os.path.exists(save_path):
    os.mkdir(save_path)

#prefix = 'LIN'

prefixes = ['mic']


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
#ica = bf.stICA(variance=variance, param={'alpha':0.001})
#ica = bf.sICA(variance=variance)
pca = bf.PCA(variance)
#icaend = bf.sICA(latent_series=True)
icaend = bf.stICA(variance, {'alpha':alpha})
icain = bf.stICA(variance, {'alpha':alpha})

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
standard_response = bf.SingleSampleResponse(method='best')
# and calculate distance between modes
combine = bf.ObjectConcat()
#combine_common = bf.ObjectConcat(unequalsample=2, unequalobj=True)
#combine_common = bf.ObjectScrambledConcat(4, 'three')
combine_common = bf.ObjectScrambledConcat(n_best)
cor_dist = bf.Distance()

for prefix in prefixes:

    filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
    colorlist = {}

    #create lists to collect results
    all_sel_modes, all_sel_modes_condensed, all_raw = [], [], []
    baselines = []
    all_stimulifilter = []

    for file_ind, filename in enumerate(filelist):

        # load timeseries, shape and labels
        meas_path = os.path.splitext(filename)[0]

        #assign each file a color:
        f_ind = file_ind / (len(filelist) - 1.)
        colorlist[os.path.basename(meas_path)] = plt.cm.jet(f_ind)

        # create timeseries
        ts = bf.TimeSeries()
        ts.load(meas_path)

        # change shape from list to tuple!!
        ts.shape = tuple(ts.shape)

        ts = temporal_downsampling(ts)
        baseline = trial_mean(baseline_cut(ts))
        baselines.append(baseline)
        pp = gauss_filter(pixel_filter(rel_change(ts, baseline)))
        pp.timecourses[np.isnan(pp.timecourses)] = 0
        pp.timecourses[np.isinf(pp.timecourses)] = 0

        if normalize:
            pp.timecourses = pp.timecourses / np.max(pp.timecourses)
        mean_resp_unsort = trial_mean(signal_cut(pp))
        mean_resp = sorted_trials(mean_resp_unsort)
        pp = sorted_trials(pp)

        
        ####################################################################
        # do individual matrix factorization
        ####################################################################
        
        icain = bf.NNMA(variance, 30, **{'sparse_par': 0, 'smoothness':0.2, 'sparse_par2':0.005})
        raw_ica = icain(pp)
        
#        # select only stimylidriven modes
#        mode_cor = modefilter(stimuli_filter(raw_ica, stimuli_selection))
#        selected_ica = select_modes(raw_ica, mode_cor)
#        selected_ica = select_modes(stim_ica, mode_cor)
#        selected_ica_and_trial = stimuli_filter(selected_ica, stimuli_selection)
#        final_modes = sorted_trials(standard_response(selected_ica_and_trial))
#        final_modes_condensed = trial_mean(signal_cut(final_modes))
       
        
        ####################################################################
        # plot and save results
        ####################################################################

        # save plot and data
        tmp_save = os.path.join(save_path, os.path.basename(meas_path))

        # draw signal overview
        resp_overview = vis.VisualizeTimeseries()
        resp_overview.subplot(mean_resp.samplepoints)
        for ind, resp in enumerate(mean_resp.shaped2D()):
            resp_overview.imshow(resp_overview.axes['base'][ind],
                                  resp,
                                  title=mean_resp.label_sample[ind],
                                  colorbar=True)
        resp_overview.fig.savefig(tmp_save + '_overview')

        # draw unsorted signal overview
        uresp_overview = vis.VisualizeTimeseries()
        uresp_overview.subplot(mean_resp_unsort.samplepoints)
        for ind, resp in enumerate(mean_resp_unsort.shaped2D()):
            uresp_overview.imshow(uresp_overview.axes['base'][ind],
                                   resp,
                                   title=mean_resp_unsort.label_sample[ind],
                                   colorbar=True)
        uresp_overview.fig.savefig(tmp_save + '_overview_unsort')
      
        # draw individual matrix factorization overview
        toplot = raw_ica
        ica_overview = vis.VisualizeTimeseries()
        ica_overview.base_and_time(toplot.num_objects)
        for ind, resp in enumerate(toplot.base.shaped2D()):
            ica_overview.imshow(ica_overview.axes['base'][ind],
                                resp,
                                title=toplot.label_sample[ind])
            ica_overview.plot(ica_overview.axes['time'][ind],
                              toplot.timecourses[:, ind])
            ica_overview.add_labelshade(ica_overview.axes['time'][ind], toplot)
            #ica_overview.add_shade('time', 'onetoall', stimuli_selection, 20)
        ica_overview.add_samplelabel(ica_overview.axes['time'][-1], toplot, rotation='45', toppos=True)
        [ax.set_title(toplot.label_objects[i]) for i, ax in enumerate(ica_overview.axes['base'])]
        ica_overview.fig.savefig(tmp_save + '_modes.svg')


        plt.close('all')

