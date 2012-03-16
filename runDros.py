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

n_best = 3
frames_per_trial = 40
variance = 5
lowpass = 2
similarity_threshold = 0.6
modesim_threshold = 0.5
medianfilter = 5
data_path = '/Users/dedan/projects/fu/data/dros_calcium_new/'
loadfolder = 'common_channels'
savefolder = 'simil' + str(int(similarity_threshold * 100)) + 'n_best' + str(n_best)
save_path = os.path.join(data_path, savefolder)
if not os.path.exists(save_path):
    os.mkdir(save_path)

prefix = 'LIN'


filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
colorlist = {}

# use only the n_best animals --> most stable odors in common
res = pickle.load(open(os.path.join(data_path, loadfolder, 'thres_res.pckl')))
best = utils.select_n_channels(res[prefix][0.3], n_best)
filelist = [filelist[i] for i in best]



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
icaend = bf.stICA(variance, {'alpha':0.9})
icain = bf.sICA(variance)

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
combine_common = bf.ObjectConcat(unequalsample=True, unequalobj=True)
cor_dist = bf.Distance()

#create lists to collect results
all_sel_modes, all_sel_modes_condensed, all_raw = [], [], []
baselines = []
all_stimulifilter = []

for file_ind, filename in enumerate(filelist):

    # load timeseries, shape and labels
    meas_path = os.path.splitext(filename)[0]

    #assign each file a color:
    colorlist[os.path.basename(meas_path)] = plt.cm.jet(file_ind / (len(filelist) - 1.))

    # create timeseries
    ts = bf.TimeSeries()
    ts.load(meas_path)

    # change shape from list to tuple!!
    ts.shape = tuple(ts.shape)

    ts = temporal_downsampling(ts)
    baseline = trial_mean(baseline_cut(ts))
    baselines.append(baseline)
    preprocessed = gauss_filter(pixel_filter(rel_change(ts, baseline)))
    preprocessed.timecourses[np.isnan(preprocessed.timecourses)] = 0
    preprocessed.timecourses[np.isinf(preprocessed.timecourses)] = 0
    mean_resp_unsort = trial_mean(signal_cut(preprocessed))
    mean_resp = sorted_trials(mean_resp_unsort)
    preprocessed = sorted_trials(preprocessed)
    stimuli_selection = stimuli_mask(mean_resp)

    #raw_ica = icain(preprocessed)
    '''
    ica = bf.NNMA(variance, 30, {'sparse_par': 0.1, 'smoothness':0.2, 'sparse_par2':0.3})
    raw_ica = ica(preprocessed)

    #stim_ica = ica(stimuli_filter(preprocessed, stimuli_selection))
    mode_cor = modefilter(stimuli_filter(raw_ica, stimuli_selection))
    #mode_cor = modefilter(stim_ica)
    selected_ica = select_modes(raw_ica, mode_cor)
    #selected_ica = select_modes(stim_ica, mode_cor)
    selected_ica_and_trial = stimuli_filter(selected_ica, stimuli_selection)
    final_modes = sorted_trials(standard_response(selected_ica_and_trial))
    final_modes_condensed = trial_mean(signal_cut(final_modes))


    all_sel_modes.append(final_modes)
    all_sel_modes_condensed.append(final_modes_condensed)
    '''
    all_raw.append(stimuli_filter(preprocessed, stimuli_selection))
    distanceself, distancecross = stimulirep(mean_resp)
    ####################################################################
    # plot and save results
    ####################################################################

    # save plot and data
    tmp_save = os.path.join(save_path, os.path.basename(meas_path))

    #draw quality overview
    qual_view = vis.VisualizeTimeseries()
    qual_view.oneaxes()
    data = zip(*distancecross.items())
    vis.violin_plot(qual_view.axes['time'][0], [i.flatten() for i in data[1]] ,
                    range(len(distancecross)), 'b')
    qual_view.axes['time'][0].set_xticks(range(len(distancecross)))
    qual_view.axes['time'][0].set_xticklabels(data[0])
    qual_view.axes['time'][0].set_title(ts.name)
    for pos, stim in enumerate(data[0]):
        qual_view.axes['time'][0].plot([pos] * len(distanceself[stim]),
                                       distanceself[stim], 'o', mew=2, mec='k', mfc='None')

    qual_view.fig.savefig(tmp_save + '_quality')

    # draw signal overview
    resp_overview = vis.VisualizeTimeseries()
    resp_overview.subplot(mean_resp.samplepoints)
    resp_overview.imshow('base', 'onetoone', mean_resp, title=True, colorbar=True)
    resp_overview.fig.savefig(tmp_save + '_overview')

    # draw unsorted signal overview
    uresp_overview = vis.VisualizeTimeseries()
    uresp_overview.subplot(mean_resp_unsort.samplepoints)
    uresp_overview.imshow('base', 'onetoone', mean_resp_unsort, title=True, colorbar=True)
    uresp_overview.fig.savefig(tmp_save + '_overview_unsort')

    '''
    # draw ica overview
    toplot = raw_ica
    ica_overview = vis.VisualizeTimeseries()
    ica_overview.base_and_time(toplot.num_objects)
    ica_overview.imshow('base', 'onetoone', toplot.base)
    ica_overview.plot('time', 'onetoone', toplot)
    ica_overview.add_labelshade('time', 'onetoall', toplot)
    ica_overview.add_shade('time', 'onetoall', stimuli_selection, 20)
    ica_overview.add_samplelabel([-1], toplot, rotation='45', toppos=True)
    [ax.set_title(toplot.label_objects[i]) for i, ax in enumerate(ica_overview.axes['base'])]
    ica_overview.fig.savefig(tmp_save + '_goodmodes.svg')
    '''
    '''
    preprocessed.save(tmp_save + '_prepocess')
    raw_ica.save(tmp_save + '_rawica')
    #stim_ica.save(tmp_save + '_stimica')
    '''

    plt.close('all')
####################################################################
# stimultanieous ICA
####################################################################

#plt.close('all')

allodors = list(set(ts.label_sample + reduce(lambda x, y: x + y, [t.label_sample for t in all_raw])))
allodors.sort()
quality_mx = np.zeros((len(all_raw), len(allodors)))
for t_ind, t in enumerate(all_raw):
    for od in set(t.label_sample):
        quality_mx[t_ind, allodors.index(od)] = 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(quality_mx, interpolation='nearest', cmap=plt.cm.bone)
ax.set_yticks(range(len(all_raw)))
ax.set_yticklabels([t.name for t in all_raw])
ax.set_xticks(range(len(allodors)))
ax.set_xticklabels(allodors, rotation='45')
plt.title(prefix + '_' + str(similarity_threshold))
plt.savefig('_'.join(tmp_save.split('_')[:-1]) + 'mask.png')


intersection = sorted_trials(combine_common(all_raw))
mo = pca(intersection)
#mo2 = icaend(mo)
mo2 = icaend(intersection)

fig = plt.figure()
fig.suptitle(np.sum(mo.eigen))
num_bases = len(filelist)
names = [t.name for t in all_raw]
for modenum in range(variance):
    single_bases = mo2.base.objects_sample(modenum)
    for base_num in range(num_bases):
        ax = fig.add_subplot(variance+1, num_bases, base_num+1)
        ax.imshow(np.mean(baselines[base_num].shaped2D(), 0), cmap=plt.cm.gray)
        ax.set_axis_off()
        ax.set_title(names[base_num])
        ax = fig.add_subplot(variance+1, num_bases, ((num_bases * modenum) + num_bases) + base_num + 1)
        ax.imshow(single_bases[base_num] * -1, cmap=plt.cm.hsv, vmin= -1, vmax=1)
        ax.set_axis_off()

fig.savefig('_'.join(tmp_save.split('_')[:-1]) + '_simultan.png')

fig = plt.figure()
fig.suptitle(np.sum(mo.eigen))
num_bases = len(filelist)
names = [t.name for t in all_raw]
for modenum in range(variance):
    single_bases = mo2.base.objects_sample(modenum)
    for base_num in range(num_bases):
        ax = fig.add_subplot(variance+1, num_bases, base_num+1)
        ax.imshow(np.mean(baselines[base_num].shaped2D(), 0), cmap=plt.cm.gray)
        ax.set_axis_off()
        ax.set_title(names[base_num])
        ax = fig.add_subplot(variance+1, num_bases, ((num_bases * modenum) + num_bases) + base_num + 1)
        ax.imshow(single_bases[base_num], cmap=plt.cm.jet)
        ax.set_axis_off()
fig.savefig('_'.join(tmp_save.split('_')[:-1]) + '_simultan_scaled.png')

fig = plt.figure()
for modenum in range(variance):
    ax = fig.add_subplot(variance, 1, modenum + 1)
    ax.plot(mo2.timecourses[:, modenum])
    ax.set_xticklabels([], fontsize=12, rotation=45)
    ax.grid(True)
ax.set_xticks(np.arange(3, mo2.samplepoints, mo2.timepoints))
ax.set_xticklabels(mo2.label_sample, fontsize=12, rotation=45)
fig.savefig('_'.join(tmp_save.split('_')[:-1]) + '_simultan_time.png')



####################################################################
# cluster modes
####################################################################

#create dictionary with mode distances
def cordist(modelist):
    #modelist.pop(4)
    #modelist.pop(0)
    alldist_dic = {}
    for i in range(len(modelist)):
        stimuli_i = modelist[i].label_sample
        if len(stimuli_i) < 8:
            print 'skipped: ', modelist[i].name
            continue
        for j in range(i + 1, len(modelist)):
            stimuli_j = modelist[j].label_sample
            if len(stimuli_j) < 8:
                print 'skipped: ', modelist[j].name
                continue
            common = set(stimuli_i).intersection(stimuli_j)
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
    lables = ['_'.join(lab.split('_nnma_mode')) for lab in lables]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(prefix + titleextra)

    d = dendrogram(linkage(np.array(modedist).squeeze() + 1E-10, 'average'), labels=lables, leaf_font_size=12)

    group_colors = []
    for i in d['ivl']:
        print i, prefix
        group_colors.append(colorlist[prefix + '_' + '_'.join(i.split('_')[:-1])])

    labelsn = ax.get_xticklabels()
    for j, i in enumerate(labelsn):
        i.set_color(group_colors[j])

'''
dendro(all_sel_modes_condensed, 'condensed')
dendro(all_sel_modes)
'''
