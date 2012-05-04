
import os
import glob
import json
import itertools as it
from NeuralImageProcessing.pipeline import TimeSeries
import NeuralImageProcessing.basic_functions as bf
import logging as l
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

comparisons = [(u'vlPRCb', u'vlPRCt'),
               (u'iPN', u'iPNsecond'),
               (u'iPNtract', u'betweenTract'),
               (u'betweenTract', u'vlPRCb'),
               (u'iPN', u'blackhole')]
main_regions = [u'iPN', u'iPNtract', u'vlPRCt']
format = 'png'
integrate = True
load_path = '/Users/dedan/projects/fu/results/simil80n_bestFalse/nnma/'
save_path = os.path.join(load_path, 'boxplots')
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(os.path.join(load_path, 'odors')):
    os.mkdir(os.path.join(load_path, 'odors'))

data = {}

# load the labels created by GUI
labeled_animals = json.load(open(os.path.join(load_path, 'regions.json')))

# load and filter filelist
filelist = glob.glob(os.path.join(load_path, '*.json'))
filelist = [f for f in filelist if not 'base' in os.path.basename(f)]
filelist = [f for f in filelist if not 'regions' in os.path.basename(f)]

# initialize processing (pipeline) components
average_over_stimulus_repetitions = bf.SingleSampleResponse()
integrator = bf.StimulusIntegrator()

# read data to dictionary
l.info('read files from: %s' % load_path)
for fname in filelist:
    ts = TimeSeries()
    ts.load(os.path.splitext(fname)[0])
    name = os.path.splitext(os.path.basename(fname))[0]
    if integrate:
        data[name] = integrator(average_over_stimulus_repetitions(ts))
    else:
        data[name] = average_over_stimulus_repetitions(ts)

# get all stimuli and region labels
all_stimuli = sorted(set(it.chain.from_iterable([ts.label_sample for ts in data.values()])))
all_region_labels = list(set(it.chain.from_iterable([labels for labels in labeled_animals.values()])))
l.debug('all_stimuli: %s' % all_stimuli)
l.debug('all_region_labels: %s' % all_region_labels)

# produce a figure for each region_label
medians = {}
for region_label in all_region_labels:

    fig = plt.figure()
    fig.suptitle(region_label)
    t_modes = []

    # iterate over region labels for each animal
    for animal, regions in labeled_animals.items():

        # load data and extract trial shape
        ts = data[animal]
        trial_shaped = ts.trial_shaped()
        trial_length = trial_shaped.shape[1]
        n_modes = trial_shaped.shape[2]

        # extract modes for region_label (several modes can belong to one region)
        modes = [i for i in range(n_modes) if regions[i] == region_label]
        for mode in modes:

            # initialize to nan (not all stimuli are found for all animals)
            pdat = np.zeros(len(all_stimuli) * trial_length)
            pdat[:] = np.nan

            # fill the results vector for the current animal
            for i, stimulus in enumerate(all_stimuli):
                if stimulus in ts.label_sample:
                    index = ts.label_sample.index(stimulus)
                    pdat[i*trial_length:i*trial_length+trial_length] = trial_shaped[index, :, mode]

            # add to results list
            t_modes.append(pdat)
    t_modes = np.array(t_modes)

    add = '_integrated' if integrate else ''
    ax = fig.add_subplot(111)
    # mask it for nans (! True in the mask means exclusion)
    t_modes_ma = np.ma.array(t_modes, mask=np.isnan(t_modes))
    medians[region_label] = np.ma.extras.median(t_modes_ma, axis=0)
    # make it a list because boxplot has a problem with masked arrays
    t_modes_ma = [[y for y in row if y] for row in t_modes_ma.T]
    ax.boxplot(t_modes_ma)
    ax.set_xticklabels(list(all_stimuli), rotation='90')
    plt.savefig(os.path.join(save_path, region_label + add + '.' + format))

    # write the data to csv files
    np.savetxt(os.path.join(save_path, region_label + add + '.csv'), t_modes, delimiter=',')


if integrate:

    # overview of the medians plot
    fig = plt.figure()
    for i, region_label in enumerate(medians.keys()):
        ax = fig.add_subplot(len(medians), 1, i + 1)
        ax.bar(range(len(medians[region_label])), medians[region_label])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(region_label, rotation='0')
    ax.set_xticks(range(len(medians[region_label])))
    ax.set_xticklabels(list(all_stimuli), rotation='90')
    plt.savefig(os.path.join(save_path, 'medians.' + format))
    np.savetxt(os.path.join(save_path, 'medians.csv'), medians.values(), delimiter=',')

    # medians comparison plot
    fig = plt.figure()
    for i, comparison in enumerate(comparisons):
        ax = fig.add_subplot(len(comparisons), 1, i + 1)
        l = len(medians[comparison[0]])
        ax.bar(range(l), medians[comparison[0]], color='r')
        ax.bar(range(l), medians[comparison[1]]*-1, color='b')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(', '.join(comparison), rotation='0')
    ax.set_xticks(range(l))
    ax.set_xticklabels(list(all_stimuli), rotation='90')
    plt.savefig(os.path.join(save_path, 'comparisons.' + format))

    # odor-region comparison plots
    all_odors = sorted(set([s.split('_')[0] for s in all_stimuli]))
    all_concentrations = sorted(set([s.split('_')[1] for s in all_stimuli]))
    for odor in all_odors:

        fig = plt.figure()
        rel_concentrations = ['_'.join([odor, c]) for c in all_concentrations
                                if '_'.join([odor, c]) in all_stimuli]
        for i, conc in enumerate(rel_concentrations):
            ax = fig.add_subplot(len(rel_concentrations), 1, i + 1)
            idx = all_stimuli.index(conc)
            plot_data = [medians[key][idx] for key in sorted(medians.keys())]
            ax.bar(range(len(medians)), plot_data)
            ax.set_yticks(range(int(np.max(np.array(medians.values()).flatten()))))
            ax.set_xticks([])
            ax.set_ylabel(conc, rotation='0')
        ax.set_xticks(range(len(all_region_labels)))
        ax.set_xticklabels(sorted(medians.keys()), rotation='90')
        plt.savefig(os.path.join(load_path, 'odors', odor + '.' + format))

    # median heatmaps
    hm_data = np.array([medians[region] for region in main_regions])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(hm_data, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks(range(len(main_regions)))
    ax.set_yticklabels(main_regions)
    plt.savefig(os.path.join(load_path, 'heatmap.' + format))

    ### 3D plot
    # 3D plot of the responses on the space of the 3 most prominent clusters.
    # each dimension in the plot is the magnitude of a odor response in a certain cluster.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(hm_data[0,:], hm_data[1,:], hm_data[2,:])


