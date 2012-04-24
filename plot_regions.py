
import os
import glob
import json
from NeuralImageProcessing.pipeline import TimeSeries
import NeuralImageProcessing.basic_functions as bf
import logging as l
import numpy as np
import pylab as plt
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

format = 'svg'
load_path = '/Users/dedan/projects/fu/results/simil80n_bestFalse/nnma/'
save_path = os.path.join(load_path, 'boxplots')
if not os.path.exists(save_path):
    os.mkdir(save_path)
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
    data[name] = integrator(average_over_stimulus_repetitions(ts))

# get all stimuli and region labels
all_stimuli = sorted(set(sum([ts.label_sample for ts in data.values()], [])))
all_region_labels = set(sum([labels for labels in labeled_animals.values()], []))
l.debug('all_stimuli: %s' % all_stimuli)
l.debug('all_region_labels: %s' % all_region_labels)

# produce a figure for each region_label
for region_label in all_region_labels:

    fig = plt.figure()
    fig.suptitle(region_label)
    integrated = []

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
            integrated.append(pdat)
    integrated = np.array(integrated)

    ax = fig.add_subplot(111)
    # mask it for nans (! True in the mask means exclusion)
    integrated = np.ma.array(integrated, mask=np.isnan(integrated))
    # make it a list because boxplot has a problem with masked arrays
    integrated = [[y for y in row if y] for row in integrated.T]
    ax.boxplot(integrated)
    ax.set_xticklabels(list(all_stimuli), rotation='90')
    plt.savefig(os.path.join(save_path, region_label + '.' + format))



