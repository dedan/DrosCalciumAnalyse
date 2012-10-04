#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys, os, glob, csv, json
import itertools as it
import numpy as np
import pylab as plt
from NeuralImageProcessing.pipeline import TimeSeries
import NeuralImageProcessing.basic_functions as bf
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

def compute_latencies(collected_modes, n_frames):
    """latency: time (in frames) of the maximum response (peak)"""
    n_animals = collected_modes['t_modes'].shape[0]
    n_stimuli = collected_modes['t_modes'].shape[1]

    res = np.zeros((n_animals, n_stimuli/n_frames))

    for animal_index in range(n_animals):
        for i, stim_begin in enumerate(range(0, n_stimuli, n_frames)):
            stimulus = collected_modes['t_modes'][animal_index, stim_begin:stim_begin+n_frames]
            if np.all(np.isnan(stimulus)):
                res[animal_index, i] = np.nan
            else:
                res[animal_index, i] = np.argmax(stimulus)
    return res

def plot_latencies(latency_matrix, region_label, all_stimuli):
    """boxplot for the distribution of latencies

       latency: time (in frames) of the maximum response (peak)
    """
    res_ma = np.ma.array(latency_matrix, mask=np.isnan(latency_matrix))
    # make it a list because boxplot has a problem with masked arrays
    res_ma = [[y for y in row if y] for row in res_ma.T]
    fig = plt.figure()
    fig.suptitle(region_label)
    ax = fig.add_subplot(111)
    ax.boxplot(res_ma)
    ax.set_xticklabels(list(all_stimuli), rotation='90')
    return fig

def collect_modes_for(region_label, regions_json_path, data):
    """collect all spatial and temporal modes for a given region_label

        * several modes from one animal can belong to one region
        * the information for this comes from the regions.json created by
          using the regions_gui
    """
    t_modes, t_modes_names, s_modes = [], [], []
    labeled_animals = json.load(open(regions_json_path))
    all_stimuli = sorted(set(it.chain.from_iterable([ts.label_sample for ts in data.values()])))

    # iterate over region labels for each animal
    for animal, regions in labeled_animals.items():

        # load data and extract trial shape
        if not animal in data:
            continue
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
                    pdat[i * trial_length:i * trial_length + trial_length] = trial_shaped[index, :, mode]

            # add to results list
            t_modes.append(pdat)
            t_modes_names.append("%s_%d" % (animal, mode))
            s_modes.append((animal, ts.base.trial_shaped2D()[mode, :, :, :].squeeze()))
    t_modes = np.array(t_modes)
    return {'t_modes': t_modes, 't_modes_names': t_modes_names, 's_modes': s_modes}


def load_lesion_data(lesion_data_path):
    """read the table of which lesion was applied into a dictionary"""
    les_dict = {}
    lesion_table = list(csv.reader(open(lesion_data_path, 'rb'), delimiter='\t'))
    for row in lesion_table[1:]:
        les_dict[row[0]] = {}
        les_dict[row[0]]['l'] = row[1]
        les_dict[row[0]]['r'] = row[2]
    return les_dict

def load_mf_results(load_path, selection, lesion_data, integrate):
    """read data (matrix factorization results) to dictionary"""

    data = {}
    # initialize processing (pipeline) components
    average_over_stimulus_repetitions = bf.SingleSampleResponse()
    if integrate:
        integrator = bf.StimulusIntegrator(method=integrate,
                                           threshold= -1000,
                                           window=integrator_window)

    # load and filter filelist
    filelist = glob.glob(os.path.join(load_path, '*.json'))
    filelist = [f for f in filelist if not 'base' in os.path.basename(f)]
    filelist = [f for f in filelist if not 'regions' in os.path.basename(f)]
    filelist = [f for f in filelist if not 'selection' in os.path.basename(f)]

    for fname in filelist:

        if selection:
            skip = True
            for sel in selection:
                if sel in name:
                    skip = False
            if skip:
                l.info('skip %s because not in selection' % name)
                continue
            l.info('taking %s because found in selection' % name)

        ts = TimeSeries()
        ts.load(os.path.splitext(fname)[0])
        name = os.path.splitext(os.path.basename(fname))[0]

        if lesion_data:
            if '_' in name:
                fname_base, side = name.split('_')
            new_labels = []
            for label in ts.label_sample:
                if label[0] == 'm':
                    new_labels.append(label[1:] + '-' + les_dict[fname_base][side])
                else:
                    new_labels.append(label)
            ts.label_sample = new_labels

        if integrate:
            data[name] = integrator(average_over_stimulus_repetitions(ts))
        else:
            data[name] = average_over_stimulus_repetitions(ts)
    return data
