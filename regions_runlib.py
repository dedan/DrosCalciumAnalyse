#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys, os, glob, csv, json, __builtin__
import itertools as it
import numpy as np
import pylab as plt
from scipy.stats.mstats_basic import scoreatpercentile
from NeuralImageProcessing.pipeline import TimeSeries
from NeuralImageProcessing import illustrate_decomposition as vis
import NeuralImageProcessing.basic_functions as bf
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

def plot_split_valenz_heatmap(valenz, config):
    # normalize valenz for colormap
    norm_val = {}
    all_vals = np.array(valenz.values())
    for val in valenz:
        norm_val[val] = (valenz[val] / (2 * np.abs(np.max(all_vals))) + 0.5)

    # splitted heatmap for valenz information
    fig = vis.VisualizeTimeseries()
    fig.subplot(1)
    ax = fig.axes['base'][0]
    plotti = np.ones((3, len(config['new_stimuli_order']))) * 0.5
    for y, odor in enumerate(config['new_stimuli_order']):
        for x, co in enumerate(config['concentrations']):
            if odor == 'MOL':
                stim = 'MOL_0'
            else:
                stim = '%s_%s' % (odor, co)
            if stim in norm_val:
                plotti[x, y] = norm_val[stim]
    fig.imshow(ax, plotti, cmap=plt.cm.RdYlGn)
    fig.overlay_image(ax, plotti == 0.5, threshold=0.1,
                      title={"label": "valenz - max: %f" % np.max(all_vals)},
                      colormap=plt.cm.gray)
    ax.set_yticks(range(len(config['concentrations'])))
    ax.set_yticklabels(config['concentrations'])
    ax.set_xticks(range(len(config['new_stimuli_order'])))
    ax.set_xticklabels(config['new_stimuli_order'], rotation='90')
    return fig.fig

def plot_splitsort_heatmaps(medians, all_stimuli, all_odors, config):
    """split and sort heatmap of medians"""
    dats = []
    hm_data = np.array([medians[region] for region in config['main_regions']])
    fig = plt.figure()
    for i in range(len(config['concentrations'])):
        plotti = np.zeros((3, len(config['new_stimuli_order'])))
        for y, odor in enumerate(config['new_stimuli_order']):
            for x, co in enumerate(config['concentrations']):
                if odor == 'MOL':
                    stim = 'MOL_0'
                else:
                    stim = '%s_%s' % (odor, co)
                if stim in all_stimuli:
                    plotti[x, y] = hm_data[i, all_stimuli.index(stim)]
        dats.append(plotti)
    for i in range(len(config['concentrations'])):
        ax = fig.add_subplot(len(config['concentrations']), 1, i + 1)
        ax.set_title("region: %s - max: %f" % (config['main_regions'][i], np.max(dats[i])))
        ax.imshow(dats[i], interpolation='nearest')
        ax.set_yticks(range(len(config['concentrations'])))
        ax.set_yticklabels(config['concentrations'])
        ax.set_xticks([])
    ax.set_xticks(range(len(all_odors)))
    ax.set_xticklabels(config['new_stimuli_order'], rotation='90')
    return fig

def plot_heatmap(medians, main_regions):
    fig = plt.figure()
    hm_data = np.array([medians[region] for region in main_regions])
    ax = fig.add_subplot(111)
    ax.imshow(hm_data, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks(range(len(main_regions)))
    ax.set_yticklabels(main_regions)
    return fig

def plot_region_comparison_for(odor, medians, all_stimuli, all_region_labels):
    """group of bar plots - all concentrations for one odor - for all regions"""
    all_concentrations = sorted(set([s.split('_')[1] for s in all_stimuli]))
    rel_concentrations = ['_'.join([odor, c]) for c in all_concentrations
                            if '_'.join([odor, c]) in all_stimuli]
    fig = plt.figure()
    for i, conc in enumerate(rel_concentrations):
        ax = fig.add_subplot(len(rel_concentrations), 1, i + 1)
        idx = all_stimuli.index(conc)
        plot_data = [medians[key].data[idx] for key in sorted(medians.keys())]
        plot_data[plot_data == 0.0] = 0.01
        ax.bar(range(len(medians)), plot_data)
        ax.set_yticks(range(int(np.max(np.array(medians.values()).flatten()))))
        ax.set_xticks([])
        ax.set_ylabel(conc, rotation='0')
    ax.set_xticks(range(len(all_region_labels)))
    ax.set_xticklabels(sorted(medians.keys()), rotation='90')
    return fig

def plot_median_comparison(medians, comparisons, all_stimuli):
    """medians comparison plot"""
    fig = plt.figure()
    for i, comparison in enumerate(comparisons):
        ax = fig.add_subplot(len(comparisons), 1, i + 1)
        l = len(medians[comparison[0]])
        ax.bar(range(l), medians[comparison[0]], color='r')
        ax.bar(range(l), medians[comparison[1]] * -1, color='b')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(', '.join(comparison), rotation='0')
    ax.set_xticks(range(l))
    ax.set_xticklabels(list(all_stimuli), rotation='90')
    return fig

def plot_median_overview(region_label, medians, all_stimuli):
    """overview of the medians plot"""
    fig = plt.figure()
    for i, region_label in enumerate(medians.keys()):
        ax = fig.add_subplot(len(medians), 1, i + 1)
        ax.bar(range(len(medians[region_label])), medians[region_label])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(region_label, rotation='0')
    ax.set_xticks(range(len(medians[region_label])))
    ax.set_xticklabels(list(all_stimuli), rotation='90')
    return fig

def plot_spatial_base(region_label, s_modes, to_turn, load_path, colormaps):
    fig = vis.VisualizeTimeseries()
    fig.subplot(len(s_modes))
    for i, (name, s_mode) in enumerate(s_modes):
        n = '_'.join(name.split('_')[:-1])
        filelist = glob.glob(os.path.join(load_path, '*' + n + '_baseline.json'))
        base_series = TimeSeries()
        base_series.load(os.path.splitext(filelist[0])[0])
        base_series.shape = tuple(base_series.shape)
        base = base_series.shaped2D()
        if n in to_turn:
            base = base[:, ::-1, :]
            s_mode = s_mode[::-1, :]

        fig.overlay_workaround(fig.axes['base'][i],
                           np.mean(base, axis=0), {'cmap':plt.cm.bone},
                           s_mode, {'threshold':0.2, 'colormap':colormaps[region_label]},
                           {'title':{"label": n}})
    return fig.fig

def get_masked_selection(t_modes, all_stimuli, stim_selection, integrate=False):
    """mask array for nans and filter out non selected stimuli"""
    stim_selection_idxs = [all_stimuli.index(i) for i in stim_selection]
    if not integrate:
        # TODO: remove magic number
        replicated_indeces = [range(i*20, i*20+20) for i in stim_selection_idxs]
        stim_selection_idxs = __builtin__.sum(replicated_indeces, [])
    # mask it for nans (! True in the mask means exclusion)
    return np.ma.array(t_modes[:, stim_selection_idxs],
                       mask=np.isnan(t_modes[:, stim_selection_idxs]))


def plot_temporal(region_label, t_modes_ma, medians, stim_selection, n_frames):
    fig = plt.figure()
    fig.suptitle(region_label)
    ax = fig.add_subplot(111)
    l = len(medians[region_label])
    p25 = scoreatpercentile(t_modes_ma, 25)
    p75 = scoreatpercentile(t_modes_ma, 75)
    ax.fill_between(range(l), p25, p75, linewidth=0, color='0.75')
    ax.plot(medians[region_label], linewidth=0.5, color='0')
    ax.set_xticks(range(0, l, n_frames))
    ax.set_xticklabels(list(stim_selection), rotation='90')
    return fig

def plot_temporal_lesion(region_label, t_modes_ma, medians, stim_selection, n_frames):
    fig = plt.figure()
    fig.suptitle(region_label)
    ax = fig.add_subplot(111)
    m = medians[region_label]
    l = len(m)
    cols = ['r', 'g', 'b']
    labels = ['intact', 'iPN', 'vlPrc']
    for i in range(3):
        idx = __builtin__.sum([range(j, j+n_frames) for
                               j in range(i*n_frames, l, 3 * n_frames)], [])
        d = m[idx]
        p25 = scoreatpercentile(t_modes_ma, 25)
        p75 = scoreatpercentile(t_modes_ma, 75)
        ax.fill_between(range(len(d)), p25[idx], p75[idx], linewidth=0, color=cols[i], alpha=0.2)
        ax.plot(d, linewidth=0.5, color=cols[i], label=labels[i])
        ax.set_xticks(range(0, len(d), n_frames))
        ax.set_xticklabels(list(stim_selection)[::3], rotation='90')
    plt.legend(labels)
    return fig

def plot_temporal_integrated(region_label, t_modes_ma):
    """"""
    fig = plt.figure()
    fig.suptitle(region_label)
    ax = fig.add_subplot(111)
    # make it a list because boxplot has a problem with masked arrays
    t_modes_ma = [[y for y in row if y] for row in t_modes_ma.T]
    ax.boxplot(t_modes_ma)
    return fig

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

def load_mf_results(load_path, selection, lesion_data, integrate, integrator_window):
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
