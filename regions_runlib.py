#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys, os, glob, csv, json, __builtin__
import itertools as it
from collections import defaultdict
import numpy as np
import pylab as plt
from scipy.stats.mstats_basic import scoreatpercentile, mquantiles
from NeuralImageProcessing.pipeline import TimeSeries
from NeuralImageProcessing import illustrate_decomposition as vis
import NeuralImageProcessing.basic_functions as bf
from sklearn import linear_model
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

def fit_models(data_dict, config):
    """fit different models to our data"""
    regressor = linear_model.LinearRegression(fit_intercept=False)
    x, y = [], []
    for odor in data_dict:
        for concen in data_dict[odor]:
            if 'valenz' in data_dict[odor][concen]:
                t = data_dict[odor][concen]
                x.append([t['data'][2], t['data'][0]])
                y.append(t['valenz'])
    fit = regressor.fit(x, y)
    alpha = fit.coef_[1]

    agg = defaultdict(list)
    for odor in data_dict:
        for concen in data_dict[odor]:
            if 'valenz' in data_dict[odor][concen]:
                t = data_dict[odor][concen]
                agg['val'].append(t['valenz'])
                for i in range(3):
                    agg[config['main_regions'][i]].append(t['data'][i])
                agg['ratio'].append(data_dict[odor][concen]['data'][2] /
                                    data_dict[odor][concen]['data'][0])
                agg['diff'].append(data_dict[odor][concen]['data'][2] -
                                   alpha * data_dict[odor][concen]['data'][0])
    idx = np.argmax(agg['ratio'])
    agg['ratio'].pop(idx)
    return agg

def plot_valenz_3d(data_dict, config):
    """3d valenz plot"""
    symbols = lambda x: 'x' if x == '0' else 'o'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for odor in data_dict:
        for concen in data_dict[odor]:
            if 'valenz_color' in data_dict[odor][concen]:
                ax.scatter(*[[i] for i in data_dict[odor][concen]['data']],
                           edgecolors=data_dict[odor][concen]['valenz_color'],
                           facecolors=data_dict[odor][concen]['valenz_color'],
                           marker=symbols(concen), label=odor)
                ax.plot([], [], 'o', c=data_dict[odor][concen]['valenz_color'], label=odor)
                s_concen = sorted([int(concen) for concen in data_dict[odor]])
                bla = np.array([data_dict[odor][str(concen)]['data'] for concen in s_concen])
                ax.plot(*[x for x in bla.T], c='0.5')
    ax.set_xlabel(config['main_regions'][0])
    ax.set_ylabel(config['main_regions'][1])
    ax.set_zlabel(config['main_regions'][2])
    plt.legend(loc=(0.0, 0.6), ncol=2, prop={"size":9})
    return fig

def plot_medians_3d(data_dict, config):
    """3d plot of the data also shown as heatmap"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for odor in data_dict:
        for concen in data_dict[odor]:
            ax.scatter(*[[i] for i in data_dict[odor][concen]['data']],
                       edgecolors=data_dict[odor][concen]['color'], facecolors='none',
                       marker=config['symbols'][concen], label=odor)
        ax.plot([], [], 'o', c=data_dict[odor][concen]['color'], label=odor)
        s_concen = sorted([int(concen) for concen in data_dict[odor]])
        bla = np.array([data_dict[odor][str(concen)]['data'] for concen in s_concen])
        ax.plot(*[x for x in bla.T], c=data_dict[odor][str(concen)]['color'])
    ax.set_xlabel(config['main_regions'][0])
    ax.set_ylabel(config['main_regions'][1])
    ax.set_zlabel(config['main_regions'][2])
    plt.legend(loc=(0.0, 0.6), ncol=2, prop={"size":9})
    return fig

def organize_data_in_dict(medians, all_stimuli, all_odors, valenz, config):
    """prepare data for 3 d plots"""
    data_dict = {}
    hm_data = np.array([medians[region] for region in config['main_regions']])
    for i in range(len(all_stimuli)):
        odor, concen = all_stimuli[i].split('_')
        if not odor in data_dict:
            data_dict[odor] = {}
        data_dict[odor][concen] = {}
        c = plt.cm.hsv(float(all_odors.index(odor)) / len(all_odors))
        data_dict[odor][concen]['color'] = c

        # add color to code valenz (if valenz available)
        if all_stimuli[i] in valenz:
            c = plt.cm.RdYlGn(valenz[all_stimuli[i]])
            data_dict[odor][concen]['valenz_color'] = c
            data_dict[odor][concen]['valenz'] = valenz[all_stimuli[i]]
        data_dict[odor][concen]['data'] = hm_data[:, i]
    return data_dict

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

def plot_medians_heatmap(medians, main_regions):
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

#TODO: include plot_single

def plot_temporal(modes, stim_layout, stim2ax, plot_single=False):
    medians = calc_scoreatpercentile(modes, 0.5).trial_shaped().squeeze()
    p25 = calc_scoreatpercentile(modes, 0.25).trial_shaped().squeeze()
    p75 = calc_scoreatpercentile(modes, 0.75).trial_shaped().squeeze()

    max_y = np.max(p75) + 0.05
    min_y = np.min(p25) - 0.05
    max_ytick = np.floor(max_y * 10) / 10
    min_ytick = np.ceil(min_y * 10) / 10

    for stim_ix, stim in enumerate(modes.label_sample):
        if stim not in stim2ax:
            continue
        ax = stim2ax[stim]
        ax.fill_between(range(modes.timepoints), p25[stim_ix], p75[stim_ix], linewidth=0, color='0.75')
        ax.plot(medians[stim_ix], linewidth=1.5, color='0')
        # create stimulus bar
        ax.fill_between(np.array(modes.stim_window), max_y, min_y, color='b',
                        alpha=0.2, linewidth=0)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_ylim([min_y, max_y])
        ax.set_yticks([0, max_ytick])
        ax.set_yticks(np.arange(min_ytick, max_ytick, 0.1), minor=True)
        if stim in stim2ax['left']:
            ax.set_yticklabels([0, max_ytick])
        else:
            ax.set_yticklabels([])
        xticks = range(0, modes.timepoints, 8)
        ax.set_xticks(xticks)
        ax.set_xticks(range(0, modes.timepoints), minor=True)
        if stim in stim2ax['bottom']:
            ax.set_xticklabels(['%d' % i for i in np.array(xticks) * 1. / modes.framerate], fontsize=10)
        else:
            ax.set_xticklabels([])

# TODO: finish corrections
def plot_temporal_lesion(modes, stim_selection):
    fig = plt.figure()
    fig.suptitle(modes.name[0])
    ax = fig.add_subplot(111)
    m = calc_scoreatpercentile(modes, 50)
    l = len(m)
    cols = ['r', 'g', 'b']
    labels = ['intact', 'iPN', 'vlPrc']
    for i in range(3):
        idx = __builtin__.sum([range(j, j + n_frames) for
                               j in range(i * n_frames, l, 3 * n_frames)], [])
        d = m[idx]
        p25 = scoreatpercentile(t_modes_ma, 25)
        p75 = scoreatpercentile(t_modes_ma, 75)
        ax.fill_between(range(len(d)), p25[idx], p75[idx], linewidth=0, color=cols[i], alpha=0.2)
        ax.plot(d, linewidth=0.5, color=cols[i], label=labels[i])
        ax.set_xticks(range(0, len(d), n_frames))
        ax.set_xticklabels(list(stim_selection)[::3], rotation='90')
    plt.legend(labels)
    return fig

def plot_temporal_integrated(modes_integrated, stim_selection):
    """"""
    fig = plt.figure()
    fig.suptitle(modes_integrated.name[0])
    ax = fig.add_subplot(111)
    # make it a list because boxplot has a problem with masked arrays
    if modes_integrated.timecourses.ndim < 2:
        timecourses = modes_integrated.timecourses.reshape((1, -1))
    else:
        timecourses = modes_integrated.timecourses
    t_modes_ma = np.ma.array(timecourses, mask=np.isnan(timecourses))
    t_modes_ma = [[y for y in row if y] for row in t_modes_ma]
    x_index = [modes_integrated.label_sample.index(lab) for lab in stim_selection
               if lab in modes_integrated.label_sample]
    t_modes_ma = [t_modes_ma[ind] for ind in x_index]
    ax.boxplot(t_modes_ma)
    ax.set_xticklabels(stim_selection, rotation='45', ha='right')
    return fig

def compute_latencies(modes):
    """latency: time (in frames) of the maximum response (peak)"""
    latencies = np.argmax(modes.trial_shaped(), axis=1).astype('float')
    latencies[np.isnan(modes.trial_shaped()[:, 0, :])] = np.nan
    return latencies

def plot_latencies(latency_matrix, region_label, all_stimuli):
    """boxplot for the distribution of latencies

       latency: time (in frames) of the maximum response (peak)
    """
    res_ma = np.ma.array(latency_matrix, mask=np.isnan(latency_matrix))
    # make it a list because boxplot has a problem with masked arrays
    res_ma = [[y for y in row if y] for row in res_ma]
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
    t_modes, t_modes_names, s_modes, s_shapes = [], [], [], []
    labeled_animals = json.load(open(regions_json_path))
    all_stimuli = sorted(set(it.chain.from_iterable([ts.label_sample for ts in data.values()])))

    # iterate over region labels for each animal
    trial_length = data[labeled_animals.keys()[0]].timepoints
    for animal, regions in labeled_animals.items():

        # load data and assert equal trial shape
        if not animal in data:
            continue
        ts = data[animal]
        assert trial_length == ts.timepoints
        n_modes = ts.num_objects

        # extract modes for region_label (several modes can belong to one region)
        modes = [i for i in range(n_modes) if regions[i] == region_label]
        for mode in modes:

            # initialize to nan (not all stimuli are found for all animals)
            pdat = np.zeros((len(all_stimuli), trial_length))
            pdat[:] = np.nan

            # fill the results vector for the current animal
            indices = [all_stimuli.index(lab) for lab in ts.label_sample]
            pdat[indices] = ts.trial_shaped()[:, :, mode]

            # add to results list
            t_modes.append(pdat.flatten())
            t_modes_names.append("%s_%d" % (animal, mode))
            s_modes.append(ts.base.timecourses[mode, :])
            s_shapes.append(ts.base.shape)
    t_modes = np.array(t_modes).T
    s_modes = np.hstack(s_modes).reshape((1, -1))

    # create timeseries object for region timecourses
    ts_temporal = ts.copy()
    ts_temporal.name = [region_label]
    ts_temporal.timecourses = t_modes
    ts_temporal.shape = t_modes.shape[1]
    ts_temporal.label_sample = all_stimuli
    ts_temporal.label_objects = t_modes_names
    # TODO: remove when all data in correct format
    # adds framerate and stimuli_window for old data
    if not(hasattr(ts_temporal, 'stim_window')):
        l.info('no stim_window given, set to (4,8)')
        ts_temporal.stim_window = (4, 8)
    if not(hasattr(ts_temporal, 'framerate')):
        l.info('no framerate given, set to 2')
        ts_temporal.framerate = 2

    # create timeseries object for region bases
    ts_spatial = bf.TimeSeries(name=[region_label], shape=s_shapes)
    ts_spatial.timecourses = s_modes
    ts_spatial.label_objects = t_modes_names
    ts_temporal.base = ts_spatial
    return ts_temporal

def load_lesion_data(lesion_data_path):
    """read the table of which lesion was applied into a dictionary"""
    les_dict = {}
    lesion_table = list(csv.reader(open(lesion_data_path, 'rb'), delimiter='\t'))
    for row in lesion_table[1:]:
        les_dict[row[0]] = {}
        les_dict[row[0]]['l'] = row[1]
        les_dict[row[0]]['r'] = row[2]
    return les_dict

def load_mf_results(load_path, selection, lesion_data):
    """read data (matrix factorization results) to dictionary"""

    data = {}
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

        average_over_stimulus_repetitions = bf.SingleSampleResponse()
        data[name] = average_over_stimulus_repetitions(ts)
    return data

def calc_scoreatpercentile(modes, percentile):
    ''' calculates percentile for Timeseries (may include nans)'''
    out = modes.copy()
    out.timecourses = np.ma.array(out.timecourses, mask=np.isnan(out.timecourses))
    out.timecourses = mquantiles(out.timecourses, percentile, axis=1)
    out.shape = (1,)
    return out

def generate_axesmatrix(fig, stimlist):
    ''' generates axes matrix according to stimlist, returns dictionary with
    maps stim_names to axes '''
    stim2ax = {}
    stim2ax['bottom'], stim2ax['left'] = [], []
    dim0 = len(stimlist[0])
    dim1 = len(stimlist)
    for col_ix in range(dim1):
        for row_ix in range(dim0):
            stim = stimlist[col_ix][row_ix]
            if stim == 'empty':
                continue
            if col_ix == 0:
                stim2ax['left'].append('stim')
            if row_ix == dim0:
                stim2ax['bottom'].append('stim')
            ax = fig.add_subplot(row_ix, col_ix, row_ix * dim1 + col_ix + 1)
            stim2ax[stim] = ax
    return stim2ax

def generate_axeslist(fig, stimlist):
    ''' generates axes list according to stimlist, returns dictionary with
    maps stim_names to axes '''
    stim2ax = {}
    stim2ax['bottom'] = stimlist
    for col_ix, stim in enumerate(stimlist):
        if col_ix == 0:
            stim2ax['left'] = [stim]
        ax = fig.add_subplot(1, len(stimlist), col_ix + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(stim, fontsize=6)
        stim2ax[stim] = ax
    return stim2ax
