#!/usr/bin/env python
# encoding: utf-8
"""
This script is used to compare different regions (activation patterns) over data sets

Input to this script is the result of the factorization process (output of runDros
or created using the new GUI in main_gui). Furthermore a json file is needed which
associates each spatial mode of the factorization to a certain region. Because
we could not find an automatic method to cluster these regions this has to be
done by hand with the help of the region_gui.

input: result of factorization, region labels created by region_gui
output: plots (in all plots activity means the median activity for a certain odor)
        * spatial mode overview of each label
        * boxplot of the temporal modes
        * csv file containing the same information as the temporal boxplot
        * a 3D scatter plot relating the activity in the 3 main regions to
          the valenz of an odor (how applealing an odor is).
        * a 3D scatter plot relating the activity in the 3 main regions to
          the concentration of an odor. All concentrations are connected by
          a line and the highest concentration is marked by a square.
        * same output as previous 3D scatter plot but as a matrix of 2D plots
          which shows the distribution of the individual activations on the diagonal
        * scatter plot to look for a linear relationship between the activity
          in the regions and the valenz (activation_vs_valenz)
        * scatter plots of simple linear models that could explain the relationship
          between a combination of regions and the valenz of an odor
          (activation(difference)_vs_valenz and activation(ratio)_vs_valenz)
        * temporal comparisons of different regions (comparisons)
        * heatmap of temporal activation for the three main regions (heatmap)
        * plot of only the activity (median activity) for each odor and region (medians)
        * csv file of the activity (median activity) for each odor and region
        * odors subfolder contains bar plots of activitation per concentration and region
        * heatmap of activations split by concentration for the 3 main regions (split_heatmap)
        * heatmap of valenz (when available) split by concentration for the 3
          main regions (split_heatmap_valenz)
        * plot of latencies of the maximum response per stimulus and region

@author: stephan.gabler@gmail.com
"""

import os, glob, json, math, csv, __builtin__, sys
from configobj import ConfigObj
import itertools as it
from collections import defaultdict
from NeuralImageProcessing.pipeline import TimeSeries
from NeuralImageProcessing import illustrate_decomposition as vis
import NeuralImageProcessing.basic_functions as bf
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats.mstats_basic import scoreatpercentile
import utils
import regions_runlib as rl
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');
reload(utils)
reload(rl)

config = ConfigObj(sys.argv[1], unrepr=True)

# load the dictionary with mapping from a region to a colormap
lut_colormaps = utils.get_all_lut_colormaps(config['main_regions'])

# set paths and create folders
load_path = os.path.join(config['results_path'], 'timeseries')
save_path = os.path.join(config['results_path'], 'region_plots')
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(os.path.join(save_path, 'odors')):
    os.mkdir(os.path.join(save_path, 'odors'))

# load list of animals to analyse
selection = []
if os.path.exists(os.path.join(load_path, 'selection.json')):
    l.info('found selection file')
    selection = json.load(open(os.path.join(load_path, 'selection.json')))

# load stimulus selection list
stim_selection = []
if os.path.exists(os.path.join(load_path, 'stim_selection.json')):
    l.info('found stimulus selection file')
    stim_selection = json.load(open(os.path.join(load_path, 'stim_selection.json')))

# load valenz information (which odor they like)
valenz = json.load(open(os.path.join(config['results_path'], 'valenz.json')))

# read lesion-tract table into dictionary for easy access
if config['lesion_data']:
    lesion_dict = rl.load_lesion_data(config['lesion_table_path'])

# read mf results
l.info('read files from: %s' % load_path)
data = rl.load_mf_results(load_path, selection, config['lesion_data'])

# get all stimuli and region labels (use selection if selection given)
all_stimuli = sorted(set(it.chain.from_iterable([ts.label_sample for ts in data.values()])))
if not stim_selection:
    stim_selection = all_stimuli
regions_file_path = os.path.join(config['results_path'], 'regions.json')
with open(regions_file_path) as f:
    all_labels_list = [labels for labels in json.load(f).values()]
    all_region_labels = list(set(it.chain.from_iterable(all_labels_list)))
l.debug('all_stimuli: %s' % all_stimuli)
l.debug('all_region_labels: %s' % all_region_labels)

# collect data for regions
medians, fulldatadic = {}, {}
for region_label in all_region_labels:

    region_savepath = os.path.join(save_path, region_label)
    t_modes = rl.collect_modes_for(region_label, regions_file_path, data)
    fulldatadic[region_label] = t_modes
    # TODO: remove this after jan fixed the timecourses object
    if t_modes.timecourses.ndim < 2:
        tmp_t_modes = t_modes.timecourses.reshape((-1, 1))
    else:
        tmp_t_modes = t_modes.timecourses
    medians[region_label] = np.ma.extras.median(tmp_t_modes, axis=1)

# produce per-region plots
for region_label in all_region_labels:

    # latency plots
    latency_matrix = rl.compute_latencies(collected_modes, config['n_frames'])
    fig = rl.plot_latencies(latency_matrix, region_label, all_stimuli)
    fig.savefig(region_savepath + '_latencies.' + config['format'])

    # TODO: write generic function to write CSVs with headers
    with open(region_savepath + '_latencies.csv', 'w') as f:
        f.write(', ' + ', '.join(all_stimuli) + '\n')
        for i, mode_name in enumerate(collected_modes['t_modes_names']):
            f.write(mode_name + ', ' + ', '.join(latency_matrix[i,:].astype('|S16')) + '\n')

    if config['integrate']:
        fig = rl.plot_temporal_integrated(region_label, t_modes_ma)
        fig.savefig(region_savepath + '_temp_integrated.' + config['format'])
    elif config['lesion_data']:
        fig = rl.plot_temporal_lesion(region_label, t_modes_ma, medians,
                                      stim_selection, config['n_frames'])
        fig.savefig(region_savepath + '_temp_lesion.' + config['format'])
    else:
        fig = rl.plot_temporal(region_label, t_modes_ma, medians,
                               stim_selection, config['n_frames'])
        fig.savefig(region_savepath + '_temp.' + config['format'])

    # write the temporal modes to csv files
    with open(region_savepath  + '_tmodes.csv', 'w') as f:
        header = __builtin__.sum([[s] * config['n_frames'] for s in all_stimuli], [])
        f.write(', ' + ', '.join(header) + '\n')
        for i, mode_name in enumerate(collected_modes['t_modes_names']):
            f.write(mode_name + ', ' + ', '.join(collected_modes['t_modes'][i,:].astype('|S16')) + '\n')

    # spatial base plots
    fig = rl.plot_spatial_base(region_label, collected_modes['s_modes'],
                               config['to_turn'], load_path, lut_colormaps)
    fig.savefig(region_savepath + '_spatial.' + config['format'])


if config['integrate']:

    fig = rl.plot_median_overview(region_label, medians, all_stimuli)
    fig.savefig(os.path.join(save_path, 'medians.' + config['format']))
    np.savetxt(os.path.join(save_path, 'medians.csv'), medians.values(), delimiter=',')

    fig = rl.plot_median_comparison(medians, config['comparisons'], all_stimuli)
    fig.savefig(os.path.join(save_path, 'comparisons.' + config['format']))

    all_odors = sorted(set([s.split('_')[0] for s in all_stimuli]))
    for odor in all_odors:
        fig = rl.plot_region_comparison_for(odor, medians, all_stimuli, all_region_labels)
        plt.savefig(os.path.join(save_path, 'odors', odor + '.' + config['format']))

    fig = rl.plot_medians_heatmap(medians, config['main_regions'])
    fig.savefig(os.path.join(save_path, 'heatmap.' + config['format']))

    fig = rl.plot_splitsort_heatmaps(medians, all_stimuli, all_odors, config)
    plt.savefig(os.path.join(save_path, 'split_heatmap.' + config['format']))

    fig = rl.plot_split_valenz_heatmap(valenz, config)
    fig.savefig(os.path.join(save_path, 'split_heatmap_valenz.' + config['format']))

    data_dict = rl.organize_data_in_dict(medians, all_stimuli, all_odors, valenz, config)
    fig = rl.plot_medians_3d(data_dict, config)
    plt.savefig(os.path.join(save_path, '3dscatter.' + config['format']))

    fig = rl.plot_valenz_3d(data_dict, config)
    plt.savefig(os.path.join(save_path, '3dscatter_valenz.' + config['format']))

    models = rl.fit_models(data_dict, config)

    # valenz vs. activation plot
    fig = plt.figure()
    N = 3
    for i in range(N):
        ax = fig.add_subplot(N, 1, i)
        ax.scatter(models[config['main_regions'][i]], models['val'])
        ax.set_title('%s %.2f' % (config['main_regions'][i],
            np.corrcoef(models[config['main_regions'][i]], models['val'])[0,1]))
        ax.set_xlabel('activation')
        ax.set_ylabel('valenz')
    plt.savefig(os.path.join(save_path, 'activation_vs_valenz.' + config['format']))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(models['diff'], models['val'])
    ax.set_title('vlPRCt - alpha * iPN %.2f' % np.corrcoef(models['diff'], models['val'])[0,1])
    ax.set_xlabel('activation difference')
    ax.set_ylabel('valenz')
    plt.savefig(os.path.join(save_path, 'activation(difference)_vs_valenz.' + config['format']))

    idx = np.argmax(models['ratio'])
    models['val'].pop(idx)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(models['ratio'], models['val'])
    ax.set_title('vlPRCt / iPN %.2f' % np.corrcoef(models['ratio'], models['val'])[0,1])
    ax.set_xlabel('activation ratio')
    ax.set_ylabel('valenz')
    plt.savefig(os.path.join(save_path, 'activation(ratio)_vs_valenz.' + config['format']))
