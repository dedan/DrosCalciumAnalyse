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

@author: stephan.gabler@gmail.com, j.soelter@fu-berlin.de
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
if os.path.exists(config['animal_selection_file']):
    l.info('found animal selection file')
    selection = json.load(open(config['animal_selection_file']))

# load stimuli (and order to visualize)
stim_selection = []
if os.path.exists(config['stimuli_order_file']):
    l.info('found stimulus selection file')
    stim_selection = json.load(open(config['stimuli_order_file']))

# load valenz information (which odor they like)
valenz = json.load(open(config['valence_file']))

# read lesion-tract table into dictionary for easy access
if config['lesion_data']:
    lesion_dict = rl.load_lesion_data(config['lesion_table_path'])

#get data
l.info('read files from: %s' % load_path)
# read mf results
data = rl.load_mf_results(load_path, selection, config['lesion_data'])
#collect bg for animals (take just first picture)
bg_dic = rl.load_baseline(load_path, selection)
for k in bg_dic:
    bg_dic[k] = bg_dic[k].shaped2D()[0]

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

    modes = rl.collect_modes_for(region_label, regions_file_path, data)
    integrator = bf.StimulusIntegrator(method=config['integration_method'],
                                       threshold= -10000,
                                       window=config['integration_window'])
    fulldatadic[region_label] = {}
    fulldatadic[region_label]['modes'] = modes
    fulldatadic[region_label]['modes_integrated'] = integrator(modes)

# produce per-region plots
for region_label in all_region_labels:

    #get_data
    region_savepath = os.path.join(save_path, region_label)
    modes = fulldatadic[region_label]['modes']
    modes_integrated = fulldatadic[region_label]['modes_integrated']

    # =========================================================================
    # latencies: calc, plot and save
    # =========================================================================
    latencies = rl.compute_latencies(modes)

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('latencies', y=0.96)
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    rl.boxplot(ax, latencies, stim_selection)
    fig.savefig(region_savepath + '_latencies.' + config['format'])
    rl.write_csv_wt_labels(region_savepath + '_latencies.csv',
                           latencies)

    # =========================================================================
    # mean activation: plot, calc and save
    # =========================================================================
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle('activation', y=0.96)
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    rl.boxplot(ax, modes_integrated, stim_selection)
    fig.savefig(region_savepath + '_activation_integrated.' + config['format'])
    rl.write_csv_wt_labels(region_savepath + '_activation_integrated.csv',
                           modes_integrated)

    # =========================================================================
    # full time activation: plot, calc and save
    # =========================================================================    
    # TODO: fix lesion plots and save results
    if config['lesion_data']:
        fig = rl.plot_temporal_lesion(region_label, t_modes_ma, medians,
                                      stim_selection)
        fig.savefig(region_savepath + '_activation_lesion.' + config['format'])
        #write to csv
    else:
        fig = plt.figure(figsize=(25, 3))
        fig.suptitle(region_label, y=0.96)
        ax2stim = rl.axesline_dic(fig, stim_selection)
        rl.plot_temporal(modes, stim_selection, ax2stim)
        fig.savefig(region_savepath + '_activation.' + config['format'])
        rl.write_csv_wt_labels(region_savepath + '_activation.csv', modes)

    # =========================================================================
    # region bases: plot
    # ========================================================================= 
    fig = plt.figure(figsize=(12, 12))
    axlist = rl.axesgrid_list(fig, len(modes.base.shape))
    rl.plot_spatial_base(axlist, modes.base, bg_dic)
    fig.savefig(region_savepath + '_spatial.' + config['format'])


#if config['integrate']:
#
#    fig = rl.plot_median_overview(region_label, medians, all_stimuli)
#    fig.savefig(os.path.join(save_path, 'medians.' + config['format']))
#    np.savetxt(os.path.join(save_path, 'medians.csv'), medians.values(), delimiter=',')
#
#    fig = rl.plot_median_comparison(medians, config['comparisons'], all_stimuli)
#    fig.savefig(os.path.join(save_path, 'comparisons.' + config['format']))
#
#    all_odors = sorted(set([s.split('_')[0] for s in all_stimuli]))
#    for odor in all_odors:
#        fig = rl.plot_region_comparison_for(odor, medians, all_stimuli, all_region_labels)
#        plt.savefig(os.path.join(save_path, 'odors', odor + '.' + config['format']))
#
#    fig = rl.plot_medians_heatmap(medians, config['main_regions'])
#    fig.savefig(os.path.join(save_path, 'heatmap.' + config['format']))
#
#    fig = rl.plot_splitsort_heatmaps(medians, all_stimuli, all_odors, config)
#    plt.savefig(os.path.join(save_path, 'split_heatmap.' + config['format']))
#
#    fig = rl.plot_split_valenz_heatmap(valenz, config)
#    fig.savefig(os.path.join(save_path, 'split_heatmap_valenz.' + config['format']))
#
#    data_dict = rl.organize_data_in_dict(medians, all_stimuli, all_odors, valenz, config)
#    fig = rl.plot_medians_3d(data_dict, config)
#    plt.savefig(os.path.join(save_path, '3dscatter.' + config['format']))
#
#    fig = rl.plot_valenz_3d(data_dict, config)
#    plt.savefig(os.path.join(save_path, '3dscatter_valenz.' + config['format']))
#
#    models = rl.fit_models(data_dict, config)
#
#    # valenz vs. activation plot
#    fig = plt.figure()
#    N = 3
#    for i in range(N):
#        ax = fig.add_subplot(N, 1, i)
#        ax.scatter(models[config['main_regions'][i]], models['val'])
#        ax.set_title('%s %.2f' % (config['main_regions'][i],
#            np.corrcoef(models[config['main_regions'][i]], models['val'])[0, 1]))
#        ax.set_xlabel('activation')
#        ax.set_ylabel('valenz')
#    plt.savefig(os.path.join(save_path, 'activation_vs_valenz.' + config['format']))
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.scatter(models['diff'], models['val'])
#    ax.set_title('vlPRCt - alpha * iPN %.2f' % np.corrcoef(models['diff'], models['val'])[0, 1])
#    ax.set_xlabel('activation difference')
#    ax.set_ylabel('valenz')
#    plt.savefig(os.path.join(save_path, 'activation(difference)_vs_valenz.' + config['format']))
#
#    idx = np.argmax(models['ratio'])
#    models['val'].pop(idx)
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.scatter(models['ratio'], models['val'])
#    ax.set_title('vlPRCt / iPN %.2f' % np.corrcoef(models['ratio'], models['val'])[0, 1])
#    ax.set_xlabel('activation ratio')
#    ax.set_ylabel('valenz')
#    plt.savefig(os.path.join(save_path, 'activation(ratio)_vs_valenz.' + config['format']))
