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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');
reload(utils)
reload(rl)

config = ConfigObj(sys.argv[1], unrepr=True)

# another colormap dictionary
# TODO: more generic solution
color_dic = defaultdict(lambda: np.array([160, 82, 45]) / 256.)
color_dic['iPN'] = np.array([48, 128, 20]) / 256.
color_dic['iPNsecond'] = np.array([173, 255, 47]) / 256.
color_dic['iPNtract'] = np.array([67, 110, 238]) / 256.
color_dic['vlPRCt'] = np.array([220, 20, 60]) / 256.
color_dic['vlPRCb'] = np.array([238, 218, 0]) / 256.
color_dic['betweenTract'] = np.array([155, 48, 255]) / 256.
color_dic['blackhole'] = np.array([142, 142, 142]) / 256.

def my_xaxeslayout(ax, ts, tickstep, labelstep):
    ax.xaxis.set_tick_params(direction='out', top=False, bottom=True, size=3)
    xticks = range(0, ts.timepoints + 1, tickstep)
    ax.set_xlim((0, ts.timepoints))
    ax.set_xticks(xticks)
    mylabels = ['%d' % i for i in np.array(xticks) * 1. / ts.framerate]
    for i in range(1, len(mylabels), labelstep):
        mylabels[i] = ''
    ax.set_xticklabels(mylabels, fontsize=8)

# set paths and create folders
save_path = os.path.join(config['results_path'], 'region_plots')
overall_savepath = os.path.join(save_path, 'overall')
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(os.path.join(save_path, 'regions')):
    os.mkdir(os.path.join(save_path, 'regions'))
if not os.path.exists(os.path.join(save_path, 'all_regions')):
    os.mkdir(os.path.join(save_path, 'all_regions'))
if not os.path.exists(overall_savepath):
    os.mkdir(overall_savepath)

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

# load 2D stimuli layout
if os.path.exists(config['stimuli_matrix_file']):
    l.info('found 2D stimulus layout file')
    stim_matrix = json.load(open(config['stimuli_matrix_file']))

# load valenz information (which odor they like)
valenz = json.load(open(config['valence_file']))

#get data
l.info('read files from: %s' % config['results_path'])
# read mf results
data = rl.load_mf_results(config['results_path'], selection, config['lesion_table_path'])
# for old data: rename labels
# TODO: remove if all data in correct format
for mf in data.values():
    change_from = ['CO2_-1', 'CO2_-5', 'CO2_1', 'CO2_5']
    change_to = ['CO2_01', 'CO2_05', 'CO2_01', 'CO2_05']
    new_labels = []
    for i_label in mf.label_sample:
        if i_label in change_from:
            new_labels.append(change_to[change_from.index(i_label)])
        else:
            new_labels.append(i_label)
    mf.label_sample = new_labels

# instantiate function to integrate stimuli
integrator = bf.StimulusIntegrator(method=config['integration_method'],
                                   threshold= -10000,
                                   window=config['integration_window'])

# get all stimuli and region labels (use selection if selection given)
all_stimuli = sorted(set(it.chain.from_iterable([ts.label_sample for ts in data.values()])))
if not stim_selection:
    stim_selection = all_stimuli
with open(config['regions_file_path']) as f:
    regions_dic = json.load(f)
    all_region_labels = list(set(__builtin__.sum(regions_dic.values(), [])))
l.debug('all_stimuli: %s' % all_stimuli)
l.debug('all_region_labels: %s' % all_region_labels)

# calc Stimulusdrive for each mode and than reduce to single stimulus response
average_over_stimulus_repetitions = bf.SingleSampleResponse()
stimulusdrive_fct = bf.CalcStimulusDrive()
stimulusdrive_ts_dic = {}
for key, dat in data.items():
    st_drive = stimulusdrive_fct(dat)
    st_drive.timecourses = st_drive.timecourses.flatten()
    stimulusdrive_ts_dic[key] = st_drive
    data[key] = average_over_stimulus_repetitions(dat)

# create dataset according to stimuli set
# also reorganice stimulusdrives to dic with single value
stim_sets = {'PAC':[], 'BEA':[], 'LIN':[], 'ISO':[]}
for id_stim, animallist in stim_sets.items():
    for animal, mf in data.items():
        if id_stim + '_-1' in mf.label_sample:
            animallist.append(animal)
stimulusdrive_dic = {}
for id_stim, animals in stim_sets.items():
    all_data = []
    for animal_id in animals:
        if animal_id not in regions_dic.keys():
            continue
        dat = data[animal_id].copy()
        if config['mode_thres']:
            dat = bf.SelectModes(config['mode_thres'])(dat,
                                            stimulusdrive_ts_dic[animal_id])
        modeIDs = [int(i[-1]) for i in dat.label_objects]
        dat.label_objects = [regions_dic[animal_id][i] for i in modeIDs]
        dat.name = '_'.join(dat.name.split('_')[:-1])
        if dat.num_objects:
            all_data.append(dat)
        else:
            l.info('no mode left for %s' % animal_id)
        for mode_id, lab  in zip(modeIDs, dat.label_objects):
            stimulusdrive_dic['_'.join([animal_id, lab])] = stimulusdrive_ts_dic[animal_id].timecourses[mode_id]
    stim_sets[id_stim] = integrator(bf.ObjectConcat(unequalsample=True)(all_data))

#collect bg for animals (take just first picture)
bg_dic = rl.load_baseline(config['results_path'], selection)
for k in bg_dic:
    bg_dic[k] = bg_dic[k].shaped2D()[0]

# collect data for regions
medians, fulldatadic = {}, {}
for region_label in all_region_labels:

    modes = rl.collect_modes_for(region_label, regions_dic, data)

    fulldatadic[region_label] = {}
    fulldatadic[region_label]['modes'] = modes
    fulldatadic[region_label]['modes_integrated'] = integrator(modes)

median_ts_list = [rl.calc_scoreatpercentile(region['modes'], 0.5)
                      for region in fulldatadic.values()]
all_region_ts_raw = bf.ObjectConcat()(median_ts_list)
# TODO: save as csv
# remove percentile string from object labels
all_region_ts_raw.label_objects = ['_'.join(ilabel.split('_')[:-1])
                                   for ilabel in all_region_ts_raw.label_objects]
median_ts_list = [rl.calc_scoreatpercentile(region['modes_integrated'], 0.5)
                      for region in fulldatadic.values()]
all_region_ts = bf.ObjectConcat()(median_ts_list)
# TODO: save as csv
# remove percentile string from object labels
all_region_ts.label_objects = ['_'.join(ilabel.split('_')[:-1])
                               for ilabel in all_region_ts.label_objects]

if not config['lesion_table_path']:
    data_dict = rl.organize_data_in_dict(all_region_ts, stim_selection, valenz, config)
    # fix the CO2 labels
    FIXES = {'01': '-5', '05': '-3', '10': '-1'}
    for conc in FIXES:
        data_dict['CO2'][FIXES[conc]] = data_dict['CO2'][conc]
        del(data_dict['CO2'][conc])

if config['do_per_animal']:
    # =============================================================================
    # per-animal plots
    # =============================================================================
    for animal in data:
        #TODO: trimming only for old data, remove later
        if config['old_data']:
            animal_trim = '_'.join(animal.split('_')[:-1])
        else:
            animal_trim = animal
        animal_savepath = os.path.join(save_path, 'regions', animal_trim)
        # exclude animals wto region data
        if animal not in regions_dic.keys():
            continue
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
        rl. plot_regions_of_animal(ax, data[animal].base, bg_dic[animal_trim],
                                   regions_dic[animal], color_dic,
                                   cut_off=config['region_plot_cutoff'])
        fig.savefig(animal_savepath + '.' + config['format'])
        plt.close('all')

if config['do_per_region']:
    # ==========================================================================
    # per-region plots
    # ==========================================================================
    for region_label in all_region_labels:
        #get_data
        region_savepath = os.path.join(save_path, region_label)
        if not os.path.exists(region_savepath):
            os.mkdir(region_savepath)
        region_savepath = os.path.join(region_savepath, region_label)
        modes = fulldatadic[region_label]['modes']
        modes_integrated = fulldatadic[region_label]['modes_integrated']

        # ======================================================================
        # latencies: calc, plot and save
        # ======================================================================
        latencies = rl.compute_latencies(modes)

        fig = plt.figure(figsize=(12, 7))
        fig.suptitle('latencies', y=0.96)
        ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
        ax.xaxis.set_tick_params(direction='out', top=False)
        rl.boxplot(ax, latencies, stim_selection)
        fig.savefig(region_savepath + '_latencies.' + config['format'])
        rl.write_csv_wt_labels(region_savepath + '_latencies.csv', latencies)

        # ======================================================================
        # mean activation: plot, calc and save
        # ======================================================================
        fig = plt.figure(figsize=(12, 7))
        fig.suptitle('activation', y=0.96)
        ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
        ax.xaxis.set_tick_params(direction='out', top=False)
        rl.boxplot(ax, modes_integrated, stim_selection)
        fig.savefig(region_savepath + '_activation_integrated.' + config['format'])
        rl.write_csv_wt_labels(region_savepath + '_activation_integrated.csv',
                               modes_integrated)

        # ======================================================================
        # full time activation: plot, calc and save
        # ======================================================================
        if config['lesion_table_path']:
            fig = plt.figure()
            fig.suptitle(region_label)
            tmp_stim_select = []
            for stim in stim_selection:
                tmp_stim = '_'.join(stim.split('_')[0:2])
                if not tmp_stim in tmp_stim_select:
                    tmp_stim_select.append(tmp_stim)
            stim2ax = rl.axesgroupline_dic(fig, stim_matrix, leftspace=0.05, topspace=0.6,
                                   bottomspace=0.1, inner_axesspace=0.2, gapspace=0.01,
                                   title_param={'va':'bottom', 'rotation':'90', 'fontsize':8})
            rl.plot_temporal(modes, stim2ax, conditions=config['conditions'])
            savename = region_savepath + '_activation_lesion.' + config['format']
            rl.write_csv_wt_labels(region_savepath + '_activation_lesion.csv', modes)
        else:
            fig = plt.figure(figsize=(25, 3))
            fig.suptitle(region_label, y=0.96)
            stim2ax = rl.axesgroupline_dic(fig, stim_matrix, leftspace=0.05, topspace=0.6,
                                   bottomspace=0.1, inner_axesspace=0.2, gapspace=0.01,
                                   title_param={'va':'bottom', 'rotation':'90', 'fontsize':8})
            rl.plot_temporal(modes, stim2ax, linecolor=color_dic[region_label])
            savename = region_savepath + '_activation.' + config['format']
            rl.write_csv_wt_labels(region_savepath + '_activation.csv', modes)

        for ax in stim2ax['bottom']:
            my_xaxeslayout(ax, modes, 4, 2)
            ax.yaxis.set_tick_params(direction='out', left=False, right=False,
                                 labelleft=False, labelright=False)
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
        for ax in stim2ax['left_inner']:
            ax.yaxis.set_tick_params(left=True)
            ax.spines['left'].set_color('k')
        for ax in stim2ax['left']:
            ax.yaxis.set_tick_params(labelleft=True)
        fig.savefig(savename)

        # ======================================================================
        # region bases: plot
        # ======================================================================
        fig = plt.figure(figsize=(12, 12))
        axlist = rl.axesgrid_list(fig, len(modes.base.shape))
        rl.plot_spatial_base(axlist, modes.base, bg_dic)
        fig.savefig(region_savepath + '_spatial.' + config['format'])
        plt.close('all')

if config['do_overall_region']:
    with open(config['region_order_file']) as f:
        regions2plot = json.load(f)
    # ==========================================================================
    # median region activation plots
    # ==========================================================================
    fig = plt.figure(figsize=(25, 2))
    stim2ax = rl.axesgroupline_dic(fig, stim_matrix, leftspace=0.05, topspace=0.6,
                                   bottomspace=0.2, inner_axesspace=0.2, gapspace=0.01,
                                   title_param={'va':'bottom', 'rotation':'90', 'fontsize':8})
    rl.plot_stim_heatmap(all_region_ts_raw, stim2ax, regions2plot,
                         imshow_args={'symetric':False, 'aspect':'auto',
                                      'cmap':utils.colormap_from_lut(config['heatmap_course'])})
    for ax in stim2ax['bottom']:
        my_xaxeslayout(ax, all_region_ts_raw, 4, 2)
        ax.yaxis.set_tick_params(direction='out', left=False, right=False,
                                 labelleft=False, labelright=False)

    for ax in stim2ax['left']:
        ax.yaxis.set_tick_params(labelleft=True)

    fig.savefig(os.path.join(overall_savepath,
                             'activation_heatmap.' + config['format']))

    # ==========================================================================
    # median integrated region activation plots
    # ==========================================================================
    fig = plt.figure(figsize=(20, 2))
    ax2stim = rl.axesgroupline_dic(fig, stim_matrix, leftspace=0.05, topspace=0.6,
                                   bottomspace=0.2, inner_axesspace= -0.01, gapspace=0.01,
                                   title_param={'va':'bottom', 'rotation':'90', 'fontsize':8})
    rl.plot_stim_heatmap(all_region_ts, ax2stim, regions2plot,
                         imshow_args={'symetric':False, 'aspect':'auto',
                                      'cmap':utils.colormap_from_lut(config['heatmap_point'])})
    for ax in ax2stim['bottom']:
        ax.xaxis.set_tick_params(top=False, bottom=False, labeltop=False,
                                labelbottom=False)
        ax.yaxis.set_tick_params(left=False, right=False, labelleft=False,
                                labelright=False)
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
    for ax in ax2stim['left_inner']:
        ax.spines['left'].set_color('k')
    for ax in ax2stim['right_inner']:
        ax.spines['right'].set_color('k')
    for ax in ax2stim['left']:
        ax.yaxis.set_tick_params(labelleft=True)

    fig.savefig(os.path.join(overall_savepath,
                             'activation_heatmap_integrated.' + config['format']))


    if not config['lesion_table_path']:

        # ======================================================================
        # hierachical clustering
        # ======================================================================
        for id_stim, set_modes in stim_sets.items():
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_axes([0.3, 0.05, 0.65, 0.9])
            dendrogram(linkage(pdist(set_modes.timecourses.T, 'cosine'), 'average'),
                       labels=set_modes.label_objects, orientation='left', color_threshold=0)
            newtext = [i.get_text() + ' (%.2f)' % stimulusdrive_dic[i.get_text()]
                        for i in ax.get_yticklabels()]
            ax.set_yticklabels(newtext)
            for lab in ax.get_yticklabels():
                region_name = lab.get_text().split('_')[-1].split()[0]
                lab.set_color(color_dic[region_name])

            ax.set_title(','.join(set([i.split('_')[0] for i in set_modes.label_sample])))
            fig.savefig(os.path.join(overall_savepath, 'cluster' + id_stim
                                     + '.' + config['format']))
        # ======================================================================
        # overview plots
        # ======================================================================

        fig = rl.plot_median_comparison(all_region_ts, config['comparisons'])
        fig.savefig(os.path.join(overall_savepath, 'comparisons.' + config['format']))

        fig = rl.plot_splitsort_heatmaps(data_dict, valenz, stim_selection, config)
        fig.savefig(os.path.join(overall_savepath, 'split_heatmap.' + config['format']))

        fig = rl.plot_valenz_3d(data_dict, config)
        plt.savefig(os.path.join(overall_savepath, '3dscatter_valenz.' + config['format']))

        models = rl.fit_models(data_dict, config)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(models['diff'], models['val'])
        ax.set_title('vlPRCt - alpha * iPN %.2f' % np.corrcoef(models['diff'], models['val'])[0, 1])
        ax.set_xlabel('activation difference')
        ax.set_ylabel('valenz')
        fig.savefig(os.path.join(overall_savepath, 'activation(difference)_vs_valenz.' + config['format']))

        idx = np.argmax(models['ratio'])
        models['val'].pop(idx)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(models['ratio'], models['val'])
        ax.set_title('vlPRCt / iPN %.2f' % np.corrcoef(models['ratio'], models['val'])[0, 1])
        ax.set_xlabel('activation ratio')
        ax.set_ylabel('valenz')
        fig.savefig(os.path.join(overall_savepath, 'activation(ratio)_vs_valenz.' + config['format']))
        plt.close('all')


if config['do_region_concentration_valenz'] and not config['lesion_table_path']:

    for region in fulldatadic.keys():
        fig = plt.figure()
        fig.suptitle(region)
        ax = fig.add_subplot(111)
        x_data, y_data = [], []
        for odor in data_dict:
            for conc in data_dict[odor]:
                if region in data_dict[odor][conc]['medians']:
                    x_data.append(int(conc))
                    y_data.append(data_dict[odor][conc]['medians'][region])
        ax.scatter(x_data, y_data)
        ax.set_xlabel('concentration')
        ax.set_ylabel('median activation')
        fig.savefig(os.path.join(save_path, 'all_regions', region + '_conc_act.' + config['format']))

        fig = plt.figure()
        fig.suptitle(region)
        ax = fig.add_subplot(111)
        x_data, y_data = [], []
        for odor in data_dict:
            for conc in data_dict[odor]:
                if region in data_dict[odor][conc]['medians'] and 'valenz' in data_dict[odor][conc]:
                    x_data.append(data_dict[odor][conc]['valenz'])
                    y_data.append(data_dict[odor][conc]['medians'][region])
        ax.scatter(x_data, y_data)
        ax.set_xlabel('valenz')
        ax.set_ylabel('median activation')
        fig.savefig(os.path.join(save_path, 'all_regions', region + '_conc_val.' + config['format']))
        plt.close('all')
