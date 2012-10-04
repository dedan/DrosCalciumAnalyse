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
from sklearn import linear_model
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
valenz_orig = json.load(open(os.path.join(config['results_path'], 'valenz.json')))

# read lesion-tract table into dictionary for easy access
if config['lesion_data']:
    lesion_dict = rl.load_lesion_data(config['lesion_table_path'])

# read mf results
l.info('read files from: %s' % load_path)
data = rl.load_mf_results(load_path, selection, config['lesion_data'],
                          config['integrate'], config['integrator_window'])

# get all stimuli and region labels
all_stimuli = sorted(set(it.chain.from_iterable([ts.label_sample for ts in data.values()])))
if not stim_selection:
    stim_selection = all_stimuli
regions_file_path = os.path.join(config['results_path'], 'regions.json')
with open(regions_file_path) as f:
    all_labels_list = [labels for labels in json.load(f).values()]
    all_region_labels = list(set(it.chain.from_iterable(all_labels_list)))
l.debug('all_stimuli: %s' % all_stimuli)
l.debug('all_region_labels: %s' % all_region_labels)

# produce a figure for each region_label
medians, fulldatadic = {}, {}
for region_label in all_region_labels:

    region_savepath = os.path.join(save_path, region_label + '_latencies.')
    collected_modes = rl.collect_modes_for(region_label, regions_file_path, data)

    # latency plots
    if not config['integrate']:
        latency_matrix = rl.compute_latencies(collected_modes, config['n_frames'])
        fig = rl.plot_latencies(latency_matrix, region_label, all_stimuli)
        fig.savefig(region_savepath + '_latencies.' + config['format'])

        # TODO: write generic function to write CSVs with headers
        with open(region_savepath + '_latencies.csv', 'w') as f:
            f.write(', ' + ', '.join(all_stimuli) + '\n')
            for i, mode_name in enumerate(collected_modes['t_modes_names']):
                f.write(mode_name + ', ' + ', '.join(latency_matrix[i,:].astype('|S16')) + '\n')

    # temporal mode plots
    t_modes_ma = rl.get_masked_selection(collected_modes['t_modes'],
                                         all_stimuli,
                                         stim_selection,
                                         config['integrate'])
    fulldatadic[region_label] = t_modes_ma
    medians[region_label] = np.ma.extras.median(t_modes_ma, axis=0)
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

    fig = rl.median_overview_plot(region_label, medians, all_stimuli)
    fig.savefig(os.path.join(save_path, 'medians.' + config['format']))
    np.savetxt(os.path.join(save_path, 'medians.csv'), medians.values(), delimiter=',')

    fig = rl.median_comparison_plot(medians, config['comparisons'], all_stimuli)
    fig.savefig(os.path.join(save_path, 'comparisons.' + config['format']))

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
            plot_data = [medians[key].data[idx] for key in sorted(medians.keys())]
            plot_data[plot_data == 0.0] = 0.01
            ax.bar(range(len(medians)), plot_data)
            ax.set_yticks(range(int(np.max(np.array(medians.values()).flatten()))))
            ax.set_xticks([])
            ax.set_ylabel(conc, rotation='0')
        ax.set_xticks(range(len(all_region_labels)))
        ax.set_xticklabels(sorted(medians.keys()), rotation='90')
        plt.savefig(os.path.join(save_path, 'odors', odor + '.' + format))

    # median heatmaps
    hm_data = np.array([medians[region] for region in main_regions])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(hm_data, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks(range(len(main_regions)))
    ax.set_yticklabels(main_regions)
    plt.savefig(os.path.join(save_path, 'heatmap.' + format))

    conc = ['-1', '-3', '-5']
    fig = plt.figure()
    dats = []

    # splitted and sorted heatmaps
    new_order = [u'MSH', u'LIN', u'BEA', u'OCO', u'ISO', u'ACP', u'BUT',
                 u'2PA', u'PAC', u'CO2', u'AAC', u'ACA', u'ABA', u'BUD',
                 u'PAA', u'GEO', u'CVA', u'MOL']
    for i in range(len(conc)):
        plotti = np.zeros((3, len(new_order)))
        for y, odor in enumerate(new_order):
            for x, co in enumerate(conc):
                if odor == 'MOL':
                    stim = 'MOL_0'
                else:
                    stim = '%s_%s' % (odor, co)
                if stim in all_stimuli:
                    plotti[x, y] = hm_data[i, all_stimuli.index(stim)]
        dats.append(plotti)
    for i in range(len(conc)):
        ax = fig.add_subplot(len(conc), 1, i + 1)
        ax.set_title("region: %s - max: %f" % (main_regions[i], np.max(dats[i])))
        ax.imshow(dats[i], interpolation='nearest')
        ax.set_yticks(range(len(conc)))
        ax.set_yticklabels(conc)
        ax.set_xticks([])
    ax.set_xticks(range(len(all_odors)))
    ax.set_xticklabels(new_order, rotation='90')
    plt.savefig(os.path.join(save_path, 'split_heatmap.' + format))

    # normalize valenz for colormap
    all_vals = np.array(valenz.values())
    for val in valenz:
        valenz[val] = (valenz[val] / (2 * np.abs(np.max(all_vals))) + 0.5)

    # splitted heatmap for valenz information
    fig = vis.VisualizeTimeseries()
    fig.subplot(1)
    ax = fig.axes['base'][0]
    plotti = np.ones((3, len(new_order))) * 0.5
    for y, odor in enumerate(new_order):
        for x, co in enumerate(conc):
            if odor == 'MOL':
                stim = 'MOL_0'
            else:
                stim = '%s_%s' % (odor, co)
            if stim in valenz:
                plotti[x, y] = valenz[stim]
    fig.imshow(ax, plotti, cmap=plt.cm.RdYlGn)
    fig.overlay_image(ax, plotti == 0.5, threshold=0.1,
                      title={"label": "valenz - max: %f" % np.max(all_vals)},
                      colormap=plt.cm.gray)
    ax.set_yticks(range(len(conc)))
    ax.set_yticklabels(conc)
    ax.set_xticks(range(len(new_order)))
    ax.set_xticklabels(new_order, rotation='90')
    plt.savefig(os.path.join(save_path, 'split_heatmap_valenz.' + format))


    # prepare data for 3 d plots
    tmp_dat = {}
    for i in range(len(all_stimuli)):
        odor, concen = all_stimuli[i].split('_')
        if not odor in tmp_dat:
            tmp_dat[odor] = {}
        tmp_dat[odor][concen] = {}
        c = plt.cm.hsv(float(all_odors.index(odor)) / len(all_odors))
        tmp_dat[odor][concen]['color'] = c

        # add color to code valenz (if valenz available)
        if all_stimuli[i] in valenz:
            c = plt.cm.RdYlGn(valenz[all_stimuli[i]])
            tmp_dat[odor][concen]['valenz_color'] = c
            tmp_dat[odor][concen]['valenz'] = valenz[all_stimuli[i]]
            tmp_dat[odor][concen]['valenz_orig'] = valenz_orig[all_stimuli[i]]
        tmp_dat[odor][concen]['data'] = hm_data[:, i]
    symbols = {'-1': 's', '-2': 's', '-3': 'o', '-5': 'o', '0': 'x'}


    # 3d plot of the data also shown as heatmap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for odor in tmp_dat:
        for concen in tmp_dat[odor]:
            ax.scatter(*[[i] for i in tmp_dat[odor][concen]['data']],
                       edgecolors=tmp_dat[odor][concen]['color'], facecolors='none',
                       marker=symbols[concen], label=odor)
        ax.plot([], [], 'o', c=tmp_dat[odor][concen]['color'], label=odor)
        s_concen = sorted([int(concen) for concen in tmp_dat[odor]])
        bla = np.array([tmp_dat[odor][str(concen)]['data'] for concen in s_concen])
        ax.plot(*[x for x in bla.T], c=tmp_dat[odor][str(concen)]['color'])
    ax.set_xlabel(main_regions[0])
    ax.set_ylabel(main_regions[1])
    ax.set_zlabel(main_regions[2])
    plt.legend(loc=(0.0, 0.6), ncol=2, prop={"size":9})
    plt.savefig(os.path.join(save_path, '3dscatter.' + format))

    # concentration matrix plot
    fig = plt.figure()
    N = 3
    for i in range(3):
        for j in range(3):
            if i > j:
                ax = fig.add_subplot(N, N, i*N+j+1)
                for odor in tmp_dat:
                    for concen in tmp_dat[odor]:
                        if int(concen) >= -2:
                            ax.scatter(tmp_dat[odor][concen]['data'][i], tmp_dat[odor][concen]['data'][j],
                                       edgecolors=tmp_dat[odor][concen]['color'], facecolors='none',
                                       marker=symbols[concen], label=odor)
                    s_concen = sorted([int(concen) for concen in tmp_dat[odor]])
                    bla = np.array([tmp_dat[odor][str(concen)]['data'] for concen in s_concen])
                    ax.plot(bla[:,i], bla[:,j], c=tmp_dat[odor][str(concen)]['color'])
                ax.set_xlabel(main_regions[i])
                ax.set_ylabel(main_regions[j])
                ax.set_xticks([])
                ax.set_yticks([])
            elif i == j:
                ax = fig.add_subplot(N, N, i*N+j+1)
                bla = np.array([tmp_dat[o][c]['data'][i]
                                for o in tmp_dat
                                for c in tmp_dat[o]])
                ax.hist(bla, color='k')
                ax.set_title(main_regions[i])
                ax.set_xticks([])
                ax.set_yticks([])

    plt.savefig(os.path.join(save_path, 'matrix.' + format))

    # 3d valenz plot
    symbols = {'-1': 'o', '-2': 'o', '-3': 'o', '-5': 'o', '0': 'x'}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for odor in tmp_dat:
        for concen in tmp_dat[odor]:
            if 'valenz_color' in tmp_dat[odor][concen]:
                ax.scatter(*[[i] for i in tmp_dat[odor][concen]['data']],
                           edgecolors=tmp_dat[odor][concen]['valenz_color'],
                           facecolors=tmp_dat[odor][concen]['valenz_color'],
                           marker=symbols[concen], label=odor)
                ax.plot([], [], 'o', c=tmp_dat[odor][concen]['valenz_color'], label=odor)
                s_concen = sorted([int(concen) for concen in tmp_dat[odor]])
                bla = np.array([tmp_dat[odor][str(concen)]['data'] for concen in s_concen])
                ax.plot(*[x for x in bla.T], c='0.5')
    ax.set_xlabel(main_regions[0])
    ax.set_ylabel(main_regions[1])
    ax.set_zlabel(main_regions[2])
    #plt.legend(loc=(0.0, 0.6), ncol=2, prop={"size":9})
    plt.savefig(os.path.join(save_path, '3dscatter_valenz.' + format))

    regressor = linear_model.LinearRegression(fit_intercept=False)
    x, y = [], []
    for odor in tmp_dat:
        for concen in tmp_dat[odor]:
            if 'valenz_orig' in tmp_dat[odor][concen]:
                t = tmp_dat[odor][concen]
                x.append([t['data'][2], t['data'][0]])
                y.append(t['valenz_orig'])
    fit = regressor.fit(x,y)
    alpha = fit.coef_[1]

    agg = defaultdict(list)
    for odor in tmp_dat:
        for concen in tmp_dat[odor]:
            if 'valenz_orig' in tmp_dat[odor][concen]:
                t = tmp_dat[odor][concen]
                agg['val'].append(t['valenz_orig'])
                for i in range(3):
                    agg[main_regions[i]].append(t['data'][i])
                agg['ratio'].append(tmp_dat[odor][concen]['data'][2] /
                                    tmp_dat[odor][concen]['data'][0])
                agg['diff'].append(tmp_dat[odor][concen]['data'][2] -
                                   alpha * tmp_dat[odor][concen]['data'][0])
    idx = np.argmax(agg['ratio'])
    agg['ratio'].pop(idx)


    # valenz vs. activation plot
    fig = plt.figure()
    N = 3
    for i in range(N):
        ax = fig.add_subplot(N, 1, i)
        ax.scatter(agg[main_regions[i]], agg['val'])
        ax.set_title('%s %.2f' % (main_regions[i], np.corrcoef(agg[main_regions[i]], agg['val'])[0,1]))
        ax.set_xlabel('activation')
        ax.set_ylabel('valenz')
    plt.savefig(os.path.join(save_path, 'activation_vs_valenz.' + format))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(agg['diff'], agg['val'])
    ax.set_title('vlPRCt - alpha * iPN %.2f' % np.corrcoef(agg['diff'], agg['val'])[0,1])
    ax.set_xlabel('activation difference')
    ax.set_ylabel('valenz')
    plt.savefig(os.path.join(save_path, 'activation(difference)_vs_valenz.' + format))

    agg['val'].pop(idx)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(agg['ratio'], agg['val'])
    ax.set_title('vlPRCt / iPN %.2f' % np.corrcoef(agg['ratio'], agg['val'])[0,1])
    ax.set_xlabel('activation ratio')
    ax.set_ylabel('valenz')
    plt.savefig(os.path.join(save_path, 'activation(ratio)_vs_valenz.' + format))
