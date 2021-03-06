#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys, os, glob, csv, json, __builtin__, re
import itertools as iter
from collections import defaultdict
import numpy as np
import pylab as plt
from matplotlib import gridspec
from scipy.stats.mstats_basic import scoreatpercentile, mquantiles
from NeuralImageProcessing.pipeline import TimeSeries
from NeuralImageProcessing import illustrate_decomposition as vis
import NeuralImageProcessing.basic_functions as bf
from sklearn import linear_model
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

#===============================================================================
# plot functions
#===============================================================================

def plot_valenz_3d(data_dict, config):
    """3d valenz plot"""
    symbols = lambda x: 'x' if x == '0' else 'o'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for odor in data_dict:
        for concen in data_dict[odor]:
            if 'valenz_color' in data_dict[odor][concen]:
                plot_data = [[data_dict[odor][concen]['medians'][r]] for r in config['main_regions']]
                ax.scatter(*plot_data,
                           edgecolors=data_dict[odor][concen]['valenz_color'],
                           facecolors=data_dict[odor][concen]['valenz_color'],
                           marker=symbols(concen), label=odor)
                ax.plot([], [], 'o', c=data_dict[odor][concen]['valenz_color'], label=odor)
                s_concen = sorted([int(concen) for concen in data_dict[odor]])
                line_data = [[data_dict[odor][str(concen)]['medians'][r] for r in config['main_regions']]
                                                                        for concen in s_concen]
                ax.plot(*[x for x in np.array(line_data).T], c='0.5')
    ax.set_xlabel(config['main_regions'][0])
    ax.set_ylabel(config['main_regions'][1])
    ax.set_zlabel(config['main_regions'][2])
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

def plot_split_valenz_heatmap(ax, valenz, stim_selection, config):
    """splitted heatmap for valenz information"""
    # normalize valenz for colormap
    norm_val = {}
    all_vals = np.array(valenz.values())
    for val in valenz:
        norm_val[val] = (valenz[val] / (2 * np.abs(np.max(all_vals))) + 0.5)
    all_odors = get_all_odors(stim_selection)

    fig = vis.VisualizeTimeseries()
    plotti = np.ones((3, len(all_odors))) * 0.5
    for y, odor in enumerate(all_odors):
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
    ax.set_xticks(range(len(all_odors)))
    ax.set_xticklabels(all_odors, rotation='90')

def plot_splitsort_heatmaps(data_dict, valenz, stim_selection, config):
    """split and sort heatmap of medians"""
    dats = []
    fig = plt.figure()
    n_conc = len(config['concentrations'])
    all_odors = get_all_odors(stim_selection)

    for region in config['main_regions']:
        sub_heatmap = np.zeros((n_conc, len(all_odors)))
        for i, odor in enumerate(all_odors):
            for j, conc in enumerate(config['concentrations']):
                if conc in data_dict[odor]:
                    sub_heatmap[j, i] = data_dict[odor][conc]['medians'][region]
                else:
                    sub_heatmap[j, i] = 0
        dats.append(sub_heatmap)
    for i in range(len(config['concentrations'])):
        ax = fig.add_subplot(len(config['concentrations']) + 1, 1, i + 1)
        ax.set_title("region: %s - max: %f" % (config['main_regions'][i], np.max(dats[i])))
        ax.imshow(dats[i], interpolation='nearest')
        ax.set_yticks(range(len(config['concentrations'])))
        ax.set_yticklabels(config['concentrations'])
        ax.set_xticks([])
    ax = fig.add_subplot(len(config['concentrations']) + 1, 1, 4)
    fig.subplots_adjust(hspace=0.4)
    plot_split_valenz_heatmap(ax, valenz, stim_selection, config)
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

def plot_median_comparison(medians, comparisons):
    """medians comparison plot"""
    fig = plt.figure()
    for i, comparison in enumerate(comparisons):
        ax = fig.add_subplot(len(comparisons), 1, i + 1)
        m1 = medians.timecourses[:, medians.label_objects.index(comparison[0])]
        m2 = medians.timecourses[:, medians.label_objects.index(comparison[1])]
        ax.bar(range(len(m1)), m1, color='r')
        ax.bar(range(len(m1)), m2 * -1, color='b')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(', '.join(comparison), rotation='0')
        ax.set_xlim((0, len(m1)))
    ax.set_xticks(range(len(m1)))
    ax.set_xticklabels(medians.label_sample, rotation='45')
    fig.subplots_adjust(hspace=0.6)
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

def plot_spatial_base(axlist, modes, bg_dic, old_data=False):
    plotter = vis.VisualizeTimeseries()
    for base_ix, base in enumerate(modes.objects_sample(0)):
        region_label = modes.name[0]
        mode_name = modes.label_objects[base_ix]
        if old_data:
            animal_name = '_'.join(mode_name.split('_')[:-2])
        else:
            animal_name = '_'.join(mode_name.split('_')[:-1])
        ax = axlist[base_ix]
        plotter.overlay_workaround(ax, bg_dic[animal_name], {'cmap':plt.cm.bone},
                           base, {'threshold':0.05},
                           {'title':{"label": mode_name, 'fontsize':8}})

#TODO: include plot_single
def plot_temporal(modes, stim2ax, plot_single=False, conditions={}, linecolor='k'):

    medians = calc_scoreatpercentile(modes, 0.5).trial_shaped().squeeze()
    p25 = calc_scoreatpercentile(modes, 0.25).trial_shaped().squeeze()
    p75 = calc_scoreatpercentile(modes, 0.75).trial_shaped().squeeze()

    num_modes = np.sum(np.logical_not(np.isnan(modes.trial_shaped()[:, 0])), 1)

    max_y = np.max(p75) + 0.05
    min_y = np.min(p25) - 0.05
    max_ytick = np.floor(max_y * 10) / 10
    min_ytick = np.ceil(min_y * 10) / 10

    for stim_ix, stim in enumerate(modes.label_sample):

        if conditions:
            stim, _, cond = stim.rpartition('_')
            color = conditions[cond]
        else:
            color = linecolor

        if stim not in stim2ax:
            continue
        ax = stim2ax[stim]
        ax.fill_between(range(modes.timepoints), p25[stim_ix], p75[stim_ix],
                        linewidth=0, color=color, alpha=0.4)
        ax.plot(medians[stim_ix], linewidth=1.5, color=color, label=str(num_modes[stim_ix]))
        ax.set_ylim((min_y, max_y))
        ax.set_yticks([0, max_ytick])

        le = ax.legend(frameon=False, markerscale=0.1, numpoints=1, prop={'size': 6})


def plot_stim_heatmap(ts, stim2ax, object_order=None, imshow_args={'symetric':True}):


    data = ts.trial_shaped()
    if object_order:
        ylabels = [ilabel for ilabel in object_order if ilabel in ts.label_objects]
        new_order = [ts.label_objects.index(ilabel) for ilabel in ylabels]
        data = data[:, :, new_order]
    else:
        ylabels = ts.label_objects

    max_color = np.max(np.abs(data))
    if imshow_args.pop('symetric'):
        min_color = -max_color
    else:
        min_color = np.min(data)

    for stim_ix, stim in enumerate(ts.label_sample):
        if stim not in stim2ax:
            l.info(stim + 'in heatmap excluded')
            continue
        ax = stim2ax[stim]
        ax.imshow(data[stim_ix].T, vmin=min_color, vmax=max_color,
                  interpolation='none', **imshow_args)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)

def boxplot(ax, modes, stim_selection):
    """make boxplots of modes for stim in stim_selection in ax object

       ax: axes object to draw plot
       modes: TimeSeries object which contains multiple object for each stim.
       Might contain nans. Should only have one value per stim, otherwise first
       is taken
       stim_selection: selection and order of stim to do the boxplot
    """
    if not(modes.timepoints == 1):
        l.warning('''Temporal extension of stimuli > 1 (%d)/n
                    only first taken into account''' % modes.timepoints)
    ax.set_title(modes.name)
    timecourses = modes.matrix_shaped()

    # make it a list because boxplot has a problem with masked arrays
    t_modes_ma = [row[~np.isnan(row)] for row in timecourses]
    stim_wt_data = [i_stim for i_stim in stim_selection if i_stim in modes.label_sample]
    x_index = [modes.label_sample.index(i_stim) for i_stim in stim_wt_data]

    t_modes_ma = [t_modes_ma[ind] for ind in x_index]
    distribution_size = [len(tm) for tm in t_modes_ma]
    ax.boxplot(t_modes_ma, sym='.k')
    for pos, size in enumerate(distribution_size):
        ax.text(pos + 1, ax.get_ylim()[1] * 0.99, str(size), fontsize=6, va='top')
    ax.set_xticklabels(stim_wt_data, rotation='45', ha='right')


def plot_regions_of_animal(ax, modes_bases, bg, mode_labels, color_dic, cut_off=0.3):
    ax.imshow(bg, cmap=plt.cm.bone)
    for base_ix, base in enumerate(modes_bases.shaped2D()):
        ax.contourf(base, [cut_off, 1], alpha=0.7,
                    colors=(color_dic[mode_labels[base_ix]],))
        ax.set_xticks([])
        ax.set_yticks([])


#===============================================================================
# reading and collectiog data
#===============================================================================

def organize_data_in_dict(medians, stim_selection, valenz, config):
    """prepare data for 3 d plots"""
    data_dict = defaultdict(dict)
    all_odors = get_all_odors(stim_selection)
    norm_val = {}
    all_vals = np.array(valenz.values())
    for val in valenz:
        norm_val[val] = (valenz[val] / (2 * np.abs(np.max(all_vals))) + 0.5)

    for stim in stim_selection:

        stim_idx = medians.label_sample.index(stim)
        odor, concen = stim.split('_')
        data_dict[odor][concen] = {}

        # add color code for odor
        c = plt.cm.hsv(float(all_odors.index(odor)) / len(all_odors))
        data_dict[odor][concen]['color'] = c

        # add color to code valenz (if valenz available)
        if stim in valenz:
            c = plt.cm.RdYlGn(norm_val[stim])
            data_dict[odor][concen]['valenz_color'] = c
            data_dict[odor][concen]['valenz'] = norm_val[stim]

        # add median value for odor, conc and region
        data_dict[odor][concen]['medians'] = {}
        for region in medians.label_objects:
            region_idx = medians.label_objects.index(region)
            tmp_median = medians.timecourses[stim_idx, region_idx]
            data_dict[odor][concen]['medians'][region] = tmp_median
    return data_dict

def collect_modes_for(region_label, regions_dic, data):
    """collect all spatial and temporal modes for a given region_label

        * several modes from one animal can belong to one region
        * the information for this comes from the regions.json created by
          using the regions_gui
    """
    t_modes, t_modes_names, s_modes, s_shapes = [], [], [], []
    labeled_animals = regions_dic
    all_stimuli = sorted(set(iter.chain.from_iterable([ts.label_sample for ts in data.values()])))

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
    ts_temporal.name = region_label
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
    ts_spatial = bf.TimeSeries(name=region_label, shape=s_shapes)
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

def load_mf_results(load_path, selection, lesion_table_path):
    """read data (matrix factorization results) to dictionary"""

    data = {}
    # load and filter filelist
    filelist = glob.glob(os.path.join(load_path, '*.json'))
    filelist = [f for f in filelist if not 'base' in os.path.basename(f)]
    filelist = [f for f in filelist if not 'regions' in os.path.basename(f)]
    filelist = [f for f in filelist if not 'selection' in os.path.basename(f)]
    if lesion_table_path:
        lesion_dict = load_lesion_data(lesion_table_path)

    for fname in filelist:

        if selection:
            skip = True
            for sel in selection:
                if sel in fname:
                    skip = False
            if skip:
                l.info('skip %s because not in selection' % fname)
                continue
            l.info('taking %s because found in selection' % fname)

        ts = TimeSeries()
        ts.load(os.path.splitext(fname)[0])
        name = os.path.splitext(os.path.basename(fname))[0]

        if lesion_table_path:
            tmp_name = re.search('\d.*?_(r|l)', name).group()
            if '_' in tmp_name:
                fname_base, side = tmp_name.split('_')
            new_labels = []
            for label in ts.label_sample:
                if label[0] == 'm':
                    new_labels.append(label[1:] + '_' + lesion_dict[fname_base][side])
                else:
                    new_labels.append(label + '_intact')
            ts.label_sample = new_labels
        data[name] = ts
    return data

def load_baseline(load_path, selection):
    """read data (baseline) to dictionary"""

    data = {}
    # get filelist
    filelist = glob.glob(os.path.join(load_path, '*baseline*.json'))
    for fname in filelist:
        fname = os.path.splitext(fname)[0]
        animal = os.path.basename(fname).split('_baseline')[0]
        if selection:
            skip = True
            for sel in selection:
                if sel in fname:
                    skip = False
            if skip:
                l.info('skip %s because not in selection' % fname)
                continue

        ts = TimeSeries()
        ts.load(fname)
        data[animal] = ts
    return data


#===============================================================================
# timeseries calculations
#===============================================================================

def compute_latencies(modes):
    """latency: time (in frames) of the maximum response (peak)"""
    latencies_timeseries = modes.copy()
    latencies = np.argmax(modes.trial_shaped(), axis=1).astype('float')
    latencies[np.isnan(modes.trial_shaped()[:, 0, :])] = np.nan
    latencies_timeseries.timecourses = latencies
    return latencies_timeseries

def calc_scoreatpercentile(modes, percentile):
    ''' calculates percentile for Timeseries (may include nans)'''
    out = modes.copy()
    out.timecourses = np.ma.array(out.timecourses, mask=np.isnan(out.timecourses))
    out.timecourses = mquantiles(out.matrix_shaped(), percentile, axis=1)
    out.shape = (1,)
    out.label_objects = ['percentile%d' % (percentile * 100)]
    return out

#===============================================================================
# figure layouter
#===============================================================================

def axesmatrix_dic(fig, stimlist):
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

def axesline_dic(fig, stimlist, leftspace=0.02):
    ''' generates axes list according to stimlist, returns dictionary with
    maps stim_names to axes '''
    stim2ax = defaultdict(list)
    gs = gridspec.GridSpec(1, len(stimlist))
    gs.update(left=leftspace, right=0.99, top=0.8)
    for col_ix, stim in enumerate(stimlist):
        ax = fig.add_subplot(gs[0, col_ix])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(stim, fontsize=6)
        stim2ax[stim] = ax
        stim2ax['bottom'].append(ax)
        if col_ix == 0:
            stim2ax['left'].append(ax)
    return stim2ax

def axesgroupline_dic(fig, stimlist, **kwargs):
    ''' generates axes list according to stimlist, returns dictionary with
    maps stim_names to axes '''

    title_param = kwargs.get('title_param', {})
    leftspace = kwargs.get('leftspace', 0.02)
    topspace = kwargs.get('topspace', 0.95)
    inner_axesspace = kwargs.get('inner_axesspace', 0.01)
    gapspace = kwargs.get('gapspace', 0.01)
    bottomspace = kwargs.get('bottomspace', 0.05)

    stim2ax = defaultdict(list)
    all_stim = __builtin__.sum(stimlist, [])
    ax_width = (0.99 - leftspace - (len(stimlist) - 1) * gapspace) / len(all_stim)

    stim_num = 0
    for col_ix, inner_stim in enumerate(stimlist):
        gs = gridspec.GridSpec(1, len(inner_stim))
        gs.update(left=leftspace + stim_num * ax_width + col_ix * gapspace,
                  right=(leftspace + (stim_num + len(inner_stim)) * ax_width
                         + col_ix * gapspace),
                  top=topspace, bottom=bottomspace,
                  wspace=inner_axesspace)
        for col2_ix, stim in enumerate(inner_stim):
            ax = fig.add_subplot(gs[0, col2_ix])
            ax.set_title(stim, **title_param)
            stim2ax[stim] = ax
            # save axespos
            if (col_ix + col2_ix) == 0:
                stim2ax['left'].append(ax)
            if col2_ix == 0:
                stim2ax['left_inner'].append(ax)
            if col2_ix == (len(inner_stim) - 1):
                stim2ax['right_inner'].append(ax)
            if (col2_ix == (len(inner_stim) - 1)) and (col_ix == (len(stimlist) - 1)):
                stim2ax['right'].append(ax)

            stim2ax['bottom'].append(ax)
            stim_num += 1
    return stim2ax

def axesgrid_list(fig, num_axes, num_col=None):
    ''' generates axes-grid list containing num_axes '''
    if not num_col:
        num_col = int(np.ceil(np.sqrt(num_axes)))
    num_row = int(np.ceil(1.* num_axes / num_col))
    gs = gridspec.GridSpec(num_row, num_col)
    axlist = []
    for col_ix, row_ix in iter.product(range(num_col), xrange(num_row)):
        # generate only as many axes as there are objects
        num_axes -= 1
        if num_axes < 0:
            break
        ax = fig.add_subplot(gs[row_ix, col_ix])
        axlist.append(ax)
    return axlist

#===============================================================================
# helper functions
#===============================================================================

def get_all_odors(stim_selection):
    """get all odors from a stim selection (stim concentration combinations)

        in the original order
    """
    all_odors = []
    for stim in stim_selection:
        odor, _, _ = stim.partition('_')
        if not odor in all_odors:
            all_odors.append(odor)
    return all_odors

#TODO: include as method in TimeSerie class
def write_csv_wt_labels(filename, ts):
    ''' write timeseries ts to csv file including row and col headers '''
    with open(filename, 'w') as f:
        headers = __builtin__.sum([[s] * ts.timepoints for s in ts.label_sample], [])
        f.write(', '.join([''] + headers) + '\n')
        for i, lab in enumerate(ts.label_objects):
            f.write(', '.join([lab] + list(ts.matrix_shaped()[:, i].astype('|S16'))) + '\n')

def flat_colormap(rgb_value):
    return lambda array: np.ones(list(array.shape) + [3]) * rgb_value

def fit_models(data_dict, config):
    """fit different models to our data"""
    main_reg = config['main_regions']
    regressor = linear_model.LinearRegression(fit_intercept=False)
    x, y = [], []
    for odor in data_dict:
        for concen in data_dict[odor]:
            if 'valenz' in data_dict[odor][concen]:
                t = data_dict[odor][concen]
                x.append([t['medians'][main_reg[2]], t['medians'][main_reg[0]]])
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
                    agg[main_reg[i]].append(t['medians'][main_reg[i]])
                agg['ratio'].append(data_dict[odor][concen]['medians'][main_reg[2]] /
                                    data_dict[odor][concen]['medians'][main_reg[0]])
                agg['diff'].append(data_dict[odor][concen]['medians'][main_reg[2]] -
                                   alpha * data_dict[odor][concen]['medians'][main_reg[0]])
    idx = np.argmax(agg['ratio'])
    agg['ratio'].pop(idx)
    return agg
