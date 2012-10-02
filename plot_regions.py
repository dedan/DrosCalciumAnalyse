
import os, glob, json, math
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
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

comparisons = [(u'vlPRCb', u'vlPRCt'),
               (u'iPN', u'iPNsecond'),
               (u'iPNtract', u'betweenTract'),
               (u'betweenTract', u'vlPRCb'),
               (u'iPN', u'blackhole')]
main_regions = [u'iPN', u'iPNtract', u'vlPRCt']
to_turn = ['120112b_neu', '111012a', '111017a_neu', '111018a', '110902a']

luts_path = os.path.join(os.path.dirname(__file__), 'colormap_luts')
filelist = glob.glob(os.path.join(luts_path, '*.lut'))
first_map = utils.colormap_from_lut(filelist.pop())
first_map = plt.cm.hsv_r
colormaps = defaultdict(lambda: first_map)
assert len(main_regions) == len(filelist)
for i, fname in enumerate(filelist):
    #colormaps[main_regions[i]] = utils.colormap_from_lut(fname)
    colormaps[main_regions[i]] = plt.cm.hsv_r

format = 'png'
integrate = True
results_path = '/Users/dedan/projects/fu/results/'
load_path = os.path.join(results_path, 'simil80n_bestFalse', 'nnma')
save_path = os.path.join(load_path, 'plots')
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(os.path.join(save_path, 'odors')):
    os.mkdir(os.path.join(save_path, 'odors'))

data = {}
fulldatadic = {}

# load the labels created by GUI
labeled_animals = json.load(open(os.path.join(load_path, 'regions.json')))

# load valenz information (which odor they like)
valenz = json.load(open(os.path.join(results_path, 'valenz.json')))
valenz_orig = json.load(open(os.path.join(results_path, 'valenz.json')))

# load and filter filelist
filelist = glob.glob(os.path.join(load_path, '*.json'))
filelist = [f for f in filelist if not 'base' in os.path.basename(f)]
filelist = [f for f in filelist if not 'regions' in os.path.basename(f)]

# initialize processing (pipeline) components
average_over_stimulus_repetitions = bf.SingleSampleResponse()
integrator = bf.StimulusIntegrator(threshold= -1000)

# read data (matrix factorization results) to dictionary
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

    t_modes, t_modes_names = [], []
    s_modes = []

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
                    pdat[i * trial_length:i * trial_length + trial_length] = trial_shaped[index, :, mode]

            # add to results list
            t_modes.append(pdat)
            t_modes_names.append("%s_%d" % (animal, mode))
            s_modes.append((ts.name, ts.base.trial_shaped2D()[mode, :, :, :].squeeze()))
    t_modes = np.array(t_modes)


    add = '_integrated' if integrate else ''

    # temporal boxplots
    fig = plt.figure()
    fig.suptitle(region_label)
    ax = fig.add_subplot(111)
    # mask it for nans (! True in the mask means exclusion)
    t_modes_ma = np.ma.array(t_modes, mask=np.isnan(t_modes))
    fulldatadic[region_label] = t_modes_ma
    medians[region_label] = np.ma.extras.median(t_modes_ma, axis=0)
    if integrate:
        # make it a list because boxplot has a problem with masked arrays
        t_modes_ma = [[y for y in row if y] for row in t_modes_ma.T]
        ax.boxplot(t_modes_ma)
    else:
        l = len(medians[region_label])
        p25 = scoreatpercentile(t_modes_ma, 25)
        p75 = scoreatpercentile(t_modes_ma, 75)
        ax.fill_between(range(l), p25, p75, linewidth=0, color='0.75')
        ax.plot(medians[region_label], linewidth=0.5, color='0')
        ax.set_xticks(range(0, l, ts.timepoints))
    ax.set_xticklabels(list(all_stimuli), rotation='90')
    plt.savefig(os.path.join(save_path, region_label + add + '.' + format))

    # spatial base plots
    fig = vis.VisualizeTimeseries()
    fig.subplot(len(s_modes))
    for i, (name, s_mode) in enumerate(s_modes):
        n = '_'.join(name.split('_')[:-1])
        filelist = glob.glob(os.path.join(load_path, '*' + n + '_baseline.json'))
        print name, filelist
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
    fig.fig.savefig(os.path.join(save_path, region_label + add + '_spatial.' + format))

    # write the data to csv files
    assert(len(t_modes_names) == t_modes.shape[0])
    with open(os.path.join(save_path, region_label + add + '.csv'), 'w') as f:
        f.write(', ' + ', '.join(all_stimuli) + '\n')
        for i, mode_name in enumerate(t_modes_names):
            print 'write'
            f.write(mode_name + ', ' + ', '.join(t_modes[i,:].astype('|S16')) + '\n')

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
        ax.bar(range(l), medians[comparison[1]] * -1, color='b')
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


    # filter out the strange CO2 labels
    co2strange = [u'CO2_1', u'CO2_5']
    co2rename = {u'CO2_10': u'CO2_-1', u'CO2_-1': u'CO2_-3'}
    idx = np.array([True if not all_stimuli[i] in co2strange else False for i in range(len(all_stimuli))])
    hm_data = hm_data[:, idx]
    all_stimuli = [s for i, s in enumerate(all_stimuli) if idx[i]]
    for i in range(len(all_stimuli)):
        if all_stimuli[i] in co2rename:
            all_stimuli[i] = co2rename[all_stimuli[i]]
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
