"""
    visualize the data created by run_crossvalidation.py

    set the inpath variable to the folder which contains the results.

    When you first run this file it will create only the dendrogram plots.
    From this one can select the modes which should be compared in a time-series
    plot. To do this write the names of the modes into a json file named
    'PREFIX_time_plot.json'

    If this file exists the time series plot is also created.


    terminology in this file:

    * fold: ica computed on data from N-1 (or for comparison N) animals
    * time_series: the temporal part of the ICA result
    * base: the spatial part of the ICA result
    * base_part: the base in the simultanious ICA consists of patches from
                 several animals. This patch extracted is called a base_part
"""

import glob, pickle, os, json
import numpy as np
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as ic
import pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage

#create dictionary with mode distances
def cordist(modelist):
    alldist_dic = {}
    for i in range(len(modelist)):
        stimuli_i = modelist[i].label_sample
        if len(stimuli_i) < 8:
            print 'skipped: ', modelist[i].name
            continue
        for j in range(i + 1, len(modelist)):
            stimuli_j = modelist[j].label_sample
            if len(stimuli_j) < 8:
                print 'skipped: ', modelist[j].name
                continue
            common = set(stimuli_i).intersection(stimuli_j)
            mask_i = np.zeros(len(stimuli_i), dtype='bool')
            mask_j = np.zeros(len(stimuli_j), dtype='bool')
            for stim in common:
                mask_i[stimuli_i.index(stim)] = True
                mask_j[stimuli_j.index(stim)] = True
            ts1 = stimuli_filter(modelist[i], bf.TimeSeries(mask_i))
            ts2 = stimuli_filter(modelist[j], bf.TimeSeries(mask_j))
            cor = cor_dist(combine([ts1, ts2]))
            alldist_dic.update(cor.as_dict('objects'))
    return alldist_dic

# helper function to convert dictionary to pdist format
def dict2pdist(dic):
    key_parts = list(set(sum([i.split(':') for i in dic.keys()], [])))
    new_pdist = []
    for i in range(len(key_parts)):
        for j in range(i + 1, len(key_parts)):
            try:
                new_pdist.append(dic[':'.join([key_parts[i], key_parts[j]])])
            except KeyError:
                new_pdist.append(dic[':'.join([key_parts[j], key_parts[i]])])
    return new_pdist, key_parts


inpath = '/Users/dedan/projects/fu/results/cross_val/nbest-5_thresh-60/'
prefixes = ['OCO', '2PA', 'LIN', 'CVA']
# prefixes = ['OCO']
stimulus_offset = 4

cor_dist = bf.Distance()
stimuli_filter = bf.SelectTrials()
combine = bf.ObjectConcat()

for prefix in prefixes:

    # load files and compute distances
    colorlist, mode_dict = {}, {}
    files = glob.glob(os.path.join(inpath, prefix + '*.pckl'))
    for i, fname in enumerate(files):
        mo = pickle.load(open(fname))
        if 'all' in os.path.splitext(os.path.basename(fname))[0]:
            mo.name = 'all'
        else:
            mo.name = os.path.splitext(os.path.basename(fname))[0].split("-")[-1]
        mode_dict[mo.name] = mo
    modedist_dic = cordist(mode_dict.values())
    modedist, labels = dict2pdist(modedist_dic)

    # compute and plot the dendrogram
    fig = plt.figure()
    axes = plt.Axes(fig, [.25,.1,.7,.8])
    fig.add_axes(axes)
    d = dendrogram(linkage(np.array(modedist).squeeze() + 1E-10, 'average'),
                   labels=labels,
                   leaf_font_size=10,
                   orientation='left')
    plt.savefig(os.path.join(inpath, prefix + '_dendro.svg'))

    # plot the bases and time-series of json file already created (by hand!!!)
    if os.path.exists(os.path.join(inpath, prefix + '_time_plot.json')):
        info = json.load(open(os.path.join(inpath, prefix + '_time_plot.json')))

        # timeseries plot
        vis = ic.VisualizeTimeseries()
        vis.fig = plt.figure()
        for i, key in enumerate(info):
            ax = vis.fig.add_subplot(len(info), 1, i+1)
            vis.axes['time'].append(ax)
            ax.set_title(key)
            for mode in info[key]:
                if 'all' in mode:
                    all_mode = mode
                    continue
                mo = mode_dict["_".join(mode.split("_")[0:-1])]
                vis.plot(ax, mo.timecourses[:, int(mode[-1])], linewidth=1.0, color='0.7')
            mo = mode_dict['all']
            vis.plot(ax, mo.timecourses[:, int(all_mode[-1])], linewidth=3.0, color='0.0')
            vis.add_onsets(ax, mo, stimulus_offset)
            vis.add_labelshade(ax, mo)
            for l in ax.get_xticklabels():
                l.set_visible(False)
            for l in ax.get_yticklabels():
                l.set_visible(False)
            if i == 0:
                vis.add_samplelabel(ax, mo, rotation='45', toppos=True)
        plt.savefig(os.path.join(inpath, prefix + '_time_series.svg'))

        # base plot
        num_bases = mode_dict.values()[0].timecourses.shape[1]
        for i, key in enumerate(info):

            fig = plt.figure()
            for modenum, mode in enumerate(info[key]):
                cluster_size = len(info[key])
                mo = mode_dict["_".join(mode.split("_")[0:-1])]
                single_bases = mo.base.objects_sample(int(mode[-1]))
                for base_num in range(len(single_bases)):
                    ax = fig.add_subplot(num_bases,
                                         cluster_size,
                                         (base_num*cluster_size) +modenum+1)
                    if base_num == 0:
                        ax.set_title(mo.name, fontsize=8)
                    ax.set_axis_off()
                    data = single_bases[base_num] * -1
                    data_max = np.max(np.abs(data))
                    ax.imshow(data, cmap=plt.cm.hsv, vmin= -data_max, vmax=data_max)
            fig.savefig(os.path.join(inpath, prefix + '_' + key + '_simultan.svg'))
plt.close('all')



