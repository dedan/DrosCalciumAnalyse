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
import pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as ic
from collections import defaultdict
import utils
reload(ic)
reload(utils)

mycolormap = defaultdict(lambda: plt.cm.hsv_r)
mycolormap.update({'key1': plt.cm.hsv_r, 'key2': plt.cm.hsv_r, 'key3': plt.cm.hsv_r,
              'key4': plt.cm.hsv_r, 'key5': plt.cm.hsv_r, 'key6': plt.cm.hsv_r,
              'iPN': plt.cm.hsv_r, 'vlPrc': plt.cm.hsv_r , 'acid':plt.cm.hsv_r,
              'iPNph': plt.cm.hsv_r })

inpath = '/Users/dedan/projects/fu/results/cross_val/nbest-5_thresh-80/'
# inpath = '/home/jan/Documents/dros/new_data/fromStephan/nbest-5_thresh-80/'
prefixes = ['OCO', '2PA', 'LIN', 'CVA']
prefixes = ['LIN']
format = 'png'
stimulus_offset = 4

for prefix in prefixes:

    # load files and compute distances
    colorlist, mode_dict = {}, {}
    files = glob.glob(os.path.join(inpath, prefix + '*.pckl'))
    for i, fname in enumerate(files):
        print fname
        res = pickle.load(open(fname))
        mo = res['base']
        #check if all files have same stimuli
        if i == 0:
            stimuli = mo.label_sample
            print mo.label_sample
        else:
            print mo.label_sample
            assert mo.label_sample == stimuli
        # create stimulus name
        if 'all' in os.path.splitext(os.path.basename(fname))[0]:
            mo.name = 'all'
        else:
            mo.name = os.path.splitext(os.path.basename(fname))[0].split("-")[-1]
        mode_dict[mo.name] = mo
    modedist_dic = utils.cordist(mode_dict.values())
    modedist, labels = utils.dict2pdist(modedist_dic)

    # compute and plot the dendrogram
    fig = plt.figure()
    axes = plt.Axes(fig, [.25, .1, .7, .8])
    fig.add_axes(axes)
    d = dendrogram(linkage(np.array(modedist).squeeze() + 1E-10, 'average'),
                   labels=labels,
                   leaf_font_size=10,
                   orientation='left')
    plt.savefig(os.path.join(inpath, prefix + '_dendro.' + format))

    # plot the bases and time-series of json file already created (by hand!!!)
    if os.path.exists(os.path.join(inpath, prefix + '_time_plot.json')):
        info = json.load(open(os.path.join(inpath, prefix + '_time_plot.json')))

        # timeseries plot
        vis = ic.VisualizeTimeseries()
        vis.fig = plt.figure()
        for i, key in enumerate(info):
            ax = vis.fig.add_subplot(len(info), 1, i + 1)
            vis.axes['time'].append(ax)
            ax.set_ylabel(key)
            for mode in info[key]:
                if 'all' in mode:
                    all_mode = mode
                    continue
                mo = mode_dict["_".join(mode.split("_")[0:-1])]
                vis.plot(ax, mo.timecourses[:, int(mode[-1])], linewidth=1.0, color='0.7')
            mo = mode_dict['all']
            vis.plot(ax, mo.timecourses[:, int(all_mode[-1])], linewidth=2.0, color='0.0')
            vis.add_onsets(ax, mo, stimulus_offset)
            vis.add_labelshade(ax, mo)
            for l in ax.get_xticklabels():
                l.set_visible(False)
            ylim = ax.get_ylim()
            ax.set_yticks([ylim[0] + 0.05, 0, ylim[1] - 0.05])
            #ax.set_yticklabels([ylim[0] + 0.05, 0, ylim[1] - 0.05])
            ax.grid(True)
            if i == 0:
                vis.add_samplelabel(ax, mo,
                                    rotation='45',
                                    toppos=True,
                                    stimuli_offset=stimulus_offset)
        plt.savefig(os.path.join(inpath, prefix + '_time_series.' + format))

        # base plot compare
        names = [os.path.splitext(os.path.basename(name))[0] for name in res['names']]
        plot_idx = {}
        for i_name, name in enumerate(names):
            plot_idx[name] = i_name

        num_bases = mode_dict.values()[0].num_objects
        for i, key in enumerate(info):

            fig = plt.figure()
            for modenum, mode in enumerate(info[key]):
                mode_name = "_".join(mode.split("_")[0:-1])
                base_names = [name for name in names if not name == mode_name]
                cluster_size = len(info[key])
                mo = mode_dict[mode_name]
                single_bases = mo.base.objects_sample(int(mode[-1]))
                for base_num, base_name in enumerate(base_names):
                    ax = fig.add_subplot(num_bases,
                                         cluster_size,
                                         (plot_idx[base_name] * cluster_size) + modenum + 1)
                    ax.set_yticks([])
                    ax.set_xticks([])
                    data = single_bases[base_num] * -1
                    data_max = np.max(np.abs(data))
                    ax.imshow(data, cmap=plt.cm.hsv, vmin= -data_max, vmax=data_max)
                    if modenum == 0:
                        ax.set_ylabel(base_name, rotation='horizontal')
                    if base_num == 0:
                        ax = fig.add_subplot(num_bases,
                                             cluster_size,
                                             (base_num * cluster_size) + modenum + 1)
                        ax.set_title(mo.name, fontsize=8)
                        ax.set_yticks([])
                        ax.set_xticks([])
            fig.savefig(os.path.join(inpath, prefix + '_' + key + '_simultan.' + format))

        # base plot alltogether
        fig = plt.figure()
        for i, key in enumerate(info):
            for modenum, mode in enumerate(info[key]):
                if 'all' in mode:
                    mo = mode_dict["_".join(mode.split("_")[0:-1])]
                    single_bases = mo.base.objects_sample(int(mode[-1]))
                    for base_num in range(len(single_bases)):
                        ax = fig.add_subplot(len(info)+1, len(single_bases),
                                             base_num + 1)
                        ax.imshow(res['baselines'][base_num],
                                  cmap=plt.cm.gray)
                        ax.set_yticks([])
                        ax.set_xticks([])
                        ax = fig.add_subplot(len(info)+1, len(single_bases),
                                             (i + 1) * len(single_bases) + base_num + 1)
                        if base_num == 0:
                            ax.set_ylabel(key)
                        data = (single_bases[base_num])
                        data_max = np.max(np.abs(data))
                        ax.imshow(data, cmap=mycolormap[key], vmin= -data_max, vmax=data_max)
                        ax.set_yticks([])
                        ax.set_xticks([])
                        if i == len(info)-1:
                            name = os.path.splitext(os.path.basename(res['names'][base_num]))[0]
                            ax.set_xlabel(name, fontsize=10)
                        ax.set_title('%.2f' % data_max, fontsize=8)
        fig.savefig(os.path.join(inpath, prefix + '_bases.' + format))


plt.close('all')


