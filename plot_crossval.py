"""
    visualize the data created by run_crossvalidation.py

    set the inpath variable to the folder which contains the results.

    When you first run this file it will create only the dendrogram plots.
    From this one can select the modes which should be compared in a time-series
    plot. To do this write the names of the modes into a json file named
    'PREFIX_time_plot.json'

    If this file exists the time series plot is also created.
"""

import glob, pickle, os, json
import numpy as np
from NeuralImageProcessing import basic_functions as bf
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
            print 'overlap: ', len(common)
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
    key_parts = (reduce(lambda x, y: x + y, [i.split(':') for i in dic.keys()]))
    key_parts = list(set(key_parts))
    print key_parts
    new_pdist, new_labels = [], []
    for i in range(len(key_parts)):
        new_labels.append(key_parts[i])
        for j in range(i + 1, len(key_parts)):
            try:
                new_pdist.append(dic[':'.join([key_parts[i], key_parts[j]])])
            except KeyError:
                new_pdist.append(dic[':'.join([key_parts[j], key_parts[i]])])
    return new_pdist, new_labels

def onpick(event):
    print event.artist.name
    print dir(event)
    # event.artist.figure.axes[0].texts = []
    # plt.annotate(event.artist.name, (event.artist._x, event.artist._y))


inpath = '/Users/dedan/projects/fu/results/cross_val/nbest-5_thresh-80/'
prefixes = ['OCO', '2PA', 'LIN', 'CVA']

cor_dist = bf.Distance()
stimuli_filter = bf.SelectTrials()
combine = bf.ObjectConcat()

for prefix in prefixes:

    modelist = []
    colorlist = {}
    mode_dict = {}

    files = glob.glob(os.path.join(inpath, prefix + '*.pckl'))

    for i, fname in enumerate(files):

        mo = pickle.load(open(fname))
        mo.name = os.path.splitext(os.path.basename(fname))[0].split("-")[-1]
        mode_dict[mo.name] = mo
        modelist.append(mo)

    modedist_dic = cordist(modelist)
    modedist, lables = dict2pdist(modedist_dic)
    lables = ['_'.join(lab.split('_nnma_mode')) for lab in lables]

    fig = plt.figure()
    axes = plt.Axes(fig, [.25,.1,.7,.8])
    fig.add_axes(axes)
    # TODO: try again to make it clickable
    fig.canvas.mpl_connect('pick_event', onpick)

    d = dendrogram(linkage(np.array(modedist).squeeze() + 1E-10, 'average'),
                   labels=lables,
                   leaf_font_size=10,
                   orientation='left')
    plt.savefig(os.path.join(inpath, prefix + '_dendro.svg'))


    if os.path.exists(os.path.join(inpath, prefix + '_time_plot.json')):
        info = json.load(open(os.path.join(inpath, prefix + '_time_plot.json')))
        fig = plt.figure()
        for i, key in enumerate(info):
            plt.subplot(len(info), 1, i+1)
            plt.title(key)
            for mode in info[key]:
                mo = mode_dict["_".join(mode.split("_")[0:2])]
                plt.plot(mo.timecourses[:, int(mode[-1])])
            for xlabel_i in fig.axes[i].get_xticklabels():
                xlabel_i.set_visible(False)
        plt.savefig(os.path.join(inpath, prefix + '_time_series.svg'))

