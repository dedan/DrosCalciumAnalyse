'''
Created on Dec 12, 2011

@author: jan
'''

import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

all_mean_resp = []
labels = []

data_path = '/Users/dedan/projects/fu/data/dros_calcium/'
save_path = os.path.join(data_path, 'out')
prefix = 'LIN'
colorlist = ['r', 'g', 'b', 'c', 'm', 'k', 'r']
measIDs = []

filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
for ind, filename in enumerate(filelist):

    # load data
    meas_path = os.path.splitext(filename)[0]
    tmp_save = os.path.join(save_path, os.path.basename(meas_path))
    measIDs.append(os.path.basename(meas_path))
    info = json.load(open(filename))
    names = [i.strip('.png') for i in info['labels'][::40]]
    timecourse = np.load(tmp_save + '_time_sica.npy')

    num_modes = timecourse.shape[1]
    stimuli = list(set(names))
    stimuli.sort()
    '''
    try:
        stimuli.remove('CVA_-1')
    except ValueError:
        pass
    '''
    ''' +++++++++++++ remove bad modes (low correlation over pseudo-trials(repeated odorstimuli)) +++++++++++++++'''

    # TODO: if more than 2 stimuli, take avarage
    s1, s2 = [], []
    splitter = np.split(timecourse, len(names))
    for i in stimuli:
        temp = names.index(i)
        s1.append(splitter[temp])
        try:
            s2.append(splitter[names.index(i, temp + 1)])
        except ValueError:
            s1.pop()
    t1, t2 = np.vstack(s1), np.vstack(s2)
    sel = []
    for a, b in zip(t1.T, t2.T):
        simil = np.corrcoef(a, b)[0, 1]
        sel.append(simil > 0.3)
    print 'selected modes: ', np.sum(sel)
    sel = np.array(sel)
    labels += [os.path.basename(meas_path) + '-' + str(i) for i in np.arange(num_modes)[sel]]
    timecourse = timecourse.reshape((-1, 20, num_modes))[:, :, sel]


    ''' +++++++++++++ calculate mean response of all stimuli +++++++++++++++'''
    resp = []
    for stim in stimuli:
        pos = -1
        stim_resp = []
        while True:
            try:
                pos = names.index(stim, pos + 1)
                stim_resp.append(timecourse[pos])
            except ValueError:
                break
        temp = np.mean(np.array(stim_resp), 0)
        #temp = np.mean(temp[6:12], 0)
        #temp = temp[5:15]
        #temp[temp < 0.002] = 0
        resp.append(temp)
    resp = np.vstack(resp)
    all_mean_resp.append(resp)
all_mean_resp = np.hstack(all_mean_resp)


''' +++++++++++++ cluster modes +++++++++++++++'''
fig = plt.figure()
ax = fig.add_axes([0.05, 0.2, 0.9, 0.75])
d = dendrogram(linkage(pdist(all_mean_resp.T, 'cosine') + 1E-10, 'average'), labels=labels, leaf_font_size=12)
group_colors = []
for i in d['ivl']:
    group_colors.append(colorlist[measIDs.index(i.split('-')[0])])

labelsn = ax.get_xticklabels()
for j, i in enumerate(labelsn):
    i.set_color(group_colors[j])
