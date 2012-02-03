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
