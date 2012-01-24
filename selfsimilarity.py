'''
Created on Dec 12, 2011

@author: jan
'''

from collections import defaultdict
import os
import pickle
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

cordist = defaultdict(list) # all similaritys
data_path = '/Users/dedan/projects/fu/data/dros_calcium/'
save_path = os.path.join(data_path, 'out')
prefix = 'LIN'

filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
for ind, filename in enumerate(filelist):

    # load data    
    meas_path = os.path.splitext(filename)[0]
    tmp_save = os.path.join(save_path, os.path.basename(meas_path))
    timecourse = np.load(tmp_save + '_data.npy')
    info = json.load(open(filename)) 
    names = [i.strip('.png') for i in info['labels'][::40]]
    
    cordist_loc = defaultdict(list)     # self-sim for one individual
    cordist_loc2 = defaultdict(list)    # cross-sim for one individual
    strength = defaultdict(list)        # response strength for one individual
    
    # calculate correlation distance
    timecourse = timecourse.reshape((-1, 20, timecourse.shape[1]))
    odorresponse = np.mean(timecourse[:, 5:12, :], 1)    
    distances = pdist(odorresponse, 'correlation')
    distances = squareform(distances)    
    
    # create all dictionaries with key: odor-pair, value: pair distance
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = ','.join(sorted([names[i], names[j]]))
            cordist[key].append(distances[i, j])
            if names[i] == names[j]:
                cordist_loc[names[i]].append(distances[i, j])
                strength[names[i]].append((np.percentile(odorresponse[i], 95),
                                           np.percentile(odorresponse[j], 95)))
            else:
                for ind in [i, j]:
                    cordist_loc2[names[ind]].append(distances[i, j])

    # plot similarity for one individual
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.15, 0.8, 0.8])
    ax.set_title(os.path.basename(meas_path))
    for ind, key in enumerate(cordist_loc2):
        if key in cordist_loc:
            vals = cordist_loc[key]
            strengths = np.array(strength[key])
            strengths *= 3
            strengths[strengths > 1] = 1
            for stre, val in zip(strengths, vals):
                ax.plot(ind, val, 'o', mfc=plt.cm.jet(stre[0]), 
                        mec=plt.cm.jet(stre[1]), mew=2)
        ax.plot([ind] * len(cordist_loc2[key]), 
                cordist_loc2[key], 'kx', alpha=0.5)
    ax.set_xlim((-0.5, len(cordist_loc2) - 0.5))
    ax.set_xticks(range(len(cordist_loc2)))
    ax.set_xticklabels([i.split(',')[0] for i in cordist_loc2.keys()], 
                       fontsize=12, rotation='vertical')
    
    figtemp = plt.figure()
    cax = plt.imshow(np.array([[0, 1], [1, 0]]))
    cbar = fig.colorbar(cax, ticks=[0, 0.3, 1])
    cbar.ax.set_yticklabels(['0', '0.1', '> 0.33'])
    fig.savefig(tmp_save + 'reproducibility.png')
    plt.close('all')


stimuli = list(set([i.split(',')[0] for i in cordist.keys()]))
stimuli.sort()
mean = np.zeros((len(stimuli), len(stimuli)))
standard = np.zeros((len(stimuli), len(stimuli)))
for ind_i, i in enumerate(stimuli):
    for ind_j, j in enumerate(stimuli):
        sortnames = [i, j]
        sortnames.sort()
        key = sortnames[0] + ',' + sortnames[1]
        if key in cordist:
            mean[ind_i, ind_j] = np.median(cordist[key])
            mean[ind_j, ind_i] = np.median(cordist[key])
            standard[ind_i, ind_j] = np.std(cordist[key])
            standard[ind_j, ind_i] = np.std(cordist[key])        
        else:
            print 'No entry for ', key


# for all animals calculate and draw mean distances of odors
mean2 = mean - np.diag(np.diag(mean))
dist = squareform(mean2)
fig = plt.figure()
d = dendrogram(linkage(dist, 'average'), color_threshold=0.2, labels=stimuli)
sortind = np.array(d['leaves'])
mean = mean[sortind][:, sortind]
standard = standard[sortind][:, sortind]
stimuli = [stimuli[i] for i in sortind]

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(mean, interpolation='nearest', vmin=0)
ax.set_xticks(range(len(stimuli)))
ax.set_xticklabels(stimuli, rotation='vertical')
ax.set_yticks(range(len(stimuli)))
ax.set_yticklabels(stimuli)
ax.set_title('median correlation dist')
fig.colorbar(im)

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(standard, interpolation='nearest', vmin=0)
ax.set_xticks(range(len(stimuli)))
ax.set_xticklabels(stimuli, rotation='vertical')
ax.set_yticks(range(len(stimuli)))
ax.set_yticklabels(stimuli)
ax.set_title('std correlation dist')
fig.colorbar(im)

