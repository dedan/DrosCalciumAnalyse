'''
Created on Dec 12, 2011

@author: jan
'''

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

cordist = {} # all similaritys
#measIDs = ['110901a', '110902a', '111012a', '111013a', '111014a', '111014b']
#measIDs = ['110817b_neu', '110817c', '110823a', '110823b', '111025a', '111025b']
#measIDs = ['111017a', '111017b', '111018a', '111018b', '111024a', '111024c', '120111a', '120111b']
measIDs = ['111026a', '111027a', '111107a', '111107b', '111108a', '111108b', '120112b']
prefix = 'LIN'
data_path = '/Users/dedan/projects/fu/data/dros_calcium/'

for ind, measID in enumerate(measIDs):
    
    #pathr = '/home/jan/Documents/dros/new_data/CVA_MSH_ACA_BEA_OCO_align/analysis/' + measID 
    #pathr = '/home/jan/Documents/dros/new_data/2PA_PAC_ACP_BUT_OCO/analysis/' #+ measID 
    #pathr = '/home/jan/Documents/dros/new_data/OCO_GEO_PAA_ISO_BUT_align/analysis/' + measID 
    #pathr = '/home/jan/Documents/dros/new_data/LIN_AAC_ABA_CO2_OCO_align/analysis/' + measID 

    timecourse = np.load(os.path.join(data_path, 'out', prefix + '_' + measID + '_data.npy'))
    info = json.load(open(data_path + prefix + '_' + measID + '.json')) 

    names = [i.strip('.png') for i in info['labels'][::40]]
    
    cordist_loc = {} #self-similarity for one individual/ key:stimuli
    cordist_loc2 = {} #cross-similarity for one individual / key: stimuli-pair
    strength = {} #response strength for one individual / key: stimuli
    
    
    timecourse = timecourse.reshape((-1, 20, timecourse.shape[1]))
    odorresponse = np.mean(timecourse[:, 5:12, :], 1)


    # calculate correlation distance
    distances = pdist(odorresponse, 'correlation')
    distances = squareform(distances)
    
    
    
    # erzeuge alle  dicionarys mit key: odor-pair, value: pair distance
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sortnames = [names[i], names[j]]
            sortnames.sort()
            try:
                cordist[sortnames[0] + ',' + sortnames[1]].append(distances[i, j])
            except KeyError:
                cordist[sortnames[0] + ',' + sortnames[1]] = [distances[i, j]]
            if sortnames[1] == sortnames[0]:
                try:
                    cordist_loc[sortnames[0]].append(distances[i, j])
                except KeyError:
                    cordist_loc[sortnames[0]] = [distances[i, j]]
                try:
                    strength[sortnames[0]].append((np.percentile(odorresponse[i], 95), np.percentile(odorresponse[j], 95)))
                except KeyError:
                    strength[sortnames[0]] = [(np.percentile(odorresponse[i], 95), np.percentile(odorresponse[j], 95))]
            else:
                for ind in range(2):
                    try:
                        cordist_loc2[sortnames[ind]].append(distances[i, j])
                    except KeyError:
                        cordist_loc2[sortnames[ind]] = [distances[i, j]]
    
    ''' ==== plot similarity in one individual === '''
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.15, 0.8, 0.8])
    ax.set_title(measID)
    for ind, key in enumerate(cordist_loc2):
        try:
            vals = cordist_loc[key]
            strengths = np.array(strength[key])
            strengths *= 3
            strengths[strengths > 1] = 1
            for stre, val in zip(strengths, vals):
                ax.plot(ind, val, 'o', mfc=plt.cm.jet(stre[0]), mec=plt.cm.jet(stre[1]), mew=2)
        except KeyError:
            pass
        ax.plot([ind] * len(cordist_loc2[key]), cordist_loc2[key], 'kx', alpha=0.5)
    ax.set_xlim((-0.5, len(cordist_loc2) - 0.5))
    ax.set_xticks(range(len(cordist_loc2)))
    ax.set_xticklabels([i.split(',')[0] for i in cordist_loc2.keys()], fontsize=12, rotation='vertical')
    
    figtemp = plt.figure()
    cax = plt.imshow(np.array([[0, 1], [1, 0]]))
    cbar = fig.colorbar(cax, ticks=[0, 0.3, 1])
    cbar.ax.set_yticklabels(['0', '0.1', '> 0.33'])

    fig.savefig(os.path.join(data_path, 'out', prefix + '_' + measID + 'reproducibility.png'))    
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
        try:
            mean[ind_i, ind_j] = np.median(cordist[key])
            mean[ind_j, ind_i] = np.median(cordist[key])
            standard[ind_i, ind_j] = np.std(cordist[key])
            standard[ind_j, ind_i] = np.std(cordist[key])        
        except KeyError:
            print 'No entry for ', key


''' === for all animals calculate and draw mean distances of odors === '''

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

