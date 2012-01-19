'''
Created on Dec 12, 2011

@author: jan
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

pathraw = '/home/jan/Documents/dros/new_data/CVA_MSH_ACA_BEA_OCO_align/analysis/'
#pathraw = '/home/jan/Documents/dros/new_data/2PA_PAC_ACP_BUT_OCO/analysis/'
#pathraw = '/home/jan/Documents/dros/new_data/LIN_AAC_ABA_CO2_OCO_align/analysis/'

path_extra = ['110901a', '110902a', '111012a', '111013a', '111014a']
#path_extra = ['110817b_neu', '110817c', '110823a', '110823b', '111025a', '111025b']
#path_extra = ['111026a', '111027a', '111107a', '111107b', '111108a']

pathes = [pathraw + extra for extra in path_extra]
pathes = [i + '/earlynorm_sica5/' for i in pathes]
measIDs = [i.strip('/')for i in path_extra]#['A', 'B', 'C', 'D', 'E']
colorlist = ['r', 'g', 'b', 'c', 'm', 'k']

all_mean_resp = []
labels = []
for j, path in enumerate(pathes):
    names = [i.strip('.png') for i in pickle.load(open(path + 'ids.pik'))]
    timecourse = np.load(path + 'time_sica.npy')
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
    labels += [measIDs[j] + '-' + str(i) for i in np.arange(num_modes)[sel]]
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
