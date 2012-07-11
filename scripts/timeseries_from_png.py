'''
Created on Aug 11, 2011

create time series objects from png images

@author: jan
'''
import sys
import os
import numpy as np
import pylab as plt
from NeuralImageProcessing.pipeline import TimeSeries

def create(path):

    dir2analyze = path + 'png/'

    files = os.listdir(dir2analyze)
    selected_files = [len(i.split('_')) > 2 for i in files]
    files2 = []
    for j, i in enumerate(selected_files):
        if i:
            files2.append(files[j])
    files = files2

    frame = np.array([int(i.split('-')[0][2:]) for i in files])
    point = np.array([int(i.split(' - ')[1][:2]) for i in files])
    odor = [i.split('_')[1] for i in files]
    conc = [i.split('_')[2].strip('.tif') for i in files]

    new_odor = []
    new_conc = []
    timeseries = []
    names = []

    for p in set(point):
        temp = []
        ind = np.where(point == p)[0]
        sel_frame = frame[ind]
        sel_files = [files[ind[i]] for i in np.argsort(sel_frame)]
        sel_odor = [odor[i] for i in ind]
        sel_conc = [conc[i] for i in ind]
        for file in sel_files:
            im = plt.imread(dir2analyze + file)
            temp.append(im.flatten())
            names.append(file)
        print '====', path, str(p) , '===='
        print len(temp)
        timeseries.append(np.array(temp))
        new_odor += sel_odor
        new_conc += sel_conc
    shape = im.shape
    timeseries = np.vstack(timeseries)
    label = [new_odor[i] + '_' + new_conc[i] for i in range(len(new_odor))]
    return timeseries, shape, label

def reorder(timeseries, labels):
    timeseries_new = []
    new_label = []
    for label in set(labels):
        print label
        ind = np.array([i == label for i in labels])
        timeseries_new.append(timeseries[ind, :])
        new_label += [label] * np.sum(ind)
    return np.vstack(timeseries_new), new_label


#measIDs = ['110901a', '110902a', '111012a', '111013a', '111014a', '111014b']
#measIDs = ['110817b_neu', '110817c', '110823a', '110823b', '111025a', '111025b']
#measIDs = ['111017a', '111017b', '111018a', '111018b', '111024a', '111024c', '120111a', '120111b']
#measIDs = ['111026a', '111027a', '111107a', '111107b', '111108a', '111108b', '120112b']
#measIDs = ['111117a', '111118a', '111124a']
#measIDs = ['120307a', '120307b', '120307c', '120308c']
measIDs = ['111108a', '111108b']

#path = '/home/jan/Documents/dros/new_data/CVA_MSH_ACA_BEA_OCO_align/'
#path = '/home/jan/Documents/dros/new_data/2PA_PAC_ACP_BUT_OCO_align/'
#path = '/home/jan/Documents/dros/new_data/OCO_GEO_PAA_ISO_BUT_align/'
#path = '/home/jan/Documents/dros/new_data/LIN_AAC_ABA_CO2_OCO_align/'
#path = '/home/jan/Documents/dros/new_data/microlesion_stabilized/'
#path = '/home/jan/Documents/dros/new_data/micro_align/'
path = '/home/jan/Documents/dros/new_data/LIN_AAC_ABA_CO2_OCO/'
#measIDs = os.listdir(path)
#measIDs.remove('analysis')
#measIDs.remove('111202a_neu')
frames_per_trial = 40

for measid in measIDs:

    dataset = path.strip('/').split('/')[-1][:3]
    pathr = path + measid + '/'
    path_save = '/home/jan/Documents/dros/new_data/raw/' + dataset + '_' + measid #+ measid + '/unnormed_nnma' + str(variance) + '/'#'percent' + str(variance).split('.')[1] + '/'

    timeseries, shape, label = create(pathr)
    label = [i.strip('.png') for i in label[::frames_per_trial]]
    ts = TimeSeries(shape=tuple(shape), series=timeseries, name=measid, label_sample=label)
    ts.save(path_save)



