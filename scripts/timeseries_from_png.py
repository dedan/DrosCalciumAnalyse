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

    files = os.listdir(path)
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


frames_per_trial = 40

measid = '110222a'
pathr = '/Users/dedan/projects/fu/data/dros_gui_test/110222a/'
timeseries, shape, label = create(pathr)
label = [i.strip('.png') for i in label[::frames_per_trial]]
ts = TimeSeries(shape=tuple(shape), series=timeseries, name=measid, label_sample=label)
ts.save('/Users/dedan/Desktop/test')



