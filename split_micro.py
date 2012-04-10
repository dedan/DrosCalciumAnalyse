"""
    script to split the microlession data

    This is done in a separate step to that the resulting data can be
    used with our usual analysis tools.

    What the script does is:

    * perform a vertical cut of the original image
    * seperate stimuli responses from before and after the lesion
    * save all the results in a separate folder
"""

import glob, pickle, os, json
import numpy as np
import pylab as plt
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import pipeline as pl
reload(bf)
reload(pl)

inpath = '/Users/dedan/projects/fu/data/dros_calcium_new/'
prefix = 'mic'
outpath = os.path.join(inpath, 'mic_split')

mikro_cutter = bf.MicroCutter()

filelist = glob.glob(os.path.join(inpath, prefix + '*.json'))

for fname in filelist:

    fbase = os.path.splitext(os.path.basename(fname))[0]

    ts = pl.TimeSeries()
    ts.load(os.path.splitext(fname)[0])
    ts.shape = tuple(ts.shape)
    normal_mask = np.array([l[0] != 'm' for l in ts.label_sample])
    normal_labels = [ts.label_sample[i] for i in np.where(normal_mask)[0]]
    lesion_labels = [ts.label_sample[i] for i in np.where(np.invert(normal_mask))[0]]
    trial_shaped_2d = ts.trial_shaped2D()

    left_normal = trial_shaped_2d[normal_mask,:,:,:ts.shape[1]/2]
    new_ts = pl.TimeSeries(series=left_normal,
                           shape=((ts.shape[0], ts.shape[1]/2)),
                           label_sample=normal_labels)
    new_ts.save(os.path.join(outpath, fbase + '_ln'))

    right_normal = trial_shaped_2d[normal_mask,:,:,ts.shape[1]/2:]
    new_ts = pl.TimeSeries(series=right_normal,
                           shape=((ts.shape[0], ts.shape[1]/2)),
                           label_sample=normal_labels)
    new_ts.save(os.path.join(outpath, fbase + '_rn'))


    left_lesion = trial_shaped_2d[np.invert(normal_mask),:,:,:ts.shape[1]/2]
    new_ts = pl.TimeSeries(series=left_lesion,
                           shape=((ts.shape[0], ts.shape[1]/2)),
                           label_sample=lesion_labels)
    new_ts.save(os.path.join(outpath, fbase + '_lm'))

    right_lesion = trial_shaped_2d[np.invert(normal_mask),:,:,:ts.shape[1]/2]
    new_ts = pl.TimeSeries(series=right_lesion,
                           shape=((ts.shape[0], ts.shape[1]/2)),
                           label_sample=lesion_labels)
    new_ts.save(os.path.join(outpath, fbase + '_rm'))
