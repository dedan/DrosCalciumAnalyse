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

#inpath = '/Users/dedan/projects/fu/data/dros_calcium_new/'
inpath = '/home/jan/Documents/dros/new_data/aligned/'
maskpath = '/home/jan/Dropbox/lesion_masks/'
prefix = 'mic'
outpath = os.path.join(inpath, 'mic_split')

#mikro_cutter = bf.MicroCutter()

filelist = glob.glob(os.path.join(inpath, prefix + '*.json'))

for fname in filelist:

    print 'splitting', fname
    fbase = os.path.splitext(os.path.basename(fname))[0]

    ts = pl.TimeSeries()
    ts.load(os.path.splitext(fname)[0])
    ts.shape = tuple(ts.shape)
    normal_mask = np.array([l[0] != 'm' for l in ts.label_sample])
    normal_labels = [ts.label_sample[i] for i in np.where(normal_mask)[0]]
    lesion_labels = [ts.label_sample[i] for i in np.where(np.invert(normal_mask))[0]]
    trial_shaped_2d = ts.trial_shaped2D()
    
    # load spatial mask, such that only LateralHorn Regions are in Analysis
    # if no mask, than skip datafile
    filename_mask = '_'.join(fbase.split('_')[1:]) + '_mask.png'
    try:
        spatial_mask = (plt.imread(os.path.join(maskpath, filename_mask))[:, :, 0]).astype('bool')
    except IOError:
        print 'no mask for: ', filename_mask
        continue
    # set everything in the mask to zero
    trial_shaped_2d[:, :, spatial_mask] = 0
     
    if fbase[-1] in 'rl':
        print 'already right-left splitted'
        normal = trial_shaped_2d[normal_mask, :, :, :]
        new_ts = pl.TimeSeries(series=normal,
                               shape=(ts.shape),
                               label_sample=normal_labels)
        new_ts.save(os.path.join(outpath, fbase[:-2] + '_' + fbase[-1] + 'n'))

        lesion = trial_shaped_2d[np.invert(normal_mask), :, :, :]
        new_ts = pl.TimeSeries(series=lesion,
                               shape=(ts.shape),
                               label_sample=lesion_labels)
        new_ts.save(os.path.join(outpath, fbase[:-2] + '_' + fbase[-1] + 'm'))

    else:
        left_normal = trial_shaped_2d[normal_mask, :, :, :ts.shape[1] / 2]
        # remove not information containig columsn
        info_in_column = np.sum(np.sum(np.sum(left_normal, 0), 0), 0) > 0
        left_normal = left_normal[:, :, :, info_in_column]
        new_ts = pl.TimeSeries(series=left_normal,
                               shape=((ts.shape[0], left_normal.shape[3])),
                               label_sample=normal_labels)
        new_ts.save(os.path.join(outpath, fbase + '_ln'))

        right_normal = trial_shaped_2d[normal_mask, :, :, ts.shape[1] / 2:]
        # remove not information containig columsn
        info_in_column = np.sum(np.sum(np.sum(right_normal, 0), 0), 0) > 0
        right_normal = right_normal[:, :, :, info_in_column]
        new_ts = pl.TimeSeries(series=right_normal,
                               shape=((ts.shape[0], right_normal.shape[3])),
                               label_sample=normal_labels)
        new_ts.save(os.path.join(outpath, fbase + '_rn'))


        left_lesion = trial_shaped_2d[np.invert(normal_mask), :, :, :ts.shape[1] / 2]
        # remove not information containig columsn
        info_in_column = np.sum(np.sum(np.sum(left_lesion, 0), 0), 0) > 0
        left_lesion = left_lesion[:, :, :, info_in_column]
        new_ts = pl.TimeSeries(series=left_lesion,
                               shape=((ts.shape[0], left_lesion.shape[3])),
                               label_sample=lesion_labels)
        new_ts.save(os.path.join(outpath, fbase + '_lm'))

        right_lesion = trial_shaped_2d[np.invert(normal_mask), :, :, ts.shape[1] / 2:]
        # remove not information containig columsn
        info_in_column = np.sum(np.sum(np.sum(right_lesion, 0), 0), 0) > 0
        right_lesion = right_lesion[:, :, :, info_in_column]
        new_ts = pl.TimeSeries(series=right_lesion,
                               shape=((ts.shape[0], right_lesion.shape[3])),
                               label_sample=lesion_labels)
        new_ts.save(os.path.join(outpath, fbase + '_rm'))
