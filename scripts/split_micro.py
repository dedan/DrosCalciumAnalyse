"""
    script to split the microlession data

    This is done in a separate step to that the resulting data can be
    used with our usual analysis tools.

    What the script does is:

    * perform a vertical cut of the original image
    * perform a vertical cut of the corresponding mask
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
maskpath = '/Users/dedan/Dropbox/lesion_masks'
# inpath = '/home/jan/Documents/dros/new_data/aligned/'
# maskpath = '/home/jan/Dropbox/lesion_masks/'
prefix = 'mic'
outpath = os.path.join(inpath, 'mic_test')


# convert already splitted masks to numpy array
for fname in glob.glob(os.path.join(maskpath, '*.png')):
    if os.path.basename(fname).split('_')[1] in 'rl':
        print 'saved', fname
        spatial_mask = (plt.imread(fname)[:, :, 0]).astype('bool')
        spatial_mask = np.logical_not(spatial_mask).astype('int')
        np.save(fname[:-4], spatial_mask)

filelist = glob.glob(os.path.join(inpath, prefix + '*.json'))
for fname in filelist:

    print 'splitting', fname
    fbase = os.path.splitext(os.path.basename(fname))[0]

    # load time series and convert to trial shape
    ts = pl.TimeSeries()
    ts.load(os.path.splitext(fname)[0])
    ts.shape = tuple(ts.shape)
    trial_shaped_2d = ts.trial_shaped2D()

    # load spatial mask
    filename_mask = '_'.join(fbase.split('_')[1:]) + '_mask.png'
    try:
        spatial_mask = (plt.imread(os.path.join(maskpath, filename_mask))[:, :, 0]).astype('bool')
    except IOError:
        print 'no mask for: ', filename_mask
        continue

    # invert mask
    spatial_mask = np.logical_not(spatial_mask).astype('int')

    if fbase[-1] in 'rl':
        print 'already right-left splitted'
    else:

        # left data and mask
        left = trial_shaped_2d[:, :, :, :ts.shape[1] / 2]
        left_mask = spatial_mask[:, :ts.shape[1] / 2]
        # remove not information containig columns
        info_in_column = np.sum(left_mask, 0) > 0
        left = left[:, :, :, info_in_column]
        left_mask = left_mask[:, info_in_column]
        new_ts = pl.TimeSeries(series=left,
                               shape=((ts.shape[0], left.shape[3])),
                               label_sample=ts.label_sample)
        new_ts.save(os.path.join(outpath, fbase + '_l'))
        np.save(os.path.join(maskpath, filename_mask.split('_')[0] + '_l_mask'), left_mask)


        right = trial_shaped_2d[:, :, :, ts.shape[1] / 2:]
        right_mask = spatial_mask[:, ts.shape[1] / 2:]
        # remove not information containig columns
        info_in_column = np.sum(spatial_mask[:, ts.shape[1] / 2:], 0) > 0
        right = right[:, :, :, info_in_column]
        right_mask = right_mask[:, info_in_column]

        new_ts = pl.TimeSeries(series=right,
                               shape=((ts.shape[0], right.shape[3])),
                               label_sample=ts.label_sample)
        new_ts.save(os.path.join(outpath, fbase + '_r'))
        np.save(os.path.join(maskpath, filename_mask.split('_')[0] + '_r_mask'), right_mask)
