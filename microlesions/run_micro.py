'''
'''

import os, glob, sys, pickle
import numpy as np
import pylab as plt
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as vis
import utils
reload(bf)
reload(vis)

frames_per_trial = 40
variance = 5
lowpass = 2
normalize = True
medianfilter = 5
alpha = 0.1
format = 'svg'
prefix = 'mic'

base_path = '/Users/dedan/projects/fu'
data_path = os.path.join(base_path, 'data', 'dros_calcium_new', 'mic_split')
savefolder = 'micro'
save_path = os.path.join(base_path, 'results', savefolder)

filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
res = {}

for file_ind, filename in enumerate(filelist):

    f_basename = os.path.splitext(os.path.basename(filename))[0]
    print f_basename

    print 'loading..'
    meas_path = os.path.splitext(filename)[0]
    ts = bf.TimeSeries()
    ts.load(meas_path)
    ts.shape = tuple(ts.shape)

    print 'computing baseline'
    # temporal downsampling by factor 4 (originally 40 frames)
    temporal_downsampling = bf.TrialMean(20)
    ts = temporal_downsampling(ts)
    # cut baseline signal (odor starts at frame 4 (original frame8))
    baseline_cut = bf.CutOut((0, 3))
    trial_mean = bf.TrialMean()
    baseline = trial_mean(baseline_cut(ts))
    plt.figure()
    plt.title(filename)
    mean_base = np.mean(baseline.shaped2D(), 0)
    plt.imshow(mean_base, cmap=plt.cm.gray)
    plt.savefig(os.path.join(save_path, 'baseline_' + f_basename + '.' + format))

    print 'preprocessing'
    rel_change = bf.RelativeChange()
    pixel_filter = bf.Filter('median', medianfilter)
    gauss_filter = bf.Filter('gauss', lowpass, downscale=3)
    pp = gauss_filter(pixel_filter(rel_change(ts, baseline)))
    pp.timecourses[np.isnan(pp.timecourses)] = 0
    pp.timecourses[np.isinf(pp.timecourses)] = 0
    if normalize:
        pp.timecourses = pp.timecourses / np.max(pp.timecourses)
    signal_cut = bf.CutOut((6, 12))
    sorted_trials = bf.SortBySamplename()
    mean_resp_unsort = trial_mean(signal_cut(pp))
    mean_resp = sorted_trials(mean_resp_unsort)
    pp = sorted_trials(pp)

    print 'ica'
    pca = bf.PCA(variance)
    icaend = bf.stICA(variance, {'alpha':alpha})
    mo = pca(pp)
    mo2 = icaend(pp)
    res[f_basename] = {'mo2': mo2, 'mo': mo, 'mean_base': mean_base}
pickle.dump(res, open(os.path.join(save_path, 'res.pckl'), 'w'))
