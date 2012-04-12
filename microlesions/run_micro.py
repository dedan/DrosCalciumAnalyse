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
normalize = False
medianfilter = 5
alpha = 0.1
format = 'svg'
prefix = 'mic'
res = {}

# #dedan specific
# base_path = '/Users/dedan/projects/fu'
# data_path = os.path.join(base_path, 'data', 'dros_calcium_new', 'mic_split')
# savefolder = 'micro'
# save_path = os.path.join(base_path, 'results', savefolder)

#jan specific
base_path = '/home/jan/Documents/dros/new_data/aligned'
data_path = os.path.join(base_path, 'mic_split')
savefolder = 'micro'
save_path = os.path.join(base_path, 'results', savefolder)


filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
filenames = [os.path.splitext(os.path.basename(filename))[0] for filename in filelist]
sessions = set(['_'.join(filename.split('_')[0:2]) for filename in filenames])

for session in sessions:

    print 'session: ', session
    session_list = glob.glob(os.path.join(data_path, session) + '*.json')
    all_raw = []
    mean_bases = []

    for filename in session_list:

        f_basename = os.path.splitext(os.path.basename(filename))[0]
        print 'loading file: ', f_basename
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
#        plt.figure()
#        plt.title(filename)
        mean_base = np.mean(baseline.shaped2D(), 0)
        mean_bases.append(mean_base)
#        plt.imshow(mean_base, cmap=plt.cm.gray)
#        plt.savefig(os.path.join(save_path, 'baseline_' + f_basename + '.' + format))

        print 'preprocessing'
        rel_change = bf.RelativeChange()
        relative_change = rel_change(ts, baseline)
        relative_change.timecourses[np.isnan(relative_change.timecourses)] = 0
        relative_change.timecourses[np.isinf(relative_change.timecourses)] = 0      
        pixel_filter = bf.Filter('median', medianfilter)
        gauss_filter = bf.Filter('gauss', lowpass, downscale=3)
        pp = gauss_filter(pixel_filter(relative_change))
        if normalize:
            pp.timecourses = pp.timecourses / np.max(pp.timecourses)
        signal_cut = bf.CutOut((6, 12))
        sorted_trials = bf.SortBySamplename()
        mean_resp_unsort = trial_mean(signal_cut(pp))
        mean_resp = sorted_trials(mean_resp_unsort)
        pp = sorted_trials(pp)
        # remove the lesion marker from the label for the combine_common function
        if pp.label_sample[0][0] == 'm':
            pp.label_sample = [l[1:] for l in pp.label_sample]
        all_raw.append(pp)

    print 'combine all pseudo-animals'
    combine_common = bf.ObjectConcat(unequalsample=1, unequalobj=True)
    intersection = sorted_trials(combine_common(all_raw))

    print 'ica'
    pca = bf.PCA(variance)
    icaend = bf.stICA(variance, {'alpha':alpha})
    mo = pca(intersection)
    mo2 = icaend(intersection)
    filenames = [os.path.splitext(os.path.basename(filename))[0] for filename in session_list]
    res[session] = {'mo2': mo2,
                    'mo': mo,
                    'mean_bases': mean_bases,
                    'filenames': filenames}
pickle.dump(res, open(os.path.join(save_path, 'res.pckl'), 'w'))
