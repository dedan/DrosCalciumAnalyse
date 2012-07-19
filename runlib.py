#!/usr/bin/env python
# encoding: utf-8
"""
this file contains parts of the old run_dros, more modular so that it can be
re-used also for the gui

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as vis
import numpy as np
import pylab as plt

# no parameters, only loaded once at import

#sorting
sorted_trials = bf.SortBySamplename()
# calculate (delta F) / F
rel_change = bf.RelativeChange()
# calculate trial mean
trial_mean = bf.TrialMean()


def preprocess(ts, config):
    # TODO: does not work with mic yet

    out = {}

    # cut baseline signal (odor starts at frame 8)
    # TODO: set this as parameter
    out['baseline'] = trial_mean(bf.CutOut((0, 6))(ts))

    # TODO: what is this sorted baseline for?
    sorted_baseline = sorted_trials(bf.CutOut((0, 1))(ts))
    #downscale sorted baseline
    ds = config['spatial_down']
    ds_baseline = sorted_baseline.shaped2D()[:, ::ds, ::ds]
    sorted_baseline.shape = tuple(ds_baseline.shape[1:])
    sorted_baseline.set_timecourses(ds_baseline)
    out['sorted_baseline'] = sorted_baseline

    # temporal downsampling by factor 2 (originally 40 frames)
    ts = bf.TrialMean(20)(ts)

    # compute relative change (w.r.t. baseline)
    pp = rel_change(ts, out['baseline'])

    # apply mask if set in config
    if 'maskfile' in config:
        spatial_mask = np.load(config['maskfile']).astype('bool')
        pp.timecourses[:, np.logical_not(spatial_mask.flatten())] = 0

    # spatial filtering
    pixel_filter = bf.Filter('median', config['medianfilter'])
    gauss_filter = bf.Filter('gauss', config['lowpass'], downscale=config['spatial_down'])
    pp = gauss_filter(pixel_filter(pp))
    pp.timecourses[np.isnan(pp.timecourses)] = 0
    pp.timecourses[np.isinf(pp.timecourses)] = 0

    if 'normalize' in config:
        pp.timecourses = pp.timecourses / np.max(pp.timecourses)

    # select stimuli such that their mean correlation distance between the mean
    # responses of repeated stimuli presentations is below similarity_threshold
    # --> use only repeatable stimuli
    pp = sorted_trials(pp)
    stimuli_mask = bf.SampleSimilarity(config['similarity_threshold'])
    mean_resp = sorted_trials(trial_mean(bf.CutOut((6, 12))(pp)))
    stimuli_selection = stimuli_mask(mean_resp)
    stimuli_filter = bf.SelectTrials()
    pp = stimuli_filter(pp, stimuli_selection)

    out['pp'] = pp
    return out


def factorize(config):
    pass

def mf_overview_plot(mf):
    '''plot overview of factorization result

        spatial bases on the left, temporal on the right
    '''
    mf_overview = vis.VisualizeTimeseries()
    mf_overview.base_and_time(mf.num_objects)
    for ind, resp in enumerate(mf.base.shaped2D()):
        mf_overview.imshow(mf_overview.axes['base'][ind],
                            resp,
                            title={'label': mf.label_sample[ind]})
        mf_overview.plot(mf_overview.axes['time'][ind],
                          mf.timecourses[:, ind])
        mf_overview.add_labelshade(mf_overview.axes['time'][ind], mf)
    mf_overview.add_samplelabel(mf_overview.axes['time'][-1], mf, rotation='45', toppos=True)
    [ax.set_title(mf.label_objects[i]) for i, ax in enumerate(mf_overview.axes['base'])]
    return mf_overview.fig
