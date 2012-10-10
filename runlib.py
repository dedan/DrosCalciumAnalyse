#!/usr/bin/env python
# encoding: utf-8
"""
this file contains parts of the old run_dros, more modular so that it can be
re-used also for the gui

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import os
from collections import defaultdict
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as vis
from NeuralImageProcessing.pipeline import TimeSeries
import numpy as np
import pylab as plt

# no parameters, only loaded once at import

#sorting
sorted_trials = bf.SortBySamplename()
# calculate (delta F) / F
rel_change = bf.RelativeChange()
# calculate trial mean
trial_mean = bf.TrialMean()

def create_timeseries_from_pngs(path, name):
    '''convert a folder of pngs (created by imageJ) into timeseries objects'''

    path = os.path.join(path, 'png')
    files = os.listdir(path)
    selected_files = [len(i.split('_')) > 2 for i in files]
    files2 = []
    for j, i in enumerate(selected_files):
        if i:
            files2.append(files[j])
    files = files2

    frames_per_trial = int(files[0].split('-')[1])
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
            im = plt.imread(os.path.join(path, file))
            temp.append(im.flatten())
            names.append(file)
        timeseries.append(np.array(temp))
        new_odor += sel_odor
        new_conc += sel_conc
    shape = im.shape
    timeseries = np.vstack(timeseries)
    label = [new_odor[i] + '_' + new_conc[i] for i in range(len(new_odor))]
    label = [i.strip('.png') for i in label[::frames_per_trial]]
    return TimeSeries(shape=tuple(shape), series=timeseries, name=name, label_sample=label)


def preprocess(ts, config):
    # TODO: does not work with mic yet
    out = {}

    # cut baseline signal (odor starts at frame 8)
    # TODO: set this as parameter
    baseline = trial_mean(bf.CutOut((0, 6))(ts))

    # TODO: what is this sorted baseline for?
    # only used for plotting
    sorted_baseline = sorted_trials(bf.CutOut((0, 1))(ts))
    #downscale sorted baseline
    ds = config['spatial_down']
    ds_baseline = sorted_baseline.shaped2D()[:, ::ds, ::ds]
    sorted_baseline.shape = tuple(ds_baseline.shape[1:])
    sorted_baseline.set_timecourses(ds_baseline)
    out['sorted_baseline'] = sorted_baseline

    # temporal downsampling by factor 2 (originally 40 frames)
    t_ds = 2
    ts = bf.TrialMean(ts.timepoints / t_ds)(ts)
    ts.framerate /= t_ds
    ts.stim_window = (np.floor(1.*ts.stim_window[0] / t_ds),
                      np.ceil(1.*ts.stim_window[1] / t_ds))

    # compute relative change (w.r.t. baseline)
    pp = rel_change(ts, baseline)

    # apply mask if set in config
    if 'maskfile' in config and config['maskfile']:
        print 'using mask from: %s' % config['maskfile']
        spatial_mask = np.load(config['maskfile']).astype('bool')
        pp.timecourses[:, np.logical_not(spatial_mask.flatten())] = 0

    # spatial filtering
    gauss_filter = bf.Filter('gauss', config['lowpass'], downscale=config['spatial_down'])
    if config['medianfilter'] > 0:
        pixel_filter = bf.Filter('median', config['medianfilter'])
        pp = gauss_filter(pixel_filter(pp))
    else:
        pp = gauss_filter(pp)
    pp.timecourses[np.isnan(pp.timecourses)] = 0
    pp.timecourses[np.isinf(pp.timecourses)] = 0

    if 'normalize' in config and config['normalize']:
        print 'normalizing'
        pp.timecourses = pp.timecourses / np.max(pp.timecourses)

    mean_resp_unsort = trial_mean(bf.CutOut((6, 12))(pp))
    pp = sorted_trials(pp)
    mean_resp = sorted_trials(mean_resp_unsort)
    out['mean_resp_unsort'] = mean_resp_unsort
    out['mean_resp'] = mean_resp
    out['pp'] = pp
    downsampling_filter = bf.Filter('gauss', 1, downscale=config['spatial_down'])
    out['baseline'] = downsampling_filter(baseline)
    return out

def mf_overview_plot(out, fig, params):
    '''plot overview of factorization result

        spatial bases on the left, temporal on the right
    '''
    mf = out['mf']
    mf_overview = vis.VisualizeTimeseries(fig)
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

def mf_overview_plot_single(out, fig, params):
    '''plot overview of factorization result

        spatial bases on the left, temporal in split axes on the right
    '''
    mf = out['mf']

    # get parameter
    overlay = params.get('overlay', True)
    workaround = params.get('workaround', False)
    if 'stimset' in params:
        stimset = params['stimset']
    else:
        stimset = list(set(mf.label_sample))
        stimset.sort()

    mf_overview = vis.VisualizeTimeseries(fig)
    mf_overview.base_and_singlestimtime(mf.num_objects, stimset)
    for mode_ix, resp in enumerate(mf.base.shaped2D()):
        if overlay:
            if workaround:
                mf_overview.overlay_workaround(mf_overview.axes['base'][mode_ix],
                               out['sorted_baseline'].shaped2D()[0],
                               {'cmap':plt.cm.bone, 'extent':[0, mf.base.shape[1], mf.base.shape[0], 0]},
                               resp, params.get('overlay', {}), {})
            else:
                ax = mf_overview.axes['base'][mode_ix]
                ax.imshow(out['sorted_baseline'].shaped2D()[0], cmap=plt.cm.bone,
                          extent=[0, mf.base.shape[1], mf.base.shape[0], 0])
                datamax = np.max(np.abs(resp))
                ax.imshow(resp, vmax=datamax, vmin= -datamax, alpha=0.4)
        else:
            mf_overview.imshow(mf_overview.axes['base'][mode_ix], resp, cmap=plt.cm.jet)
        # set timeplot parameter
        max_y = np.max(mf.timecourses) + 0.05
        min_y = np.min(mf.timecourses) - 0.05
        max_ylabel = np.floor(max_y * 10) / 10
        stimrep = defaultdict(lambda: 0)

        # define plotparam
        plotparam = {0:{'linewidth':2, 'color':'k'},
                     1:{'linewidth':2, 'color':'k', 'alpha':0.8},
                     2:{'linewidth':2, 'color':'k', 'alpha':0.6},
                     3:{'linewidth':2, 'color':'k', 'alpha':0.4}}

        # draw stimuli plots
        for trial_idx, stim_trial in enumerate(mf.trial_shaped()):
            trial_name = mf.label_sample[trial_idx]
            current_ax = mf_overview.axes['time'][mode_ix][trial_name]
            current_ax.hold(True)

            # generate axes layout
            if stimrep[trial_name] == 0:
                # create stimulus bar
                current_ax.fill_between(np.array(mf.stim_window), max_y, min_y, color='k', alpha=0.2)
                # set axes layout
                current_ax.spines['top'].set_color('none')
                current_ax.spines['right'].set_color('none')
                current_ax.xaxis.set_ticks_position('bottom')
                current_ax.yaxis.set_ticks_position('left')
                current_ax.set_ylim([min_y, max_y])
                current_ax.set_yticks([0, max_ylabel])
                current_ax.set_yticks(np.arange(min_y, max_y, 0.1), minor=True)
                if mf.label_sample[trial_idx] == stimset[0]:
                    current_ax.set_yticklabels([0, max_ylabel])
                else:
                    current_ax.set_yticklabels([])
                current_ax.set_xticks(range(0, mf.timepoints, 8))
                current_ax.set_xticks(range(0, mf.timepoints), minor=True)
                if mode_ix == mf.num_objects - 1:
                    current_ax.set_xticklabels(np.arange(0, mf.timepoints, 8) / mf.framerate, fontsize=10, rotation='45')
                else:
                    current_ax.set_xticklabels([])
                if mode_ix == 0:
                    current_ax.set_title(trial_name, fontsize=10)

            mf_overview.plot(current_ax, stim_trial[:, mode_ix], **plotparam[stimrep[trial_name]])
            stimrep[trial_name] += 1
    [ax.set_ylabel(mf.label_objects[i]) for i, ax in enumerate(mf_overview.axes['base'])]
    return mf_overview.fig

def raw_response_overview(out, fig, params):
    '''overview of responses to different odors'''
    resp_overview = vis.VisualizeTimeseries(fig)
    resp_overview.subplot(out['mean_resp'].samplepoints)
    for ind, resp in enumerate(out['mean_resp'].shaped2D()):
        max_data = np.max(np.abs(resp))
        resp_norm = resp / max_data
        threshold = params.get('threshold', 0.3)
        resp_overview.overlay_workaround(resp_overview.axes['base'][ind],
                                          out['sorted_baseline'].shaped2D()[ind],
                                          {'cmap':plt.cm.bone},
                                          resp_norm, {'threshold':threshold},
                                        {'title':{'label':out['mean_resp'].label_sample[ind], 'size':10}})
        if hasattr(out['mask'], 'timecourses') and not out['mask'].timecourses[ind]:
            resp_overview.imshow(resp_overview.axes['base'][ind],
                                 np.ones(resp_norm.shape),
                                 alpha=0.8)
        resp_overview.axes['base'][ind].set_ylabel('%.2f' % max_data)
    return resp_overview.fig

def raw_unsort_response_overview(out, fig, params):
    uresp_overview = vis.VisualizeTimeseries(fig)
    uresp_overview.subplot(out['mean_resp_unsort'].samplepoints)
    # TODO: what is this strength used for?
    out['mean_resp_unsort'].strength = []
    for ind, resp in enumerate(out['mean_resp_unsort'].shaped2D()):
        max_data = np.max(np.abs(resp))
        normedresp = resp / max_data
        uresp_overview.imshow(uresp_overview.axes['base'][ind],
                               normedresp,
                               title={'label':out['mean_resp_unsort'].label_sample[ind], 'size':10},
                               colorbar=False, vmin= -1, vmax=1)
        uresp_overview.axes['base'][ind].set_ylabel('%.2f' % max_data)
        out['mean_resp_unsort'].strength.append(max_data)
    return uresp_overview.fig

def reconstruction_error_plot(out, fig, params):
    '''plot the responses with the reconstructed response subtracted'''
    tmp_reconstruction = np.dot(out['mf'].timecourses, out['mf'].base.timecourses)
    reconstruction = out['mf'].copy()
    reconstruction.shape = out['mf'].base.shape
    reconstruction.set_timecourses(tmp_reconstruction)
    ts_difference = bf.Combine(np.subtract)
    recon_error = ts_difference(out['mean_resp'],
                                trial_mean(bf.CutOut((6, 12))(reconstruction)))

    recon_overview = vis.VisualizeTimeseries(fig)
    recon_overview.subplot(recon_error.samplepoints)
    for ind, resp in enumerate(recon_error.shaped2D()):
        max_data = np.max(np.abs(resp))
        normedresp = resp / max_data
        recon_overview.imshow(recon_overview.axes['base'][ind],
                               normedresp,
                               title={'label': recon_error.label_sample[ind], 'size':10},
                               colorbar=False, vmin= -1, vmax=1)
        recon_overview.axes['base'][ind].set_ylabel('%.2f' % max_data)
    return recon_overview.fig

def quality_overview_plot(out, fig, params):
    '''draw quality overview

       quality is reproducability of responses over odor presentations and especially
       interesting for stimuli which are not also similar to all others. The violet
       violin plot shows cross label similarity and the circles are the similarity
       between repeated odor presentations
    '''
    stimulirep = bf.SampleSimilarityPure()
    distanceself, distancecross = stimulirep(out['mean_resp'])
    qual_view = vis.VisualizeTimeseries(fig)
    qual_view.oneaxes()
    data = zip(*distancecross.items())
    vis.violin_plot(qual_view.axes['time'][0], [i.flatten() for i in data[1]] ,
                     range(len(distancecross)), 'b')
    qual_view.axes['time'][0].set_xticks(range(len(distancecross)))
    qual_view.axes['time'][0].set_xticklabels(data[0], rotation=45)
    for pos, stim in enumerate(data[0]):
        qual_view.axes['time'][0].plot([pos] * len(distanceself[stim]),
                                        distanceself[stim], 'o', mew=2, mec='k', mfc='None')
    return qual_view.fig

def simultan_ica_baseplot(mo, mo2, all_raw, baselines):
    '''plot the spatial bases after simultaneous ica'''
    fig = plt.figure()
    fig.suptitle(np.sum(mo.eigen))
    num_bases = len(all_raw)
    names = [t.name for t in all_raw]
    for modenum in range(mo.num_objects):
        single_bases = mo2.base.objects_sample(modenum)
        for base_num in range(num_bases):
            ax = fig.add_subplot(mo.num_objects + 1, num_bases, base_num + 1)
            ax.imshow(np.mean(baselines[base_num].shaped2D(), 0), cmap=plt.cm.gray)
            ax.set_axis_off()
            ax.set_title(names[base_num])
            data_max = np.max(np.abs(single_bases[base_num]))
            ax = fig.add_subplot(mo.num_objects + 1, num_bases,
                                 ((num_bases * modenum) + num_bases) + base_num + 1)
            ax.imshow(single_bases[base_num] * -1, cmap=plt.cm.hsv, vmin= -data_max, vmax=data_max)
            ax.set_title('%.2f' % data_max, fontsize=8)
            ax.set_axis_off()
    return fig

def simultan_ica_timeplot(mo2, single_response):
    '''plot temporal bases after spatial ica'''
    fig = plt.figure()
    modefilter = bf.CalcStimulusDrive()
    stim_driven = modefilter(mo2).timecourses
    for modenum in range(mo2.num_objects):
        ax = fig.add_subplot(mo2.num_objects, 1, modenum + 1)
        ax.plot(single_response.timecourses[:, modenum])
        ax.set_xticklabels([], fontsize=12, rotation=45)
        ax.set_ylabel("%1.2f" % stim_driven[0, modenum])
        ax.grid(True)
    ax.set_xticks(np.arange(3, single_response.samplepoints, single_response.timepoints))
    ax.set_xticklabels(single_response.label_sample, fontsize=12, rotation=45)
    return fig
