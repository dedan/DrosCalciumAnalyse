'''
Created on Aug 11, 2011

@author: jan
'''

import runlib
import os, glob, sys, json, pickle
from configobj import ConfigObj
import itertools as it
from collections import defaultdict
import numpy as np
import pylab as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as vis
import utils
import matplotlib as mpl
reload(bf)
reload(vis)
reload(runlib)

config = ConfigObj(sys.argv[1], unrepr=True)

add = config['filename_add']
if config['normalize']:
    add += '_maxnorm'
save_base_path = os.path.join(config['save_path'],
                          'simil' + str(int(config['similarity_threshold'] * 100))
                          + 'n_best' + str(config['n_best']) + add + '_new')
if not os.path.exists(save_base_path):
    os.mkdir(save_base_path)
plots_folder = os.path.join(save_base_path, config['individualMF']['method'])
data_folder = os.path.join(plots_folder, 'data')
odor_plots_folder = os.path.join(plots_folder, 'odors')
if not os.path.exists(plots_folder):
    os.mkdir(plots_folder)
    os.mkdir(data_folder)
    os.mkdir(odor_plots_folder)
else:
    answer = raw_input('output folder already exists, overwrite results? (y/n): ')
    if not answer == 'y':
        print 'abort run, output folder contains files'
        sys.exit()
print 'results are written to : %s' % save_base_path

total_resp = []
for prefix in config['prefixes']:

    filelist = glob.glob(os.path.join(config['data_path'], prefix) + '*.json')
    colorlist = {}

    # use only the n_best animals --> most stable odors in common
    if config['n_best']:
        res = pickle.load(open(os.path.join(config['data_path'], config['load_path'], 'thres_res.pckl')))
        best = utils.select_n_channels(res[prefix][config['choose_threshold']], config['n_best'])
        filelist = [filelist[i] for i in best]

    #create lists to collect results
    all_sel_modes, all_sel_modes_condensed, all_raw = [], [], []
    baselines, all_stimulifilter = [], []

    filelist = filelist[0:2]
    for file_ind, filename in enumerate(filelist):

        print prefix, filename
        meas_path = os.path.splitext(filename)[0]
        fname = os.path.basename(meas_path)
        plot_name_base = os.path.join(plots_folder, fname)

        #assign each file a color:
        f_ind = file_ind / (len(filelist) - 1.)
        colorlist[os.path.basename(meas_path)] = plt.cm.jet(f_ind)

        # create timeseries, change shape and preprocess
        ts = bf.TimeSeries()
        ts.load(meas_path)
        ts.shape = tuple(ts.shape)
        out = runlib.preprocess(ts, config)
        all_raw.append(out['pp'])
        total_resp.append(out['mean_resp_unsort'])

        # do matrix factorization
        if config['individualMF']['do']:
            mf_func = utils.create_mf(config['individualMF'])
            mf = mf_func(out['pp'])
            baselines.append(out['baseline'])


        # save results
        if 'save' in config['individualMF']['do']:
            mf.save(os.path.join(data_folder, fname + '_' + config['individualMF']['method']))
            out['sorted_baseline'].save(os.path.join(data_folder, fname + '_baseline'))

        # plot overview of matrix factorization
        if 'plot' in config['individualMF']['do']:
            mf_overview = runlib.mf_overview_plot(mf)
            mf_overview.savefig(plot_name_base + '_' + config['individualMF']['method'] +
                                '_overview.' + config['format'])

        # overview of raw responses
        if config['plot_signals']:
            raw_resp_overview = runlib.raw_response_overview(out, prefix)
            raw_resp_overview.savefig(plot_name_base + '_raw_overview.' + config['format'])
            raw_resp_unsort_overview = runlib.raw_unsort_response_overview(prefix, out)
            raw_resp_unsort_overview.savefig(plot_name_base + '_raw_unsort_overview.' + config['format'])

        # calc reproducibility and plot quality
        if config['stimuli_rep']:
            stimulirep = bf.SampleSimilarityPure()
            distanceself, distancecross = stimulirep(out['mean_resp'])
            qual_view = runlib.quality_overview_plot(distanceself, distancecross, ts.name)
            qual_view.savefig(plot_name_base + '_quality.' + config['format'])


    # simultanieous ICA of one odor-set
    if config['commonMF']['do']:
        if config['combine']['scramble']:
            combine_common = bf.ObjectScrambledConcat(config['n_best'])
        else:
            combine_common = bf.ObjectConcat(**config['combine']['param'])
        intersection = bf.SortBySamplename()(combine_common(all_raw))
        variance = config['commonMF']['param']['variance']
        mo = bf.PCA(variance)(intersection)
        mf = utils.create_mf(config['commonMF'])
        mo2 = mf(intersection)
        single_response = bf.SingleSampleResponse(config['condense'])(mo2)

    # plot spatial and temporal bases of simultaneous ica
    if 'plot' in config['commonMF']['do']:
        ica_baseplots = runlib.simultan_ica_baseplot(mo, mo2, all_raw, baselines)
        ica_baseplots.savefig(os.path.join(plots_folder, prefix + '_simultan_bases.' + config['format']))
        ica_timeplots = runlib.simultan_ica_timeplot(mo2, single_response)
        ica_timeplots.savefig(os.path.join(plots_folder, prefix + '_simultan_time.' + config['format']))

# plot odor overview
if config['plot_signals']:
    vmin = -0.1
    vmax = 0.1
    datadic = defaultdict(list)
    for meas in total_resp:
        for stim_ind, stim in enumerate(meas.shaped2D()):
            datadic[meas.label_sample[stim_ind]].append((meas.name, stim, meas.strength[stim_ind]))
    for odor in datadic.keys():
        odor_overview = vis.VisualizeTimeseries()
        data = datadic[odor]
        odor_overview.subplot(len(data))
        for ind, stim in enumerate(data):
            toplot = stim[1].copy()
            toplot[toplot > vmax] = vmax
            odor_overview.imshow(odor_overview.axes['base'][ind], toplot ,
                                 title={'label':stim[0], 'size':10},
                                 colorbar=False, vmin=vmin, vmax=vmax, cmap=plt.cm.Blues_r)
        plt.suptitle(odor)
        # axc = odor_overview.fig.add_axes([0.05, 0.04, 0.9, 0.045])
        # cb1 = mpl.colorbar.ColorbarBase(axc, cmap=mpl.cm.Blues_r,
        #                                    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        #                                    orientation='horizontal')
        odor_overview.fig.savefig(os.path.join(odor_plots_folder,
                                 odor + '_allmeas_neg.' + config['format']))
        plt.close('all')
