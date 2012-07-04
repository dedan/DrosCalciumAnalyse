'''
Created on Aug 11, 2011

@author: jan
'''

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

config = ConfigObj(sys.argv[1], unrepr=True)

add = config['filename_add']
if config['normalize']:
    add += '_maxnorm'

savefolder = os.path.join(config['save_path'],
                          'simil' + str(int(config['similarity_threshold'] * 100))
                          + 'n_best' + str(config['n_best']) + add)
if not os.path.exists(savefolder):
    os.mkdir(savefolder)

def create_mf(mf_dic):
    # creates a matrixfactorization according to mf_dic specification
    mf_methods = {'nnma':bf.NNMA, 'nnman':bf.NNMA, 'sica': bf.sICA, 'stica': bf.stICA}
    mf = mf_methods[mf_dic['method']](**mf_dic['param'])
    return mf

#####################################################
#        initialize the processing functions
#####################################################
total_resp = []

# calculate trial mean
trial_mean = bf.TrialMean()
#sorting
sorted_trials = bf.SortBySamplename()
# calculate (delta F) / F
rel_change = bf.RelativeChange()
# spatial filtering
pixel_filter = bf.Filter('median', config['medianfilter'])
gauss_filter = bf.Filter('gauss', config['lowpass'], downscale=config['spatial_down'])

saveplace = os.path.join(savefolder, config['individualMF']['method'])
if not os.path.exists(saveplace):
    os.mkdir(saveplace)
else:
    answer = raw_input('output folder already exists, overwrite results? (y/n): ')
    if not answer == 'y':
        print 'abort run, output folder contains files'
        sys.exit()

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
    baselines = []
    all_stimulifilter = []

    for file_ind, filename in enumerate(filelist):
        print prefix, filename
        # load timeseries, shape and labels
        meas_path = os.path.splitext(filename)[0]
        # savelocation
        savename_ind = os.path.join(savefolder, os.path.basename(meas_path))

        #assign each file a color:
        f_ind = file_ind / (len(filelist) - 1.)
        colorlist[os.path.basename(meas_path)] = plt.cm.jet(f_ind)

        # create timeseries
        ts = bf.TimeSeries()
        ts.load(meas_path)
        # change shape from list to tuple!!
        ts.shape = tuple(ts.shape)

        ####################################################################
        # preprocess
        ####################################################################

        # cut baseline signal (odor starts at frame 4 (original frame8))
        baseline = trial_mean(bf.CutOut((0, 6))(ts))
        baselines.append(baseline)

        sorted_baseline = sorted_trials(bf.CutOut((0, 1))(ts))
        #downscale sorted baseline
        ds = config['spatial_down']
        ds_baseline = sorted_baseline.shaped2D()[:, ::ds, ::ds]
        sorted_baseline.shape = tuple(ds_baseline.shape[1:])
        sorted_baseline.set_timecourses(ds_baseline)

        # temporal downsampling by factor 2 (originally 40 frames)
        ts = bf.TrialMean(20)(ts)

        # preprocess timeseries
        pp = rel_change(ts, baseline)
        if 'mask' in config['filename_add']:
            maskpath = os.path.dirname(meas_path)
            maskfile = '_'.join(os.path.basename(meas_path).split('_')[1:]) + '_mask.npy'
            spatial_mask = np.load(os.path.join(maskpath, maskfile)).astype('bool')
            pp.timecourses[:, np.logical_not(spatial_mask.flatten())] = 0

        pp = gauss_filter(pixel_filter(pp))
        pp.timecourses[np.isnan(pp.timecourses)] = 0
        pp.timecourses[np.isinf(pp.timecourses)] = 0

        if config['normalize']:
            pp.timecourses = pp.timecourses / np.max(pp.timecourses)

        # calcualte mean response
        mean_resp_unsort = trial_mean(bf.CutOut((6, 12))(pp))
        mean_resp = sorted_trials(mean_resp_unsort)
        # select stimuli such that their mean correlation distance is below similarity_threshold
        pp = sorted_trials(pp)
        if not prefix == 'mic':
            stimuli_mask = bf.SampleSimilarity(config['similarity_threshold'])
            stimuli_selection = stimuli_mask(mean_resp)
            stimuli_filter = bf.SelectTrials()
            pp = stimuli_filter(pp, stimuli_selection)

        # collect results
        all_raw.append(pp)

        ####################################################################
        # do individual matrix factorization
        ####################################################################
        if config['individualMF']['do']:
            mf_func = create_mf(config['individualMF'])
            mf = mf_func(pp)
            path = os.path.dirname(savename_ind)
            fname = os.path.basename(savename_ind)
            if 'save' in config['individualMF']['do']:
                mf.save(os.path.join(saveplace, fname + '_' + config['individualMF']['method']))
                sorted_baseline.save(os.path.join(saveplace, fname + '_baseline'))
            if 'plot' in config['individualMF']['do']:

                mf_overview = vis.VisualizeTimeseries()
                mf_overview.base_and_time(mf.num_objects)
                for ind, resp in enumerate(mf.base.shaped2D()):
                    mf_overview.imshow(mf_overview.axes['base'][ind],
                                        resp,
                                        title=mf.label_sample[ind])
                    mf_overview.plot(mf_overview.axes['time'][ind],
                                      mf.timecourses[:, ind])
                    mf_overview.add_labelshade(mf_overview.axes['time'][ind], mf)
                    #ica_overview.add_shade('time', 'onetoall', stimuli_selection, 20)
                mf_overview.add_samplelabel(mf_overview.axes['time'][-1], mf, rotation='45', toppos=True)
                [ax.set_title(mf.label_objects[i]) for i, ax in enumerate(mf_overview.axes['base'])]
                mf_overview.fig.savefig(os.path.join(saveplace, fname + '_' + config['individualMF']['method'] + '.' + config['format']))


        ####################################################################
        # plot signals
        ####################################################################
        if config['plot_signals']:
            # draw signal overview
            resp_overview = vis.VisualizeTimeseries()
            if prefix == 'mic':
                resp_overview.subplot(mean_resp.samplepoints, dim2=4)
            else:
                resp_overview.subplot(mean_resp.samplepoints)
            for ind, resp in enumerate(mean_resp.shaped2D()):
                max_data = np.max(np.abs(resp))
                resp /= max_data
                resp_overview.imshow(resp_overview.axes['base'][ind],
                                     sorted_baseline.shaped2D()[ind],
                                     cmap=plt.cm.bone)
                resp_overview.overlay_image(resp_overview.axes['base'][ind],
                                            resp, threshold=0.3,
                                            title={'label':mean_resp.label_sample[ind], 'size':10})
                resp_overview.axes['base'][ind].set_ylabel('%.2f' % max_data)
            resp_overview.fig.savefig(savename_ind + '_overview.' + config['format'])

            # draw unsorted signal overview
            uresp_overview = vis.VisualizeTimeseries()
            if prefix == 'mic':
                uresp_overview.subplot(mean_resp_unsort.samplepoints, dim2=4)
            else:
                uresp_overview.subplot(mean_resp_unsort.samplepoints)
            mean_resp_unsort.strength = []
            for ind, resp in enumerate(mean_resp_unsort.shaped2D()):
                max_data = np.max(np.abs(resp))
                normedresp = resp / max_data
                uresp_overview.imshow(uresp_overview.axes['base'][ind],
                                       normedresp,
                                       title={'label':mean_resp_unsort.label_sample[ind], 'size':10},
                                       colorbar=False, vmin= -1, vmax=1)
                uresp_overview.axes['base'][ind].set_ylabel('%.2f' % max_data)
                mean_resp_unsort.strength.append(max_data)
            total_resp.append(mean_resp_unsort)
            uresp_overview.fig.savefig(savename_ind + '_overview_unsort.' + config['format'])

        ####################################################################
        # calc reproducibility and plot
        ####################################################################
        if config['stimuli_rep']:
            stimulirep = bf.SampleSimilarityPure()
            distanceself, distancecross = stimulirep(mean_resp)
            #draw quality overview
            qual_view = vis.VisualizeTimeseries()
            qual_view.oneaxes()
            data = zip(*distancecross.items())
            vis.violin_plot(qual_view.axes['time'][0], [i.flatten() for i in data[1]] ,
                             range(len(distancecross)), 'b')
            qual_view.axes['time'][0].set_xticks(range(len(distancecross)))
            qual_view.axes['time'][0].set_xticklabels(data[0])
            qual_view.axes['time'][0].set_title(ts.name)
            for pos, stim in enumerate(data[0]):
                qual_view.axes['time'][0].plot([pos] * len(distanceself[stim]),
                                                distanceself[stim], 'o', mew=2, mec='k', mfc='None')
            qual_view.fig.savefig(savename_ind + '_quality.' + config['format'])


        plt.close('all')

    ####################################################################
    # odorset quality overview
    ####################################################################
    if config['odorset_quality']:
        allodors = list(set(ts.label_sample + sum([t.label_sample for t in all_raw], [])))
        allodors.sort()
        quality_mx = np.zeros((len(all_raw), len(allodors)))
        for t_ind, t in enumerate(all_raw):
            for od in set(t.label_sample):
                quality_mx[t_ind, allodors.index(od)] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(quality_mx, interpolation='nearest', cmap=plt.cm.bone)
        ax.set_yticks(range(len(all_raw)))
        ax.set_yticklabels([t.name for t in all_raw])
        ax.set_xticks(range(len(allodors)))
        ax.set_xticklabels(allodors, rotation='45')
        ax.set_title(prefix + '_' + str(config['similarity_threshold']))
        fig.savefig('_'.join(savename_ind.split('_')[:-1]) + 'mask.' + config['format'])


    ####################################################################
    # simultanieous ICA
    ####################################################################
    if config['commonMF']['do']:
        if config['combine']['scramble']:
            combine_common = bf.ObjectScrambledConcat(config['n_best'])
        else:
            combine_common = bf.ObjectConcat(**config['combine']['param'])
        intersection = sorted_trials(combine_common(all_raw))
        variance = config['commonMF']['param']['variance']
        mo = bf.PCA(variance)(intersection)
        mf = create_mf(config['commonMF'])
        mo2 = mf(intersection)
        single_response = bf.SingleSampleResponse(config['condense'])(mo2)
        if 'plot' in config['commonMF']['do']:
            # plot bases
            fig = plt.figure()
            fig.suptitle(np.sum(mo.eigen))
            num_bases = len(filelist)
            names = [t.name for t in all_raw]
            for modenum in range(variance):
                single_bases = mo2.base.objects_sample(modenum)
                for base_num in range(num_bases):
                    ax = fig.add_subplot(variance + 1, num_bases, base_num + 1)
                    ax.imshow(np.mean(baselines[base_num].shaped2D(), 0), cmap=plt.cm.gray)
                    ax.set_axis_off()
                    ax.set_title(names[base_num])
                    data_max = np.max(np.abs(single_bases[base_num]))
                    ax = fig.add_subplot(variance + 1, num_bases,
                                         ((num_bases * modenum) + num_bases) + base_num + 1)
                    ax.imshow(single_bases[base_num] * -1, cmap=plt.cm.hsv, vmin= -data_max, vmax=data_max)
                    ax.set_title('%.2f' % data_max, fontsize=8)
                    ax.set_axis_off()
            fig.savefig('_'.join(savename_ind.split('_')[:-1]) + '_bases.' + config['format'])
            # plot timecourses
            fig = plt.figure()
            stim_driven = modefilter(mo2).timecourses
            for modenum in range(variance):
                ax = fig.add_subplot(variance, 1, modenum + 1)
                ax.plot(single_response.timecourses[:, modenum])
                ax.set_xticklabels([], fontsize=12, rotation=45)
                ax.set_ylabel("%1.2f" % stim_driven[0, modenum])
                ax.grid(True)
            ax.set_xticks(np.arange(3, single_response.samplepoints, single_response.timepoints))
            ax.set_xticklabels(single_response.label_sample, fontsize=12, rotation=45)
            fig.savefig('_'.join(savename_ind.split('_')[:-1]) + '_simultan_time.' + config['format'])
            plt.close('all')

####################################################################
# odor overview
####################################################################

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
            #max(np.abs(np.min(toplot)), 0.01)
            odor_overview.imshow(odor_overview.axes['base'][ind], toplot ,
            title={'label':stim[0], 'size':10},
            colorbar=False, vmin=vmin, vmax=vmax, cmap=plt.cm.Blues_r)
            print np.sum(stim[1] > 0)
            #odor_overview.axes['base'][ind].set_ylabel('%.2f' % minval)
        plt.suptitle(odor)
        axc = odor_overview.fig.add_axes([0.05, 0.04, 0.9, 0.045])
        cb1 = mpl.colorbar.ColorbarBase(axc, cmap=mpl.cm.Blues_r,
                                           norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                                           orientation='horizontal')
        odor_overview.fig.savefig(os.path.join(os.path.dirname(savename_ind),
                                 odor + '_allmeas_neg.' + config['format']))
        plt.close('all')
