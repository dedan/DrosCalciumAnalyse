'''
Created on Aug 11, 2011

@author: jan
'''

import os
import json
import glob
import pylab as plt
import numpy as np

import basic_blocks as bb
import illustrate_decomposition as vis

reload(bb)
reload(vis)

variance = 9
lowpass = 0.5
similarity_threshold = 0.2
modesim_threshold = 0.3
medianfilter = 8
data_path = '/home/jan/Documents/dros/new_data/numpyfiles/'
#data_path='/Users/dedan/projects/fu/data/dros_calcium/test_data/'
savefolder = 'med' + str(medianfilter) + 'simil' + str(similarity_threshold * 10) + 'mode' + str(modesim_threshold * 10)
save_path = os.path.join(data_path, savefolder)
if not os.path.exists(save_path):
    os.mkdir(save_path)
prefix = 'LIN'

#filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
filelist = [os.path.join(data_path, 'LIN_111026a.json')]
for filename in filelist:

    #####################################################
    #           build the processing pipeline
    # TODO: should not be done in the loop
    #####################################################

    # blank block to put in timeseries 
    blank = bb.Command()
    provider = bb.Command(command=lambda x: x)
    provider.add_sender(blank)
    
    # temporal downsampling by factor 2 (originally 40 frames)
    temporal_down = bb.Mean(20)
    temporal_down.add_sender(provider)
    
    # odor signal starts at original frame 8
    baseline_cut = bb.CutOut((0, 3))
    baseline_cut.add_sender(temporal_down)
    
    # mean of baseline singal
    base_mean = bb.Mean()
    base_mean.add_sender(baseline_cut)
    
    # calculate (delta F) / F
    rel_change = bb.RelativeChange()
    rel_change.add_sender(base_mean, 'base_image')
    rel_change.add_sender(temporal_down)
    
    # MedianFilter
    pixel_filter = bb.MedianFilter(size=medianfilter, downscale=2)
    pixel_filter.add_sender(rel_change)
    
    #Smoothing with GaussFilter
    filter = bb.GaussFilter(sigma=lowpass)
    filter.add_sender(pixel_filter)      
    
    # concatenate Timeseries objects
    staple = bb.Collector('end')
    staple.add_sender(pixel_filter)

    # do ICA
    ica = bb.sICA(variance=variance)
    ica.add_sender(staple)
           
    #sort by stimuliname
    sorted_ica = bb.SortBySamplename()
    sorted_ica.add_sender(ica)
    
    # create mean stimulus response of each odor
    signal_cut = bb.CutOut((6, 13))
    signal_cut.add_sender(pixel_filter)
    
    mean_resp = bb.Mean()
    mean_resp.add_sender(signal_cut)
    
    staple_mean_resp = bb.Collector('end')
    staple_mean_resp.add_sender(mean_resp)
    
    #sort by stimuliname
    sorted_resp = bb.SortBySamplename()
    sorted_resp.add_sender(staple_mean_resp)
    
    # create sample filter 
    filter_mask = bb.SampleSimilarity(similarity_threshold)
    filter_mask.add_sender(staple_mean_resp)
    
    # apply sample filter on ica data
    maskedstim = bb.SelectTrials()
    maskedstim.add_sender(ica)
    maskedstim.add_sender(filter_mask, 'mask')
    
    # create mode filter 
    modefilter = bb.CalcStimulusDrive()
    modefilter.add_sender(maskedstim)
    
    # apply mode filter on ica data
    selectedmodes = bb.SelectModes(modesim_threshold)
    selectedmodes.add_sender(maskedstim)
    selectedmodes.add_sender(modefilter, 'filtervalue')
    
    #collapse identical stimuli to mean timecourse
    standard_response = bb.MeanSampleResponse()
    standard_response.add_sender(selectedmodes)

    # collectors that do not receive a go signal to store computations       
    col_ica = bb.Collector('never')
    col_ica.add_sender(sorted_ica)
    col_fil = bb.Collector('never')
    col_fil.add_sender(staple)
    col_resp = bb.Collector('never')
    col_resp.add_sender(sorted_resp)
    col_selection = bb.Collector('never')
    col_selection.add_sender(standard_response)
    col_modecor = bb.Collector('never')
    col_modecor.add_sender(modefilter)    
    
    
    

    ####################################################################
    #       create odor objects and pass them in the pipeline
    ####################################################################
        
    # load timeseries, shape and labels
    meas_path = os.path.splitext(filename)[0]
    timeseries = np.load(meas_path + '.npy')
    info = json.load(open(meas_path + '.json')) 
    label = info['labels']
    shape = info['shape']

    # divide whole stimuli set into seperate stimuli
    frames_per_trial = 40
    odor_resp = np.vsplit(timeseries, timeseries.shape[0] / frames_per_trial)
    label = label[::frames_per_trial]
    
    # create objects and add them to the pipeline
    for ind, resp in enumerate(odor_resp):
        ts = bb.TimeSeries(shape=shape,
                           series=resp,
                           name=[label[ind].strip('.png')],
                           label_sample=[label[ind].strip('.png')])
        provider.receive_event(bb.Event('image_series', ts))
    provider.receive_event(bb.Event('signal', 'end'))
    

    ####################################################################
    # plot and save results
    ####################################################################    
    
    # save plot and data
    tmp_save = os.path.join(save_path, os.path.basename(meas_path))
    
    
    # draw signal overview
    resp_overview = vis.VisualizeTimeseries()
    resp_overview.subplot(col_resp.image_container.samplepoints)
    resp_overview.imshow('base', 'onetoone', col_resp.image_container, title=True, colorbar=True)
    
    # draw ica overview
    ica_overview = vis.VisualizeTimeseries()
    ica_overview.base_and_time(col_ica.image_container.num_objects)
    ica_overview.imshow('base', 'onetoone', col_ica.image_container.base)
    ica_overview.plot('time', 'onetoone', col_ica.image_container)
    ica_overview.add_labelshade('time', 'onetoall', col_ica.image_container)
    ica_overview.add_samplelabel([-1], col_ica.image_container, rotation='45', toppos=True)
    
    plt.savefig(tmp_save + '_overview.png')
    np.save(tmp_save + '_data.npy', col_fil.image_container.timecourses)
    np.save(tmp_save + '_base_sica.npy', col_ica.image_container.base)
    np.save(tmp_save + '_time_sica.npy', col_ica.image_container.timecourses)
    np.save(tmp_save + '_selectedtime.npy', col_selection.image_container.timecourses)
    np.save(tmp_save + '_modecor.npy', col_modecor.image_container.timecourses)
    selected_metainfo = {'objects': col_selection.image_container.label_objects, 'label':col_selection.image_container.label_sample}
    json.dump(selected_metainfo, open(tmp_save + '_selected.json', 'w'))  
        
    
    
