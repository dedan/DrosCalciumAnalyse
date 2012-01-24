'''
Created on Aug 11, 2011

@author: jan
'''

import os
import json
import glob
import imageprocesses2 as ip
import pylab as plt
import numpy as np
import myview_wtogui_dros as vis
reload(vis)

variance = 9
lowpass = 0.5
data_path = '/Users/dedan/projects/fu/data/dros_calcium/'
save_path = os.path.join(data_path, 'out')
prefix = 'LIN'

filelist = glob.glob(os.path.join(data_path, prefix) + '*.json')
for filename in filelist:

    #####################################################
    #           build the processing pipeline
    # TODO: should not be done in the loop
    #####################################################

    # blank block to put in timeseries 
    blank = ip.Command()
    provider = ip.Command(command=lambda x: x)
    provider.add_sender(blank)
    
    # temporal downsampling by factor 2 (originally 40 frames)
    temporal_down = ip.Mean(20)
    temporal_down.add_sender(provider)
    
    # odor signal starts at original frame 8
    baseline_cut = ip.CutOut((0, 3))
    baseline_cut.add_sender(temporal_down)
    
    # mean of baseline singal
    base_mean = ip.Mean()
    base_mean.add_sender(baseline_cut)
    
    # calculate (delta F) / F
    rel_change = ip.RelativeChange()
    rel_change.add_sender(base_mean, 'base_image')
    rel_change.add_sender(temporal_down)
    
    # MedianFilter
    pixel_filter = ip.MedianFilter(size=10, downscale=2)
    pixel_filter.add_sender(rel_change)
    
    #Smoothing with GaussFilter
    filter = ip.GaussFilter(sigma=lowpass)
    filter.add_sender(pixel_filter)      
    
    # concatenate Timeseries objects
    staple = ip.Collector('end')
    staple.add_sender(pixel_filter)

    # do ICA
    ica = ip.sICA(variance=variance)
    ica.add_sender(staple)
    
    # create mean stimulus response of each odor
    signal_cut = ip.CutOut((6, 13))
    signal_cut.add_sender(pixel_filter)
    mean_resp = ip.Mean()
    mean_resp.add_sender(signal_cut)

    # collectors that do not receive a go signal to store computations       
    col_ica = ip.Collector('never')
    col_ica.add_sender(ica)
    col_fil = ip.Collector('never')
    col_fil.add_sender(staple)
    col_resp = ip.Collector('never')
    col_resp.add_sender(mean_resp)
    
    
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
        ts = ip.TimeSeries(shape=shape, 
                           series=resp, 
                           name=[label[ind].strip('.png')], 
                           label_sample=[label[ind].strip('.png')])
        provider.receive_event(ip.Event('image_series', ts))
    provider.receive_event(ip.Event('signal', 'end'))
    

    ####################################################################
    # plot and save results
    ####################################################################    
    
    # draw signal overview
    plt.figure(figsize=(17, 15))
    dim0 = np.ceil(np.sqrt(len(col_resp.buffer)))
    dim1 = np.ceil(len(col_resp.buffer) / dim0)
    for j, v in enumerate(np.argsort(label)):
        plt.subplot(dim0, dim1, j + 1)
        plt.imshow(col_resp.buffer[v].reshape(col_fil.image_container.shape))
        plt.title(label[v].split('.')[0])
        # plt.colorbar()
        plt.axis('off')

    # save plot and data
    tmp_save = os.path.join(save_path, os.path.basename(meas_path))

    plt.savefig(tmp_save + 'overview.png')
    np.save(tmp_save + '_data.npy', col_fil.image_container.data)
    np.save(tmp_save + '_base_sica.npy', col_ica.image_container.base)
    np.save(tmp_save + '_time_sica.npy', col_ica.image_container.time)    
    
    # # visualize ica components
    # vis1 = vis.initme(path_save, '_sica', extra, reorderflag=True)
    # vis1.figcon.savefig(path_save + 'contour_sica' + extra + '.png')
    # vis1.figbase.savefig(path_save + 'bases_sica' + extra + '.png')
    # for j, i in enumerate(vis1.model['im']):
    #     vis1.showim(j)
    #     plt.draw()
    #     vis1.fig.savefig(path_save + 'mode' + str(j) + extra + '.png')
    # vis1.show_all()
    # vis1.figall.savefig(path_save + 'decomposition' + extra + '.png')
   
    # plt.close('all')
