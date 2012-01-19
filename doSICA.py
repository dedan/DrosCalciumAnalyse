'''
Created on Aug 11, 2011

@author: jan
'''
import sys
sys.path.append("/home/jan/workspace/DataProcessing/src") 


import imageprocesses2 as ip


import createTimeSeries as c
reload(c)
import pylab as plt
import numpy as np
import pickle
import myview_wtogui_dros as vis
import csv
reload(vis)

variance = 9
#measIDs = ['111108b']
#measIDs = ['110901a', '110902a', '111012a', '111013a', '111014a', '111014b']
#measIDs = ['110817b_neu', '110817c', '110823a', '110823b', '111025a', '111025b']
#measIDs = ['111017a', '111017b', '111018a', '111018b', '111024a', '111024c','120111a', '120111b']
measIDs = ['111026a', '111027a', '111107a', '111107b', '111108a', '111108b', '120112b']
#measIDs = ['111117a', '111118a', '111124a']
for measid in measIDs:
    #path = '/home/jan/Documents/dros/new_data/CVA_MSH_ACA_BEA_OCO_align/' 
    #path = '/home/jan/Documents/dros/new_data/2PA_PAC_ACP_BUT_OCO/' 
    #path = '/home/jan/Documents/dros/new_data/OCO_GEO_PAA_ISO_BUT_align/' 
    path = '/home/jan/Documents/dros/new_data/LIN_AAC_ABA_CO2_OCO_align/' 
    #path = '/home/jan/Documents/dros/new_data/microlesion_stabilized/' 
    pathr = path + measid + '/'
    path_save = '/home/jan/Documents/dros/new_data/raw_f/' + measid + '_' #+ measid + '/unnormed_nnma' + str(variance) + '/'#'percent' + str(variance).split('.')[1] + '/'
    extra = ''#'_f'
    lowpass = 0.5





    
    blank = ip.Command()
    
    # blank block to put in times  
    provider = ip.Command(command=lambda x: x)
    provider.add_sender(blank)
    
    # temporal downsampling (of original 40)
    temporal_down = ip.Mean(20)
    temporal_down.add_sender(provider)
    
    #odor signal starts at original frame 8
    baseline_cut = ip.CutOut((0, 3))
    baseline_cut.add_sender(temporal_down)
    
    # mean of baseline singal
    base_mean = ip.Mean()
    base_mean.add_sender(baseline_cut)
    
    # calculate delta F / F
    rel_change = ip.RelativeChange()
    rel_change.add_sender(base_mean, 'base_image')
    rel_change.add_sender(temporal_down)
    
    # MedianFilter
    pixel_filter = ip.MedianFilter(size=10, downscale=2)
    pixel_filter.add_sender(rel_change)
    
    #Smoothing with GaussFilter
    filter = ip.GaussFilter(sigma=lowpass)
    #filter.add_sender(rel_change)
    filter.add_sender(pixel_filter)      
    
    '''
    ====================== renorm data  =================================
    
    def norm(x):
        
        pixel = x.shape[1]
        o = x.reshape((20, pixel))
 
        o[np.abs(o) < 0.03] = 0
        print o.shape
        norm = np.sqrt(np.sum(np.abs(o) ** 2))
        print norm, norm.shape, o.shape
        o /= norm + 1E-15#(norm.reshape((-1, 20, 1)) + 1E-15)
        x = o.reshape((-1, pixel))
        if mask != None:
            x[:, mask] = 1E-10 * np.random.randn(np.sum(mask))
        return x

    wto_noise = ip.Command(command=norm)
    wto_noise.add_sender(filter)
            
    =======================================================================
    '''
    
    #concatenate Timeseries objects
    staple = ip.Collector('end')
    staple.add_sender(pixel_filter)

    '''  
    nma = ip.NNMA(latents=variance)
    #nma.add_sender(staple)
    nma.param['sparse_par'] = 0.05
    nma.param['sparse_par3'] = 0.5
    nma.param['smoothness'] = 0.05
    nma.maxcount = 50
    nma.param['negbase'] = variance
    
    #ica = nma
    '''
    
    ica = ip.sICA(variance=variance)
    ica.add_sender(staple)
    
    # collectors that do not receive a go signal to store computations       
    col_ica = ip.Collector('never')
    col_ica.add_sender(ica)
       
    col_fil = ip.Collector('never')
    col_fil.add_sender(staple)
    
    
    # create mean stimulus response of each odor
    signal_cut = ip.CutOut((6, 13))
    signal_cut.add_sender(pixel_filter)
    
    mean_resp = ip.Mean()
    mean_resp.add_sender(signal_cut)

    col_resp = ip.Collector('never')
    col_resp.add_sender(mean_resp)
    
    
    """  ==== create odor objects and pass them in the pipeline  ==== """
    
    #TODO: load timeseries, shape and labels
    
    # divide whole stimuli set into seperate stimuli
    frames_per_trial = 40
    odor_resp = np.vsplit(timeseries, timeseries.shape[0] / frames_per_trial)
    label = label[::frames_per_trial]
    
    
    for ind, resp in enumerate(odor_resp):
        # create odor object
        ts = ip.TimeSeries(shape=shape, series=resp, name=[label[ind].strip('.png')], label_sample=[label[ind].strip('.png')])
        # put object in pipeline
        provider.receive_event(ip.Event('image_series', ts))
    
    # put 'end'-signal into pipeline
    provider.receive_event(ip.Event('signal', 'end'))
    
    
    ''' === plot and save results === '''
    
    
    # draw signal overview
    plt.figure(figsize=(17, 15))
    dim0 = np.ceil(np.sqrt(len(col_resp.buffer)))
    dim1 = np.ceil(len(col_resp.buffer) / dim0)
    for j, v in enumerate(np.argsort(label)):
        plt.subplot(dim0, dim1, j + 1)
        plt.imshow(col_resp.buffer[v].reshape(col_fil.image_container.shape))
        plt.title(label[v].split('.')[0])
        plt.colorbar()
        plt.axis('off')
    plt.savefig(path_save + 'overview' + extra + '.png')
    
    # save data
    np.save(path_save + 'data' + extra, col_fil.image_container.data)
    '''
    np.save(path_save + 'shape' + extra, col_fil.image_container.shape)
    np.save(path_save + 'base_sica' + extra, col_ica.image_container.base)
    np.save(path_save + 'time_sica' + extra, col_ica.image_container.time)
   
    np.save(path_save + 'base_basevar' + extra, col_basevar.image_container.base)
    np.save(path_save + 'time_basevar' + extra, col_basevar.image_container.time)
    np.save(path_save + 'maxproj' + extra , np.max(col_fil.image_container.data, 0))
    '''
   
    
    # visualize ica components
    '''
    vis1 = vis.initme(path_save, '_sica', extra, reorderflag=True)
    vis1.figcon.savefig(path_save + 'contour_sica' + extra + '.png')
    vis1.figbase.savefig(path_save + 'bases_sica' + extra + '.png')
    for j, i in enumerate(vis1.model['im']):
        vis1.showim(j)
        plt.draw()
        vis1.fig.savefig(path_save + 'mode' + str(j) + extra + '.png')
    vis1.show_all()
    vis1.figall.savefig(path_save + 'decomposition' + extra + '.png')
    
    csvwriter = csv.writer(open(path_save + 'time_sica.csv', 'w'))
    csvwriter.writerows(vis1.model['time'])
    csvwriter = csv.writer(open(path_save + 'odors.csv', 'w'))
    csvwriter.writerows(vis1.model['names'])
    
   
    plt.close('all')
