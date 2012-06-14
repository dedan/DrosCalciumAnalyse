'''
Created on Aug 11, 2011

@author: jan
'''
import sys
sys.path.append('/home/jan/repos/NeuralImageProcessing/NeuralImageProcessing')

import os
from pipeline import TimeSeries
import createTimeSeries as c
reload(c)


#measIDs = ['110901a', '110902a', '111012a', '111013a', '111014a', '111014b']
#measIDs = ['110817b_neu', '110817c', '110823a', '110823b', '111025a', '111025b']
#measIDs = ['111017a', '111017b', '111018a', '111018b', '111024a', '111024c', '120111a', '120111b']
#measIDs = ['111026a', '111027a', '111107a', '111107b', '111108a', '111108b', '120112b']
#measIDs = ['111117a', '111118a', '111124a']
#measIDs = ['120307a', '120307b', '120307c', '120308c']
measIDs = ['111108a', '111108b']

#path = '/home/jan/Documents/dros/new_data/CVA_MSH_ACA_BEA_OCO_align/' 
#path = '/home/jan/Documents/dros/new_data/2PA_PAC_ACP_BUT_OCO_align/' 
#path = '/home/jan/Documents/dros/new_data/OCO_GEO_PAA_ISO_BUT_align/' 
#path = '/home/jan/Documents/dros/new_data/LIN_AAC_ABA_CO2_OCO_align/' 
#path = '/home/jan/Documents/dros/new_data/microlesion_stabilized/'
#path = '/home/jan/Documents/dros/new_data/micro_align/'
path = '/home/jan/Documents/dros/new_data/LIN_AAC_ABA_CO2_OCO/' 
#measIDs = os.listdir(path)
#measIDs.remove('analysis')
#measIDs.remove('111202a_neu')
frames_per_trial = 40

for measid in measIDs:
 
    dataset = path.strip('/').split('/')[-1][:3]
    pathr = path + measid + '/'
    path_save = '/home/jan/Documents/dros/new_data/raw/' + dataset + '_' + measid #+ measid + '/unnormed_nnma' + str(variance) + '/'#'percent' + str(variance).split('.')[1] + '/'
        
    timeseries, shape, label = c.create(pathr)
    label = [i.strip('.png') for i in label[::frames_per_trial]]   
    ts = TimeSeries(shape=tuple(shape), series=timeseries, name=measid, label_sample=label)
    ts.save(path_save)
    

    
