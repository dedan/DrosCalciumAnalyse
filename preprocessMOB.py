'''
Created on 17.09.2010

@author: jan

AnalysisScript
'''


import numpy as np
import glob

from os.path import join as pjoin
from os.path import basename as pbase
from scipy.io import loadmat
from configobj import ConfigObj

import basic_functions as bf
import special_functions as sf

reload(bf)
reload(sf)

measureIDs = ['111210sph', '111212sph', '111221sph', '111222sph', '120107']#']
raw_path = '/media/Iomega_HDD/Experiments/Messungen/'

for measureID in measureIDs:
    path = pjoin(raw_path, measureID)
    prepropath = pjoin(path, 'analysis', 'prepro')
    cfgfiles = glob.glob(pjoin(raw_path, 'configfiles', 'prepro', '*.ini'))
    
    for cfgfile in cfgfiles:
        #read in config      
        if glob.glob(pjoin(prepropath, pbase(cfgfile).strip('.ini') + '.npy')):
            print pbase(cfgfile), ' already done for ', measureID
            continue
        cfg = ConfigObj(pjoin(path, cfgfile), unrepr=True)
        data_selection = {'key': 'molID',
                          'properties': ['concentration', 'extraInfo', 'filename', 'fileID', 'shape'],
                          'table': 'FILES',
                          'select_columns': ['measureID'] + [cfg['select_columns']],
                          'select_values': [' = "' + measureID + '"'] + [cfg['select_values']]}
        
        bulb = loadmat(pjoin(path, 'raw', 'outline.mat'))['outline']
        
        loader = sf.DBnpyLoader(data_selection, pjoin(path, 'converted'))
        timeseries = loader()
        
        "============================================================================="
        "define operations"
        
        baseline_cut = bf.CutOut(cfg['baseline_cut'])    
        signal_cut = bf.CutOut(cfg['signal_cut'])
        trial_mean = bf.TrialMean()
        rel_change = bf.RelativeChange()
        temporal_downsampling = bf.TrialMean(cfg['temporal_down']) 
        lowpass = bf.Filter(cfg['lowfilt'], cfg['lowfiltextend'], downscale=cfg['spatial_down'])
        if not(cfg['highfilt'] == 'None'):
            highpass = bf.Filter(cfg['highfilt'], cfg['highfiltextend'], downscale=cfg['spatial_down'])
            bandpass = bf.Combine(np.subtract)
        
        
        "=============================================================================="
        "preprocess data"
        timeseries = temporal_downsampling(timeseries)
        timeseries = rel_change(timeseries, trial_mean(baseline_cut(timeseries))) 
        if not(cfg['highfilt'] == 'None'):
            preprocessed = bandpass(lowpass(timeseries), highpass(timeseries))
        else:
            preprocssed = lowpass(timeseries)
        
        def set_positive(timeseries):
            x = timeseries.timecourses
            bulb2 = bulb.reshape((128, 168))
            bulb2 = np.invert(bulb2[::2, ::2].flatten().astype('bool'))
            x[:, bulb2] = 0 
            #offset = np.max(x).reshape((-1, 1))
            #x -= offset 
            x *= -1000
            return timeseries
        
        preprocessed = set_positive(preprocessed)    
        
        "=============================================================================="
        
        preprocessed.save(pjoin(prepropath, pbase(cfgfile).strip('.ini')))


    
    
    
