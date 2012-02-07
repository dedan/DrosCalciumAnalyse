'''
Created on 17.09.2010

@author: jan

AnalysisScript
'''


import numpy as np
import glob

import os.path.join as pjoin
import os.path.basename as pbase
from scipy.io import loadmat
from ConfigParser import ConfigParser

import basic_functions as bf
import special_functions as sf

reload(bf)
reload(sf)

measureIDs = ['111221sph']
raw_path = '/media/Iomega_HDD/Experiments/Messungen/'

for measureID in measureIDs:
    path = pjoin(raw_path, measureID)
    cfgfiles = glob.glob(pjoin(path, 'prepro' + '*.ini'))
    
    for cfgfile in cfgfiles:
        #read in config
        cfg = ConfigParser()
        cfg.read(pjoin(path, cfgfile + '.ini'))
        
        data_selection = {'key': 'molID',
                          'properties': ['concentration', 'extraInfo', 'filename', 'fileID', 'shape'],
                          'table': 'FILES',
                          'select_columns': ['measureID'] + cfg.select_columns,
                          'select_values': [' = "' + measureID + '"'] + cfg.select_values}
        
        bulb = loadmat(pjoin(path, 'raw', 'outline.mat'))['outline']
        
        loader = sf.DBnpyLoader(data_selection, path + 'converted/')
        timeseries = loader()
        
        "============================================================================="
        "define operations"
        
        baseline_cut = bf.CutOut(cfg.baseline_cut)    
        signal_cut = bf.CutOut(cfg.signal_cut)
        trial_mean = bf.TrialMean()
        rel_change = bf.RelativeChange()
        temporal_downsampling = bf.TrialMean(cfg.temporal_down) 
        lowpass = bf.Filter(cfg.lowfilt, cfg.lowfiltextend, downscale=cfg.spatial_down)
        highpass = bf.Filter(cfg.highfilt, cfg.highfiltextend, downscale=cfg.spatial_down)
        bandpass = bf.Combine(np.subtract)
        
        
        "=============================================================================="
        "preprocess data"
        timeseries = temporal_downsampling(timeseries)
        timeseries = rel_change(timeseries, trial_mean(baseline_cut(timeseries))) 
        preprocessed = bandpass(lowpass(timeseries), highpass(timeseries))
        
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
        
        preprocessed.save(pjoin(path, 'analysis', pbase(cfgfile).strip('ini')))


    
    
    
