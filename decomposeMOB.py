'''
Created on 17.09.2010

@author: jan

AnalysisScript
'''

import glob
import ConfigParser

from pipeline import TimeSeries
import os.path.join as pjoin
import os.path.basename as pbase
import basic_functions as bf
import special_functions as sf


reload(bf)
reload(sf)


measureIDs = ['111221sph']
methods = ['sica, nnma, roifilt']
raw_path = '/media/Iomega_HDD/Experiments/Messungen'

for measureID in measureIDs:
    for method in methods:
        path = pjoin(raw_path, measureID, 'analysis')
        
        datafiles = glob.glob(pjoin(path, 'prepro*.npy'))
        cfgfiles = glob.glob(pjoin(path, method + '*.ini'))
        
        for datafile in datafiles:
            datafile = datafile.strip('.npy')
            ts = TimeSeries()
            ts.load(datafile)
            for cfgfile in cfgfiles:
                cfg = ConfigParser()
                cfg.read(cfgfile)
                if method == 'nnma':
                    decompose = bf.NNMA(cfg.latents, cfg.maxcount, param=cfg.nma_param)
                elif method == 'sica':
                    decompose = bf.sICA(cfg.variance)
                elif method == 'roifilt':
                    roiload = sf.LoadRoi(path)
                    roi_smooth = bf.Filter('gauss', cfg.smoothfiltextend)
                    roiprojection = sf.Project()
                    
                    rois = roiload(ts.shape)
                    decomposition = roiprojection(ts, roi_smooth(rois))
                         
                decomposition = decompose(ts)                
                decomposition.save(pjoin(path, pbase(cfgfile).strip('.ini')) + '_' 
                             + pbase(datafile.strip('prepro_')))

    
    
    
