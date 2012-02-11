'''
Created on 17.09.2010

@author: jan

AnalysisScript
'''

import glob
from configobj import ConfigObj
from pipeline import TimeSeries
from os.path import join as pjoin
from os.path import basename as pbase
import basic_functions as bf
import special_functions as sf


reload(bf)
reload(sf)


measureIDs = ['111222sph', '120107'] #['111210sph', '111212sph'] #, ' #,['111221sph']
methods = ['sica', 'roifilt' , 'nnma']
raw_path = '/media/Iomega_HDD/Experiments/Messungen'


for measureID in measureIDs:
    for method in methods:
        base_path = pjoin(raw_path, measureID)
        path = pjoin(base_path, 'analysis')
        
        datafiles = glob.glob(pjoin(path, 'prepro', 'prepro*.npy'))
        cfgfiles = glob.glob(pjoin(raw_path, 'configfiles', 'decompose', method + '*.ini'))
        
        for datafile in datafiles:
            datafile = datafile.split('.')[0]
            ts = TimeSeries()
            ts.load(datafile)
            for cfgfile in cfgfiles:
                print pbase(datafile), pbase(cfgfile)
                configname = pbase(cfgfile).split('.')[0] + '_' + pbase(datafile).strip('prepro_')
                if glob.glob(pjoin(path, 'decompose', configname + '.npy')):
                    print configname, ' already done for ', measureID
                    continue
                
                cfg = ConfigObj(pjoin(path, cfgfile), unrepr=True)
                if method == 'nnma':
                    decompose = bf.NNMA(cfg['latents'], cfg['maxcount'], param=cfg['nma_param'])
                    decomposition = decompose(ts)
                elif method == 'sica':
                    decompose = bf.sICA(cfg['variance'])
                    decomposition = decompose(ts)
                elif method == 'roifilt':
                    roiload = sf.LoadRoi(base_path)
                    roiprojection = sf.Project()
                    rois = roiload(ts.shape)
                    if 'erode' in cfg:
                        roi_erode = bf.Filter('erosion', cfg['erode']['filtextend'])
                        rois = roi_erode(rois)
                    rois.timecourses = rois.timecourses.astype('float')
                    if 'smooth' in cfg:                   
                        roi_smooth = bf.Filter('gauss', cfg['smooth']['filtextend'])
                        rois = roi_smooth(rois)
                    decomposition = roiprojection(ts, rois)
                         
                                
                decomposition.save(pjoin(path, 'decompose', configname))

    
    
    
