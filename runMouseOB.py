'''
Created on 17.09.2010

@author: jan

AnalysisScript
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
import basic_functions as bf
import special_functions as sf
from scipy.io import loadmat
from scipy.ndimage import filters as filters
reload(bf)
reload(sf)
#globalminima = pickle.load(open('minima.pik'))

measureID = '111221sph'

data_selection = {'key': 'molID',
                  'properties': ['concentration', 'extraInfo', 'filename', 'fileID', 'shape'],
                  'table': 'FILES',
                  'select_columns': ['measureID', 'stimulus'],
                  'select_values': [' = "' + measureID + '"', ' = "o"']}

path = '/media/Iomega_HDD/Experiments/Messungen/' + measureID + '/'
bulb = loadmat(path + 'raw/outline.mat')['outline']

nma_param = {'sparse_par': 1, 'sparse_par2' :7, 'negbase': 0, 'smoothness': 0.5}

loader = sf.DBnpyLoader(data_selection, path)
timeseries = loader()

"============================================================================="
"define operations"

baseline_cut = bf.CutOut((0, 10))    
signal_cut = bf.CutOut((30, 60))
trial_mean = bf.TrialMean()
rel_change = bf.RelativeChange()
temporal_downsampling = bf.TrialMean(5)  
lowpass = bf.Filter(filters.median_filter, {'size':3}, downscale=2)
highpass = bf.Filter(filters.gaussian_filter, {'size':10}, downscale=2)
bandpass = bf.Combine(np.subtract)
nma = bf.NNMA(latents=81, maxcount=70, nma_param)
ica = bf.sICA(variance=150)

"=============================================================================="
"preprocess data"
timeseries = temporal_downsampling(timeseries)
timeseries = rel_change(timeseries, trial_mean(baseline_cut(timeseries))) 
preprocessed = bandpass(lowpass(timeseries), highpass(timeseries))

"=============================================================================="
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

icadata = ica(preprocessed)
nmadata = nma(preprocessed)

#preprocessed.save(path + 'preprocess')
#icadata.save(path + 'icamodes')
#nmadata.save(path + 'nnmadata')
    
    
    
