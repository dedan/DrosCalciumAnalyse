'''
Created on 17.09.2010

@author: jan

AnalysisScript
'''

import glob
from configobj import ConfigObj
import numpy as np
from pipeline import TimeSeries
from os.path import join as pjoin
from os.path import basename as pbase
import basic_functions as bf
import special_functions as sf
import illustrate_decomposition as id
from collections import defaultdict

reload(bf)
reload(sf)
reload(id)

measureIDs = ['111221sph']
methods = ['roifilt', 'nnma' , 'sica', ]# 'roifilt']
raw_path = '/media/Iomega_HDD/Experiments/Messungen'

methodcolor = {'roifilt':'b', 'nnma':'r', 'sica': 'g'}

trialmean = bf.TrialMean()
stimulusdrive = bf.CalcStimulusDrive()
combine = bf.SampleConcat()
spatial_cor = bf.Distance(direction='spatial')

for measureID in measureIDs:
    


    for method in methods:
        
        stimulusdrives = defaultdict(list)
        print stimulusdrives.keys()
        spcorrelation = defaultdict(list)
        
        path = pjoin(raw_path, measureID, 'analysis')       
        datafiles = glob.glob(pjoin(path, 'decompose', method + '*5.npy'))
        
               
        for datafile in datafiles:
            datafile = datafile.strip('.npy')
            ts = TimeSeries()
            ts.load(datafile)
            ts.name = pbase(datafile)
            ts.base.name = ts.name
            ts.label_sample = [i.split('_')[0] for i in ts.label_sample]
            if datafile.find('t15') > 0:
                signal_cut = bf.CutOut((6, 15))
            else:
                signal_cut = bf.CutOut((2, 5))

            ts = signal_cut(ts)

            #ts.timecourses[ts.timecourses < 0] = 0
            #drivenheit = stimulusdrive(trialmean(ts))
            drivenheit = stimulusdrive(ts)
            key = '_'.join(ts.name.split('_')[:2])
            stimulusdrives[key].append(drivenheit)
            if method == 'nnma':
                spcorrelation[key].append(spatial_cor(ts.base))

  
    
        for key in stimulusdrives:
            series = combine(stimulusdrives[key])
            stimulusdrives[key] = series
            sortlabel = ['_'.join(i.split('_')[2:]) for i in series.label_sample]
            sortind = np.argsort(sortlabel)
            series.timecourses = series.timecourses[sortind]
            series.label_sample = ['_'.join(series.label_sample[i].split('_')[2:]) for i in sortind]
                
        if method == 'nnma':        
            for key in spcorrelation:
                series = combine(spcorrelation[key])       
                spcorrelation[key] = series
                sortlabel = ['_'.join(i.split('_')[2:]) for i in series.label_sample]
                sortind = np.argsort(sortlabel)
                series.timecourses = series.timecourses[sortind]
                series.label_sample = ['_'.join(series.label_sample[i].split('_')[2:]) for i in sortind]
                assert series.label_sample == stimulusdrives[key].label_sample, 'labels are not equal'
                vis_cor = id.VisualizeTimeseries()
                vis_cor.oneaxes() 
                vis_cor.add_violine('time', 'onetoall', series, rotation='45')
                vis_cor.add_violine('time', 'onetoall', stimulusdrives[key], color=methodcolor[method], rotation='45')
                vis_cor.axes['time'][0].set_ylim((0, 1.3))
                vis_cor.axes['time'][0].set_title(key)
        else: 
            for key in stimulusdrives:        
                vis = id.VisualizeTimeseries()
                vis.oneaxes()    
                vis.add_violine('time', 'onetoall', stimulusdrives[key], color=methodcolor[method], rotation='45')
                vis.axes['time'][0].set_ylim((0, 1.3))
                vis.axes['time'][0].set_title(key)
