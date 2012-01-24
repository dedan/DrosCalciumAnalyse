'''
Created on 18.02.2011

@author: jan
'''

import sys
import scipy.ndimage.filters as filter
import numpy as np
from scipy.io import loadmat
from matplotlib.cm import bone
from matplotlib.patches import Polygon
from matplotlib import collections
import matplotlib.nxutils as nx
import pickle
import scipy.stats
import dataimport
import Image
import pylab as plt
import matplotlib
from scipy.spatial.distance import pdist
import imageprocesses2 as ip
from matplotlib.nxutils import points_inside_poly

class VisualizeTimeseries(object):
    
    def __init__(self):
        self.fig = None
        self.axes_base = []
        self.axes_time = []
    
    def create_subplot(self, num_objects):
        if not(self.fig):
            self.fig = plt.figure(figsize=(12, 12))
        subplot_dim1 = np.ceil(np.sqrt(num_objects))
        subplot_dim2 = np.ceil(num_objects / subplot_dim1)                
        for axind in xrange(num_objects):
            axhandle = self.fig.add_subplot(subplot_dim1, subplot_dim2, axind)
            self.axes_base.append(axhandle)
    
    def add_alltoall(self, axes, objects, drawfunc):
        for ax in axes:
            for obj in objects:
                drawfunc(ax, obj)
    
    def add_onetoall(self, axes, objects, drawfunc, args):
        for ax in axes:
            drawfunc(ax, objects, **args)
    
    def add_onetoone(self, axes, objects, drawfunc):
        for ax_ind, ax in enumerate(axes):
            drawfunc(ax, objects[ax_ind])
           
    def contourfaces(self, ax, im):   
        ax.contourf(im, [0.3, 1], colors=['r'], alpha=0.2)

    def contour(self, ax, im):  
        ax.contourf(im, [0.3], colors=['w'])
    
    def overlay_image(self, ax, im):    
        im_rgba = plt.cm.jet(im / 2 + 0.5)
        im_rgba[:, :, 3] = 0.8
        im_rgba[np.abs(im) < 0.1, 3] = 0 
        ax.imshow(im_rgba, aspect='equal', interpolation='nearest')
    
    def imshow(self, ax, im):
        ax.imshow(im, aspect='equal', interpolation='nearest', cmap=plt.cm.bone) 
        ax.set_axis_off()
        
    def plot(self, ax, timecourse):
        ax.plot(timecourse, '-')
        ax.set_xticks([])
         
    def add_shade(self, ax, labels, timepoints=1, rotate=True):     
        # create changing shade with changing label
        shade = []
        shade_color = 0                  
        reference_label = labels[0]
        for label in labels:
            if not(label == reference_label):
                reference_label = label
                shade_color = 1 - shade_color
            shade.append(shade_color)      
        shade = np.outer(np.array(shade), np.ones((timepoints))).flatten()
        shade[np.hstack((np.array([0]), np.diff(shade))) == -1] = 1
        shade = np.hstack((shade, np.array([1]))) 
        shade = collections.BrokenBarHCollection.span_where(
                                np.arange(len(shade) + 1) - 0.5,
                                *ax.get_ylim(), where=shade > 0,
                                facecolor='k', alpha=0.2)
        ax.add_collection(shade)
        
    def add_xlabels(self, ax, labels, rotation='0'):      
        # add xlabels
        ax.set_xticks(range(0, len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlim([0, len(labels)])
            
        for tick in ax.xaxis.iter_ticks():
            tick[0].label2On = True
            tick[0].label1On = False
            tick[0].label2.set_rotation(rotation)
            tick[0].label2.set_ha('left')
            tick[0].label2.set_size('x-small')
            tick[0].label2.set_stretch('extra-condensed')
            tick[0].label2.set_family('sans-serif')

class TimeSeriesDecomposition():
    pass
        
def stimulus_dependencey(timeseries):
    stim_set = set(timeseries.stimuli)
    # create dictionary with key: stimulus and value: trial where stimulus was given
    stim_pos = {}
    for stimulus in stim_set:
        stim_pos[stimulus] = np.where([i == stimulus for i in timeseries.stimuli])[0]
    min_len = min([len(i)for i in stim_pos.values()])
    # create list of lists, where each sublist contains for all stimuli one exclusive trial
    indices = []
    for i in range(min_len):
        indices.append([j[i] for j in stim_pos.values()])
    # create pseudo-trial timecourses
    trial_timecourses = np.array([timeseries.timecourse[i].reshape((-1, timeseries.num_objects)) for i in indices])    
    # calculate correlation of pseudo-trials, aka stimulus dependency
    cor = [] 
    for object_num in range(timeseries.num_objects):
        cor.append(np.mean(pdist(trial_timecourses[:, :, object_num], 'correlation')))
    timeseries.stimulus_dependency = np.array(cor)

def initmouseob(path='/media/Iomega_HDD/Experiments/Messungen/111210sph/',
           dateikenn='_nnma'):    
    
    """ ======== load in decomposition ======== """
    measID = path.strip('/').split('/')[-1]
    db = dataimport.instantJChemInterface()
                 
    base = np.load(path + 'base' + dateikenn + '.npy')
    norm = np.max(base, 1)
    base /= norm.reshape((-1, 1))
    timecourse = np.load(path + 'time' + dateikenn + '.npy') * norm
    
    shape = np.load(path + 'shape.npy')   
    bg = np.asarray(Image.open(path + 'bg.png').convert(mode='L').resize(
                                                    (shape[1], shape[0])))

    namelist = pickle.load(open(path + 'ids.pik'))
    names = db.make_table_dict('cd_id', ['Name'], 'MOLECULE_PROPERTIES')
    labels = []   
    for label in namelist:
        name_parts = label.split('_')
        odor_name = names[int(name_parts[0])][0][0]
        if len(name_parts) > 2:
            odor_name += name_parts[2].strip()
        if len(name_parts) > 3:
            odor_name += name_parts[3].strip()
        labels.append(odor_name)
    
    decomposition = ip.TimeSeries(timecourse, name=[measID], shape=shape,
                 typ='Decomposition', label_sample=labels)
    decomposition.base = base
    decomposition.bg = bg
    
    """ ======== load in data ======== """
    preprocessed_timecourse = np.load(path + 'data.npy')
    preprocessed = ip.TimeSeries(preprocessed_timecourse, name=[measID],
                  shape=shape, typ='Timeseries', label_sample=labels)
    
    
    """ ======== load in 2p selected ROIs ======== """
    rois_vert = loadmat(path + 'nrois.mat')
    num_rois = rois_vert['nxis'].shape[1]
    
    temp_grid = np.indices(shape)
    grid = np.array(zip(temp_grid[1].flatten(), temp_grid[0].flatten()))
    
    rois = []
    for roi_ind in range(num_rois):
        x_edge = rois_vert['nxis'][:, roi_ind] / 16
        y_edge = rois_vert['nyis'][:, roi_ind] / 16
        num_edges = np.sum(x_edge != 0)
        verts = np.array(zip(x_edge, y_edge))[:num_edges]
        rois.append(points_inside_poly(grid, verts))
    rois = np.array(rois)   
    roidata = ip.TimeSeries('', name=[measID],
                  shape=shape, typ='Decomposition', label_sample=labels)
    roidata.base = rois
    
    # filter rois, such that their main weight is in the center
    for roi_ind, roi in roidata.bases_2D():
        roidata.base[roi_ind] = filter.gaussian_filter(roi, 2).flatten()
               
    # create roi timecourse as convolution with data
    roidata.timecourses = np.dot(preprocessed_timecourse, roidata.base.T)
    
    """ ====== combine Timeseries ===== """
    combi = TimeSeriesDecomposition()
    combi.factorization = decomposition
    combi.data = preprocessed
    combi.roi = roidata
    return combi
    
