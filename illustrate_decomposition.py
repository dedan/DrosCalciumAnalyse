'''
Created on 18.02.2011

@author: jan
'''


import numpy as np
from matplotlib import collections
import pylab as plt


class VisualizeTimeseries(object):
    
    def __init__(self):
        self.fig = None
        self.axes = {'base':[], 'time':[]}
        self.mappings = {'onetoone':self.onetoone, 'onetoall':self.onetoall, 'alltoall':self.alltoall}
    
    def base_and_time(self, num_objects):
        if not(self.fig):
            self.fig = plt.figure(figsize=(20, 13))
        
        height = 0.9 / num_objects
        for i in range(num_objects):
            #create timeaxes
            ax = self.fig.add_axes([0.2, height * i + 0.05, 0.75, height])
            ax.set_xticklabels([])
            self.axes['time'].append(ax)
            #create baseaxes
            ax = self.fig.add_axes([0.05, height * i + 0.05, min(height, 0.15), min(height, 0.15)])
            ax.set_axis_off()
            self.axes['base'].append(ax)
    
    def subplot(self, num_objects):
        if not(self.fig):
            self.fig = plt.figure(figsize=(13, 13))
        subplot_dim1 = np.ceil(np.sqrt(num_objects))
        subplot_dim2 = np.ceil(num_objects / subplot_dim1)                
        for axind in xrange(num_objects):
            axhandle = self.fig.add_subplot(subplot_dim1, subplot_dim2, axind + 1)
            self.axes['base'].append(axhandle)
    

    
    def alltoall(self, num_axes, num_objects):
        for ax_ind in range(num_axes):
            for obj_ind in range(num_objects):
                yield ax_ind, obj_ind
 
    def onetoall(self, num_axes, num_objects):
        for ax_ind in range(num_axes):
            yield ax_ind, range(num_objects)
    
    def onetoone(self, num_axes, num_objects):
        for ax_ind in range(num_axes):
            yield ax_ind, ax_ind   
              

           
    def contourfaces(self, where, how, timeseries):   
        axes = self.axes[where]
        for ax_ind, im_ind in self.mappings[how](len(axes), timeseries.samplepoints):
            axes[ax_ind].contourf(timeseries.shaped2D()[im_ind], [0.3, 1], colors=['r'], alpha=0.2)

    def contour(self, where, how, timeseries):   
        axes = self.axes[where]
        for ax_ind, im_ind in self.mappings[how](len(axes), timeseries.samplepoints): 
            axes[ax_ind].contour(timeseries.shaped2D()[im_ind], [0.3], colors=['w'])
    
    def overlay_image(self, where, how, timeseries):    
        axes = self.axes[where]
        for ax_ind, im_ind in self.mappings[how](len(axes), timeseries.samplepoints):
            ax = axes[ax_ind]
            im = timeseries.shaped2D()[im_ind]
            im_rgba = plt.cm.jet(im / 2 + 0.5)
            im_rgba[:, :, 3] = 0.8
            im_rgba[np.abs(im) < 0.1, 3] = 0 
            ax.imshow(im_rgba, aspect='equal', interpolation='nearest')
    
    def imshow(self, where, how, timeseries, title=False, colorbar=False):   
        axes = self.axes[where]
        for ax_ind, im_ind in self.mappings[how](len(axes), timeseries.samplepoints):
            ax = axes[ax_ind]
            im = ax.imshow(timeseries.shaped2D()[im_ind], aspect='equal', interpolation='nearest', cmap=plt.cm.jet) 
            ax.set_axis_off()
            if title:
                ax.set_title(timeseries.label_sample[im_ind])
            if colorbar:
                self.fig.colorbar(im, ax=ax)
                       
    def plot(self, where, how, timeseries):   
        axes = self.axes[where]
        for ax_ind, obj_ind in self.mappings[how](len(axes), timeseries.num_objects):
            ax = axes[ax_ind]
            ax.plot(timeseries.timecourses[:, obj_ind], '-')
            ax.xticklabels = []
            
         
    def add_labelshade(self, where, how, timeseries, rotate=True):     
        # create changing shade with changing label
        shade = []
        shade_color = 0
        labels = timeseries.label_sample                  
        reference_label = labels[0]
        for label in labels:
            if not(label == reference_label):
                reference_label = label
                shade_color = 1 - shade_color
            shade.append(shade_color)      
        shade = np.outer(np.array(shade), np.ones((timeseries.timepoints))).flatten()
        shade[np.hstack((np.array([0]), np.diff(shade))) == -1] = 1
        shade = np.hstack((shade, np.array([1]))) 
        
        axes = self.axes[where]
        for ax_ind, obj_ind in self.mappings[how](len(axes), timeseries.num_objects):
            ax = axes[ax_ind]
            axshade = collections.BrokenBarHCollection.span_where(
                                np.arange(len(shade) + 1) - 0.5,
                                *ax.get_ylim(), where=shade > 0,
                                facecolor='k', alpha=0.2)
            ax.add_collection(axshade)
        
    def add_samplelabel(self, where, timeseries, rotation='0', toppos=False):      
            for ax_ind in where:
                ax = self.axes['time'][ax_ind]
                ax.set_xticks(range(0, timeseries.samplepoints, timeseries.timepoints))
                ax.set_xticklabels(timeseries.label_sample)
            
                for tick in ax.xaxis.iter_ticks():
                    tick[0].label2On = toppos
                    tick[0].label1On = not(toppos)
                    tick[0].label2.set_rotation(rotation)
                    tick[0].label2.set_ha('left')
                    tick[0].label2.set_size('x-small')
                    tick[0].label2.set_stretch('extra-condensed')
                    tick[0].label2.set_family('sans-serif')
                    
    def add_axescolor(self, where, how, timeseries, ec='g', lw=2):
        axes = self.axes[where]
        for ax_ind, obj_ind in self.mappings[how](len(axes), timeseries.num_objects):
            ax = axes[ax_ind]
            for spine in ax.spines.values():
                spine.set_edgecolor(ec)
                spine.set_linewidth(lw)


'''
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
'''    
