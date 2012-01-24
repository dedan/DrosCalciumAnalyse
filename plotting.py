'''
Created on 18.02.2011

@author: jan
'''

import sys
import scipy.ndimage.filters as filter
import numpy as np
from matplotlib.cm import bone
from matplotlib.patches import Polygon
from matplotlib import collections
import matplotlib.nxutils as nx
import pickle
import scipy.stats
import Image
import pylab as plt
import matplotlib

class SelectRoi():
    
    def __init__(self, model=None):
        self.model = model
        print model['im'].shape
        num_bases = model['im'].shape[0]
        self.image_series = np.vsplit(model['im'], num_bases)
        print len(self.image_series)
        self.fig = plt.figure(figsize=(17, 5))
        #self.axbase = fig.add_subplot(111) 
        self.axbase = self.fig.add_axes([0.01, 0.1, 0.32, 0.8])

        '''
        if self.model['green']:
            self.axbase.imshow(self.model['green'], cmap=bone)
        if self.model['bg']:
            bg = bone(self.model['bg'])
            bg[:, :, 3] = (self.model['bg'] > 40).astype('float') * 0.7
            self.axbase.imshow(bg)
        '''
        #fig2=plt.figure()
        #self.axtime = fig2.add_axes([0.1,0.1,0.8,0.5])#
        self.axtime = self.fig.add_axes([0.35, 0.1, 0.6, 0.45])
        #self.axcol = fig.add_axes([0.02, 0.1, 0.05, 0.8])
        self.rois = {}
        self.pos = -1
        self.show_bases()
        self.show_contours()
        self.var = []
        temp = np.indices(self.model['shape'])
        self.grid = np.array(zip(temp[1].flatten(), temp[0].flatten()))
        self.degrouped_bases = []

    def show_contours(self):
        self.figcon = plt.figure()
        self.axcontour = self.figcon.add_subplot(111)
        self.axcontour.imshow(self.model['bg'], cmap=plt.cm.bone, extent=[0, self.model['shape'][1], self.model['shape'][0], 0])
        
        for j in range(len(self.image_series)):
            i = self.image_series[j]
            self.axcontour.contourf(filter.uniform_filter((i.reshape(self.model['shape'])), 3), [0.4, 1], colors=['r'], alpha=0.2)
            #self.axcontour.contourf(filter.uniform_filter((i.reshape(self.model['shape'])), 3), [-1, -0.4], colors=['b'], alpha=0.2)
        #self.axcontour.set_ylim(self.axcontour.get_ylim()[1], 0)       
    def show_bases(self):
        num_bases = len(self.image_series)
        subplot_dim1 = np.ceil(np.sqrt(num_bases))
        subplot_dim2 = np.ceil(num_bases / subplot_dim1)
        self.ax = {}
        self.figbase = plt.figure()
          
        for baseind, base in enumerate(self.image_series):
            axhandle = self.figbase.add_subplot(subplot_dim1, subplot_dim2, baseind + 1)
            axhandle.set_axis_off()
            self.ax[axhandle] = baseind
            axhandle.imshow(base.reshape(self.model['shape']), cmap=plt.cm.jet, aspect='equal', vmin= -1, vmax=1)
  
    def show_all(self):
        num_bases = len(self.image_series)
        height = 0.9 / num_bases
        self.figall = plt.figure(figsize=(20, 13))
        pixel = self.model['data'].shape[1]
        time = self.model['time'].copy()
        
        #------------------------------------------------ ''' === renorm === '''
        #---------------------------------------- print '!!! time renormed !!! '
        # norm = np.sqrt(np.sum(np.sum(np.abs(self.model['data'].reshape((-1, self.model['timepoints'], pixel))) ** 2, 2), 1))
        #-------- time = time.reshape((-1, self.model['timepoints'], num_bases))
        #---------------------------------------- #time[np.abs(time) < 0.02] = 0
        #-------------------------------------- time /= norm.reshape((-1, 1, 1))
        #---------------------------------- time = time.reshape((-1, num_bases))
        #------------------------------------------------ ''' ============== '''
        
        for i in range(num_bases):
            #ax = self.figall.add_axes([0.2, height * i + 0.05, 0.77, height])
            ax = self.figall.add_axes([0.4, height * i + 0.05, 0.55, height])
            self.showtime(i, ax, time)
            if i > 0:
                ax.set_xticklabels = []
        for i in range(num_bases):
            ax = self.figall.add_axes([0.05, height * i + 0.05, min(height, 0.15), min(height, 0.15)])
            im = self.image_series[i].reshape(self.model['shape'])
            ax.imshow(im, aspect='equal', interpolation='nearest', vmin= -1, vmax=1)
            ax.set_axis_off()
            
    def showim(self, pos):
        timepoints = self.model['timepoints']
        
        self.axbase.clear()
        self.axtime.clear()
        

        self.axbase.imshow(self.model['bg'], cmap=plt.cm.bone, extent=[0, self.model['shape'][1], self.model['shape'][0], 0])
        #bg = bone(self.model['bg'])
        #bg[:,:,3]=self.model['bg']>50
        #self.axbase.imshow(bg)
        #self.axbase.contour(self.model['bg'],[70], colors='y', linewidths=2)
        if True: #pos:    
            im = self.image_series[pos].reshape(self.model['shape'])
            print np.min(im), np.max(im)
            im_rgba = plt.cm.jet(im / 2 + 0.5)
            im_rgba[:, :, 3] = 1
            #im_rgba[np.abs(im) < 0.3, 3] = 0 
            bild = self.axbase.imshow(im_rgba, aspect='equal', interpolation='nearest') #, vmin= -1, vmax=1)
            #self.axbase.set_title('explained var %f2' % self.model['var'][pos])
            #matplotlib.colorbar.Colorbar(self.axcol, bild)
        #for j in vis.model['glom']:    
        
        
       
        
        
        #show time course
        '''
        time=np.hsplit(self.model['time'][:,self.pos], len(self.model['ind']))
        for i in range(54):
            self.axtime.plot(np.hstack([time[k] for k in np.where(self.model['ind']==i)[0]]))
        '''
        self.showtime(pos, self.axtime)
        
    def showtime(self, pos, tax, time=None, rotate=True):
        if time == None:
            time = self.model['time']
        names = self.model['names']
        shade = []
        to_add = 0
        reference_odor = self.model['names'][0]
        timepoints = self.model['timepoints']
        
        #------------------------------------------------------- '''REDUCTION'''
        #------------------------------------- time1, time2 = np.vsplit(time, 2)
        #---------------------------------------- names = names[:len(names) / 2]
        #------------------------------------------------------ print time.shape
        
        
        onsets = range(4, time.shape[0], timepoints)
        
        labels = [''] * len(names) * timepoints  
        labels[0] = reference_odor
        for odor_num, odor in enumerate(names):
            if not(odor == reference_odor):
                labels[odor_num * timepoints] = odor
                reference_odor = odor
                to_add = 1 - to_add
            shade.append(to_add)
        
        shade = np.outer(np.array(shade), np.ones((timepoints))).flatten()
        shade[np.hstack((np.array([0]), np.diff(shade))) == -1] = 1
        shade = np.hstack((shade, np.array([1]))) 
        if True: #pos:
            for i in onsets:
                tax.axvline(i, color='r', linestyle=':')
            #tax.plot(time1[:, pos], '-')
            #tax.plot(time2[:, pos], '-g')
            tax.plot(time[:, pos], '-')
            
            tax.set_ylim(np.min(time[:, pos]) * 1.1, np.max(time[:, pos]) * 1.1)
            shade = collections.BrokenBarHCollection.span_where(
                                np.arange(len(shade) + 1) - 0.5,
                                *tax.get_ylim(), where=shade > 0,
                                facecolor='k', alpha=0.2)
            tax.add_collection(shade)
            # add labels
            tax.set_xticks(range(0, len(labels)))
            tax.set_xticklabels(labels)
            tax.set_xlim([0, len(labels)])
            
            
            for tick in tax.xaxis.iter_ticks():
                tick[0].label2On = True
                tick[0].label1On = False
                if rotate:
                    tick[0].label2.set_rotation('30')
                tick[0].label2.set_ha('left')
                tick[0].label2.set_size('x-small')
                tick[0].label2.set_stretch('extra-condensed')
                tick[0].label2.set_family('sans-serif')

    def contourbase(self, pos, fig):
        fig.clf()
        stim = len(self.model['names'])
        for j in range(stim):
            dim1 = np.mod(j, 9)
            dim2 = np.ceil(j / 9. + 0.01)
            ax = fig.add_axes([0.02 + (dim1 * 0.1), 0.95 - (dim2 * 0.2), 0.1, 0.15])
            sig = np.mean(self.model['data'][(self.model['timepoints'] * j + 8):(self.model['timepoints'] * (j + 1) - 10)], 0).reshape(self.model['shape'])
            ax.imshow(sig / np.max(np.abs(sig)), cmap=plt.cm.gray)
            ax.contour(filter.uniform_filter(self.model['im'][pos].reshape(self.model['shape']), 2), [-0.4, 0.4], colors=['b', 'y'], linewidths=1)
            ax.set_axis_off()
            ax.set_title(self.model['names'][j], fontsize='x-small')
        tax = fig.add_axes([0.05, 0.02, 0.8, ax.get_position().ymin - 0.08])
        self.showtime(pos, tax, rotate=False)
        
    def overview(self):
        plt.figure() 
        plt.imshow(np.max(self.model['im'], 0).reshape(self.model['shape']), cmap=plt.cm.gray)
        for v in self.model['im']:
            plt.contour(filter.uniform_filter(v.reshape(self.model['shape']), 2), [-0.3, 0.3], colors=['b', 'y'])

        
                                     
def reorder(names, timepoints):
    ind = np.argsort(names)
    ind_full = np.hstack([np.arange(timepoints) + i * timepoints for i in ind])    
    return  ind, ind_full                

def initme(path, method, extra, reorderflag=True, timepoints=20):

    datafile = 'data' + extra
    model = {}
    model['timepoints'] = timepoints
    s = np.load(path + 'data' + extra + '.npy')    
    
    
    model['shape'] = s
    model['im'] = np.load(path + 'base' + method + extra + '.npy')
    print model['im'].shape
    norm = np.max(np.abs(model['im']), 1)
    model['im'] /= norm.reshape((-1, 1))
    #temp = Image.open(path + 'bg.png').convert(mode='L') #.resize((s[1], s[0]))
    model['bg'] = np.load(path + 'maxproj' + extra + '.npy').reshape(s)#np.asarray(temp)
    print model['bg'], type(model['bg'])
    model['time'] = np.load(path + 'time' + method + extra + '.npy') * norm
    timemin = np.min(model['time'], 0)
    timemax = np.max(model['time'], 0)
    '''
    signchange = -1 * (np.abs(timemin) > np.abs(timemax))
    signchange[signchange == 0] = 1
    print signchange.shape
    model['time'] *= signchange
    model['im'] *= signchange.reshape((-1, 1))
    '''
    print model['time'].shape
    #model['ind']=np.load('ind.npy')
    #names = db.make_table_dict('cd_id', ['Name'], 'MOLECULE_PROPERTIES')
    model['names'] = [i.split('.')[0] for i in pickle.load(open(path + 'ids.pik'))]
    print model['names']          
    #model['names'] = [str(i) for i in [8, 9, 14, 18, 19, 24, 29, 33]]
    var = np.ones(model['im'].shape[0])
    '''
    for base in range(model['im'].shape[0]):
        if np.sum(np.abs(model['im'][base])) > 0 :
            var.append(1)#np.sum(np.outer(np.abs(model['time'][:, base]), np.abs(model['im'][base, :]))))
        else: var.append(0) 
    '''
    #varind = np.argsort(var)[::-1]
    y = np.load(path + datafile + '.npy')   
    model['data'] = y
    yhat = np.dot(model['time'], model['im'])
    #explained_var = np.sum((yhat-np.mean(yhat,0))**2)/np.sum((y-np.mean(y,0))**2)*100
    #explained_var = (1 - np.sum(np.abs(yhat - y)) / np.sum(y)) * 100
    #print 'total explained var: ', explained_var
    #var = [var[i] for i in varind]
    var = np.array(var)
    model['var'] = var[var != 0]
    #varind = varind[var != 0]
    #model['im'] = model['im'][varind, :]
    #model['time'] = model['time'][:, varind]
    if reorderflag:
        ind, ind_full = reorder(model['names'], model['timepoints'])
        model['time'] = model['time'][ind_full]
        model['names'] = [model['names'][i] for i in ind]
        model['data'] = model['data'][ind_full]
    pickle.dump(model['names'], open('sortedids.pik', 'w'))
    np.save(path + 'sortedbase' + method + extra, model['im'])
    np.save(path + 'sortedtime' + method + extra, model['time'])
    vis = SelectRoi(model)
    return vis
        
