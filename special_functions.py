'''
Created on 14.09.2010

@author: jan
'''

from pipeline import *
import copy as cp
import dataimport
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.nxutils import points_inside_poly
from scipy.io import loadmat
import matplotlib.collections as collections
import nnmaRRI as nnma
import sklearn.decomposition as sld

reload(nnma)
                    
class DBnpyLoader():
   
    def __init__(self, data_selection, data_location, db=dataimport.instantJChemInterface()):
        '''data_selection is a dictionary that contains which property from which table is returned'''
        self.data_selection = data_selection
        self.data_location = data_location
        self.db = db
        self.props = self.db.make_table_dict(self.data_selection['key'],
                                             self.data_selection['properties'],
                                             self.data_selection['table'],
                                             col_select=self.data_selection['select_columns'],
                                             val_select=self.data_selection['select_values'])

    def __call__(self):
        timecourses, label_sample = [], []      
        for key in self.props.keys():
            odormeas = self.props[key]
            for single_trial in odormeas:         
                # create property dictionary
                prop_dict = {}        
                for propnum, prop in enumerate(self.data_selection['properties']):
                    prop_dict[prop] = single_trial[propnum]                     
                # load data
                filename = self.data_location + prop_dict['filename'] + '.npy'
                timecourses.append(np.load(filename))
                # create label
                trial_label = str(key) + '_' + str(prop_dict['fileID'])
                if prop_dict['concentration']:
                    trial_label += '_' + str(int(prop_dict['concentration']))
                if prop_dict['extraInfo']:
                    trial_label += '_' + str(prop_dict['extraInfo'])
                label_sample.append(trial_label)
        
        shape = tuple([int(i) for i in prop_dict['shape'].strip('()').split(', ')])        
        name = '_'.join([i.split('"')[1] for i in self.data_selection['select_values']])
        
        image_series = TimeSeries(np.vstack(timecourses), shape=shape, name=name,
                                  label_sample=label_sample)
        return image_series


class LoadRoi():
    
    """ ======== load in 2p selected ROIs ======== """
    
    def __init__(self, path): 
        self.path = path
    
    def _call__(self, shape):
        rois_vert = loadmat(self.path + 'nrois.mat')
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
        rois = TimeSeries('', name=[self.path], typ='mask',
                          label_sample=['ROI' + str(i) for i in range(rois.shape[0])])

    
class Project():
    
    def __call__(self, timeseries, base):               
        # create roi timecourse as convolution with data
        proj_timecourses = np.dot(timeseries.timecourses, base.T)
        out = timeseries.copy()
        out.timecourses = proj_timecourses
        out.label_objects = base.label_samples
        out.shape = (len(base.label_samples),)
        out.type = 'latent_series'
        out.base = base.copy()
        return out
        
  
'''      
class Response2DB(Block):
    
    def __init__(self, name='DBwriter', meas_id='', db=''):
        Block.__init__(self, name)
        self.db = db
        self.meas_id = meas_id
        
    def set_receiving(self):
        self.input = {'pca':'', 'latent':''} 
    
    def execute(self):
        #TODO: make measureID automatically
         
        try:
            id_response = max(self.db.fetch_data(['ID'], 'MOUSERESPONSE'))[0] + 1
        except:
            id_response = 1
        
        latent1_pos = np.argmax(np.abs((self.input['latent'].timecourses[:, 0])))
        latent1 = self.input['latent'].timecourses[latent1_pos, 0]
        

        dict_Activation = {}
        dict_molID = {}
        dict_glomID = {}
        dict_extra = {}
         
        for ind in range(self.input['pca'].timecourses.shape[1]):
            row = id_response + ind
            dict_glomID[row] = self.meas_id + '_' + str(ind)
            dict_Activation[row] = latent1 * self.input['pca'].timecourses[0, ind]
            try:
                split_pos = self.input['pca'].name[0].index('_')
                dict_molID[row] = int(self.input['pca'].name[0][:split_pos])
                dict_extra[row] = self.input['pca'].name[0][split_pos + 1:]
            except ValueError:
                dict_molID[row] = int(self.input['pca'].name[0])
                dict_extra[row] = ''
        
        self.db.write_data('MOUSERESPONSE', 'glomID', 's', 'ID', dict_glomID)         
        self.db.write_data('MOUSERESPONSE', 'Activation', 'f', 'ID', dict_Activation)    
        self.db.write_data('MOUSERESPONSE', 'molID', 'd', 'ID', dict_molID)
        self.db.write_data('MOUSERESPONSE', 'extra', 's', 'ID', dict_extra)

class Response2Mx(Block):
    
    def __init__(self, name='DBwriter', meas_id='', db=''):
        Block.__init__(self, name)
        self.db = db
        self.meas_id = meas_id
        
    def set_receiving(self):
        self.input = {'pca':'', 'latent':''} 
    
    def execute(self):
        #TODO: make measureID automatically
         
        latent1_pos = np.argmax(np.abs((self.input['latent'].timecourses[:, 0])))
        latent1 = self.input['latent'].timecourses[latent1_pos, 0]
        activation = latent1 * self.input['pca'].timecourses[0]
        out = self.input['latent'].copy()
        out.timecourses = activation
        out.variance = self.input['pca'].eigen[0]
        out.label_sample = [common_substr(out.label_sample).strip('_')] 
        self.sent_event(out)
        
class Activation2DB(Block):
    
    def __init__(self, name='DBwriter', db=dataimport.instantJChemInterface()):
        Block.__init__(self, name)
        self.db = db
        
    def set_receiving(self):
        self.input = {'pca':'', 'latent':''} 
    
    def execute(self):
        
        molID = self.db.make_table_dict('Name', ['cd_id'], 'MOLECULE_PROPERTIES')
        id_glom = max(self.db.fetch_data(['ID'], 'GLOMERULI'))[0] + 1
        id_activation = max(self.db.fetch_data(['ID'], 'ACTIVATION'))[0] + 1
        
        self.db.write_data('GLOMERULI', 'LOCATION', 's', 'ID', {id_glom: str(self.input['pca'].name).replace(' ', '')})
        self.db.write_data('GLOMERULI', 'measureID', 's', 'ID', {id_glom: self.input['pca'].origin[:-2]})
        self.db.write_data('GLOMERULI', 'explainedVar1', 'f', 'ID', {id_glom: self.input['pca'].eigen[0]})
        self.db.write_data('GLOMERULI', 'explainedVar2', 'f', 'ID', {id_glom: self.input['pca'].eigen[1]})
        
        latent1 = np.mean(self.input['latent'].timecourses[:, 0])
        latent2 = np.max(self.input['latent'].timecourses[:, 1]) - np.min(self.input['latent'].timecourses[:, 1])
        sign_latent2 = np.sign(np.argmin(self.input['latent'].timecourses[:, 1]) - np.argmax(self.input['latent'].timecourses[:, 1]))
        
        self.db.write_data('GLOMERULI', 'PCA1_mean', 'f', 'ID', {id_glom: abs(latent1)})
        self.db.write_data('GLOMERULI', 'PCA2_amp', 'f', 'ID', {id_glom: latent2})
         
        dict_glom = {}
        dict_file = {}
        dict_pca1 = {}
        dict_pca2 = {}
        dict_molID = {}
         
        for ind, measure in enumerate(self.input['pca'].id):
            row = id_activation + ind
            dict_glom[row] = id_glom
            dict_file[row] = self.input['pca'].origin + measure
            dict_pca1[row] = latent1 * self.input['pca'].timecourses[0, ind]
            dict_pca2[row] = sign_latent2 * latent2 * self.input['pca'].timecourses[1, ind]
            dict_molID[row] = molID[self.input['pca'].label_dim1[ind]][0][0]
            
        self.db.write_data('ACTIVATION', 'glomID', 'd', 'ID', dict_glom)    
        self.db.write_data('ACTIVATION', 'filename', 's', 'ID', dict_file)   
        self.db.write_data('ACTIVATION', 'PCA1', 'f', 'ID', dict_pca1)   
        self.db.write_data('ACTIVATION', 'PCA2', 'f', 'ID', dict_pca2)   
        self.db.write_data('ACTIVATION', 'molID', 'd', 'ID', dict_molID)

class VisPCA2(Block):
    
    def __init__(self, name='VisPCA', save_path='', db='', stop=False):
        Block.__init__(self, name)
        self.save_path = save_path
        self.numpcs = 4
        self.db = db
        self.stop = stop
        
    def set_receiving(self):
        self.input = {'loadings':'', 'pcs':''}
        
    def execute(self):
        molID = self.input['loadings'].name[0]
        print molID
         #molID = self.db.make_table_dict('CAS', ['cd_id'], 'MOLECULE_PROPERTIES')
        fig = plt.figure()
        fig.set_size_inches([21, 12])
        fig.suptitle(self.input['loadings'].name[0])
        numpc = self.numpcs #len(self.input['loadings'].eigen)
        
        #cm = plt.get_cmap('hsv')
        colors = [plt.cm.hsv(1. * i / (numpc + 1)) for i in range(numpc)]
               
        ax_pc = fig.add_axes([0.05, 0.05, 0.24, 0.8])
        ax_pc.set_xticks(range(0, len(self.input['loadings'].name)))
        ax_pc.set_xticklabels(self.input['loadings'].name)
        norm = []
        for pc in range(self.numpcs):
            argnorm = np.argmax(np.abs(self.input['pcs'].timecourses[:, pc]))
            norm.append(self.input['pcs'].timecourses[argnorm, pc])
            p = ax_pc.plot(self.input['pcs'].timecourses[:, pc] / norm[-1], color=colors[pc])
            if pc == 0:
                p[0].set_linewidth(3)
        
        ax_eig = fig.add_axes([0.29, 0.05, 0.01, 0.8])
        ax_eig.bar([0] * numpc, self.input['loadings'].eigen[:numpc], width=1, bottom=[0] + list(np.cumsum(self.input['loadings'].eigen[:numpc - 1])), color=colors)
        ax_eig.set_xticks([])
        ax_eig.set_yticks([])
        ax_eig.set_ylim([0, 1])
        
        
        subplotheight = 0.8 / self.numpcs
        ylowlims = []
        yhighlims = []
        ax_handles = []
        for pc in range(self.numpcs):
            ax_handles.append(fig.add_axes([0.33, 0.85 - (pc + 1) * subplotheight, 0.61, subplotheight]))
            ax_handles[pc].yaxis.tick_right()
            ax_handles[pc].yaxis.set_label_position('left')        
            ax_handles[pc].plot(self.input['loadings'].timecourses[pc] * norm[pc], 's-', color=colors[pc])
            ax_handles[pc].set_xticks([])
            ax_handles[pc].set_xticklabels([])   
            ylow, yhigh = ax_handles[pc].get_ylim()
            ylowlims.append(ylow)
            yhighlims.append(yhigh)
        ylow = np.min(ylowlims)
        yhigh = np.max(yhighlims)        
                  
        for pc in range(self.numpcs):
            ax_handles[pc].plot([-0.5, len(self.input['loadings'].timecourses[0]) + 0.5], [0, 0], 'k:')
            ax_handles[pc].set_ylim((ylow - 1, yhigh + 1))
            print ax_handles[pc].get_ylim()
            #ylow, yhigh = ax_handles[pc].get_ylim()
            ax_handles[pc].set_yticks([ylow, yhigh, 0])
            ax_handles[pc].set_ylabel('loading ' + str(pc + 1))
            ax_handles[pc].set_xticks(range(0, len(self.input['loadings'].timecourses[0])))
            ax_handles[pc].set_xlim([-0.5, len(self.input['loadings'].timecourses[0]) + 0.5])
            
            ax_handles[pc].grid(True)
        for tick in ax_handles[0].xaxis.iter_ticks():
            tick[0].label2On = True
            tick[0].label1On = False
            tick[0].label2.set_rotation('45')
            tick[0].label2.set_ha('left')
            tick[0].label2.set_size('small')
            tick[0].label2.set_stretch('extra-condensed')
            tick[0].label2.set_family('sans-serif')
        ax_handles[0].set_xticklabels(self.input['loadings'].label_dim1)
        
        if self.save_path:
            filename = self.input['loadings'].name[0] + '.png'
            dpi = 50
            if self.name:
                filename = self.name + '_' + filename
            fig.savefig(self.save_path + filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        
        if self.stop:
            plt.show()
            raw_input()

class VisPCA(Block):
    
    def __init__(self, name='VisPCA', numpcs=3, locations='', save_path='', stop=False):
        Block.__init__(self, name)
        self.numpcs = numpcs
        self.locations = locations
        self.save_path = save_path
        self.stop = stop
        
    def set_receiving(self):
        self.input = {'activations':'', 'loadings':'', 'pcs':''}
        
    def execute(self):
        fig = plt.figure()
        fig.set_size_inches([21, 12])
        
        cm = plt.get_cmap('hsv')
        colors = [cm(1. * i / (self.numpcs + 3)) for i in np.linspace(0, self.numpcs + 3, self.numpcs + 3)]
        
        datalength = len(self.input['activations'].timecourses)
        num_odors = len(self.input['activations'].name)
        timepoints = datalength / num_odors
        
        
        ax = fig.add_axes([0.33, 0.55, 0.61, 0.2])
        ax.plot(self.input['activations'].timecourses, '-')        
        
        ax.set_xticks(range(0, datalength, timepoints))
        ax.set_xlim([0, datalength])
        for tick in ax.xaxis.iter_ticks():
            tick[0].label2On = True
            tick[0].label1On = False
            tick[0].label2.set_rotation('30')
            tick[0].label2.set_ha('left')
            tick[0].label2.set_size('x-small')
            tick[0].label2.set_stretch('extra-condensed')
            tick[0].label2.set_family('sans-serif')
            
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('left')
        ax.set_ylabel('Delta R / R')
        
        ax.grid(True)
        
        labels = [''] * len(self.input['activations'].id)  
        shade = []
        shade_pc = []
        add_shade = True
        reference_odor = ''
        for odor_num, odor in enumerate(self.input['activations'].name):
            if not(odor == reference_odor):
                reference_odor = odor
                add_shade = not(add_shade)
                labels[odor_num] += odor
            shade += [add_shade] * timepoints
            shade_pc += [add_shade]
        shade = np.array(shade + [False])
        shade_pc = np.array(shade_pc + [False])
        shade[np.nonzero(shade)[0] + 1] = True
        shade_pc[np.nonzero(shade_pc)[0] + 1] = True
        a = collections.BrokenBarHCollection.span_where(range(len(self.input['activations'].timecourses)),
                                                        *ax.get_ylim(), where=shade, facecolor='k', alpha=0.2)
        ax.add_collection(a)
        ax.set_xticklabels(labels)
        
        ax_pc = fig.add_axes([0.05, 0.05, 0.24, 0.5])
        for pc in range(self.numpcs):
            ax_pc.plot(self.input['pcs'].timecourses[:, pc], color=colors[pc])
        
        ax_eig = fig.add_axes([0.29, 0.05, 0.01, 0.5])
        print self.input['loadings'].eigen[0]
        ax_eig.bar([0] * (self.numpcs + 3), self.input['loadings'].eigen[:self.numpcs + 3], width=1, bottom=[0] + list(np.cumsum(self.input['loadings'].eigen[:self.numpcs + 2])), color=colors)
        ax_eig.set_xticks([])
        ax_eig.set_yticks([])
        ax_eig.set_ylim([0, 1])
        
        ax_pict = fig.add_axes([0.05, 0.6, 0.25, 0.2])
        ax_pict.set_xlim([0, 167])
        ax_pict.set_ylim([127, 0])
        ax_pict.imshow(self.locations.reshape(128, 168), extent=(0, 167, 127, 0))
        ax_pict.plot(self.input['activations'].label_dim1[1], self.input['activations'].label_dim1[0], 'o' , mew=2, mec='w', mfc='none', ms=10)
        ax_pict.set_axis_off()
        
        
        subplotheight = 0.5 / self.numpcs
        ylowlims = []
        yhighlims = []
        ax_handles = []
        for pc in range(self.numpcs):
            ax_handles.append(fig.add_axes([0.33, 0.55 - (pc + 1) * subplotheight, 0.61, subplotheight]))
            ax_handles[pc].yaxis.tick_right()
            ax_handles[pc].yaxis.set_label_position('left')
         
            ax_handles[pc].plot(np.arange(0.5, num_odors + 0.5), self.input['loadings'].timecourses[pc], 's-', color=colors[pc])
            ax_handles[pc].set_xlim([0, num_odors + 0.01])
            ax_handles[pc].set_xticks([])
            ax_handles[pc].set_xticklabels([])
            ylow, yhigh = ax_handles[pc].get_ylim()
            ylowlims.append(ylow)
            yhighlims.append(yhigh)
                  
        for pc in range(self.numpcs):
            ax_handles[pc].set_ylim([min(ylowlims), max(yhighlims)])
            ax_handles[pc].set_yticks([min(ylowlims) + 0.1, max(yhighlims) - 0.1, 0])
            ax_handles[pc].set_ylabel('loading ' + str(pc + 1))
            a = collections.BrokenBarHCollection.span_where(range(num_odors),
                         *ax_handles[pc].get_ylim(), where=shade_pc, facecolor='k', alpha=0.2)
            ax_handles[pc].add_collection(a)
            ax_handles[pc].plot([0, num_odors], [0, 0], 'k:')
        
        if self.save_path:
            filename = 'response' + str(self.input['loadings'].name).replace(' ', '') + '.png'
            if self.name:
                dpi = 50
                filename = self.name + '_' + filename
            fig.savefig(self.save_path + filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        if self.stop:
            plt.show()                   

class Visualize(Block):
    
    def __init__(self, internal_mask_input=False, name='', alpha_reverse=False,
                  alpha_scale=3, alpha_scales='', save_path='', color_scale='', stop=False, ms=10,
                  internal_mask2=True, bg=None, name_dict={}):
        self.internal_mask_input = internal_mask_input
        self.internal_mask2 = internal_mask2
        Block.__init__(self, name)
        self.alpha_reverse = alpha_reverse
        self.alpha_scale = alpha_scale
        self.save_path = save_path
        self.color_scale = color_scale
        self.stop = stop
        self.ms = ms
        self.bg = bg
        self.alpha_scales = alpha_scales
        self.name_dict = name_dict
        
    def set_receiving(self):
        self.input = {'image_series':'', 'pointmask': self.internal_mask_input, 'pointmask2':self.internal_mask2}
    
        
    def execute(self):
        no_images = self.input['image_series'].timecourses.shape[0]
       
        subplot_dim1 = np.ceil(np.sqrt(no_images))
        subplot_dim2 = np.ceil(no_images / subplot_dim1)
        fig = plt.figure()
        if not(self.color_scale == 'ind'):
            if self.color_scale:
                vmin, vmax = self.color_scale
            else:
                vmax = np.max(self.input['image_series'].timecourses)
                vmin = np.min(self.input['image_series'].timecourses)
            if self.alpha_scales:
                amin, amax = self.alpha_scales
            else:
                amax, amin = (vmax, vmin)                 
            normer = Normalize(vmin=vmin, vmax=vmax)
            normer2 = Normalize(vmin=amin, vmax=amax)
            im = plt.imshow(np.zeros(self.input['image_series'].timecourses.shape),
                    vmin=vmin, vmax=vmax, cmap=cm.jet)
        
        ax = []
        for image_ind, image in enumerate(self.input['image_series'].shaped2D()):
            if self.color_scale == 'ind':
                vmax = np.max(image)
                vmin = np.min(image)
                normer = Normalize(vmin=vmin, vmax=vmax)
                normer2 = Normalize(vmin=vmin, vmax=vmax)
            ax.append(plt.subplot(int(subplot_dim2), int(subplot_dim1), int(image_ind + 1)))
            rgba_map = cm.ScalarMappable(norm=normer, cmap=cm.jet)    
            rgba_im = rgba_map.to_rgba(image)
            alpha = 1 - normer2(image) if self.alpha_reverse else normer2(image) 
            alpha[alpha < 0] = 0
            alpha[alpha > 1] = 1
            alpha = alpha ** self.alpha_scale
            alpha[alpha < 0.5] = 0.5
            rgba_im[:, :, 3] = alpha
            plt.imshow(rgba_im)
            plt.title(self.name_dict[int(self.input['image_series'].label_sample[image_ind].split('_')[0])][0][0], fontsize=10)
            if self.bg != None:
                plt.contour(self.bg, [100], extent=(0, 84, 0, 64), linewidth=2)
            #key = self.input['image_series'].label_sample[image_ind].strip('_')
            #plt.title(self.name_dict[int(key)][0][0])

            
            
        if not(self.input['pointmask'] == True):
            for image_ind, mask in enumerate(self.input['pointmask'].shaped2D()):    
                mask_pos = np.where(mask)        
                plt.axes(ax[image_ind])
                plt.plot(mask_pos[1], mask_pos[0], 'wx', ms=self.ms) #, mfc='none', ms=15)
                plt.xlim((0, self.input['image_series'].shape[1]))
                plt.ylim((self.input['image_series'].shape[0], 0))
        for sax in ax:     
            sax.set_axis_off()
            
        if not(self.input['pointmask2'] == True):
            for image_ind, mask in enumerate(self.input['pointmask2'].shaped2D()):    
                mask_pos = np.where(mask)        
                plt.axes(ax[image_ind])
                plt.plot(mask_pos[1], mask_pos[0], 'o', ms=self.ms , mfc='none', mec='w')
                plt.xlim((0, self.input['image_series'].shape[1]))
                plt.ylim((self.input['image_series'].shape[0], 0))
        for sax in ax:     
            sax.set_axis_off()      
        if not(self.color_scale == 'ind'):
            cax = fig.add_axes([0.91, 0.1, 0.01, 0.8])
            plt.colorbar(im, cax, orientation='vertical', format='%.1e')
        if self.save_path:
            if self.name:
                dpi = 300
                filename = self.name + '_' + common_substr(self.input['image_series'].name)
            fig.savefig(self.save_path + filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        if self.stop:
            raw_input()
        self.sent_event(fig)

class PCAProjection(Block):
    
    def __init__(self, name='PCA', variance=None):
        Block.__init__(self, name)
        self.variance = variance
        
    def set_receiving(self):
        self.input = {'image_series':''}
    
    def execute(self):
        self.pca = sld.PCA(self.variance)
        self.pca.fit(self.input['image_series'].timecourses)
        out = self.input['image_series'].copy()
        #back and forth projection
        out.timecourses = np.dot(np.dot(out.timecourses, self.pca.components_.T), self.pca.components_)
        out.eigen = self.pca.explained_variance_ratio_
        self.sent_event(out)
        
class PCA(Block):
    
    def __init__(self, name='PCA', components=None):
        Block.__init__(self, name)
        self.components = components
        
    def set_receiving(self):
        self.input = {'image_series':''}
    
    def execute(self):
        self.pca = sld.PCA(n_components=self.components)
        self.pca.fit(self.input['image_series'].timecourses)
        out = self.input['image_series'].copy()
        out.type = 'latent_series'
        out.base = self.pca.components_
        out.time = self.pca.transform(self.input['image_series'].timecourses)
        out.timecourses = out.base
        out.eigen = self.pca.explained_variance_ratio_
        self.sent_event(out)
        
            
    def execute(self):
        image = self.input['image_series'].timecourses
        if image.shape[1] > image.shape[0]:
            image = image - np.mean(image, 0)
            cov = np.dot(image, image.T) / image.shape[0]
            evalue, evec = scilin.eigh(cov)
            trans_evec = np.dot(image.T, evec) / np.sqrt(image.shape[0] * evalue)
            trans_evec = trans_evec.T[::-1]
            evalue = evalue[::-1]
        else:
            image = image - np.mean(image, 1).reshape((-1,1))
            cov = np.dot(image.T, image) / image.shape[1]
            evalue, trans_evec = scilin.eigh(cov)
        out = self.input['image_series'].copy()
        out.type = 'basis'
        out.timecourses = trans_evec
        print 'pca output shape: ', trans_evec.shape
        out.eigen = evalue / np.sum(evalue)
        self.sent_event(out)
                 

class sICA(Block):
    
    def __init__(self, name='sICA', latent_series=False, variance=None):
        self.latent_series = latent_series
        Block.__init__(self, name)
        self.variance = variance
        
    def set_receiving(self):
        if self.latent_series:
            self.input = {'latent_series':''}
        else:
            self.input = {'image_series':''}
    
    def execute(self):
        if self.variance > 1:
            self.pca = sld.PCA(self.variance)
        else:    
            self.pca = sld.PCA()
        if self.latent_series:
            base = self.pca.fit_transform(self.input['latent_series'].base.T)
            time = np.dot(self.pca.components_, self.input['latent_series'].time.T)#self.pca.transform(self.input['latent_series'].time)
            print 'lat', base.shape, time.shape
        else:
            print 'imag'
            base = self.pca.fit_transform(self.input['image_series'].timecourses.T)
            base2 = np.dot(self.pca.components_, self.input['image_series'].timecourses)
            time = self.pca.components_
        normed_base = base / np.sqrt(self.pca.explained_variance_)
        normed_time = time * np.sqrt(self.pca.explained_variance_.reshape((-1, 1)))
        if self.variance:
            ind = np.cumsum(self.pca.explained_variance_ratio_) <= self.variance
            print ind
            normed_base = normed_base[:, ind]
            normed_time = normed_time[ind, :]
            base2 = base2[:, ind]
        
        print 'BASEshape', normed_base.shape   
        self.ica = sld.FastICA(whiten=False)
        self.ica.fit(normed_base)
        
        if self.latent_series:
            out = self.input['latent_series'].copy()
        else:
            out = self.input['image_series'].copy()
        out.base = self.ica.components_.T
        #out.base2 = self.ica.transform(base2.T)
        #out.time2 = self.ica.transform(time)
        new_norm = np.diag(out.base[:, np.argmax(np.abs(out.base), 1)])
        print out.base.shape, new_norm.shape
        out.base /= new_norm.reshape((-1, 1))
        print np.max(out.base)
        out.time = self.ica.transform(normed_time.T)
        out.time *= new_norm
        out.timecourses = out.base
        out.type = 'latent_series'
        self.sent_event(out)

class ICA(Block):
    
    def __init__(self, name='ICA', components=None):
        Block.__init__(self, name)
        self.components = components
        
    def set_receiving(self):
        self.input = {'image_series':''}
               
    def execute(self):
        
        timecourses = self.input['image_series'].timecourses
        fcanode = mdp.nodes.FastICANode()
        fcanode.train(timecourses)
        fcanode.stop_training()
       
        if self.components:
            self.ica = sld.FastICA(self.components)
        else:
            self.ica = sld.FastICA()
        #print self.input['image_series'].timecourses.T.shape
        self.ica.fit(self.input['image_series'].timecourses)
        projmatrix = self.ica.unmixing_matrix_
        print projmatrix.shape
                #latent = np.dot(timecourses, fcanode.get_projmatrix())
        #projection = np.dot(self.input['projection'], fcanode.get_projmatrix())
        out = self.input['image_series'].copy()
        #out.timecourses = fcanode.get_projmatrix()
        out.timecourses = projmatrix
        self.sent_event(out)
'''
                    

        
'''              
class Projection(Block):
    
    def __init__(self, components, name='projection', proj_dim=0):
        Block.__init__(self, name)
        self.components = components
        self.dim = proj_dim
        
    def set_receiving(self):
        self.input = {'image_series':'', 'base':''}
              
    def execute(self):
        out = self.input['image_series'].copy()
        print self.input['image_series'].timecourses.shape, self.input['base'].timecourses.shape
        if self.dim == 0:
            out.timecourses = np.dot(out.timecourses, self.input['base'].timecourses[:self.components].T)
        elif self.dim == 1:
            out.timecourses = np.dot(out.timecourses, self.input['base'].timecourses[:self.components])
        print 'projection output shape: ', out.timecourses.shape
        self.sent_event(out)

class Align(Block):
    
    def __init__(self, name='align'):
        Block.__init__(self, name)
        
    def set_receiving(self):
        self.input = {'image_series':''}
              
    def execute(self):
        x = []
        y = []
        xt = yt = 0
        base_image = self.input['image_series'].shaped2D()[0]
        for image in self.input['image_series'].shaped2D():
            xt, yt = align.align_mutual_information(image, base_image, xt, yt)
            print xt, yt
            x.append(xt)
            y.append(yt)
        out = self.input['image_series'].copy()
        out.timecourses = np.array([x, y]).transpose()
        self.sent_event(out)

'''      
"============================================================================="
''' helper functions'''
    
def common_substr(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0]) - i + 1):
                if j > len(substr) and is_substr(data[0][i:i + j], data):
                    substr = data[0][i:i + j]
        return substr
    else:
        return data[0]

def is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True

class GUISelector:
    def __init__(self, mask):
        self.mask = mask
    
    def __call__(self, event):
        x, y = (event.xdata, event.ydata)
        self.mask[y, x] = True
        plt.text(int(x), int(y), 'x', color='w', ha='center', va='center')
