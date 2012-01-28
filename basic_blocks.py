
from pipeline import *
import numpy as np
import scipy.ndimage.filters as filter
import sklearn.decomposition as sld
from scipy.spatial.distance import pdist


class Command(Block):
    ''' executes the ommited function '''
    
    def __init__(self, name='Command', command=''):
        Block.__init__(self, name)
        self.command = command

    def set_receiving(self):
        self.input = {'image_series':''}
        
    def execute(self):
        out = self.input['image_series'].copy()
        out.timecourses = self.command(out.timecourses)
        self.sent_event(out)         
            
class Filter(Block):
    ''' filter scelet'''
    
    def __init__(self, name='Filter', flat=False, normalize=False):
        Block.__init__(self, name)
        self.flat = flat
        self.normalize = normalize
        
    def set_receiving(self):    
        self.input = {'image_series':''}
        
    def execute(self):    
        filtered_images = []
        if self.flat:
            print 'flattened'
            for image in self.input['image_series'].timecourses:
                filtered_images.append(self.filter(image))
        else:
            for image in self.input['image_series'].shaped2D():
                im = self.filter(image)
                if self.downscale:
                    im = im[::self.downscale, ::self.downscale]
                filtered_images.append(im.flatten())
        shape = im.shape
        out = self.input['image_series'].copy()
        out.shape = shape
        out.timecourses = np.vstack(filtered_images)
        if self.normalize:
            test = np.zeros((200, 200))
            test[100, 100] = 1
            test = self.filter(test)
            out.timecourses /= test[100, 100] 
        print self.name, out.label_sample
        self.sent_event(out)
        
    def filter(self):
        pass
    
class GaussFilter(Filter):
    ''' low_pass with gaussian filter'''
    
    def __init__(self, sigma=2, flat=False, normalize=False, downscale=None):
        Filter.__init__(self, name='GaussFilter', flat=flat, normalize=normalize)
        self.sigma = sigma
        self.downscale = downscale
        
    def filter(self, image):
        return filter.gaussian_filter(image, self.sigma, mode='mirror')

class UniformFilter(Filter):
    ''' low_pass with gaussian filter'''
    
    def __init__(self, size=10, flat=False, downscale=''):
        Filter.__init__(self, name='UniformFilter', flat=flat)
        self.size = size
        self.downscale = downscale
        
    def filter(self, image):
        return filter.uniform_filter(image, self.size)

class MedianFilter(Filter):
    ''' low_pass with median filter'''
    
    def __init__(self, size=10, flat=False, downscale=''):
        Filter.__init__(self, name='MedianFilter', flat=flat)
        self.size = size
        self.downscale = downscale
        
    def filter(self, image):
        return filter.median_filter(image, self.size)

class PercentileFilter(Filter):
    ''' low_pass with percentile filter'''
    
    def __init__(self, size=10, percentile=80, flat=False):
        Filter.__init__(self, name='PercentileFilter', flat=flat)
        self.size = size
        self.percentile = percentile
        
    def filter(self, image):
        return filter.percentile_filter(image, self.percentile, self.size)

class MaximumFilter(Filter):
    ''' low_pass with maximum filter'''
    
    def __init__(self, size=10, flat=False):
        Filter.__init__(self, name='MaximumFilter', flat=flat)
        self.size = size
        
    def filter(self, image):
        return filter.maximum_filter(image, self.size)

class CutOut(Block):
    ''' cuts out images of a series from range[0] to range[1]'''
    
    def __init__(self, range, name='CutOut'):
        Block.__init__(self, name)
        self.range = range
        
    def set_receiving(self):    
        self.input = {'image_series':''}
        
    def execute(self):
        view = self.input['image_series'].timecourses
        image_cut = self.input['image_series'].copy()
        image_cut.timecourses = view[self.range[0]:self.range[1]].copy()
        self.sent_event(image_cut)

class Mean(Block):
    ''' splits the array in parts and calculates their means'''
    
    def __init__(self, parts=1, name='mean'):
        Block.__init__(self, name)
        self.parts = parts
        
    def set_receiving(self):    
        self.input = {'image_series':''}
        
    def execute(self):
        print self.input['image_series'].timecourses.shape
        averaged_im = [np.mean(im, 0) for im in np.vsplit(self.input['image_series'].timecourses, self.parts)]
        out = self.input['image_series'].copy()
        out.timecourses = np.vstack(averaged_im)
        print 'before', out.label_sample
        out.label_sample = [common_substr(out.label_sample).strip('_')]
        print 'after', out.label_sample
        self.sent_event(out)

class spatialMean(Block):
    ''' splits the array in parts and calculates their means'''
    
    def __init__(self, parts=1, name='mean'):
        Block.__init__(self, name)
        self.parts = parts
        
    def set_receiving(self):    
        self.input = {'image_series':''}
        
    def execute(self):
        averaged_im = [np.mean(im, 1) for im in np.hsplit(self.input['image_series'].timecourses, self.parts)]
        out = self.input['image_series'].copy()
        out.timecourses = np.hstack(averaged_im)
        print out.timecourses.shape
        self.sent_event(out)        

class Add(Block):
    ''' adds the image_series'''
    
    def __init__(self, name='Add'):
        Block.__init__(self, name)
        
    def set_receiving(self):    
        self.input = {'image_series':''}
        
    def execute(self):
        oldtype = self.input['image_series'].type
        out = self.input['image_series'].copy()
        out.timecourses = np.sum(self.input['image_series'].timecourses.astype('float'), 0)
        out.type = 'hist' if  oldtype == 'mask' else oldtype                  
        self.sent_event(out)

class RelativeChange(Block):
    ''' gives relative change of image_series to base_image '''
    
    def __init__(self, name='rel_change'):
        Block.__init__(self, name)
        
    def set_receiving(self):    
        self.input = {'image_series':'', 'base_image':''}
        
    def execute(self):    
        relative_change = self.input['image_series'].copy()
        relative_change.timecourses = ((self.input['image_series'].timecourses - self.input['base_image'].timecourses)
                           / self.input['base_image'].timecourses)
        self.sent_event(relative_change)

class Prod(Block):
    ''' gives product of two image_series,
        this corresponds to logical and of two masks 
    '''
    
    def __init__(self, name='Prod'):
        Block.__init__(self, name)
        
    def set_receiving(self):    
        self.input = {'image_series':'', 'image_series_2':''}
        
    def execute(self):    
        prod = self.input['image_series'].copy()
        prod.timecourses = (self.input['image_series'].timecourses * 
                      self.input['image_series_2'].timecourses)
        self.sent_event(prod)

class Change(Block):
    ''' gives absolute change of image_series to base_image '''
    
    def __init__(self, name='Change'):
        Block.__init__(self, name)
        
    def set_receiving(self):    
        self.input = {'image_series':'', 'base_image':''}
        
    def execute(self):    
        change = self.input['image_series'].copy()
        change.timecourses -= self.input['base_image'].timecourses
        print change.shape
        self.sent_event(change)

class Splitter(Block):
    
    def __init__(self, name='Splitter', hsplit=False):
        Block.__init__(self, name)
        self.hsplit = hsplit
                
    def set_receiving(self):
        self.input = {'image_series':''} 
                
    def execute(self):
        num_pieces = (len(self.input['image_series'].name) 
                      if not(self.hsplit) else len(self.input['image_series'].label_dim1)) 
        print num_pieces, self.input['image_series'].timecourses.shape
        if self.hsplit:
            data = np.hsplit(self.input['image_series'].timecourses, num_pieces)
        else:
            data = np.vsplit(self.input['image_series'].timecourses, num_pieces)
        for piece in range(num_pieces):
            out = self.input['image_series'].copy()
            out.timecourses = data[piece]
            if self.hsplit:
                out.label_dim1 = out.label_dim1[piece]
            else:
                out.name = out.name[piece]
                out.id = out.id[piece]
            print out.timecourses.shape
            self.sent_event(out)

class Collector(Block):
    
    def __init__(self, finish_signal, name='Collector', hstack=False):
        Block.__init__(self, name)
        self.finish_signal = finish_signal
        self.finish_count = 0
        self.buffer = []
        self.image_container = []
        self.hstack = hstack
        
    def set_receiving(self):
        for key in self.input:
            self.input[key] = '' 
    
    def receive_event(self, event):
        if event.key == 'signal':
            self.process_signal(event)
        elif not(self.input[event.key]):
            #print self.name, ' filled'
            self.input['image_series'] = event.value
            print self.name, ' executed'
            self.execute()
            self.set_receiving()
        else:
            print self.name, ": !!!FullBuffer!!!"
            raise FullBufferException()
                
    def execute(self): 
        self.buffer.append(self.input['image_series'].timecourses.copy())
        if not(self.image_container):
            self.image_container = self.input['image_series'].copy()
            #self.image_container.label_sample = self.input['image_series'].name
        else:
            self.image_container.label_sample += self.input['image_series'].label_sample
            if not(self.image_container.label_dim1 == self.input['image_series'].label_dim1):
                print "!!!!! Labels are not equal !!!!!"
                print self.image_container.label_dim1
                print self.input['image_series'].label_dim1
    
    def process_signal(self, signal_event):
        print self.name, ' received: ', signal_event.value
        if signal_event.value == self.finish_signal:
            self.finish_count += 1
            if self.finish_count == len(self.input):
                if self.hstack:
                    self.image_container.timecourses = np.hstack(self.buffer)
                else:
                    self.image_container.timecourses = np.vstack(self.buffer)
                self.buffer = []
                self.image_container.name = [common_substr(self.image_container.label_sample)]
                self.sent_event(self.image_container)
                self.image_container = []
                self.finish_count = 0   
                self.sent_signal(signal_event.value)
        else:
            self.sent_signal(signal_event.value)

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
        
    '''            
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
    '''              

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

class SampleSimilarity(Block):
    """calculate similarity of samples with the same Labels

    return mask such that only samples with a self similarity below threshold are included.
    Outliers in the self similarity matrix are removed by a greedy elimination as long
    as the mean exceeds the threshold.
    """

    def __init__(self, threshold, name='SampleSimilarity', metric='correlation'):
        Block.__init__(self, name)
        self.metric = metric
        self.threshold = threshold


    def execute(self):
        image_series = self.input['image_series']
        sample_set = set(image_series.label_sample)
        mask = np.zeros(len(image_series.label_sample), dtype='bool')
        for label in sample_set:
            label_index = [i for i, tmp_label in enumerate(image_series.label_sample) if tmp_label == label]
            mean_distance = np.mean(pdist(image_series.timecourses[label_index], self.metric))
            while mean_distance > self.threshold and len(label_index) > 2:
                new_means = []
                for l in label_index:
                    tmp_label_index = [ll for ll in label_index if not ll == l]
                    distances = pdist(image_series.timecourses[tmp_label_index], self.metric)
                    new_means.append(np.mean(distances))
                min_index = np.argmin(new_means)
                label_index.pop(min_index)
                mean_distance = new_means[min_index]
            if mean_distance < self.threshold:
                mask[label_index] = True
        out = image_series.copy()
        out.timecourses = mask
        self.sent_event(out)         


# helper functions 
    
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
