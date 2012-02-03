from pipeline import TimeSeries
import numpy as np
import sklearn.decomposition as sld
from scipy.spatial.distance import pdist

       
            
class Filter():
    ''' filter series with filterop in 2D'''
    
    def __init__(self, filterop, args, downscale):
        self.downscale = downscale
        self.filterop = filterop
        self.args = args
        
               
    def __call__(self, timeseries):    
        filtered_images = []
        for image in timeseries.shaped2D():
            im = self.filterop(image, **self.args)
            if self.downscale:
                im = im[::self.downscale, ::self.downscale]
            filtered_images.append(im.flatten())
        shape = im.shape
        out = timeseries.copy()
        out.shape = shape
        out.timecourses = np.vstack(filtered_images)
        return out

class CutOut():
    ''' cuts out images of a series from range[0] to range[1]'''
    
    def __init__(self, cut_range):
        self.cut_range = cut_range
      
    def __call__(self, timeseries):
        image_cut = timeseries.copy()
        image_cut.set_timecourses(image_cut.trial_shaped()[:, self.cut_range[0]:self.cut_range[1]])
        return image_cut

class TrialMean():
    ''' splits each trial in equal parts and calculates their means'''
    
    def __init__(self, parts=1):
        self.parts = parts
       
    def __call__(self, timeseries):
        splits = np.vsplit(timeseries.timecourses, self.parts * timeseries.num_trials)
        averaged_im = [np.mean(im, 0) for im in splits]
        out = timeseries.copy()
        out.timecourses = np.vstack(averaged_im)
        return out

class RelativeChange():
    ''' gives relative change of trials to base_image '''
    
    def __init__(self):
        pass
        
    def __call__(self, timeseries, baseseries):    
        relative_change = timeseries.copy()
        relative_change.set_timecourses((timeseries.trial_shaped() - baseseries.trial_shaped())
                           / baseseries.trial_shaped())
        return relative_change

class Combine():
    ''' combines two imageseries by cominefct '''
    
    def __init__(self, combine_fct):
        self.combine_fct = combine_fct
                    
    def __call__(self, timeseries1, timeseries2):    
        change = timeseries1.copy()
        change.timecourses = self.combine_fct(timeseries1.timecourses, timeseries2.timecourses)
        return change

class PCA():
    
    def __init__(self, variance=None):
        self.variance = variance
    
    def __call__(self, timeseries):
        self.pca = sld.PCA(n_components=self.variance)
        self.pca.fit(timeseries.timecourses)
        out = timeseries.copy()
        out.type = 'latent_series'
        out.label_objects = ['mode' + str(i) for i in range(self.pca.components_.shape[0])]
        out.shape = (len(out.label_objects),)
        out.timecourses = self.pca.transform(timeseries.timecourses)
        out.eigen = self.pca.explained_variance_ratio_
        out.base = TimeSeries(self.pca.components_, shape=self.input['image_series'].shape, name='base',
                              label_sample=out.label_objects)
        return out
        
class sICA():
    
    def __init__(self, variance=None):
        self.variance = variance
           
    def __call__(self, timeseries):
        self.pca = sld.PCA(n_components=self.variance)      
        base = self.pca.fit_transform(timeseries.timecourses.T)
        time = self.pca.components_
        normed_base = base / np.sqrt(self.pca.explained_variance_)
        normed_time = time * np.sqrt(self.pca.explained_variance_.reshape((-1, 1)))
 
        self.ica = sld.FastICA(whiten=False)
        self.ica.fit(normed_base)
        

        out = timeseries.copy()
        base = self.ica.components_.T
        new_norm = np.diag(base[:, np.argmax(np.abs(base), 1)])
        base /= new_norm.reshape((-1, 1))
        
        time = self.ica.transform(normed_time.T)
        time *= new_norm
        
        timesign = np.sign(np.sum(time, 0))
        time *= timesign
        base *= timesign.reshape((-1, 1))
        
        out.timecourses = time
        out.label_objects = ['mode' + str(i) for i in range(base.shape[0])]
        out.shape = (len(out.label_objects),)
        out.typ = 'latent_series'  
        out.base = TimeSeries(base, shape=timeseries.shape, name='base',
                              label_sample=out.label_objects)
        
        return out

class SampleSimilarity():
    """calculate similarity of samples with the same Labels

    return mask such that only samples with a self similarity below threshold are included.
    Outliers in the self similarity matrix are removed by a greedy elimination as long
    as the mean exceeds the threshold.
    """

    def __init__(self, threshold, metric='correlation'):
        self.metric = metric
        self.threshold = threshold

    def __call__(self, timeseries):
        sample_set = set(timeseries.label_sample)
        mask = np.zeros(len(timeseries.label_sample), dtype='bool')
        for label in sample_set:
            label_index = [i for i, tmp_label in enumerate(timeseries.label_sample) if tmp_label == label]
            mean_distance = np.mean(pdist(timeseries.timecourses[label_index], self.metric))
            while mean_distance > self.threshold and len(label_index) > 2:
                new_means = []
                for l in label_index:
                    tmp_label_index = [ll for ll in label_index if not ll == l]
                    distances = pdist(timeseries.timecourses[tmp_label_index], self.metric)
                    new_means.append(np.mean(distances))
                min_index = np.argmin(new_means)
                label_index.pop(min_index)
                mean_distance = new_means[min_index]
            if mean_distance < self.threshold:
                mask[label_index] = True
        out = timeseries.copy()
        out.timecourses = mask
        return out       

class SelectTrials():
     
    def __init__(self):
        pass
        
    def __call__(self, timeseries, mask):
        mask = mask.timecourses
        selected_timecourses = timeseries.trial_shaped()[mask]
        out = timeseries.copy()
        out.set_timecourses(selected_timecourses)
        out.label_sample = [out.label_sample[i] for i in np.where(mask)[0]]
        return out 

class CalcStimulusDrive():
    
    def __init__(self, metric='correlation'):
        self.metric = metric
    
    def __call__(self, timeseries):
        labels = timeseries.label_sample
        
        stim_set = set(labels)
        # create dictionary with key: stimulus and value: trial where stimulus was given
        stim_pos = {}
        for stimulus in stim_set:
            stim_pos[stimulus] = np.where([i == stimulus for i in labels])[0]
        min_len = min([len(i)for i in stim_pos.values()])
        # create list of lists, where each sublist contains for all stimuli one exclusive trial
        indices = []
        for i in range(min_len):
            indices.append([j[i] for j in stim_pos.values()])
        # create pseudo-trial timecourses
        trial_timecourses = np.array([timeseries.trial_shaped()[i].reshape(-1, timeseries.num_objects) for i in indices])    
        # calculate correlation of pseudo-trials, aka stimulus dependency
        cor = [] 
        for object_num in range(timeseries.num_objects):
            cor.append(np.mean(pdist(trial_timecourses[:, :, object_num], self.metric)))
        out = timeseries.copy()
        out.timecourses = np.array(cor)
        return out 

class SelectModes():
     
    def __init__(self, threshold):
        self.threshold = threshold
        
    def __call__(self, timeseries, filtervalues):
        mask = filtervalues.timecourses < self.threshold
        selected_timecourses = timeseries.timecourses[:, mask]
        out = timeseries.copy()
        out.timecourses = selected_timecourses     
        out.label_objects = [out.label_objects[i] for i in np.where(mask)[0]]
        out.shape = (len(out.label_objects),)
        return out

class MeanSampleResponse():
    
    def __init__(self):
        pass
        
    def __call__(self, timeseries):
        timecourses = timeseries.trial_shaped()
        labels = timeseries.label_sample
        label_set = list(set(labels))       
        new_labels, new_timecourses = [], []
        for label in label_set:
            label_index = [i for i, tmp_label in enumerate(labels) if tmp_label == label]
            mean_timecourse = np.mean(timecourses[label_index], 0) 
            new_labels.append(label)
            new_timecourses.append(mean_timecourse)
        new_timecourses = np.hstack(new_timecourses)
    
        out = timeseries.copy()
        out.timecourses = new_timecourses
        out.label_sample = new_labels
        return out    

class SortBySamplename():
    
    def __init__(self):
        pass
        
    def __call__(self, timeseries):
        timecourses = timeseries.trial_shaped()
        labels = timeseries.label_sample
        label_ind = np.argsort(labels)
        timecourses = timecourses[label_ind]
        out = timeseries.copy()
        out.set_timecourses(timecourses)
        out.label_sample = [labels[i] for i in label_ind]
        return out  

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
