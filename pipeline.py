import copy as cp
import numpy as np
import json

class Event(object):
    
    def __init__(self, key, value):
        self.key = key
        self.value = value
        
class FullBufferException(Exception):
    pass

class Block(object):
    
    def __init__(self, name='block'):
        self.name = name
        self.event_listener = {}
        self.listener_inputtype = {}
        self.event_sender = {}
        self.input = {}
        self.signals = []
        
    def add_listener(self, listener, asinput='image_series'):
        self.event_listener[id(listener)] = listener
        self.listener_inputtype[id(listener)] = asinput
        listener.event_sender[id(self)] = self
        
    def add_sender(self, sender, asinput='image_series'):
        self.event_sender[id(sender)] = sender
        if asinput in self.input:
            raise Exception('slot already used')
        self.input[asinput] = '' 
        sender.event_listener[id(self)] = self
        sender.listener_inputtype[id(self)] = asinput
    
    def release_from_listener(self):
        for listener in self.event_listener.itervalues():
            listener.event_sender.pop(id(self))
        self.event_listener = {}
        self.listener_inputtype = {}
    
    def release_from_sender(self):
        for sender in self.event_sender.itervalues():
            sender.event_listener.pop(id(self))
            sender.listener_inputtype.pop(id(self))
        self.event_sender = {}
 
    def sent_event(self, element):
        for key, listener in self.event_listener.iteritems():
            try:
                listener.receive_event(Event(self.listener_inputtype[key], element))
            except FullBufferException:
                print self.name, ' to ', listener, ": !!!FullBuffer, not sent!!!"

    def sent_signal(self, element):
        for listener in self.event_listener.values():
            listener.receive_event(Event('signal', element))
                           
    def receive_event(self, event):
        if event.key == 'signal':
            self.process_signal(event)
        elif not(self.input[event.key]):
            #print self.name, ' filled'
            self.input[event.key] = event.value
            if all(self.input.values()):
                print self.name, ' executed'
                self.execute()
                self.set_receiving()
        else:
            print self.name, ": !!!FullBuffer!!!"
            raise FullBufferException()
        
    def process_signal(self, signal_event):
        self.signals.append(signal_event)
        if len(self.signals) == len(self.event_sender):
            self.signals = []
            self.sent_signal(signal_event.value)
        
    def set_receiving(self):
        """reset input buffers after processing"""
        for key in self.input:
            self.input[key] = '' 

class TimeSeries(object):
    ''' dim0: timepoints, dim1: objects'''
    
    def __init__(self, series='', name=['standard_series'], shape=(),
                 typ='2DImage', label_sample=''):
        self.timecourses = series
        self.shape = shape
        self.typ = typ
        self.name = name
        self.label_sample = label_sample 
        self.label_objects = []   
    
    @property
    def timepoints(self):
        return self.timecourses.shape[0] / len(self.label_sample)
    
    @property
    def num_trials(self):
        return len(self.label_sample)
    
    @property
    def num_objects(self):
        return np.prod(self.shape)
    
    @property
    def samplepoints(self):
        return self.timecourses.shape[0]
    
    def set_timecourses(self, timecourses):
        self.timecourses = timecourses.reshape(-1, self.num_objects)
               
    def shaped2D(self):
        return self.timecourses.reshape(-1, *self.shape)
    
    def trial_shaped(self):
        return self.timecourses.reshape(len(self.label_sample), -1, np.prod(self.shape))
    
    def trial_shaped2D(self):
        return self.timecourses.reshape(len(self.label_sample), -1, *self.shape)
    
    def as_dict(self, aggregate=None):
        if aggregate == 'objects':
            outdict = zip(self.label_objects, self.timecourses.T)
        if aggregate == 'trials':
            outdict = zip(self.label_sample, self.trial_shaped())
        return outdict
                
    def copy(self):
        out = cp.copy(self)
        out.name = cp.copy(self.name)
        out.label_sample = cp.copy(self.label_sample)
        out.label_objects = cp.copy(self.label_objects)        
        if self.typ == 'latent_series':
            out.base = self.base.copy() 
        return out
    
    def save(self, filename):
        data = self.__dict__.copy()
        np.save(filename, data.pop('timecourses'))
        if self.typ == 'latent_series':
            self.base.save(filename + '_base')
            data.pop('base')
        json.dump(data, open(filename + '.json', 'w'))
            
    def load(self, filename):
        self.__dict__.update(json.load(open(filename + '.json')))
        self.timecourses = np.load(filename + '.npy')
        if self.typ == 'latent_series':
            self.base = TimeSeries()
            self.base.load(filename + '_base')
