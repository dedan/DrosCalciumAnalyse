
import os
import glob
import json
from NeuralImageProcessing.pipeline import TimeSeries
import NeuralImageProcessing.basic_functions as bf
import logging as l
import numpy as np
import pylab as plt
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

load_path = '/Users/dedan/projects/fu/results/simil80n_bestFalse/nnma/'

labeled_animals = json.load(open(os.path.join(load_path, 'regions.json')))

filelist = glob.glob(os.path.join(load_path, '*.json'))
filelist = [f for f in filelist if not 'base' in os.path.basename(f)]
filelist = [f for f in filelist if not 'regions' in os.path.basename(f)]

data = {}
average_over_stimulus_repetitions = bf.SingleSampleResponse()
integrator = bf.StimulusIntegrator()
l.info('read files from: %s' % load_path)
for fname in filelist:

    ts = TimeSeries()
    ts.load(os.path.splitext(fname)[0])
    name = os.path.splitext(os.path.basename(fname))[0]
    data[name] = average_over_stimulus_repetitions(ts)

all_stimuli = set(sum([ts.label_sample for ts in data.values()], []))
all_region_labels = set(sum([labels for labels in labeled_animals.values()], []))
l.debug('all_stimuli: %s' % all_stimuli)
l.debug('all_region_labels: %s' % all_region_labels)

for region_label in all_region_labels:

    plt.figure()
    plt.suptitle(region_label)
    integrated = []
    for animal, regions in labeled_animals.items():

        ts = data[animal]
        trial_shaped = ts.trial_shaped()
        trial_length = trial_shaped.shape[1]
        n_modes = trial_shaped.shape[2]
        modes = [i for i in range(n_modes) if regions[i] == region_label]


        for mode in modes:
            pdat = np.zeros(len(all_stimuli) * trial_length)

            for i, stimulus in enumerate(all_stimuli):
                if stimulus in ts.label_sample:
                    index = ts.label_sample.index(stimulus)
                    pdat[i*trial_length:i*trial_length+trial_length] = trial_shaped[index, :, mode]
            plt.plot(pdat)
plt.show()



