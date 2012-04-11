"""
"""

import glob, pickle, os, json
import numpy as np
import pylab as plt
from NeuralImageProcessing import basic_functions as bf
reload(bf)

inpath = '/Users/dedan/projects/fu/results/cross_val/nbest-5_thresh-80/'
prefixes = ['OCO', '2PA', 'LIN', 'CVA']
prefixes = ['OCO']
res = {}

integrator = bf.StimulusIntegrator(threshold=0)

for prefix in prefixes:

    res[prefix] = {}
    data = pickle.load(open(os.path.join(inpath, prefix + '_all.pckl')))
    integrated = integrator(data['base'])

    info = json.load(open(os.path.join(inpath, prefix + '_time_plot.json')))
    for i, key in enumerate(info):
        mode = [mode for mode in info[key] if 'all' in mode][0]
        res[prefix][key] = {}
        for j, label in enumerate(integrated.label_sample):
            res[prefix][key][label] = integrated.timecourses[j, int(mode[-1])]
json.dump(res, open(os.path.join(inpath, 'integrals.json'), 'w'), indent=2)





