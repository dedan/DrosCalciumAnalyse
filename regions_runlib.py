#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys, os, glob, csv
from NeuralImageProcessing.pipeline import TimeSeries
import NeuralImageProcessing.basic_functions as bf
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

def load_lesion_data(lesion_data_path):
    """read the table of which lesion was applied into a dictionary"""
    les_dict = {}
    lesion_table = list(csv.reader(open(lesion_data_path, 'rb'), delimiter='\t'))
    for row in lesion_table[1:]:
        les_dict[row[0]] = {}
        les_dict[row[0]]['l'] = row[1]
        les_dict[row[0]]['r'] = row[2]
    return les_dict

def load_mf_results(load_path, selection, lesion_data, integrate):
    """read data (matrix factorization results) to dictionary"""

    data = {}
    # initialize processing (pipeline) components
    average_over_stimulus_repetitions = bf.SingleSampleResponse()
    if integrate:
        integrator = bf.StimulusIntegrator(method=integrate,
                                           threshold= -1000,
                                           window=integrator_window)

    # load and filter filelist
    filelist = glob.glob(os.path.join(load_path, '*.json'))
    filelist = [f for f in filelist if not 'base' in os.path.basename(f)]
    filelist = [f for f in filelist if not 'regions' in os.path.basename(f)]
    filelist = [f for f in filelist if not 'selection' in os.path.basename(f)]

    for fname in filelist:

        if selection:
            skip = True
            for sel in selection:
                if sel in name:
                    skip = False
            if skip:
                l.info('skip %s because not in selection' % name)
                continue
            l.info('taking %s because found in selection' % name)

        ts = TimeSeries()
        ts.load(os.path.splitext(fname)[0])
        name = os.path.splitext(os.path.basename(fname))[0]

        if lesion_data:
            if '_' in name:
                fname_base, side = name.split('_')
            new_labels = []
            for label in ts.label_sample:
                if label[0] == 'm':
                    new_labels.append(label[1:] + '-' + les_dict[fname_base][side])
                else:
                    new_labels.append(label)
            ts.label_sample = new_labels

        if integrate:
            data[name] = integrator(average_over_stimulus_repetitions(ts))
        else:
            data[name] = average_over_stimulus_repetitions(ts)
    return data
