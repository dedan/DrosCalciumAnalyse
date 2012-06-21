'''
    some of the data are flip along the horizontal axis.

    This script corrects this.

    USAGE: Just write an attribute 'turn' in the json file of the recording
        session you want to turn and run the turnscript with datapath set
        to the folder in which the file resides. The script will flip the
        image data along the horizontal axis and delete the 'turn' attribute
        of the json file, that the data won't be flipped again if the file
        is run again in the same folder..

'''

import glob, os, json
from NeuralImageProcessing import basic_functions as bf
import pylab as plt
import numpy as np


datapath = '/Users/dedan/projects/fu/data/dros_calcium_new/'

filelist = glob.glob(os.path.join(datapath, '*.json'))

for dfile in filelist:
    info = json.load(open(dfile))

    if 'turn' in info:
        plt.figure()

        print 'have to turn'
        print '\t%s' % dfile
        ts = bf.TimeSeries()
        ts.load(os.path.splitext(dfile)[0])

        bla = ts.trial_shaped2D()[:,:,::-1,:]
        npy_name = os.path.splitext(dfile)[0] + '.npy'
        np.save(npy_name, bla.reshape(*np.shape(ts.timecourses)))

        del info['turn']
        json.dump(info, open(dfile, 'w'))

