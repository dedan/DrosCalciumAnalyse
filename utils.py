import os, glob
import itertools as it
from collections import defaultdict
import numpy as np
import pylab as plt
from NeuralImageProcessing import basic_functions as bf
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import array
reload(bf)

def create_mf(mf_dic):
    '''creates a matrixfactorization according to mf_dic specification'''
    mf_methods = {'nnma':bf.NNMA, 'nnman':bf.NNMA, 'sica': bf.sICA, 'stica': bf.stICA}
    mf = mf_methods[mf_dic['method']](**mf_dic['param'])
    return mf

def colormap_from_lut(filename):
    """create a colormap from a .lut file like we get them from antonia"""
    with open(filename, 'rb') as f:
        bytes = f.read()
        data = array.array('B')
        data.fromstring(bytes)
        data = np.reshape(data, (3, -1)) / 255.
    return ListedColormap(zip(data[0,:], data[1,:], data[2,:]))

def get_all_lut_colormaps(regions):
    """creates a mapping from a region name to a colormap (from luts folder)
       the first map is returned for unknown regions
    """
    luts_path = os.path.join(os.path.dirname(__file__), 'colormap_luts')
    filelist = glob.glob(os.path.join(luts_path, '*.lut'))
    first_map = colormap_from_lut(filelist.pop())
    first_map = plt.cm.hsv_r
    colormaps = defaultdict(lambda: first_map)
    assert len(regions) == len(filelist)
    for i, fname in enumerate(filelist):
        colormaps[regions[i]] = colormap_from_lut(fname)
    return colormaps


def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect * 0.3)

def select_n_channels(data, n):
    """ select n rows which have the most possible odors (columns) in common

        Not for all animals the same odors are available because sometimes they
        are not consistend over stimulus repetition and are therefore sorted out.
        This function helps to find the subset of odors which are available for
        n out of size(data, 0) animals.
    """
    best = -1
    for comb in it.combinations(range(np.size(data, 0)), n):
        s = (np.sum(data[comb, :], 0) == n).astype('int')
        if np.sum(s) > best:
            best = np.sum(s)
            best_comb = comb
    return best_comb


def cordist(modelist):
    """create dictionary with mode distances"""
    stimuli_filter = bf.SelectTrials()
    cor_dist = bf.Distance()
    combine = bf.ObjectConcat()
    alldist_dic = {}
    for i in range(len(modelist)):
        stimuli_i = modelist[i].label_sample
        if len(stimuli_i) < 8:
            print 'skipped: ', modelist[i].name
            continue
        for j in range(i + 1, len(modelist)):
            stimuli_j = modelist[j].label_sample
            if len(stimuli_j) < 8:
                print 'skipped: ', modelist[j].name
                continue
            common = set(stimuli_i).intersection(stimuli_j)
            mask_i = np.zeros(len(stimuli_i), dtype='bool')
            mask_j = np.zeros(len(stimuli_j), dtype='bool')
            for stim in common:
                mask_i[stimuli_i.index(stim)] = True
                mask_j[stimuli_j.index(stim)] = True
            ts1 = stimuli_filter(modelist[i], bf.TimeSeries(mask_i))
            ts2 = stimuli_filter(modelist[j], bf.TimeSeries(mask_j))
            cor = cor_dist(combine([ts1, ts2]))
            alldist_dic.update(cor.as_dict('objects'))
    return alldist_dic


def dict2pdist(dic):
    """helper function to convert dictionary to pdist format"""
    key_parts = list(set(sum([i.split(':') for i in dic.keys()], [])))
    new_pdist = []
    for i in range(len(key_parts)):
        for j in range(i + 1, len(key_parts)):
            try:
                new_pdist.append(dic[':'.join([key_parts[i], key_parts[j]])])
            except KeyError:
                new_pdist.append(dic[':'.join([key_parts[j], key_parts[i]])])
    return new_pdist, key_parts

def create_colormap(cmap_name, from_rgb, over_rgb, to_rgb):
    cdict = {'red': ((0., from_rgb[0] / 256., from_rgb[0] / 256.), (0.5, 1., 1.),
                     (0.75, over_rgb[0] / 256., over_rgb[0] / 256.),
                     (1., to_rgb[0] / 256., to_rgb[0] / 256.)),
             'green':((0., from_rgb[1] / 256., from_rgb[1] / 256.), (0.5, 1., 1.),
                      (0.75, over_rgb[1] / 256., over_rgb[1] / 256.),
                       (1., to_rgb[1] / 256., to_rgb[1] / 256.)),
             'blue':((0., from_rgb[2] / 256., from_rgb[2] / 256.), (0.5, 1., 1.),
                     (0.75, over_rgb[2] / 256., over_rgb[2] / 256.),
                     (1., to_rgb[2] / 256., to_rgb[2] / 256.))}
    return LinearSegmentedColormap(cmap_name, cdict, 256)

redmap = create_colormap('redmap', (205, 0, 205), (255, 165, 0), (139, 0, 0))
bluemap = create_colormap('bluemap', (205, 0, 205), (100, 149, 237), (16, 78, 139))
greenmap = create_colormap('greenmap', (205, 0, 205), (154, 205, 50), (105, 139, 34))
cyanmap = create_colormap('cyanmap', (205, 0, 205), (78, 238, 148), (0, 205, 102))
yellowmap = create_colormap('yellomap', (205, 0, 205), (238, 201, 0), (238, 201, 0))
violetmap = create_colormap('violetmap', (205, 0, 205), (148, 0, 211), (104, 34, 139))
brownmap = create_colormap('brownmap', (205, 0, 205), (205, 133, 63), (139, 90, 43))
