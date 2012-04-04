
import itertools as it
import numpy as np
from NeuralImageProcessing import basic_functions as bf

def force_aspect(ax, aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect*0.3)


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
