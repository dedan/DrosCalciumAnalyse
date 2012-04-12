
import os, pickle, json
import numpy as np
import pylab as plt
from NeuralImageProcessing import basic_functions as bf

#jan specific
base_path = '/home/jan/Documents/dros/new_data/aligned'
data_path = os.path.join(base_path, 'mic_split')
savefolder = 'micro'
save_path = os.path.join(base_path, 'results', savefolder)

##dedan specific
#base_path = '/Users/dedan/projects/fu'
#data_path = os.path.join(base_path, 'data', 'dros_calcium_new', 'mic_split')
#savefolder = 'micro'
#save_path = os.path.join(base_path, 'results', savefolder)

parts = ['lm', 'ln', 'rm', 'rn']
titles = {'lm': 'left, lesioned',
          'ln': 'left, normal',
          'rm': 'right, lesioned',
          'rn': 'right, normal'}
format = 'svg'

integrator = bf.StimulusIntegrator(threshold=0)
results = pickle.load(open(os.path.join(save_path, 'res.pckl')))
res_integrals = {}

for session in results.keys():

    print session
    result = results[session]
    base_size = result['mo2'].num_objects
    num_bases = len(result['filenames'])
    integrated = integrator(result['mo2'])
    res_integrals[session] = {}

    # spatial base plot
    fig = plt.figure()
    fig.suptitle(session)
    for modenum in range(base_size):
        single_bases = result['mo2'].base.objects_sample(modenum)
        res_integrals[session][modenum] = {}
        for base_num in range(num_bases):
            res_integrals[session][modenum][base_num] = {}
            ax = fig.add_subplot(base_size + 1, num_bases, base_num + 1)
            ax.imshow(result['mean_bases'][base_num], cmap=plt.cm.gray)
            ax.set_axis_off()
            ax.set_title(titles[result['filenames'][base_num][-2:]])
            ax = fig.add_subplot(base_size + 1, num_bases,
                                 ((num_bases * modenum) + num_bases) + base_num + 1)
            data = single_bases[base_num] * -1
            data_max = np.max(np.abs(data))
            ax.imshow(data, cmap=plt.cm.hsv, vmin= -data_max, vmax=data_max)
            ax.set_axis_off()
            ax.set_title('%.2f' % data_max, fontsize=10)

            # temporal base integrals
            for j, label in enumerate(integrated.label_sample):
                tmp = integrated.timecourses[j, modenum] * data_max
                res_integrals[session][modenum][base_num][label] = tmp

    plt.savefig(os.path.join(save_path, 'ica_' + session + '.' + format))

    # temporal base plot
    fig = plt.figure()
    for modenum in range(base_size):
        ax = fig.add_subplot(base_size, 1, modenum + 1)
        ax.plot(result['mo2'].timecourses[:, modenum])
        ax.set_xticklabels([], fontsize=12, rotation=45)
        ax.grid(True)
    ax.set_xticks(np.arange(3, result['mo2'].samplepoints, result['mo2'].timepoints))
    ax.set_xticklabels(result['mo2'].label_sample, fontsize=12, rotation=45)
    plt.savefig(os.path.join(save_path, 'time_' + session + '.' + format))

plt.close('all')
res_integrals['description'] = 'res_integrals[session][modenum][base_num][label]'
json.dump(res_integrals, open(os.path.join(inpath, 'integrals.json'), 'w'), indent=2)
