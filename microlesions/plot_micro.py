
import os, pickle
import numpy as np
import pylab as plt

base_path = '/Users/dedan/projects/fu'
data_path = os.path.join(base_path, 'data', 'dros_calcium_new', 'mic_split')
savefolder = 'micro'
save_path = os.path.join(base_path, 'results', savefolder)
parts = ['lm', 'ln', 'rm', 'rn']
titles = {'lm': 'left, lesioned',
          'ln': 'left, normal',
          'rm': 'right, lesioned',
          'rn': 'right, normal'}
format = 'png'

results = pickle.load(open(os.path.join(save_path, 'res.pckl')))

for session in results.keys():

    result = results[session]
    print session
    fig = plt.figure()
    fig.suptitle(session)

    base_size = result['mo2'].num_objects
    num_bases = len(result['filenames'])
    for modenum in range(base_size):
        single_bases = result['mo2'].base.objects_sample(modenum)
        for base_num in range(num_bases):
            ax = fig.add_subplot(base_size+1, num_bases, base_num+1)
            ax.imshow(result['mean_bases'][base_num], cmap=plt.cm.gray)
            ax.set_axis_off()
            ax.set_title(titles[result['filenames'][base_num][-2:]])
            ax = fig.add_subplot(base_size+1, num_bases,
                                 ((num_bases * modenum) + num_bases) + base_num + 1)
            data = single_bases[base_num] * -1
            data_max = np.max(np.abs(data))
            ax.imshow(data, cmap=plt.cm.hsv, vmin= -data_max, vmax=data_max)
            ax.set_axis_off()
    plt.savefig(os.path.join(save_path, 'ica_' + session + '.' + format))

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
