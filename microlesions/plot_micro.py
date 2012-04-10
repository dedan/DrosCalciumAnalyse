
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
sessions = set(['_'.join(key.split('_')[0:2]) for key in results.keys()])

for session in sessions:

    print session
    fig_base = plt.figure()
    fig_base.suptitle(session)

    for i_part, part in enumerate(parts):

        result = results['_'.join([session, part])]
        fig = plt.figure()
        fig.suptitle(titles[part])

        base_size = result['mo2'].num_objects
        for modenum in range(base_size):

            ax = fig_base.add_subplot(base_size + 1, len(parts), i_part+1)
            ax.imshow(result['mean_base'], cmap=plt.cm.gray)
            ax.set_axis_off()
            ax.set_title(titles[part])
            ax = fig_base.add_subplot(base_size + 1,
                                 len(parts),
                                 (modenum + 1) * len(parts) + i_part+1)
            base2d = result['mo2'].base.shaped2D()
            ax.imshow(base2d[modenum, :, :] * -1, cmap=plt.cm.hsv, vmin= -1, vmax=1)
            ax.set_axis_off()

            # plot timecourses
            ax = fig.add_subplot(base_size, 1, modenum + 1)
            ax.plot(result['mo2'].timecourses[:, modenum])
            ax.set_xticklabels([], fontsize=12, rotation=45)
            ax.grid(True)
            ax.set_xticks(np.arange(3, result['mo2'].samplepoints, result['mo2'].timepoints))
            ax.set_xticklabels(result['mo2'].label_sample, fontsize=12, rotation=45)
        plt.figure(fig.number)
        plt.savefig(os.path.join(save_path, 'time_' + '_'.join([session, part]) + '.' + format))


    plt.figure(fig_base.number)
    plt.savefig(os.path.join(save_path, 'ica_' + session + '.' + format))

plt.close('all')
