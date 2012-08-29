import os, sys, glob, json, time, datetime
import numpy as np
import pylab as plt
from NeuralImageProcessing import basic_functions as bf
from NeuralImageProcessing import illustrate_decomposition as vis
from DrosCalciumAnalyse import runlib, utils
from PyQt4 import QtCore
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from main_window import Ui_MainGuiWin
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

class MainGui(QtGui.QMainWindow, Ui_MainGuiWin):
    '''gui main class'''

    def __init__(self, parent=None):
        """initialize the gui, connect signals, add axes objects, etc.."""
        super(MainGui, self).__init__(parent)
        self.setupUi(self)
        self.config_file = 'gui_config.json'
        self.method_controls = {'nnma': [self.spars_par1_label, self.spars_par1_spinner,
                                         self.spars_par2_label, self.spars_par2_spinner,
                                         self.smoothness_label, self.smoothness_spinner,
                                         self.maxcount_label, self.maxcount_spinner],
                                'stica': [self.alpha_label, self.alpha_spinner],
                                'sica': []}

        # connect signals to slots
        self.factorize_button.clicked.connect(self.factorize)
        for spinner in self.findChildren((QtGui.QSpinBox, QtGui.QDoubleSpinBox)):
            spinner.valueChanged.connect(self.save_controls)
        for check_box in self.findChildren(QtGui.QCheckBox):
            check_box.stateChanged.connect(self.save_controls)
        self.methods_box.currentIndexChanged.connect(self.mf_method_changed)
        self.load_controls()

    def select_data_folder(self, path=''):
        """select data folder, either from given path or dialog"""
        if not path:
            caption = 'select your data folder'
            self.fname = str(QtGui.QFileDialog.getExistingDirectory(caption=caption))
        else:
            self.fname = path

        subfolders = [f for f in os.listdir(self.fname)
                        if os.path.isdir(os.path.join(self.fname, f))]
        to_convert = []
        for subfolder in subfolders:
            if not os.path.exists(os.path.join(self.fname, subfolder, 'timeseries.npy')):
                to_convert.append(subfolder)
        if to_convert:
            message_text = ('%d image files have to be converted to our numpy format\n' +
                            'this is only done once') % len(to_convert)
            QtGui.QMessageBox.information(self, 'File conversion', message_text)
            progdialog = QtGui.QProgressDialog('converting image files..',
                                        'cancel',
                                        0, len(to_convert), self)
            progdialog.setMinimumDuration(0)
            progdialog.setWindowModality(QtCore.Qt.WindowModal)

            for i, folder in enumerate(to_convert):
                progdialog.setValue(i)
                folder_path = os.path.join(self.fname, folder)
                QtCore.QCoreApplication.processEvents()

                ts = runlib.create_timeseries_from_pngs(folder_path, folder)
                ts.save(os.path.join(folder_path, 'timeseries'))
                if progdialog.wasCanceled():
                    print 'hui ui ui'
                    break

            progdialog.setValue(len(to_convert))
        message = '%d files found in %s' % (len(subfolders), self.fname)
        self.statusbar.showMessage(message, msecs=5000)

    def load_controls(self):
        """initialize the control elements (widgets) from config file"""
        config = json.load(open(self.config_file))
        self.filter_box.setChecked(config['similarity_filter'])
        self.normalize_box.setChecked(config['normalize'])
        self.lowpass_spinner.setValue(config['lowpass'])
        self.median_spinner.setValue(config['medianfilter'])
        self.spatial_spinner.setValue(config['spatial_down'])
        self.similarity_spinner.setValue(config['similarity_threshold'])
        self.methods_box.clear()
        self.methods_box.insertItems(0, config['methods'].keys())
        self.methods_box.setCurrentIndex(self.methods_box.findText(config['selected_method']))
        self.mf_overview_box.setChecked(config['mf_overview'])
        self.raw_overview_box.setChecked(config['raw_overview'])
        self.raw_unsort_overview_box.setChecked(config['raw_unsort_overview'])
        self.quality_box.setChecked(config['quality'])
        self.spars_par1_spinner.setValue(config['methods']['nnma']['spars_par1'])
        self.spars_par2_spinner.setValue(config['methods']['nnma']['spars_par2'])
        self.smoothness_spinner.setValue(config['methods']['nnma']['smoothness'])
        self.maxcount_spinner.setValue(config['methods']['nnma']['maxcount'])
        self.alpha_spinner.setValue(config['methods']['stica']['alpha'])
        self.n_modes_spinner.setValue(config['n_modes'])
        self.config = config

    def save_controls(self, export_file=''):
        '''after each click, save settings to config file'''
        print 'save_controls called, export file is: %s' % export_file
        config = {}
        config['similarity_filter'] = self.filter_box.isChecked()
        config['normalize'] = self.normalize_box.isChecked()
        config['lowpass'] = self.lowpass_spinner.value()
        config['medianfilter'] = self.median_spinner.value()
        config['spatial_down'] = self.spatial_spinner.value()
        config['similarity_threshold'] = self.similarity_spinner.value()
        config['selected_method'] = str(self.methods_box.currentText())
        config['mf_overview'] = self.mf_overview_box.isChecked()
        config['raw_overview'] = self.raw_overview_box.isChecked()
        config['raw_unsort_overview'] = self.raw_unsort_overview_box.isChecked()
        config['quality'] =  self.quality_box.isChecked()
        config['methods'] = {'nnma': {}, 'stica': {}, 'sica': {}}
        config['methods']['nnma']['spars_par1'] = self.spars_par1_spinner.value()
        config['methods']['nnma']['spars_par2'] = self.spars_par2_spinner.value()
        config['methods']['nnma']['smoothness'] = self.smoothness_spinner.value()
        config['methods']['nnma']['maxcount'] = self.maxcount_spinner.value()
        config['methods']['stica']['alpha'] = self.alpha_spinner.value()
        config['n_modes'] = self.n_modes_spinner.value()
        self.config = config
        json.dump(config, open(self.config_file, 'w'))
        if isinstance(export_file, str) and os.path.exists(os.path.dirname(export_file)):
            json.dump(config, open(export_file, 'w'))

    # TODO: add load and save settings to the menu

    def mf_method_changed(self):
        """display the suitable options for the selected method"""
        current_method = str(self.methods_box.currentText())
        for method in self.config['methods']:
            for ui_elem in self.method_controls[method]:
                ui_elem.setVisible(method == current_method)
        self.save_controls()

    # TODO: maybe start a new thread for this?
    def factorize(self):

        # TODO: select this in combobox
    # "formats": ["jpg", "png", "svg"],
    # "selected_format": "png",

        baselines = []

        # TODO: open a dialog to ask for output folder
        out_folder = '/Users/dedan/projects/fu/results/simil80n_bestFalsemask_dev/'
        if not os.path.exists(out_folder):
            self.statusbar.showMessage('folder does not exist', msecs=3000)
            return

        mf_params = {'method': self.config['selected_method'],
                     'param': self.config['methods'][self.config['selected_method']]}
        mf_params['param']['variance'] = self.config['n_modes']
        print mf_params

        timestamp = datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        out_folder = os.path.join(out_folder, timestamp)
        data_folder = os.path.join(out_folder, 'data')
        odor_plots_folder = os.path.join(out_folder, 'odors')
        os.mkdir(out_folder)
        os.mkdir(data_folder)
        os.mkdir(odor_plots_folder)
        self.save_controls(export_file=os.path.join(out_folder, 'config.json'))
        self.config['selected_format'] = '.png'

        filelist = [os.path.join(self.fname, f, 'timeseries') for f in os.listdir(self.fname)]
        filelist = [f for f in filelist if os.path.exists(f+'.json')]
        # TODO: use only the n_best animals --> most stable odors in common (thresh_res.pckl)

        # filelist = filelist[0:2]

        self.statusbar.showMessage('hardcore computation stuff going on..')
        progdialog = QtGui.QProgressDialog('hardcore computation stuff going on..',
                                            'cancel',
                                            0, len(filelist), self)
        progdialog.setMinimumDuration(0)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)
        for file_ind, filename in enumerate(filelist):

            disp_name = os.path.basename(filename)
            progdialog.setValue(file_ind)
            if progdialog.wasCanceled():
                print 'hui ui ui'
                break

            meas_path = os.path.splitext(filename)[0]
            fname = os.path.basename(meas_path)
            plot_name_base = os.path.join(out_folder, fname)

            # create timeseries, change shape and preprocess
            ts = bf.TimeSeries()
            progdialog.setLabelText('%s: loading' % disp_name)
            QtCore.QCoreApplication.processEvents()
            ts.load(meas_path)

            ts.shape = tuple(ts.shape)
            progdialog.setLabelText('%s: preprocessing' % disp_name)
            QtCore.QCoreApplication.processEvents()
            out = runlib.preprocess(ts, self.config)

            # do matrix factorization
            progdialog.setLabelText('%s: factorization' % disp_name)
            QtCore.QCoreApplication.processEvents()
            mf_func = utils.create_mf(mf_params)
            mf = mf_func(out['pp'])
            mf.base.shape = tuple(mf.base.shape)

            baselines.append(out['baseline'])

            # save results
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.axis('off')
            for i, spatial_mode in enumerate(mf.base.shaped2D()):
                ax.imshow(spatial_mode, interpolation='nearest')
                fig.savefig(os.path.join(data_folder, 'spatial_%s_%d.png' % (fname, i+1)))
            np.savetxt(os.path.join(data_folder, 'temporal_%s.csv' % fname),
                       mf.timecourses.T, delimiter=',')

            # plot overview of matrix factorization
            if self.config['mf_overview']:
                mf_overview = runlib.mf_overview_plot(mf)
                mf_overview.savefig(plot_name_base + '_overview.' + self.config['selected_format'])

            # overview of raw responses
            if self.config['raw_overview']:
                raw_resp_overview = runlib.raw_response_overview(out)
                raw_resp_overview.savefig(plot_name_base + '_raw_overview.' +
                                          self.config['selected_format'])
                raw_resp_unsort_overview = runlib.raw_unsort_response_overview(out)
                raw_resp_unsort_overview.savefig(plot_name_base +
                                                 '_raw_unsort_overview.' +
                                                 self.config['selected_format'])

            # calc reproducibility and plot quality
            if self.config['quality']:
                stimulirep = bf.SampleSimilarityPure()
                distanceself, distancecross = stimulirep(out['mean_resp'])
                qual_view = runlib.quality_overview_plot(distanceself, distancecross, ts.name)
                qual_view.savefig(plot_name_base + '_quality.' + self.config['selected_format'])
        progdialog.setValue(len(filelist))
        self.statusbar.showMessage('yeah, finished!', msecs=2000)


class PlotCanvas(FigureCanvas):
    '''a class only containing the figure to manage the qt layout'''
    def __init__(self):
        self.fig = Figure()
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotWidget(QtGui.QWidget):
    '''all plotting related stuff and also the context menu'''
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = PlotCanvas()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        ax = self.canvas.fig.add_subplot(111)
        ax.plot(np.random.rand(10))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    my_gui = MainGui()
    my_gui.show()
    app.setActiveWindow(my_gui)

    debugging = False
    if debugging:
        my_gui.select_data_folder('/Users/dedan/projects/fu/data/dros_gui_test/')
        my_gui.factorize()
    else:
        my_gui.select_data_folder('/Users/dedan/projects/fu/data/dros_gui_test/')

    sys.exit(app.exec_())


