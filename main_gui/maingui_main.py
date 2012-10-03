#!/usr/bin/env python
# encoding: utf-8
"""
Created by Stephan Gabler (stephan.gabler@gmail.com)
and Jan Sölter (jan_soelter@yahoo.com) at FU-Berlin.

Copyright (c) 2012. All rights reserved.
"""
import os, sys, glob, json, time, datetime, pickle
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

debugging = True

class MainGui(QtGui.QMainWindow, Ui_MainGuiWin):
    '''gui main class'''

    def __init__(self, parent=None):
        """initialize the gui, connect signals, add axes objects, etc.."""
        super(MainGui, self).__init__(parent)
        self.factorized = False
        self.setupUi(self)
        self.results = {}
        self.export_methods = {}
        self.config_file = 'gui_config.json'
        self.method_controls = {'nnma': [self.spars_par1_label, self.spars_par1_spinner,
                                         self.spars_par2_label, self.spars_par2_spinner,
                                         self.smoothness_label, self.smoothness_spinner,
                                         self.maxcount_label, self.maxcount_spinner],
                                'stica': [self.alpha_label, self.alpha_spinner],
                                'sica': []}
        self.plot_methods = {'sorted overview': runlib.raw_response_overview,
                             'unsorted overview': runlib.raw_unsort_response_overview,
                             'quality': runlib.quality_overview_plot,
                             'mf_overview': runlib.mf_overview_plot}

        # init gui
        basic_plot_methods = ['quality', 'sorted overview', 'unsorted overview']
        self.plot_selection_box.insertItems(0, basic_plot_methods)
        self.plot_threshold_box.insertItems(0, [str(x/10.) for x in range(11)])
        self.plot_threshold_box.setCurrentIndex(3)
        self.format_box.insertItems(0, ['png', 'jpg', 'svg', 'pdf'])
        self.export_method_clicked()

        # connect signals to slots
        self.format_box.currentIndexChanged.connect(self.save_controls)
        self.plot_export_button.clicked.connect(self.export_results)
        self.filter_box.toggled.connect(self.recalculate_filter)
        self.recompute_filter_button.clicked.connect(self.recalculate_filter)
        self.session_box.currentIndexChanged.connect(self.update_plot)
        self.plot_selection_box.currentIndexChanged.connect(self.update_plot)
        self.plot_selection_box.currentIndexChanged.connect(self.change_plot_parameters)
        self.preprocess_button.clicked.connect(self.preprocess)
        self.factorize_button.clicked.connect(self.factorize)
        for spinner in self.findChildren((QtGui.QSpinBox, QtGui.QDoubleSpinBox)):
            spinner.valueChanged.connect(self.save_controls)
        for check_box in self.findChildren(QtGui.QCheckBox):
            check_box.stateChanged.connect(self.save_controls)
        for check_box in self.export_box.findChildren(QtGui.QCheckBox):
            check_box.stateChanged.connect(self.export_method_clicked)
        self.plot_threshold_box.currentIndexChanged.connect(self.update_plot)
        self.methods_box.currentIndexChanged.connect(self.mf_method_changed)
        self.load_controls()

        # load cached preprocessing results in debugging mode
        if debugging:
            if os.path.exists('cache.pckl'):
                self.results = pickle.load(open('cache.pckl'))
                self.activate_controls()

    def change_plot_parameters(self):
        """enable or disable the correct plot parameters for a certain plot"""
        method = str(self.plot_selection_box.currentText())
        if method == 'sorted overview':
            self.plot_threshold_box.setEnabled(True)
        else:
            self.plot_threshold_box.setEnabled(False)

    def recalculate_filter(self):
        """ select stimuli such that their mean correlation distance between the mean
            responses of repeated stimuli presentations is below similarity_threshold
        """
        l.debug('filtering with %f' % self.config['similarity_threshold'])
        stimuli_mask = bf.SampleSimilarity(self.config['similarity_threshold'])
        stimuli_filter = bf.SelectTrials()
        for res in self.results.values():
            if self.filter_box.isChecked():
                res['mask'] = stimuli_mask(res['mean_resp'])
                print res['mask']
            else:
                res['mask'] = []
        self.update_plot()
        self.mf_overview_box.setChecked(False)
        self.mf_overview_box.setEnabled(False)
        self.mf_save_box.setChecked(False)
        self.mf_save_box.setEnabled(False)
        if self.factorized:
            self.factorize_label.setText('filtering changed, factorize again!!!')


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
            message_text = ('%d data folder has to be converted to our numpy format\n' +
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

        filelist = [os.path.join(self.fname, f, 'timeseries') for f in os.listdir(self.fname)]
        self.filelist = [os.path.basename(os.path.dirname(f)) for f in filelist if os.path.exists(f+'.json')]
        self.session_box.insertItems(0, self.filelist)
        # TODO: use only the n_best animals --> most stable odors in common (thresh_res.pckl)


    def load_controls(self):
        """initialize the control elements (widgets) from config file"""
        config = json.load(open(self.config_file))
        self.normalize_box.setChecked(config['normalize'])
        self.lowpass_spinner.setValue(config['lowpass'])
        self.median_spinner.setValue(config['medianfilter'])
        self.spatial_spinner.setValue(config['spatial_down'])
        self.similarity_spinner.setValue(config['similarity_threshold'])
        self.methods_box.clear()
        self.methods_box.insertItems(0, config['methods'].keys())
        self.methods_box.setCurrentIndex(self.methods_box.findText(config['selected_method']))
        self.format_box.setCurrentIndex(self.format_box.findText(config['format']))
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
        config['normalize'] = self.normalize_box.isChecked()
        config['lowpass'] = self.lowpass_spinner.value()
        config['medianfilter'] = self.median_spinner.value()
        config['spatial_down'] = self.spatial_spinner.value()
        config['similarity_threshold'] = self.similarity_spinner.value()
        config['selected_method'] = str(self.methods_box.currentText())
        config['format'] = str(self.format_box.currentText())
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

    def export_method_clicked(self):
        '''register which export (plot) methods are selected and enable button'''
        self.export_methods['mf_overview'] = self.mf_overview_box.isChecked()
        self.export_methods['sorted overview'] = self.raw_overview_box.isChecked()
        self.export_methods['unsorted overview'] = self.raw_unsort_overview_box.isChecked()
        self.export_methods['quality'] = self.quality_box.isChecked()
        self.export_methods['mf_matrices'] = self.mf_save_box.isChecked()
        print self.export_methods.values()
        if any(self.export_methods.values()):
            self.plot_export_button.setEnabled(True)
        else:
            self.plot_export_button.setEnabled(False)

    def export_results(self):
        """save all selected plots"""
        caption = 'select output folder'
        out_folder = str(QtGui.QFileDialog.getExistingDirectory(caption=caption))
        params = {'threshold': float(self.plot_threshold_box.currentText())}
        json.dump(self.config, open(os.path.join(out_folder, 'config.json'), 'w'))
        if not os.path.exists(os.path.join(out_folder, 'timeseries')):
            os.mkdir(os.path.join(out_folder, 'timeseries'))
        progdialog = QtGui.QProgressDialog('export results..',
                                    'cancel',
                                    0, len(self.filelist), self)
        progdialog.setMinimumDuration(0)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)

        fig = plt.figure()
        for i, session in enumerate(self.filelist):
            progdialog.setValue(i)
            for plot_method in self.plot_methods:
                fig.clear()
                if self.export_methods[plot_method]:

                    if not os.path.exists(os.path.join(out_folder, plot_method)):
                        os.mkdir(os.path.join(out_folder, plot_method))

                    self.plot_methods[plot_method](self.results[session],
                                                   fig,
                                                   params)
                    plot_name = session + '_' + plot_method.replace(' ', '_')
                    plot_name += '.' + self.config['format']
                    fig.savefig(os.path.join(out_folder, plot_method, plot_name))
            if  self.export_methods['mf_matrices'] and 'mf' in self.results[session]:
                self.results[session]['mf'].save(os.path.join(out_folder, 'timeseries', session))
                self.results[session]['baseline'].save(os.path.join(out_folder, 'timeseries', session + '_baseline'))

        progdialog.setValue(len(self.filelist))


    def preprocess(self):

        self.results = {}
        self.statusbar.showMessage('extrem preprocessing stuff going on..')
        progdialog = QtGui.QProgressDialog('', 'cancel', 0, len(self.filelist), self)
        progdialog.setMinimumDuration(0)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)

        for file_ind, filename in enumerate(self.filelist):
            progdialog.setValue(file_ind)
            if progdialog.wasCanceled():
                break
            # create timeseries, change shape and preprocess
            ts = bf.TimeSeries()
            progdialog.setLabelText('%s: loading' % filename)
            QtCore.QCoreApplication.processEvents()
            ts.load(os.path.join(self.fname, filename, 'timeseries'))
            ts.shape = tuple(ts.shape)
            progdialog.setLabelText('%s: preprocessing' % filename)
            QtCore.QCoreApplication.processEvents()
            self.results[filename] = runlib.preprocess(ts, self.config)
            self.results[filename]['mask'] = []
        progdialog.setValue(len(self.filelist))
        self.statusbar.showMessage('juhuuu, finished preprocessing', msecs=3000)
        self.activate_controls()
        if self.factorized:
            self.factorize_label.setText('preprocessig changed, factorize again!!!')
        ind = self.plot_selection_box.findText('mf_overview')
        if ind >= 0:
            self.plot_selection_box.removeItem(ind)
        self.update_plot()
        self.raw_unsort_overview_box.setEnabled(True)
        self.raw_overview_box.setEnabled(True)
        self.quality_box.setEnabled(True)

        # debugging caching
        if debugging:
            pickle.dump(self.results, open('cache.pckl', 'w'))

    def activate_controls(self):
        """activate the widgets after preprocessing"""
        self.filter_box.setEnabled(True)
        self.factorize_box.setEnabled(True)
        self.export_box.setEnabled(True)
        self.session_box.setEnabled(True)
        self.plot_selection_box.setEnabled(True)
        self.mf_overview_box.setChecked(False)
        self.mf_overview_box.setEnabled(False)
        self.mf_save_box.setChecked(False)
        self.mf_save_box.setEnabled(False)
        self.plot_export_button.setEnabled(False)

    def update_plot(self):
        """this is called when a new session or new kind of plot is selected"""
        l.debug('update plot called')
        if self.results:
            self.plot_widget.fig.clear()
            session = str(self.session_box.currentText())
            plot_method = str(self.plot_selection_box.currentText())
            params = {'threshold': float(self.plot_threshold_box.currentText())}
            self.plot_methods[plot_method](self.results[session], self.plot_widget.fig, params)
            self.plot_widget.canvas.draw()

    # TODO: maybe start a new thread for this?
    def factorize(self):

        stimuli_filter = bf.SelectTrials()
        self.factorize_label.setText('')
        mf_params = {'method': self.config['selected_method'],
                     'param': self.config['methods'][self.config['selected_method']]}
        mf_params['param']['variance'] = self.config['n_modes']
        l.info(mf_params)

        self.statusbar.showMessage('hardcore computation stuff going on..')
        progdialog = QtGui.QProgressDialog('hardcore computation stuff going on..',
                                            'cancel',
                                            0, len(self.filelist), self)
        progdialog.setMinimumDuration(0)
        progdialog.setWindowModality(QtCore.Qt.WindowModal)
        for file_ind, filename in enumerate(self.filelist):

            res = self.results[filename]
            if hasattr(res['mask'], 'timecourses'):
                pp = stimuli_filter(res['pp'], res['mask'])
            else:
                pp = res['pp']

            progdialog.setValue(file_ind)
            if progdialog.wasCanceled():
                break

            # do matrix factorization
            progdialog.setLabelText('%s: factorization' % filename)
            QtCore.QCoreApplication.processEvents()
            mf_func = utils.create_mf(mf_params)
            mf = mf_func(pp)
            mf.base.shape = tuple(mf.base.shape)
            self.results[filename]['mf'] = mf
        progdialog.setValue(len(self.filelist))
        self.statusbar.showMessage('yeah, finished!', msecs=2000)
        self.plot_selection_box.insertItem(0, 'mf_overview')
        self.plot_selection_box.setCurrentIndex(0)
        self.mf_overview_box.setEnabled(True)
        self.mf_save_box.setEnabled(True)
        self.factorized = True


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
        self.fig = self.canvas.fig
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    my_gui = MainGui()
    my_gui.show()
    app.setActiveWindow(my_gui)

    if debugging:
        my_gui.select_data_folder('/Users/dedan/projects/fu/data/dros_gui_test/')
    else:
        my_gui.select_data_folder()

    sys.exit(app.exec_())


