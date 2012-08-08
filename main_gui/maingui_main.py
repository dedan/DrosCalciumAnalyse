import os, sys, glob, json
from PyQt4 import QtCore
from PyQt4 import QtGui
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
        # self.preprocessing_box.setEnabled(False)
        # self.filter_box.setEnabled(False)
        # self.factorize_box.setEnabled(False)
        # self.plots_box.setEnabled(False)
        # self.run_button.setEnabled(False)

        self.config_file = 'gui_config.json'

        self.load_controls()

        self.method_controls = {'nnma': [self.spars_par1_label, self.spars_par1_spinner,
                                         self.spars_par2_label, self.spars_par2_spinner,
                                         self.smoothness_label, self.smoothness_spinner,
                                         self.maxcount_label, self.maxcount_spinner],
                                'ica': [self.alpha_label, self.alpha_spinner]}

        # connect signals to slots
        self.connect(self.select_data_folder_button,
                     QtCore.SIGNAL("clicked()"),
                     self.select_data_folder)
        for spinner in self.findChildren((QtGui.QSpinBox, QtGui.QDoubleSpinBox)):
            spinner.valueChanged.connect(self.save_controls)
        for check_box in self.findChildren(QtGui.QCheckBox):
            check_box.stateChanged.connect(self.save_controls)
        self.methods_box.currentIndexChanged.connect(self.save_controls)

        # TODO: add or remove controls depending on the MF method selected

        # TODO: load and save also new mf settings

    def select_data_folder(self):
        caption = 'select your data folder'
        fname = str(QtGui.QFileDialog.getExistingDirectory(caption=caption))
        self.data_folder_label.setText(fname)
        json_files = glob.glob(os.path.join(fname, '*.json'))
        npy_files = glob.glob(os.path.join(fname, '*.npy'))
        if len(json_files) == 0 or len(npy_files) == 0 or len(json_files) != len(npy_files):
            self.data_folder_label.setText('no valid data found in: %s' % fname)
        else:
            message = '%d files found in %s' % (len(json_files), fname)
            self.statusbar.showMessage(message, msecs=5000)
            self.preprocessing_box.setEnabled(True)
            self.filter_box.setEnabled(True)
            self.factorize_box.setEnabled(True)
            self.plots_box.setEnabled(True)

    def load_controls(self):
        config = json.load(open(self.config_file))
        self.normalize_box.setChecked(config['normalize'])
        self.lowpass_spinner.setValue(config['lowpass'])
        self.median_spinner.setValue(config['median'])
        self.spatial_spinner.setValue(config['spatial'])
        self.similarity_spinner.setValue(config['similarity'])
        self.methods_box.clear()
        self.methods_box.insertItems(0, config['methods'].keys())
        self.methods_box.setCurrentIndex(self.methods_box.findText(config['selected_method']))
        self.mf_overview_box.setChecked(config['mf_overview'])
        self.raw_overview_box.setChecked(config['raw_overview'])
        self.raw_unsort_overview_box.setChecked(config['raw_unsort_overview'])
        self.quality_box.setChecked(config['quality'])
        self.signals_box.setChecked(config['signals'])
        self.spars_par1_spinner.setValue(config['methods']['nnma']['spars_par1'])
        self.spars_par2_spinner.setValue(config['methods']['nnma']['spars_par2'])
        self.smoothness_spinner.setValue(config['methods']['nnma']['smoothness'])
        self.maxcount_spinner.setValue(config['methods']['nnma']['maxcount'])
        self.alpha_spinner.setValue(config['methods']['ica']['alpha'])
        self.n_modes_spinner.setValue(config['n_modes'])


    # TODO: after each click, save settings to config file
    def save_controls(self, export_file=''):
        print 'save_controls called, export file is: %s' % export_file
        config = {}
        config['normalize'] = self.normalize_box.isChecked()
        config['lowpass'] = self.lowpass_spinner.value()
        config['median'] = self.median_spinner.value()
        config['spatial'] = self.spatial_spinner.value()
        config['similarity'] = self.similarity_spinner.value()
        config['selected_method'] = str(self.methods_box.currentText())
        config['mf_overview'] = self.mf_overview_box.isChecked()
        config['raw_overview'] = self.raw_overview_box.isChecked()
        config['raw_unsort_overview'] = self.raw_unsort_overview_box.isChecked()
        config['quality'] =  self.quality_box.isChecked()
        config['signals'] = self.signals_box.isChecked()
        config['methods'] = {'nnma': {}, 'ica': {}}
        config['methods']['nnma']['spars_par1'] = self.spars_par1_spinner.value()
        config['methods']['nnma']['spars_par2'] = self.spars_par2_spinner.value()
        config['methods']['nnma']['smoothness'] = self.smoothness_spinner.value()
        config['methods']['nnma']['maxcount'] = self.maxcount_spinner.value()
        config['methods']['ica']['alpha'] = self.alpha_spinner.value()
        config['n_modes'] = self.n_modes_spinner.value()

        json.dump(config, open(self.config_file, 'w'))
        if isinstance(export_file, str) and os.path.exists(export_file):
            json.dump(config, open(export_file, 'w'))

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    my_gui = MainGui()
    my_gui.show()
    my_gui.save_controls()
    app.setActiveWindow(my_gui)
    sys.exit(app.exec_())
