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

        self.methods = {"nnma": {}, "ica": {}}
        self.config_file = 'gui_config.json'

        self.load_controls()

        # connect signals to slots
        self.connect(self.select_data_folder_button,
                     QtCore.SIGNAL("clicked()"),
                     self.select_data_folder)
        self.lowpass_spinner.valueChanged.connect(self.save_controls)
        self.similarity_spinner.valueChanged.connect(self.save_controls)
        self.spatial_spinner.valueChanged.connect(self.save_controls)
        self.median_spinner.valueChanged.connect(self.save_controls)
        self.mf_overview_box.stateChanged.connect(self.save_controls)
        self.raw_overview_box.stateChanged.connect(self.save_controls)
        self.raw_unsort_overview_box.stateChanged.connect(self.save_controls)
        self.quality_box.stateChanged.connect(self.save_controls)
        self.signals_box.stateChanged.connect(self.save_controls)
        self.normalize_box.stateChanged.connect(self.save_controls)
        self.methods_box.currentIndexChanged.connect(self.save_controls)

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
        self.methods_box.insertItems(0, self.methods.keys())
        self.methods_box.setCurrentIndex(self.methods_box.findText(config['selected_method']))
        self.mf_overview_box.setChecked(config['mf_overview'])
        self.raw_overview_box.setChecked(config['raw_overview'])
        self.raw_unsort_overview_box.setChecked(config['raw_unsort_overview'])
        self.quality_box.setChecked(config['quality'])
        self.signals_box.setChecked(config['signals'])

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
