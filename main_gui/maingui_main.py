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
        self.preprocessing_box.setEnabled(False)
        self.filter_box.setEnabled(False)
        self.factorize_box.setEnabled(False)
        self.plots_box.setEnabled(False)
        self.run_button.setEnabled(False)

        self.config = json.load(open('gui_config.json'))
        self.init_controls()

        # connect signals to slots
        self.connect(self.select_data_folder_button,
                     QtCore.SIGNAL("clicked()"),
                     self.select_data_folder)

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

    def init_controls(self):
        self.normalize_box.setChecked(self.config['normalize'])
        self.lowpass_spinner.setValue(self.config['lowpass'])
        self.median_spinner.setValue(self.config['median'])
        self.spatial_spinner.setValue(self.config['spatial'])
        self.similarity_spinner.setValue(self.config['similarity'])
        self.methods_box.clear()
        self.methods_box.insertItems(0, self.config['methods'].keys())
        self.mf_overview_box.setChecked(self.config['mf_overview'])
        self.raw_overview_box.setChecked(self.config['raw_overview'])
        self.raw_unsort_overview_box.setChecked(self.config['raw_unsort_overview'])
        self.quality_box.setChecked(self.config['quality'])
        self.signals_box.setChecked(self.config['signals'])

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    my_gui = MainGui()
    my_gui.show()
    app.setActiveWindow(my_gui)
    sys.exit(app.exec_())
