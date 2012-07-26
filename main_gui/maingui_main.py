import os, sys
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

        # connect signals to slots
        self.connect(self.select_data_folder_button,
                     QtCore.SIGNAL("clicked()"),
                     self.select_data_folder)
        # self.connect(self.selectFolderButton, QtCore.SIGNAL("clicked()"), self.select_folder)
        # self.connect(self.filesListBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.load_file)
        # self.connect(self.nextButton, QtCore.SIGNAL("clicked()"), self.next_button_click)

    def select_data_folder(self):
        caption = 'select your data folder'
        fname = QtGui.QFileDialog.getExistingDirectory(caption=caption)
        self.data_folder_label.setText(fname)
        # TODO: check whether there is really data in the folder

        # TODO: when data found, enable the other boxes

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    my_gui = MainGui()
    my_gui.show()
    app.setActiveWindow(my_gui)
    sys.exit(app.exec_())
