import sys

from PyQt4 import QtCore
from PyQt4 import QtGui


from layout import Ui_RegionGui # Module generated from reading ui file 'layout.ui',
# should include MplWidget (area where you can perfmorm matplotlib stuff)


class MyGui(QtGui.QMainWindow, Ui_RegionGui):

    def __init__(self, parent=None):
        """initialize the gui, color the boxes, etc.."""
        super(MyGui, self).__init__(parent)
        self.setupUi(self)
        self.colors = ['#4682B4', '#008080', '#FFA500', '#6B8E23', '#B22222', '#DEB887']
        self.boxes = [self.ComboBox_1, self.ComboBox_2, self.ComboBox_3,
                      self.ComboBox_4, self.ComboBox_5, self.ComboBox_6]
        for i, box in enumerate(self.boxes):
            box.addItems(['bla', 'blub'])
            box.setStyleSheet("QComboBox { color: %s; }" % self.colors[i]);

        # connect signals to slots
        QtCore.QObject.connect(self.SelectButton,
                               QtCore.SIGNAL("clicked()"),
                               self.select_file)
        QtCore.QObject.connect(self.LoadButton,
                               QtCore.SIGNAL("clicked()"),
                               self.open_file)

    def select_file(self):
        """open file select dialog and enter returned path to the line edit"""
        fname = QtGui.QFileDialog.getOpenFileName()
        if fname:
            self.FilePath.setText(fname)

    def open_file(self):
        print self.FilePath.text()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    my_view = MyGui()
    my_view.show()
    sys.exit(app.exec_())
