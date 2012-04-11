import sys

from PyQt4 import QtCore
from PyQt4 import QtGui


from layout import Ui_RegionGui # Module generated from reading ui file 'layout.ui',
# should include MplWidget (area where you can perfmorm matplotlib stuff)


class MyGui(QtGui.QMainWindow, Ui_RegionGui):

    def __init__(self, parent=None):
        super(MyGui, self).__init__(parent)
        self.setupUi(self)


    # def selectroi_example(self):
    #     #basecanvas area defined in layout
    #     self.selectcid = self.basecanvas.canvas.mpl_connect('pick_event', self.onpick)

    # def onpick(self, event):
    #     event.artist.set_ec('k')
    #     event.artist.set_linewidth(1)
    #     self.axbase.figure.canvas.draw()
    #     self.basecanvas.canvas.mpl_disconnect(self.selectcid)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    my_view = MyGui()
    my_view.show()
    sys.exit(app.exec_())
