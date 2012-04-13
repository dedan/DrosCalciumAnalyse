import sys, os
import json
import pylab as plt
import numpy as np

from PyQt4 import QtCore
from PyQt4 import QtGui
from NeuralImageProcessing import pipeline

from layout import Ui_RegionGui # Module generated from reading ui file 'layout.ui',
from DrosCalciumAnalyse import utils

debug = True


class MyGui(QtGui.QMainWindow, Ui_RegionGui):

    def __init__(self, regions_file, parent=None):
        """initialize the gui, color the boxes, etc.."""
        super(MyGui, self).__init__(parent)
        self.data = pipeline.TimeSeries()
        self.regions = json.load(open(regions_file))

        # gui init stuff
        self.setupUi(self)
        self.boxes = [self.ComboBox_1, self.ComboBox_2, self.ComboBox_3,
                      self.ComboBox_4, self.ComboBox_5, self.ComboBox_6]
        self.labels = [self.label_1, self.label_2, self.label_3,
                       self.label_4, self.label_5, self.label_6]
        self.maps = [utils.create_colormap('iPN', (0., 0., 1.), (1., 1., 0.), (1., 0., 0.)),
                     utils.create_colormap('vlPrc', (0., 0., 1.), (0., 1., 1.), (0., 1., 0.)),
                    utils.create_colormap('iPN', (0., 0., 1.), (1., 1., 0.), (1., 0., 0.)),
                    utils.create_colormap('vlPrc', (0., 0., 1.), (0., 1., 1.), (0., 1., 0.)),
                     utils.create_colormap('acid', (0., 0., 1.), (1., 0., 0.5), (1., 0., 1.))]


        for i, label in enumerate(sorted(self.regions.keys())):
            self.boxes[i].addItems(['bla', 'blub'])
            color = self.regions[label]['color']
            self.boxes[i].setStyleSheet("QComboBox { color: %s; }" % color)
            self.labels[i].setStyleSheet("QLabel { color: %s; }" % color)
            self.labels[i].setText(label)


        # connect signals to slots
        QtCore.QObject.connect(self.SelectButton,
                               QtCore.SIGNAL("clicked()"),
                               self.select_file)
        QtCore.QObject.connect(self.LoadButton,
                               QtCore.SIGNAL("clicked()"),
                               self.open_file)

        if debug:
            self.FilePath.setText('/Users/dedan/projects/fu/results/test/onemode/OCO_111018a_nnma.json')



    def select_file(self):
        """open file select dialog and enter returned path to the line edit"""
        fname = QtGui.QFileDialog.getOpenFileName()
        if fname:
            self.FilePath.setText(fname)

    def open_file(self):
        print 'loading: ', self.FilePath.text()
        self.data.load(os.path.splitext(str(self.FilePath.text()))[0])
        print self.data.base.shape
        bases = self.data.base.trial_shaped2D().squeeze()
        self.SpatialBase.canvas.ax.clear()

        for i in range(self.data.num_objects):

            im = bases[i,:,:]
            im_rgba = self.maps[i](im / 2 + 0.5)
            im_rgba[:, :, 3] = 0.8
            im_rgba[np.abs(im) < 0.1, 3] = 0
            ax = self.SpatialBase.canvas.fig.add_subplot(self.data.num_objects+1, 1, i+1)
            ax.contour(im, [0.3], colors=['k'])
            ax.contourf(im, [0.3, 1], colors=["#4682B4"], alpha=1.)

            ax.imshow(im_rgba, aspect='equal', interpolation='nearest')
            ax.set_yticks([])
            ax.set_xticks([])

            ax = self.SpatialBase.canvas.fig.add_subplot(self.data.num_objects+1, 1, self.data.num_objects+1)
            ax.contour(im, [0.3], colors=['k'])
            ax.imshow(im_rgba, aspect='equal', interpolation='nearest')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title('overlay')

            ax = self.TemporalBase.canvas.fig.add_subplot(self.data.num_objects, 1, i)
            ax.plot(self.data.timecourses[:, i])
            ax.set_yticks([])
            ax.set_xticks([])
        self.SpatialBase.canvas.draw()
        self.TemporalBase.canvas.draw()


if __name__ == '__main__':
    # TODO: check whether regions.json is passed as an argument to the script
    # if not, display a dialog that lets you chose the location of the regions.json
    # and pass the result of the dialog to the gui.
    app = QtGui.QApplication(sys.argv)
    if debug:
        regions_file = '/Users/dedan/projects/fu/dros_calcium/region_gui/test_regions.json'
    else:
        regions_file = sys.argv[1]
    my_view = MyGui(regions_file)
    my_view.show()
    sys.exit(app.exec_())
