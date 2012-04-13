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
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

config = {"labels": {"label1": "#4682B4",
                     "label2": "#008080",
                     "label3": "#FFA500",
                     "label4": "#6B8E23",
                     "label5": "#B22222",
                     "label6": "#DEB887"}}

class MyGui(QtGui.QMainWindow, Ui_RegionGui):

    def __init__(self, regions_file, parent=None):
        """initialize the gui, color the boxes, etc.."""
        super(MyGui, self).__init__(parent)
        self.data = pipeline.TimeSeries()
        self.regions_file = regions_file
        self.regions = json.load(open(regions_file))

        # gui init stuff
        self.setupUi(self)
        self.boxes = [self.ComboBox_1, self.ComboBox_2, self.ComboBox_3,
                      self.ComboBox_4, self.ComboBox_5, self.ComboBox_6]
        self.labels = [self.label_1, self.label_2, self.label_3,
                       self.label_4, self.label_5, self.label_6]

        # initialize the boxes
        size = self.ComboBox_1.style().pixelMetric(QtGui.QStyle.PM_SmallIconSize)
        pixmap = QtGui.QPixmap(size-3,size-3)
        for box in self.boxes:
            for i, label in enumerate(sorted(config["labels"].keys())):
                box.addItem(label, QtGui.QColor(i))
                pixmap.fill(QtGui.QColor(config["labels"][label]));
                box.setItemData(i, pixmap, QtCore.Qt.DecorationRole)

        # connect signals to slots
        QtCore.QObject.connect(self.SelectButton,
                               QtCore.SIGNAL("clicked()"),
                               self.select_file)
        QtCore.QObject.connect(self.LoadButton,
                               QtCore.SIGNAL("clicked()"),
                               self.open_file)

        if debug:
            test_path = '/Users/dedan/projects/fu/results/test/onemode/OCO_111018a_nnma.json'
            self.FilePath.setText(test_path)

    # TODO: add functions to update the regions.json

    def select_file(self):
        """open file select dialog and enter returned path to the line edit"""
        fname = QtGui.QFileDialog.getOpenFileName()
        if fname:
            self.FilePath.setText(fname)

    def open_file(self):
        """load the serialized TimeSeries object that contains the ICA results"""
        fname = os.path.splitext(str(self.FilePath.text()))[0]
        name = "_".join(os.path.basename(fname).split("_")[0:2])
        l.info('loading: %s' % fname)
        l.debug('name: %s' % name)
        self.data.load(fname)

        # TODO: update the number of active comboboxes and labels

        # TODO: load json
        plot_colors = ['white'] * self.data.num_objects
        if name in self.regions:
            l.debug('name found in regions')
            for i, label in enumerate(self.regions[name]):
                idx = self.boxes[i].findText(label)
                if idx < 0:
                    l.warning('unknown label')
                self.boxes[i].setCurrentIndex(idx)
                plot_colors[i] = config['labels'][label]
        l.debug(plot_colors)
        self.draw_plots(plot_colors)


    def draw_plots(self, plot_colors):
        sc = self.SpatialBase.canvas
        tc = self.TemporalBase.canvas
        sc.ax.clear()
        tc.ax.clear()
        bases = self.data.base.trial_shaped2D().squeeze()
        aspect_ratio = self.data.base.shape[0] / float(self.data.base.shape[1])
        n_objects = self.data.num_objects

        for i in range(n_objects):

            ax = sc.fig.add_subplot(n_objects + 1, 1, i + 1, aspect=aspect_ratio)
            ax.contour(bases[i,:,:], [0.3], colors=['k'])
            ax.contourf(bases[i,:,:], [0.3, 1], colors=[plot_colors[i]], alpha=1.)
            ax.set_yticks([])
            ax.set_xticks([])

            ax = sc.fig.add_subplot(n_objects + 1, 1, n_objects + 1, aspect=aspect_ratio)
            ax.contour(bases[i,:,:], [0.3], colors=['k'])
            ax.contourf(bases[i,:,:], [0.3, 1], colors=[plot_colors[i]], alpha=0.4)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title('overlay')

            ax = tc.fig.add_subplot(n_objects, 1, i)
            ax.plot(self.data.timecourses[:, i])
            ax.set_yticks([])
            ax.set_xticks([])
        sc.draw()
        tc.draw()


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
    my_view.open_file()
    my_view.show()
    sys.exit(app.exec_())
