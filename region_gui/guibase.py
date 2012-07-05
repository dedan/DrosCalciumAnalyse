import sys, os
import json
import glob
import pylab as plt
import numpy as np

from PyQt4 import QtCore
from PyQt4 import QtGui
from NeuralImageProcessing import pipeline
from NeuralImageProcessing.illustrate_decomposition import VisualizeTimeseries as Vis
from matplotlib.colors import rgb2hex

from layout import Ui_RegionGui # Module generated from reading ui file 'layout.ui',
from DrosCalciumAnalyse import utils

debug = True
import logging as l
l.basicConfig(level=l.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S');

config = {"labels": {"vlPRCt": utils.redmap,
                     "vlPRCb": utils.brownmap,
                     "iPN": utils.yellowmap,
                     "iPNsecond": utils.bluemap,
                     "iPNtract": utils.greenmap,
                     "betweenTract": utils.cyanmap,
                     "blackhole": utils.violetmap,
                     "!nolabel": plt.cm.hsv_r
                     }}

class MyGui(QtGui.QMainWindow, Ui_RegionGui):

    def __init__(self, regions_file, num_modes, parent=None):
        """initialize the gui, connect signals, add axes objects, etc.."""
        super(MyGui, self).__init__(parent)
        self.data = pipeline.TimeSeries()
        self.baseline = pipeline.TimeSeries()
        
        # gui init stuff
        self.setupUi(self)
        self.boxes = [self.ComboBox_1, self.ComboBox_2, self.ComboBox_3,
                      self.ComboBox_4, self.ComboBox_5, self.ComboBox_6]
        self.labels = [self.label_1, self.label_2, self.label_3,
                       self.label_4, self.label_5, self.label_6]

        self.select_region_file(regions_file)

        # initialize the boxes
        size = self.ComboBox_1.style().pixelMetric(QtGui.QStyle.PM_SmallIconSize)
        pixmap = QtGui.QPixmap(size - 3, size - 3)
        for box in self.boxes:
            for i, label in enumerate(sorted(config["labels"].keys())):
                box.addItem(label, QtGui.QColor(i))
                pixmap.fill(QtGui.QColor(rgb2hex(config["labels"][label](1.))))
                box.setItemData(i, pixmap, QtCore.Qt.DecorationRole)
            # connect callback
            self.connect(box,
                         QtCore.SIGNAL("currentIndexChanged(int)"),
                         self.selection_changed)

        # connect signals to slots
        self.connect(self.selectFolderButton, QtCore.SIGNAL("clicked()"), self.select_folder)
        self.connect(self.filesListBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.load_file)
        self.connect(self.nextButton, QtCore.SIGNAL("clicked()"), self.next_button_click)
        
        # create plot arena
        self.baseaxes = [self.SpatialBase.canvas.fig.add_subplot(num_modes, 1, i + 1)
                          for i in range(num_modes)]
        self.timeaxes = [self.TemporalBase.canvas.fig.add_subplot(num_modes, 1, i + 1)
                          for i in range(num_modes)]
        # instantiate plot functions
        self.vis = Vis()
        
        if debug:
            test_path = '/Users/dedan/projects/fu/results/test/onemode/OCO_111018a_nnma.json'
            # self.FilePath.setText(test_path)

    def next_button_click(self):
        box = self.filesListBox
        box.setCurrentIndex((box.currentIndex() + 1) % (len(box) - 1))

    def selection_changed(self):
        """replot and save to regions.json when a combobox changed"""
        l.info('selection made')
        box = self.sender()
        self.regions[self.data.name] = [str(box.currentText()) for box in self.boxes]
        json.dump(self.regions, open(self.regions_file, 'w'))
        self.draw_spatial_plots()

    def select_region_file(self, regions_file=None):

        if regions_file:
            self.regions_file = regions_file
            self.regions = json.load(open(regions_file))
        else:
            fname = QtGui.QFileDialog.getOpenFileNameAndFilter(caption='select regions.json',
                                                               filter='*.json')
            fname = str(fname[0])
            if fname and os.path.exists(fname) and fname[-4:] == 'json':
                self.regions_file = fname
                self.regions = json.load(open(self.regions_file))
            else:
                l.error('no regions.json selected --> quitting')
                sys.exit(-1)

    def select_folder(self):
        """open file select dialog and enter returned path to the line edit"""
        fname = str(QtGui.QFileDialog.getExistingDirectory())
        if fname:
            self.folder = fname
            filelist = glob.glob(os.path.join(self.folder, '*.json'))
            filelist = [os.path.splitext(os.path.basename(f))[0] for f in filelist]
            filelist = [f for f in filelist if not 'base' in f]
            filelist = [f for f in filelist if not 'regions' in f]
            self.filesListBox.clear()
            self.filesListBox.addItems(filelist)
            self.nextButton.setEnabled(True)
            self.filesListBox.setEnabled(True)

    def load_file(self):
        """load the serialized TimeSeries object that contains the MF results"""
        fname = os.path.join(self.folder, str(self.filesListBox.currentText()))
        l.info('loading: %s' % fname)
        self.data.load(fname)
        self.data.shape = tuple(self.data.shape)
        self.data.base.shape = tuple(self.data.base.shape)
        self.data.name = os.path.basename(fname)
        self.baseline.load('_'.join(fname.split('_')[:-1]) + '_baseline')
        self.baseline.shape = tuple(self.baseline.shape)
        # TODO: update the number of active comboboxes and labels
        # init gui when labels already exist
        if self.data.name in self.regions:
            l.debug('name found in regions')
            for i, label in enumerate(self.regions[self.data.name]):
                idx = self.boxes[i].findText(label)
                if idx < 0:
                    l.warning('unknown label')
                old_state = self.boxes[i].blockSignals(True);
                self.boxes[i].setCurrentIndex(idx)
                self.boxes[i].blockSignals(old_state)
        else:
            for box in self.boxes:
                box.setCurrentIndex(0)
        self.draw_spatial_plots()
        self.draw_temporal_plots()

    def draw_spatial_plots(self):
        # TODO: only replot the changed subplots
        bases = self.data.base.shaped2D()
        #aspect_ratio = self.data.base.shape[0] / float(self.data.base.shape[1])
        for i in range(self.data.num_objects):
            colormap = config['labels'][str(self.boxes[i].currentText())]
            ax = self.baseaxes[i]
            ax.hold(False)
            self.vis.imshow(ax, np.mean(self.baseline.shaped2D(), 0),
                                     cmap=plt.cm.bone_r)
            ax.hold(True)
            self.vis.overlay_image(ax, bases[i], threshold=0.2, colormap=colormap)
        self.SpatialBase.canvas.draw()

    def draw_temporal_plots(self):
        for i in range(self.data.num_objects):
            ax = self.timeaxes[i]
            ax.hold(False)
            self.vis.plot(ax, self.data.timecourses[:, i])
            self.vis.add_labelshade(ax, self.data)
        self.vis.add_samplelabel(self.timeaxes[0], self.data, rotation='45', toppos=True)
        self.TemporalBase.canvas.draw()



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    regions_file = sys.argv[1] if len(sys.argv) > 1 else ""
    num_modes = sys.argv[2] if len(sys.argv) > 2 else 5
    my_view = MyGui(regions_file, num_modes)
    my_view.show()
    app.setActiveWindow(my_view)
    sys.exit(app.exec_())
