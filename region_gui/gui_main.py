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
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

        self.setupUi(self)
        self.select_region_file(regions_file)
        self.labels = sorted(config["labels"].keys())

        # add plot widget
        self.plots = PlotWidget(self, self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.plots.sizePolicy().hasHeightForWidth())
        self.plots.setSizePolicy(sizePolicy)
        self.horizontalLayout_4.addWidget(self.plots)
        self.vis = Vis()
        self.vis.fig = self.plots.canvas.fig

        # connect signals to slots
        self.connect(self.selectFolderButton, QtCore.SIGNAL("clicked()"), self.select_folder)
        self.connect(self.filesListBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.load_file)
        self.connect(self.nextButton, QtCore.SIGNAL("clicked()"), self.next_button_click)

    def next_button_click(self):
        box = self.filesListBox
        box.setCurrentIndex((box.currentIndex() + 1) % (len(box) - 1))

    # def selection_changed(self):
    #     """replot and save to regions.json when a combobox changed"""
    #     l.info('selection made')
    #     box = self.sender()
    #     self.regions[self.data.name] = [str(box.currentText()) for box in self.boxes]
    #     json.dump(self.regions, open(self.regions_file, 'w'))
    #     self.draw_spatial_plots()

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

    def select_folder(self, folder=None):
        """open file select dialog and enter returned path to the line edit"""
        if folder:
            fname = folder
        else:
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
        """load the serialized TimeSeries object that contains the MF results

            * change figure to contain correct number of modes
            * draw correct number of boxes and labels
        """
        fname = os.path.join(self.folder, str(self.filesListBox.currentText()))
        l.info('loading: %s' % fname)

        # tupelization magic (set TimeSeries to correct size)
        self.data.load(fname)
        self.data.shape = tuple(self.data.shape)
        self.data.base.shape = tuple(self.data.base.shape)
        self.data.name = os.path.basename(fname)
        self.baseline.load('_'.join(fname.split('_')[:-1]) + '_baseline')
        self.baseline.shape = tuple(self.baseline.shape)

        # initialize gui for current number of modes
        n_modes = self.data.shape[0]
        # self.init_boxes_and_labels(n_modes)
        self.vis.base_and_time(n_modes, height=0.8)

        # init gui when labels already exist
        if self.data.name in self.regions:
            l.debug('already labeled, load this labeling..')
            self.current_labels = self.regions[self.data.name]
        else:
            self.current_labels = [self.labels[0] for i in range(n_modes)]
        self.draw_spatial_plots()
        self.draw_temporal_plots()

    def draw_spatial_plots(self):
        # TODO: only replot the changed subplots
        bases = self.data.base.shaped2D()
        for i in range(self.data.num_objects):
            colormap = config['labels'][self.current_labels[i]]
            ax = self.vis.axes['base'][i]
            ax.hold(False)
            self.vis.imshow(ax, np.mean(self.baseline.shaped2D(), 0),
                                     cmap=plt.cm.bone_r)
            ax.hold(True)
            self.vis.overlay_image(ax, bases[i], threshold=0.2, colormap=colormap)
            ax.set_ylabel(self.current_labels[i], rotation='0')
        self.plots.canvas.draw()

    def draw_temporal_plots(self):
        for i in range(self.data.num_objects):
            ax = self.vis.axes['time'][i]
            ax.hold(False)
            self.vis.plot(ax, self.data.timecourses[:, i])
            self.vis.add_labelshade(ax, self.data)
            ax.set_xticks([])
            ax.set_yticks([])
        # TODO: samplelabel not shown
        self.vis.add_samplelabel(self.vis.axes['time'][self.data.num_objects-1], self.data, rotation='45', toppos=True)
        self.plots.canvas.draw()

class PlotCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure()
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class PlotWidget(QtGui.QWidget):

    def __init__(self, gui, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.gui = gui
        self.current_plot = None
        self.canvas = PlotCanvas()
        self.canvas.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.fig.canvas.mpl_connect('axes_enter_event', self.enter_axes)
        self.canvas.fig.canvas.mpl_connect('axes_leave_event', self.leave_axes)
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
        self.menu = QtGui.QMenu(self.canvas.fig.canvas)
        for label in self.gui.labels:
            self.menu.addAction(label)

    def on_click(self, event):
        if self.current_plot != None:
            local_pos = (event.x, self.canvas.fig.get_figheight() * self.canvas.fig.get_dpi() - event.y)
            global_pos = self.mapToGlobal(QtCore.QPoint(*local_pos))
            action = self.menu.exec_(global_pos)

            if action:
                print self.current_plot
                self.gui.vis.axes['base'][self.current_plot].set_ylabel(action.text())
                self.gui.current_labels = [p.get_ylabel() for p in self.gui.vis.axes['base']]
                self.gui.regions[self.gui.data.name] = self.gui.current_labels
                json.dump(self.gui.regions, open(self.gui.regions_file, 'w'))
                self.gui.draw_spatial_plots()

    def enter_axes(self, event):
        self.current_plot = event.inaxes.get_gid()
        self.current_transform = event.inaxes.transData.transform

    def leave_axes(self, event):
        self.current_plot = None
        self.current_transform = None



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    regions_file = sys.argv[1] if len(sys.argv) > 1 else ""
    num_modes = sys.argv[2] if len(sys.argv) > 2 else 5
    my_view = MyGui(regions_file, num_modes)
    my_view.show()
    app.setActiveWindow(my_view)
    my_view.select_folder('/Users/dedan/projects/fu/results/simil80n_bestFalsemask/sica/')
    sys.exit(app.exec_())
