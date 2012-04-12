from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pylab as plt

class MplCanvas(FigureCanvas):

    def __init__(self):
        self.fig = Figure()
        axes = plt.Axes(self.fig, [0, 0, 1, 1])
        self.fig.add_axes(axes)
        self.ax = self.fig.add_subplot(111)
        x = np.arange(0.0, 3.0, 0.01)
        y = np.cos(2*np.pi*x)
        self.ax.plot(x, y)
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class MplWidget(QtGui.QWidget):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
