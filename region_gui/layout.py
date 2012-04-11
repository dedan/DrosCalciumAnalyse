# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qtdesigner.ui'
#
# Created: Wed Apr 11 15:11:54 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_RegionGui(object):
    def setupUi(self, RegionGui):
        RegionGui.setObjectName(_fromUtf8("RegionGui"))
        RegionGui.resize(800, 600)
        self.centralwidget = QtGui.QWidget(RegionGui)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.mpl = MplWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mpl.sizePolicy().hasHeightForWidth())
        self.mpl.setSizePolicy(sizePolicy)
        self.mpl.setObjectName(_fromUtf8("mpl"))
        self.horizontalLayout_2.addWidget(self.mpl)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        RegionGui.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(RegionGui)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        RegionGui.setMenuBar(self.menubar)

        self.retranslateUi(RegionGui)
        QtCore.QMetaObject.connectSlotsByName(RegionGui)

    def retranslateUi(self, RegionGui):
        RegionGui.setWindowTitle(QtGui.QApplication.translate("RegionGui", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))

from mplwidget import MplWidget
