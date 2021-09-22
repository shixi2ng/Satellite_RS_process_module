import gdal
from osgeo import gdal_array,osr
import sys
import collections
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import tarfile
import shutil
import datetime
from datetime import date
import rasterio
import math
import copy
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
import time
from itertools import chain
from collections import Counter
import glob
from sklearn.metrics import confusion_matrix
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtGui, QtCore
import PyQt5


x_temp = np.linspace(0, 365, 10000)
y_temp = np.linspace(0, 1, 10000)
a = np.array([1,2,3,4,5])
b = np.ones([5]) * 0.5
app = QtGui.QApplication([])

pg.setConfigOption('background', 'w')
pg.setConfigOptions(antialias=True)
win = pg.GraphicsLayoutWidget()
win.ci.layout.setRowStretchFactor(0, 10)
win.ci.layout.setRowStretchFactor(1, 10)
win.ci.layout.setRowStretchFactor(2, 10)
win.ci.layout.setRowStretchFactor(3, 10)
columns = 5
rows = 5
win.resize(1900, 1000)
year_t = 0
year_range = range(2000, 2021)
line_pen = pg.mkPen((0, 0, 255), width=5)
y_tick = [list(zip((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1), (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)))]
x_tick = [list(zip((15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')))]
r_temp = 0
c_temp = 0
while r_temp < 5:
    c_temp = 0
    while c_temp < 5:
        l_temp = win.addLayout(row=r_temp, col=c_temp)
        plot_temp = pg.PlotWidget(name='Plot1')
        l_temp.addWidget(plot_temp)
        plot_temp.plot(x_temp, y_temp, pen=line_pen, name="Phenology_index")
        y_axis = plot_temp.getAxis('left')
        y_axis.setTicks(y_tick)
        plot_temp.resize(1000, 500)
        c_temp += 1
    r_temp += 1

# for r_temp in range(rows):
#     for c_temp in range(columns):
#         if year_t < len(year_range):
#             year = year_range[year_t]
#             plot_temp = win.addPlot(row=r_temp, col=c_temp, title='Annual phenology of Year ' + str(year))
#             plot_temp.setRange(xRange=(0, 365), yRange=(0, 0.95))
#             plot_temp.setLabel('left', 'NDVI')
#             plot_temp.setLabel('bottom', 'DOY')
#             x_axis = plot_temp.getAxis('bottom')
#             x_axis.setTicks(x_tick)
#             R_square = 50.000000
#             msg_r_square = (r'R^2 = ' + str(R_square)[0:5] + '%')
#             plot_temp.plot(x_temp, y_temp, pen=line_pen, name="Phenology_index")
#             # t1 = pg.TextItem(text=msg_r_square)
#             # t1.setPos(260, 0.92)
#             # plot_temp.addItem(t1)
#             scatter_array = np.stack((a, b), axis=1)
#             s1 = pg.ScatterPlotItem(pos=scatter_array, size=14, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0))
#             plot_temp.addItem(s1)
#         year_t += 1

exporter = pg.exporters.ImageExporter(win.scene())
QtGui.QApplication.processEvents()
# exporter.parameters()['width'] = 1080
# exporter.parameters()['height'] = 1920
exporter.export('E:\\A_Vegetation_Identification\\test\\test\\' + 'annual_' + str(2000) + '.jpg')


