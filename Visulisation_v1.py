import sys
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtGui, QtCore
import PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import numpy as np
import pandas
import os
import gdal
import cv2
import random
import Landsat_main_v1
from scipy.optimize import curve_fit
import shutil
import copy


counter = 0


# Aim to create a processing popup when the main gui dealing with long time procedure
class ProcessingScreen(QMainWindow):
    def __init__(self):
        super(ProcessingScreen, self).__init__()
        uic.loadUi('ui\\processing.ui', self)
        self.setWindowFlag(PyQt5.QtCore.Qt.FramelessWindowHint)
        self.setAttribute(PyQt5.QtCore.Qt.WA_TranslucentBackground)
        self.show()


# Create a initialising screen, has no use but fancy.
class InitialisingScreen(QMainWindow):
    def __init__(self):
        # Main
        super(InitialisingScreen, self).__init__()
        uic.loadUi('ui\\initialising.ui', self)
        # Remove the title bar
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Around shadow
        self.around_shadow = QGraphicsDropShadowEffect(self)
        self.around_shadow.setBlurRadius(20)
        self.around_shadow.setXOffset(0)
        self.around_shadow.setYOffset(0)
        self.around_shadow.setColor(QColor(0, 0, 0, 60))
        self.mainframe.setGraphicsEffect(self.around_shadow)
        self.main_window = ''
        self.timer = PyQt5.QtCore.QTimer()
        self.timer.timeout.connect(self.initialise)
        # the interval of timer is set as 19 millisecond
        self.timer.start(19)
        # Creat a timer to show different label at the bottom
        PyQt5.QtCore.QTimer.singleShot(1000, lambda: self.initlabel.setText('Loading data.'))
        PyQt5.QtCore.QTimer.singleShot(1300, lambda: self.initlabel.setText('Loading data..'))
        PyQt5.QtCore.QTimer.singleShot(1600, lambda: self.initlabel.setText('Loading data...'))
        PyQt5.QtCore.QTimer.singleShot(1900, lambda: self.initlabel.setText('Done'))
        self.show()

    # Set a timer jumping to main gui
    def initialise(self):
        global counter
        self.progressBar.setValue(counter)
        # 0 to 100 % Timer
        if counter > 100:
            self.timer.stop()
            self.main = Visualisation_gui()
            self.close()
        counter += 1


# Create a Qgraphics scene embedded in the Qgraphcis view
class Demo_scene(QGraphicsScene):
    def __init__(self, ra_factor):
        super(Demo_scene, self).__init__()
        self.ra_f = ra_factor

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent'):
        pen = PyQt5.QtGui.QPen(PyQt5.QtCore.Qt.black)
        brush = PyQt5.QtGui.QBrush(PyQt5.QtCore.Qt.black)
        x = event.scenePos().x()
        y = event.scenePos().y()
        if self.ra_f == 'Pixel':
            self.addEllipse(x, y, 4, 4, pen, brush)
            print(x, y)
            return x, y


# Create a Qgraphics view for the demo
class Demo_Viewer(QGraphicsView):
    # Create a click signal
    imageClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(Demo_Viewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = Demo_scene(self)
        self._image = QGraphicsPixmapItem()
        self._scene.addItem(self._image)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def clear_scene(self):
        self._scene.clear()
        self._image = QGraphicsPixmapItem()
        self._scene.addItem(self._image)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._image.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._image.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._image.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._image.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._image.isUnderMouse():
            self.imageClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(Demo_Viewer, self).mousePressEvent(event)


# Create a Qgraphics view for the phenology information
class Phe_Viewer(QGraphicsView):

    def __init__(self, parent):
        super(Phe_Viewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self.phenology_image = QGraphicsPixmapItem()
        self.phenology_scene = QGraphicsScene()
        self.phenology_scene.addItem(self.phenology_image)
        self.setScene(self.phenology_scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def clear_scene(self):
        self.phenology_scene.clear()
        self.phenology_image = QGraphicsPixmapItem()
        self.phenology_scene.addItem(self.phenology_image)

    def hasPhoto(self):
        return self.phenology_scene.items() != []

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self.phenology_image.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.phenology_image.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.phenology_image.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self.phenology_image.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)


# The main GUI
class Visualisation_gui(QMainWindow):

    def __init__(self):
        # Input the gui template created in the QT designer
        super(Visualisation_gui, self).__init__()
        uic.loadUi('ui\\visulisation_backup.ui', self)

        # module configurations
        pg.setConfigOption('background', 'w')
        pg.setConfigOptions(antialias=True)

        # Attributes related to the input filepath
        self.rootpath = ''
        self.demo_resize_factor = np.nan
        self.orifile_path = ''
        self.keydic_path = ''
        self.shpfile_path = ''
        self.sa_demo_folder = ''
        self.levelfactor = None
        self.fundamental_dic = None
        self.vi_list = None
        self.sa_list = None
        self.sa_current = self.sa_combobox.currentText()
        self.vi_current = self.sa_combobox.currentText()
        self.demo_image = np.array([])
        self.date = ''
        self.shpfile = ''
        self.rgb_dic = {}
        self.all_date = []
        self.gamma_para = 1.52

        # Study area demo related attributes
        self.demo_pos_x = np.nan
        self.demo_pos_y = np.nan

        self.sa_demo = Demo_Viewer(self)
        # self.demoscene = Demo_scene(self.ra_domain)
        self.sa_demo.setGeometry(QtCore.QRect(17, 45, 451, 381))
        self.sa_demo.setStyleSheet("QGraphicsView{\n""border-color: rgb(0, 0, 0);\n""background-color: rgb(245, 245, 245)\n""}")
        self.sa_demo.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.sa_demo.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.sa_demo.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.sa_demo.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.sa_demo.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.sa_demo.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.sa_demo.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.sa_demo.setObjectName("sa_demo")

        # Phenology-related attributes
        self.sdc_dic = {}
        self.vi_temp_dic = np.array([])
        self.doy_list = []
        self.year_list = []
        self.inundated_para = ''
        self.inundated_dic = {}
        self.inundated_map = np.array([])
        self.inundated_dc = np.array([])
        self.inundated_doy = np.array([])
        self.rescale_factor = False
        self.doy = np.nan
        self.ori_vi_image = np.array([]).astype(np.float64)
        self.vi_demo_image = np.array([])
        self.phenology_resize_factor = np.nan
        self.phenology_information_view = Phe_Viewer(self)
        self.phenology_information_view.setGeometry(QtCore.QRect(477, 45, 865, 631))
        self.phenology_information_view.setStyleSheet("QGraphicsView{\n""border-color: rgb(0, 0, 0);\n""background-color: rgb(245, 245, 245)\n""}")
        self.phenology_information_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.phenology_information_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.phenology_information_view.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.phenology_information_view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.phenology_information_view.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.phenology_information_view.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.phenology_information_view.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.phenology_information_view.setObjectName("phenology_information_view")

        # Phenology factor
        self.phenology_view_factor = ''
        self.ra_domain = ''
        self.regression_type = ''
        self.begin_year = ''
        self.end_year = ''
        self.vi_sa_dic = np.array([])

        # Phenology fig related attributes
        self.line_pen = pg.mkPen((0, 0, 255), width=5)
        self.x_tick = [list(zip((15, 44, 75, 105, 136, 166, 197, 228, 258, 289, 320, 351), ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')))]
        self.scene_determined_factor = False
        self.phe_rows = np.nan
        self.phe_columns = np.nan
        self.phe_win = ''
        self.phe_dic = {}
        self.varied_factor = ''
        self.doy_array_for_phenology = np.array([])
        self.vi_sa_array_for_phenology = np.array([])
        self.phenology_plot_x = np.array([])
        self.phenology_plot_y = np.array([])
        self.year_range = range(0)
        self.inundated_process_factor = False
        self.r_square = np.nan
        self.curve_plot_x = np.array([])
        self.curve_plot_y = np.array([])
        self.phenology_view_name = ''

        self.vi_sa_array_for_phenology_pixel = np.array([])
        self.phenology_window_height = self.phenology_information_view.geometry().height()
        self.phenology_window_width = self.phenology_information_view.geometry().width()

        # Output related attributes
        self.data_output_factor = False
        self.output_phe_fig_path = ''
        self.output_visual_path = ''
        self.output_x_data = np.array([])
        self.output_y_data = np.array([])
        self.output_x_id = ''
        self.output_y_id = ''

        self.update_root_path()
        self.show()
        self.enable_initialisation()

        # close
        self.actionclose.triggered.connect(exit)

        # input data connection
        self.actionUpdate_the_root_path.triggered.connect(self.update_root_path)
        self.actionUpdate_Key_dictionary_path.triggered.connect(self.update_key_dic_path)
        self.actionUpdate_Original_file_path.triggered.connect(self.update_ori_file_path)
        self.actionUpdate_fundamental_dic.triggered.connect(self.update_fundamental_dic)
        self.actionUpdate_shpfile_path.triggered.connect(self.update_shpfile_path)

        # demo related connection
        self.try_Button.clicked.connect(self.update_demo_image)
        self.manual_input_date_Edit.returnPressed.connect(self.update_demo_image)
        self.random_input_Button.clicked.connect(self.random_show_demo)
        self.sa_combobox.activated.connect(self.update_shp_infor)
        self.gamma_correction_button.clicked.connect(self.gamma_correction)
        self.gamma_para_spinbox.valueChanged.connect(self.gamma_para_update)
        self.sa_demo.imageClicked.connect(self.photoClicked)

        # vi related connection
        self.vi_combobox.currentTextChanged.connect(self.update_vi_sa_dic)
        self.inundated_combobox.currentTextChanged.connect(self.update_inundated_combobox_dic)
        self.doy_line_box.returnPressed.connect(self.update_doy)
        self.display_button.clicked.connect(self.update_doy)
        self.display_button.clicked.connect(self.update_vi_display)
        self.rescale_button.clicked.connect(self.rescale)

        # phenology related connection
        self.time_span_begin.currentTextChanged.connect(self.update_time_span_begin)
        self.time_span_begin.currentTextChanged.connect(self.year_change_generate_fig)
        self.time_span_end.currentTextChanged.connect(self.update_time_span_end)
        self.time_span_end.currentTextChanged.connect(self.year_change_generate_fig)
        self.generate_button.clicked.connect(self.generate_fig_button)
        self.ra_domain_box.currentTextChanged.connect(self.pixel_change_generate_fig)
        self.phenology_view_box.currentTextChanged.connect(self.phenology_view_change_generate_fig)
        self.regression_box.currentTextChanged.connect(self.regression_change_generate_fig)
        self.inundated_combobox.currentTextChanged.connect(self.inundated_change_generate_fig)

        # output connection
        self.output_fig_button.clicked.connect(self.output_figure)
        self.output_data_button.clicked.connect(self.output_data)

    # All the functions of the main GUI were separated into
    # Section 1 Update displaying the phenology curve
    # (4) input data and close connection

    # Section 1 Update displaying the phenology curve
    def generate_fig_button(self):
        self.update_phenology_factor()
        self.determine_scene()
        self.preprocess_phe_data()
        self.plot_phenology_figure()
        self.update_output_button()

    def year_change_generate_fig(self):
        if str(self.begin_year) != '0000' and str(self.begin_year) != '0000':
            self.varied_factor = 'year'
            self.generate_fig_button()

    def phenology_view_change_generate_fig(self):
        self.varied_factor = 'phenology_view'
        self.generate_fig_button()

    def pixel_change_generate_fig(self):
        self.varied_factor = 'pixel'
        self.generate_fig_button()

    def regression_change_generate_fig(self):
        self.varied_factor = 'regression'
        self.generate_fig_button()

    def inundated_change_generate_fig(self):
        self.varied_factor = 'inundated'
        self.generate_fig_button()

    def update_time_span_begin(self):
        self.begin_year = self.time_span_begin.currentText()
        self.end_year = self.time_span_end.currentText()
        if self.begin_year == '' or self.end_year == '':
            pass
        elif int(self.end_year) < int(self.begin_year):
            self.begin_year = self.end_year
            self.time_span_begin.setCurrentText(self.begin_year)
            self.caution_msg_box('The begin year should not larger than the end year!')

    def update_time_span_end(self):
        self.begin_year = self.time_span_begin.currentText()
        self.end_year = self.time_span_end.currentText()
        if self.begin_year == '' or self.end_year == '':
            pass
        elif int(self.end_year) < int(self.begin_year):
            self.end_year = self.begin_year
            self.time_span_end.setCurrentText(self.end_year)
            self.caution_msg_box('The end year should not smaller than the begin year!')

    def photoClicked(self, pos):
        if self.sa_demo.dragMode() == QtWidgets.QGraphicsView.NoDrag and self.ra_domain == 'Pixel':
            self.demo_pos_x = pos.x()
            self.demo_pos_y = pos.y()
            self.x_coordinate.setText(str(int(self.demo_pos_x / self.demo_resize_factor)))
            self.y_coordinate.setText(str(int(self.demo_pos_y / self.demo_resize_factor)))
            self.pixel_change_generate_fig()

    def update_phenology_factor(self):
        if self.phenology_view_box.isEnabled():
            self.phenology_view_factor = self.phenology_view_box.currentText()
            self.ra_domain = self.ra_domain_box.currentText()
            self.inundated_para = self.inundated_combobox.currentText()
            if self.ra_domain == 'Entire SA':
                self.generate_button.setEnabled(True)
                if self.sa_demo.dragMode() == QtWidgets.QGraphicsView.NoDrag:
                    self.sa_demo.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            elif self.ra_domain == 'Pixel':
                self.generate_button.setEnabled(False)
                if self.sa_demo.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
                    self.sa_demo.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.regression_type = self.regression_box.currentText()

            if self.vi_temp_dic == np.array([]):
                self.update_vi_sa_dic()
                if self.vi_temp_dic == np.array([]):
                    self.caution_msg_box('There is something wrong concerning about the sdc dic')
                    return
            self.update_inundated_combobox_dic()

    def determine_scene(self):
        if self.phenology_view_box.isEnabled() and self.begin_year != '' and self.end_year != '':
            self.year_range = range(int(self.begin_year), int(self.end_year) + 1)
            if not self.scene_determined_factor or self.varied_factor == 'year' or self.varied_factor == 'phenology_view':
                reconstructed_factor = True
            elif self.varied_facor == 'year' and self.phenology_factor == 'Annual':
                reconstructed_factor = True
            else:
                reconstructed_factor = False

            # Reconstruct the window
            if reconstructed_factor:
                # Reform the size
                if self.phenology_view_factor == 'Annual':
                    self.phe_columns = int(np.ceil(np.sqrt(len(self.year_range))))
                    self.phe_rows = int(len(self.year_range) // self.phe_columns + 1 * (np.mod(len(self.year_range), self.phe_columns) != 0))
                elif self.phenology_view_factor == 'Overview':
                    self.phe_rows = 1
                    self.phe_columns = 1
                else:
                    self.caution_msg_box('Unknown Error occurred during determine scene!')
                    return

                # Reconstructed the win
                self.phe_win = pg.GraphicsLayoutWidget(show=False, title="Annual phenology")
                self.phe_win.clear()
                self.phe_win.setRange(pg.Qt.QtCore.QRectF(25 * self.phe_columns, 15 * self.phe_rows, 475 * self.phe_columns, 285 * self.phe_rows), disableAutoPixel=False)
                self.phe_win.resize(525 * self.phe_columns, 325 * self.phe_rows)
                font = QtGui.QFont('Times New Roman')
                font.setPixelSize(18)
                label_style = {"font-size": "18pt", "font-family": "Times New Roman"}
                self.phe_dic = {}
                if self.phenology_view_factor == 'Overview':
                    self.phe_win.setRange(pg.Qt.QtCore.QRectF(25, 15, 475, 285), disableAutoPixel=False)
                    # self.phe_win.resize(525 * self.phe_columns, 325 * self.phe_rows)
                    self.phe_dic['plot_temp_0000'] = self.phe_win.addPlot(row=0, col=0)
                    self.phe_dic['plot_temp_0000'].setLabel('left', self.vi_current, **label_style)
                    self.phe_dic['plot_temp_0000'].setLabel('bottom', 'DOY', **label_style)
                    self.phe_dic['plot_temp_0000'].setTitle(title="<font face='Times New Roman' size='15' color='black'> Phenology Overview </font>", **label_style)
                    x_axis = self.phe_dic['plot_temp_0000'].getAxis('bottom')
                    x_axis.setTicks(self.x_tick)
                    self.phe_dic['curve_temp_0000'] = pg.PlotCurveItem(pen=self.line_pen, name="Phenology_index")
                    self.phe_dic['plot_temp_0000'].addItem(self.phe_dic['curve_temp_0000'])
                    self.phe_dic['plot_temp_0000'].setRange(xRange=(0, 365), yRange=(0, 0.95))
                    self.phe_dic['scatterplot_temp_0000'] = pg.ScatterPlotItem(size=0.01, pxMode=False)
                    self.phe_dic['scatterplot_temp_0000'].setPen(pg.mkPen('r', width=10))
                    self.phe_dic['scatterplot_temp_0000'].setBrush(pg.mkBrush(255, 0, 0))
                    self.phe_dic['plot_temp_0000'].addItem(self.phe_dic['scatterplot_temp_0000'])
                elif self.phenology_view_factor == 'Annual':
                    self.phe_win.setRange(pg.Qt.QtCore.QRectF(25 * self.phe_columns, 15 * self.phe_rows, 475 * self.phe_columns, 285 * self.phe_rows), disableAutoPixel=False)
                    self.phe_win.resize(525 * self.phe_columns, 325 * self.phe_rows)
                    year_index_temp = 0
                    for r_temp in range(self.phe_rows):
                        for c_temp in range(self.phe_columns):
                            if year_index_temp < len(self.year_range):
                                year = self.year_range[year_index_temp]
                                self.phe_dic['plot_temp_' + str(year)] = self.phe_win.addPlot(row=r_temp, col=c_temp)
                                self.phe_dic['plot_temp_' + str(year)].setLabel('left', self.vi_current, **label_style)
                                self.phe_dic['plot_temp_' + str(year)].setLabel('bottom', 'DOY', **label_style)
                                self.phe_dic['plot_temp_' + str(year)].setTitle(title="<font face='Times New Roman' size='15' color='black'>Annual phenology of Year " + str(year) + '</font>', **label_style)
                                # self.phe_dic['plot_temp_' + str(year)].setTitle(title='Annual phenology of Year ' + str(year), **label_style)
                                x_axis = self.phe_dic['plot_temp_' + str(year)].getAxis('bottom')
                                x_axis.setTicks(self.x_tick)
                                self.phe_dic['plot_temp_' + str(year)].getAxis('bottom').setStyle(tickFont=font)
                                self.phe_dic['plot_temp_' + str(year)].getAxis('left').setStyle(tickFont=font)
                                self.phe_dic['curve_temp_' + str(year)] = pg.PlotCurveItem(pen=self.line_pen, name="Phenology_index")
                                self.phe_dic['plot_temp_' + str(year)].addItem(self.phe_dic['curve_temp_' + str(year)])
                                self.phe_dic['scatterplot_temp_' + str(year)] = pg.ScatterPlotItem(size=0.01, pxMode=False)
                                self.phe_dic['scatterplot_temp_' + str(year)].setPen(pg.mkPen('r', width=10))
                                self.phe_dic['scatterplot_temp_' + str(year)].setBrush(pg.mkBrush(255, 0, 0))
                                self.phe_dic['plot_temp_' + str(year)].addItem(self.phe_dic['scatterplot_temp_' + str(year)])
                                self.phe_dic['text_temp_' + str(year)] = pg.TextItem()
                                self.phe_dic['text_temp_' + str(year)].setPos(260, 0.92)
                                self.phe_dic['plot_temp_' + str(year)].addItem(self.phe_dic['text_temp_' + str(year)])
                                self.phe_dic['plot_temp_' + str(year)].setRange(xRange=(0, 365), yRange=(0, 0.95))
                            year_index_temp += 1
                else:
                    self.caution_msg_box('Impossible Error occurred during determine scene!')
                    return
            else:
                pass

    def preprocess_phe_data(self):
        if self.phenology_view_box.isEnabled() and self.begin_year != '' and self.end_year != '':
            if self.vi_temp_dic.shape[2] == len(self.doy_list):
                # Preprocess inundated data
                if self.inundated_dic != {}:
                    # for i in range(len(self.doy_list)):
                    #     if len(Landsat_main_v1.file_filter(self.inundated_dic[self.inundated_para + '_' + self.sa_current], [str(self.doy_list[i]), '.TIF'], and_or_factor='and', exclude_word_list=['.xml'])) != 0:
                    #         temp_ds = gdal.Open(Landsat_main_v1.file_filter(self.inundated_dic[self.inundated_para + '_' + self.sa_current], [str(self.doy_list[i]), '.TIF'], and_or_factor='and', exclude_word_list=['.xml'])[0])
                    #         temp_inundated_map = temp_ds.GetRasterBand(1).ReadAsArray()
                    #         try:
                    #             vi_temp = self.vi_sa_array_for_phenology[:, :, i].reshape([self.vi_sa_array_for_phenology.shape[0], self.vi_sa_array_for_phenology.shape[1]])
                    #             vi_temp[temp_inundated_map > 0] = np.nan
                    #             self.vi_sa_array_for_phenology[:, :, i] = vi_temp.reshape([self.vi_sa_array_for_phenology.shape[0], self.vi_sa_array_for_phenology.shape[1], 1])
                    #         except:
                    #             pass
                    self.vi_sa_array_for_phenology = copy.copy(self.vi_temp_dic)
                    for i in range(len(self.doy_list)):
                        if int(self.doy_list[i]) in self.inundated_doy:
                            pos_index = np.argwhere(self.inundated_doy == int(self.doy_list[i]))
                            dc_temp = self.vi_sa_array_for_phenology[:, :, i]
                            inundated_temp = self.inundated_dc[:, :, pos_index[0]].reshape([self.inundated_dc.shape[0], self.inundated_dc.shape[1]])
                            dc_temp[inundated_temp > 0] = np.nan
                            self.vi_sa_array_for_phenology[:, :, i] = dc_temp
                    # self.inundated_process_factor = True
                elif self.inundated_dic == {}:
                    self.vi_sa_array_for_phenology = copy.copy(self.vi_temp_dic)
                # Generate plot data
                if self.varied_factor == 'year':
                    year_list_temp = np.array([int(self.doy_list[i]) // 1000 for i in range(len(self.doy_list))])
                    try:
                        pos_min = np.min(np.argwhere(year_list_temp == int(self.begin_year)))
                        pos_max = np.max(np.argwhere(year_list_temp == int(self.end_year)))
                        self.doy_array_for_phenology = np.array(self.doy_list)[pos_min: pos_max + 1]
                        self.vi_sa_array_for_phenology = self.vi_temp_dic[:, :, pos_min: pos_max + 1]
                    except:
                        self.caution_msg_box('The year range is invalid')
            else:
                self.caution_msg_box('Consistency error!')
                return

    def plot_phenology_figure(self):
        self.output_x_data = np.array([])
        self.output_y_data = np.array([])
        self.data_output_factor = False
        if self.phenology_view_box.isEnabled() and self.begin_year != '' and self.end_year != '':
            if self.doy_array_for_phenology.shape[0] != 0:
                if self.ra_domain == 'Entire SA':
                    vi_entire_temp = np.nanmean(np.nanmean(self.vi_sa_array_for_phenology, axis=0), axis=0)
                    if self.phenology_view_factor == 'Annual':
                        year_list_temp = np.array([int(self.doy_array_for_phenology[i]) // 1000 for i in range(len(self.doy_array_for_phenology))])
                        for year in self.year_range:
                            if year in year_list_temp:
                                try:
                                    year_pos_min = np.min(np.argwhere(year_list_temp == year))
                                    year_pos_max = np.max(np.argwhere(year_list_temp == year))
                                    self.phenology_plot_y = vi_entire_temp[year_pos_min: year_pos_max + 1]
                                    np.append(self.output_y_data, self.phenology_plot_y)
                                    self.phenology_plot_x = np.mod(self.doy_array_for_phenology[year_pos_min: year_pos_max + 1], 1000)
                                    np.append(self.output_x_data, self.doy_array_for_phenology[year_pos_min: year_pos_max + 1])
                                    self.output_x_id = 'YEAR + DOY'
                                    self.output_y_id = 'Entire SA Annual ' + self.vi_current
                                    self.phe_dic['scatterplot_temp_' + str(year)].setData(self.phenology_plot_x, self.phenology_plot_y)
                                    self.r_square, self.curve_plot_x, self.curve_plot_y = self.regression_analysis(self.phenology_plot_x, self.phenology_plot_y, self.regression_type)
                                    self.phe_dic['curve_temp_' + str(year)].setData(self.curve_plot_x, self.curve_plot_y)
                                    self.data_output_factor = True
                                except:
                                    self.caution_msg_box('Some Unknown Error Occurred during the phenology figure output')
                    elif self.phenology_view_factor == 'Overview':
                        try:
                            self.phenology_plot_y = vi_entire_temp
                            self.output_y_data = self.phenology_plot_y
                            self.phenology_plot_x = np.mod(self.doy_array_for_phenology, 1000)
                            self.output_x_data = self.phenology_plot_x
                            self.output_x_id = 'YEAR + DOY'
                            self.output_y_id = 'Entire SA ' + self.vi_current + ' Overview'
                            self.phe_dic['scatterplot_temp_0000'].setData(self.phenology_plot_x, self.phenology_plot_y)
                            self.r_square, self.curve_plot_x, self.curve_plot_y = self.regression_analysis(self.phenology_plot_x, self.phenology_plot_y, self.regression_type)
                            self.phe_dic['curve_temp_0000'].setData(self.curve_plot_x, self.curve_plot_y)
                            self.data_output_factor = True
                        except:
                            self.caution_msg_box('Some Unknown Error Occurred during the phenology figure output')
                elif self.ra_domain == 'Pixel':
                    if self.demo_pos_x != 0 and self.demo_pos_y != 0 and ~np.isnan(self.demo_pos_x) and ~np.isnan(self.demo_pos_y):
                        self.vi_sa_array_for_phenology_pixel = self.vi_sa_array_for_phenology[int(self.demo_pos_y / self.demo_resize_factor), int(self.demo_pos_x / self.demo_resize_factor), :]
                        if self.phenology_view_factor == 'Annual':
                            year_list_temp = np.array([int(self.doy_array_for_phenology[i]) // 1000 for i in range(len(self.doy_array_for_phenology))])
                            for year in self.year_range:
                                if year in year_list_temp:
                                    try:
                                        year_pos_min = np.min(np.argwhere(year_list_temp == year))
                                        year_pos_max = np.max(np.argwhere(year_list_temp == year))
                                        self.phenology_plot_y = self.vi_sa_array_for_phenology_pixel[year_pos_min: year_pos_max + 1]
                                        np.append(self.output_y_data, self.phenology_plot_y)
                                        self.phenology_plot_x = np.mod(self.doy_array_for_phenology[year_pos_min: year_pos_max + 1], 1000)
                                        np.append(self.output_x_data, self.doy_array_for_phenology[year_pos_min: year_pos_max + 1])
                                        self.output_x_id = 'YEAR + DOY'
                                        self.output_y_id = 'Pixel ' + str(self.demo_pos_x) + ' ' + str(self.demo_pos_y) + ' Annual ' + self.vi_current
                                        self.phe_dic['scatterplot_temp_' + str(year)].setData(self.phenology_plot_x, self.phenology_plot_y)
                                        self.r_square, self.curve_plot_x, self.curve_plot_y = self.regression_analysis(self.phenology_plot_x, self.phenology_plot_y, self.regression_type)
                                        self.phe_dic['curve_temp_' + str(year)].setData(self.curve_plot_x, self.curve_plot_y)
                                        self.data_output_factor = True
                                    except:
                                        self.caution_msg_box('Some Unknown Error Occurred during the phenology figure output')
                        elif self.phenology_view_factor == 'Overview':
                            try:
                                self.phenology_plot_x = np.mod(self.doy_array_for_phenology, 1000)
                                self.phenology_plot_y = self.vi_sa_array_for_phenology_pixel
                                self.output_y_data = self.phenology_plot_y
                                self.output_x_data = self.phenology_plot_x
                                self.output_x_id = 'YEAR + DOY'
                                self.output_y_id = 'Pixel ' + str(self.demo_pos_x) + ' ' + str(self.demo_pos_y) + ' ' + self.vi_current + ' Overview'
                                self.phe_dic['scatterplot_temp_0000'].setData(self.phenology_plot_x, self.phenology_plot_y)
                                self.r_square, self.curve_plot_x, self.curve_plot_y = self.regression_analysis(self.phenology_plot_x, self.phenology_plot_y, self.regression_type)
                                self.phe_dic['curve_temp_0000'].setData(self.curve_plot_x, self.curve_plot_y)
                                self.data_output_factor = True
                            except:
                                self.caution_msg_box('Some Unknown Error Occurred during the phenology figure output')
                exporter = pg.exporters.ImageExporter(self.phe_win.scene())
                Landsat_main_v1.create_folder(self.sa_demo_folder + self.phenology_view_factor + '//')
                exporter.export(self.sa_demo_folder + self.phenology_view_factor + '//' + str(self.vi_current) + '.png')
                # image_temp = cv2.imread(self.sa_demo_folder + self.phenology_view_factor + '//' + str(self.vi_current) + '.png')
                # factor_min = min((self.phenology_window_height / (image_temp.shape[0] - 1)), (self.phenology_window_width / (image_temp.shape[1] - 1)))
                # image_temp_t = cv2.resize(image_temp, (int(factor_min * image_temp.shape[1] - 1), int(factor_min * image_temp.shape[0] - 1)))
                # cv2.imwrite(self.sa_demo_folder + self.phenology_view_factor + '//' + str(self.vi_current) + '_resize.png', image_temp_t)
                self.phenology_information_view.clear_scene()
                self.phenology_information_view.setPhoto(QPixmap(self.sa_demo_folder + self.phenology_view_factor + '//' + str(self.vi_current) + '.png'))
                self.output_phe_fig_path = self.sa_demo_folder + self.phenology_view_factor + '//' + str(self.vi_current) + '.png'
                self.phenology_view_name = 'phenology'
            else:
                self.caution_msg_box('There is no data within the defined range!')
                return

    def regression_analysis(self, x_data, y_data, regression_type):
        all_supported_curve_fitting_method = ['SPLF', 'TTF']
        VI_curve_fitting = {}
        nan_pos = np.argwhere(np.isnan(y_data))
        x_data = np.delete(x_data, nan_pos)
        y_data = np.delete(y_data, nan_pos)
        x_temp = np.linspace(0, 365, 1000)
        x_fail_temp = np.linspace(-10, -1, 100)
        y_fail_temp = np.linspace(-10, -1, 100)
        if x_data.shape[0] != 0:
            if regression_type is None:
                return np.nan, x_fail_temp, y_fail_temp
            elif regression_type == 'SPLF':
                VI_curve_fitting['CFM'] = 'SPL'
                VI_curve_fitting['para_num'] = 7
                VI_curve_fitting['para_ori'] = [0.10, 0.8802, 108.2, 7.596, 280.4, 7.473, 0.00225]
                VI_curve_fitting['para_boundary'] = ([0.08, 0.7, 100, 6.2, 260, 4.5, 0.0015], [0.20, 1.0, 130, 11.5, 300, 8.8, 0.0028])
                curve_fitting_algorithm = seven_para_logistic_function
                if x_data.shape[0] < VI_curve_fitting['para_num']:
                    return np.nan, x_fail_temp, y_fail_temp
                else:
                    paras, extras = curve_fit(curve_fitting_algorithm, x_data, y_data, maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
                    predicted_y_data = seven_para_logistic_function(x_data, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                    output_y_data = seven_para_logistic_function(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
                    R_square = (1 - np.sum((predicted_y_data - y_data) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)) * 100
                    return R_square, x_temp, output_y_data
            elif regression_type == 'TTF':
                curve_fitting_algorithm = two_term_fourier
                VI_curve_fitting['CFM'] = 'TTF'
                VI_curve_fitting['para_num'] = 6
                VI_curve_fitting['para_ori'] = [0, 0, 0, 0, 0, 0.017]
                VI_curve_fitting['para_boundary'] = ([0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
                if x_data.shape[0] < VI_curve_fitting['para_num']:
                    return np.nan, x_fail_temp, y_fail_temp
                else:
                    paras, extras = curve_fit(curve_fitting_algorithm, x_data, y_data, maxfev=5000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
                    predicted_y_data = two_term_fourier(x_data, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5])
                    output_y_data = two_term_fourier(x_temp, paras[0], paras[1], paras[2], paras[3], paras[4], paras[5])
                    R_square = (1 - np.sum((predicted_y_data - y_data) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)) * 100
                    return R_square, x_temp, output_y_data
            elif regression_type == 'Preferred':
                R_square_list = []
                x_temp_list = []
                output_y_date_list = []
                for i in range(len(all_supported_curve_fitting_method)):
                    R_square_t, x_temp_t, output_y_data_t = self.regression_analysis(x_data, y_data, all_supported_curve_fitting_method[i])
                    R_square_list.append(R_square_t)
                    x_temp_list.append(x_temp_t)
                    output_y_date_list.append(output_y_data_t)
                r_square_array = np.array(R_square_list)
                if np.isnan(r_square_array).all():
                    return np.nan, x_fail_temp, y_fail_temp
                else:
                    index = np.argwhere(r_square_array == np.nanmax(r_square_array))
                    return R_square_list[index[0, 0]], x_temp_list[index[0, 0]], output_y_date_list[index[0, 0]]
            else:
                return np.nan, x_fail_temp, y_fail_temp
        else:
            # self.caution_msg_box('There has no valid data!')
            return np.nan, x_fail_temp, y_fail_temp

    # Section 2 Update displaying vi figure
    def update_vi_sa_dic(self):
        self.vi_current = self.vi_combobox.currentText()
        if self.vi_current != 'None':
            try:
                if len(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')) == 1:
                    self.sdc_dic = np.load(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')[0], allow_pickle=True).item()
                    try:
                        self.vi_temp_dic = np.load(self.sdc_dic[self.vi_current + '_path'] + self.vi_current + '_sequenced_datacube.npy', allow_pickle=True)
                        self.update_vi_related_button(True)
                        self.update_display_button()
                    except:
                        self.caution_msg_box('Please generate the ' + self.vi_current + ' sequenced datacube of the ' + self.sa_current + ' before visualization!')
                        self.default_vi_related_factors()
                elif len(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')) == 0:
                    self.caution_msg_box('Please generate the sdc datacube of the ' + self.sa_current + ' before visualization!')
                    self.default_vi_related_factors()
                else:
                    self.caution_msg_box('There are more than two sdc dic in the key dictionary folder!')
                    self.default_vi_related_factors()
            except:
                self.caution_msg_box('Unknown error occurred during update_vi_sa_dic')
                self.default_vi_related_factors()
        else:
            self.sdc_dic = {}
            self.update_vi_related_button(False)

    def update_inundated_combobox_dic(self):
        self.inundated_para = self.inundated_combobox.currentText()
        if self.inundated_para != 'None' and self.inundated_para != '':
            try:
                self.inundated_dic = np.load(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', self.inundated_para, self.sa_current], and_or_factor='and')[0], allow_pickle=True).item()
                self.inundated_dc = np.load(self.inundated_dic['inundated_dc_file'])
                self.inundated_doy = np.load(self.inundated_dic['inundated_doy_file'])
                if self.inundated_dc.shape[2] != self.inundated_doy.shape[0]:
                    self.caution_msg_box('Consistency Error')
                    self.inundated_dc = np.array([])
                    self.inundated_doy = np.array([])
                    self.inundated_dic = {}
            except:
                self.caution_msg_box('Unknown error during updated inundated dic')
        else:
            self.inundated_dic = {}
        self.update_vi_display()

    def update_doy(self):
        self.doy = self.doy_line_box.text()
        if len(self.doy) == 7:
            try:
                self.doy = int(self.doy)
            except:
                self.caution_msg_box('Please input the doy in the format of YYYYDOY!')
                return
        elif(len(self.doy)) == 8:
            try:
                self.doy = int(Landsat_main_v1.date2doy(self.doy))
            except:
                self.caution_msg_box('Please input the doy in the format of YYYYDOY!')
                return
        else:
            self.caution_msg_box('Please input the doy in the format of YYYYDOY!')
            return

        self.doy_list = [int(i) for i in self.doy_list]
        if self.doy not in self.doy_list:
            for i in range(200000):
                if self.doy - i in self.doy_list:
                    self.doy = self.doy - i
                    break
                elif self.doy + i in self.doy_list:
                    self.doy = self.doy + i
                    break
            self.caution_msg_box('Caution the input doy is invalid, the nearest date ' + str(self.doy) + ' is used instead!')
        else:
            self.doy = str(self.doy)
        self.doy_line_box.setText(str(self.doy))
        self.update_vi_display()

    def update_vi_display(self):
        if np.isnan(float(self.doy)) or self.vi_current == 'None' or self.vi_temp_dic.shape[0] == 0 or self.phenology_view_name == 'phenology':
            pass
        else:
            self.inundated_map = {}
            doy_pos = np.argwhere(np.array(self.doy_list) == int(self.doy))
            vi_temp = self.vi_temp_dic[:, :, doy_pos[0]]

            try:
                if vi_temp.shape[2] == 1:
                    vi_temp = np.reshape(vi_temp, (vi_temp.shape[0], vi_temp.shape[1]))
            except:
                pass

            if self.inundated_para != 'None' and self.inundated_dic != {}:
                try:
                    # temp_ds = gdal.Open(Landsat_main_v1.file_filter(self.inundated_dic[self.inundated_para + '_' + self.sa_current], [str(self.doy)])[0])
                    # self.inundated_map = temp_ds.GetRasterBand(1).ReadAsArray()
                    # if self.inundated_map.shape[0] != 0:
                    #     if self.inundated_map.shape[0] == vi_temp.shape[0] and self.inundated_map.shape[1] == vi_temp.shape[1]:
                    #         vi_temp[self.inundated_map > 0] = np.nan
                    # else:
                    #     vi_temp = vi_temp
                    inundated_pos = np.argwhere(self.inundated_doy == self.doy)
                    self.inundated_map = self.inundated_dc[:, :, inundated_pos[0]]
                    if self.inundated_map.shape[0] == vi_temp.shape[0] and self.inundated_map.shape[1] == vi_temp.shape[1]:
                        if self.inundated_map.shape[0] == vi_temp.shape[0] and self.inundated_map.shape[1] == vi_temp.shape[1]:
                            vi_temp[self.inundated_map > 0] = np.nan
                    else:
                        vi_temp = vi_temp
                        self.caution_msg_box('Inconsistency detected between vi and inundated image!')
                except:
                    self.caution_msg_box('Unknown Error occurred during inundated map input!')
            elif self.inundated_para == 'None':
                self.inundated_dic = {}

            if np.sum(~np.isnan(vi_temp)) == 0:
                if self.doy != str(self.doy_list[0]):
                    self.caution_msg_box('This is a void image, Please re-input the date!')
                else:
                    return
            else:
                self.ori_vi_image = np.stack([vi_temp, vi_temp, vi_temp], axis=2).astype(np.float64)
                self.phenology_window_height = self.phenology_information_view.geometry().height()
                self.phenology_window_width = self.phenology_information_view.geometry().width()
                self.phenology_resize_factor = min((self.phenology_window_height - 1)/self.ori_vi_image.shape[0], (self.phenology_window_width - 1)/self.ori_vi_image.shape[1])
                if self.ori_vi_image.dtype != np.uint16:
                    self.vi_demo_image = cv2.resize(self.ori_vi_image.astype(np.uint16), (int(self.phenology_resize_factor * self.ori_vi_image.shape[1]) - 1, int(self.phenology_resize_factor * self.ori_vi_image.shape[0]) - 1))
                    cv2.imwrite(self.sa_demo_folder + 'phenology_temp.png', self.vi_demo_image)
                else:
                    self.vi_demo_image = cv2.resize(self.ori_vi_image, (int(self.phenology_resize_factor * self.ori_vi_image.shape[1]) - 1, int(self.phenology_resize_factor * self.ori_vi_image.shape[0]) - 1))
                    cv2.imwrite(self.sa_demo_folder + 'phenology_temp.png', self.vi_demo_image)
                if self.rescale_factor:
                    self.rescale()
                else:
                    self.phenology_information_view.clear_scene()
                    self.phenology_information_view.setPhoto(QPixmap(self.sa_demo_folder + 'phenology_temp.png'))
                    self.phenology_view_name = 'VI'
                    self.rescale_factor = False
                    self.update_rescale_button()
                    self.update_output_button()

    def rescale(self):
        for i in range(3):
            layer_max = np.sort(np.delete(np.unique(self.ori_vi_image[:, :, i]), np.argwhere(np.isnan(np.unique(self.ori_vi_image[:, :, i])))))[-2]
            layer_min = np.sort(np.delete(np.unique(self.ori_vi_image[:, :, i]), np.argwhere(np.isnan(np.unique(self.ori_vi_image[:, :, i])))))[0]
            layer_temp = (self.ori_vi_image[:, :, i] - layer_min) / (layer_max - layer_min)
            self.ori_vi_image[:, :, i] = 65536 * layer_temp.astype(np.float64)
            self.ori_vi_image.astype(np.float64)

        if self.ori_vi_image.dtype != np.uint16:
            self.ori_vi_image[np.isnan(self.ori_vi_image)] = 65535
            self.ori_vi_image[self.ori_vi_image == np.inf] = 65535
            self.ori_vi_image = self.ori_vi_image.astype(np.uint16)

        self.vi_demo_image = cv2.resize(self.ori_vi_image.astype(np.uint16), (int(self.phenology_resize_factor * self.ori_vi_image.shape[1]) - 1, int(self.phenology_resize_factor * self.ori_vi_image.shape[0]) - 1))
        cv2.imwrite(self.sa_demo_folder + 'phenology_temp_rescaled.png', self.vi_demo_image)
        self.phenology_information_view.clear_scene()
        self.phenology_information_view.setPhoto(QPixmap(self.sa_demo_folder + 'phenology_temp_rescaled.png'))
        self.phenology_view_name = 'VI'
        self.rescale_factor = True
        self.update_rescale_button()

    def update_rescale_button(self):
        if self.vi_current != 'None':
            if not self.display_button.isEnabled:
                self.rescale_button.setEnabled(False)
            else:
                if not self.rescale_factor and self.phenology_information_view.phenology_scene.items() != []:
                    self.rescale_button.setEnabled(True)
                else:
                    self.rescale_button.setEnabled(False)
        else:
            self.rescale_button.setEnabled(False)

    def update_display_button(self):
        if self.vi_current != 'None':
            if self.phenology_information_view.phenology_scene.items() == []:
                self.display_button.setEnabled(True)
            else:
                if self.rescale_factor is True:
                    self.display_button.setEnabled(False)
                else:
                    if float(self.doy) == np.nan:
                        self.display_button.setEnabled(False)
                    else:
                        self.display_button.setEnabled(True)
        else:
            self.display_button.setEnabled(False)

    # Section 3 Update the Demo FIGURE
    def show_demo(self):
        if len(Landsat_main_v1.file_filter(self.orifile_path, ['.TIF'])) == 0:
            self.caution_msg_box('Please double check the original tif file path')

        if not os.path.exists(self.shpfile):
            self.caution_msg_box('The shapefile doesnot exists, please manually input!')
            self.update_sa_related_button(False)
        else:
            if self.date != '':
                if len(self.date) == 8:
                    if len(Landsat_main_v1.file_filter(self.orifile_path, [self.date, '.TIF'], and_or_factor='and')) == 0:
                        self.caution_msg_box('This is not a valid date! Try again!')
                        self.update_sa_related_button(False)
                    else:
                        self.sa_demo_folder = self.rootpath + '/Landsat_phenology_demo/' + str(self.sa_current) + '/'
                        Landsat_main_v1.create_folder(self.sa_demo_folder)
                        ori_file = Landsat_main_v1.file_filter(self.orifile_path, [self.date, '.TIF'], and_or_factor='and')
                        if 'LC08' in ori_file[0]:
                            self.rgb_dic = {'r': 'B4', 'g': 'B3', 'b': 'B2'}
                        elif 'LE07' in ori_file[0] or 'LT05' in ori_file[0]:
                            self.rgb_dic = {'r': 'B3', 'g': 'B2', 'b': 'B1'}
                        else:
                            self.caution_msg_box('Unkown error occurred!')
                            self.update_sa_related_button(False)
                            return
                        r_ds = gdal.Open(Landsat_main_v1.file_filter(self.orifile_path, [self.date, self.rgb_dic['r'], '.TIF'], and_or_factor='and')[0])
                        g_ds = gdal.Open(Landsat_main_v1.file_filter(self.orifile_path, [self.date, self.rgb_dic['g'], '.TIF'], and_or_factor='and')[0])
                        b_ds = gdal.Open(Landsat_main_v1.file_filter(self.orifile_path, [self.date, self.rgb_dic['b'], '.TIF'], and_or_factor='and')[0])
                        Landsat_main_v1.remove_all_file_and_folder(Landsat_main_v1.file_filter(self.sa_demo_folder, ['.']))
                        gdal.Warp(self.sa_demo_folder + 'r_' + self.sa_current + '.TIF', r_ds, cutlineDSName=self.shpfile, cropToCutline=True, dstNodata=65536, xRes=30, yRes=30)
                        gdal.Warp(self.sa_demo_folder + 'g_' + self.sa_current + '.TIF', g_ds, cutlineDSName=self.shpfile, cropToCutline=True, dstNodata=65536, xRes=30, yRes=30)
                        gdal.Warp(self.sa_demo_folder + 'b_' + self.sa_current + '.TIF', b_ds, cutlineDSName=self.shpfile, cropToCutline=True, dstNodata=65536, xRes=30, yRes=30)
                        r_sa_ds = gdal.Open(self.sa_demo_folder + 'r_' + self.sa_current + '.TIF')
                        g_sa_ds = gdal.Open(self.sa_demo_folder + 'g_' + self.sa_current + '.TIF')
                        b_sa_ds = gdal.Open(self.sa_demo_folder + 'b_' + self.sa_current + '.TIF')
                        r_sa_raster = r_sa_ds.GetRasterBand(1).ReadAsArray()
                        g_sa_raster = g_sa_ds.GetRasterBand(1).ReadAsArray()
                        b_sa_raster = b_sa_ds.GetRasterBand(1).ReadAsArray()
                        demo_window_height = self.sa_demo.geometry().height()
                        demo_window_width = self.sa_demo.geometry().width()
                        new_image = np.stack((r_sa_raster, g_sa_raster, b_sa_raster), axis=2)
                        self.demo_resize_factor = min((demo_window_height - 1)/new_image.shape[0], (demo_window_width - 1)/new_image.shape[1])
                        self.demo_image = cv2.resize(new_image, (int(self.demo_resize_factor * new_image.shape[1]) - 1, int(self.demo_resize_factor * new_image.shape[0]) - 1))
                        cv2.imwrite(self.sa_demo_folder + 'temp.png', self.demo_image)
                        self.sa_demo.clear_scene()
                        self.sa_demo.setPhoto(QPixmap(self.sa_demo_folder + 'temp.png'))
                        # self.sa_demo._scene.addPixmap(QPixmap(self.sa_demo_folder + 'temp.png'))
                        # self.update_pix_mode()
                        self.update_sa_related_button(True)
                else:
                    self.caution_msg_box('Please manual input the date in YYYYMMDD format!')
                    self.update_sa_related_button(False)
            else:
                self.caution_msg_box('Please manual input the date of the demo!')
                self.update_sa_related_button(False)

    def random_show_demo(self):
        i = random.randint(0, int(len(self.all_date)))
        self.manual_input_date_Edit.setText(self.all_date[i])
        self.update_demo_image()

    def update_demo_image(self):
        self.date = self.manual_input_date_Edit.text()
        self.show_demo()

    def gamma_para_update(self):
        self.gamma_para = self.gamma_para_spinbox.value()
        if self.gamma_para != 1.52:
            self.gamma_correction()

    def gamma_correction(self):
        r_sa_ds = gdal.Open(self.sa_demo_folder + 'r_' + self.sa_current + '.TIF')
        g_sa_ds = gdal.Open(self.sa_demo_folder + 'g_' + self.sa_current + '.TIF')
        b_sa_ds = gdal.Open(self.sa_demo_folder + 'b_' + self.sa_current + '.TIF')
        r_sa_raster = r_sa_ds.GetRasterBand(1).ReadAsArray()
        g_sa_raster = g_sa_ds.GetRasterBand(1).ReadAsArray()
        b_sa_raster = b_sa_ds.GetRasterBand(1).ReadAsArray()
        new_image = np.stack((r_sa_raster, g_sa_raster, b_sa_raster), axis=2)
        if new_image.dtype == np.uint16:
            img_max_v = 65535
            img_dtype = np.uint16
        elif new_image.dtype == np.uint8:
            img_max_v = 255
            img_dtype = np.uint8
        else:
            self.caution_msg_box('not supported data type!')
            return
        try:
            image_temp = new_image.astype(np.float)
            r_max = np.sort(np.unique(image_temp[:, :, 0]))[-2]
            r_min = np.sort(np.unique(image_temp[:, :, 0]))[0]
            g_max = np.sort(np.unique(image_temp[:, :, 1]))[-2]
            g_min = np.sort(np.unique(image_temp[:, :, 1]))[0]
            b_max = np.sort(np.unique(image_temp[:, :, 2]))[-2]
            b_min = np.sort(np.unique(image_temp[:, :, 2]))[0]
            image_temp[:, :, 0] = img_max_v * (image_temp[:, :, 0] - r_min) / (r_max - r_min)
            image_temp[:, :, 1] = img_max_v * (image_temp[:, :, 1] - g_min) / (g_max - g_min)
            image_temp[:, :, 2] = img_max_v * (image_temp[:, :, 2] - b_min) / (b_max - b_min)
            image_temp[image_temp >= img_max_v] = img_max_v
            image_temp = (img_max_v * ((image_temp / img_max_v) ** (1 / self.gamma_para))).astype(img_dtype)
        except:
            self.caution_msg_box('Gamma cannot equal to 0 or invalid!')
            return
        image_temp[image_temp >= img_max_v] = img_max_v
        self.demo_image = cv2.resize(image_temp, (int(self.demo_resize_factor * image_temp.shape[1]) - 1, int(self.demo_resize_factor * image_temp.shape[0]) - 1))
        cv2.imwrite(self.sa_demo_folder + 'temp_gamma.png', self.demo_image)
        self.sa_demo.clear_scene()
        self.sa_demo.setPhoto(QPixmap(self.sa_demo_folder + 'temp_gamma.png'))
        # self.update_pix_mode()
        # self.demoscene.clear()
        # self.demoscene.addPixmap(QPixmap(self.sa_demo_folder + 'temp_gamma.png'))
        # self.sa_demo.setScene(self.demoscene)
        self.gamma_correction_button.setEnabled(False)

    # Section 4 Input paths and close the programme
    def update_root_path(self):
        try:
            os.path.exists("E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\")
            path_temp = "E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\"
        except:
            path_temp = 'C:\\'
        self.rootpath = QFileDialog.getExistingDirectory(self, "Please select the root path", path_temp)
        self.orifile_path = self.rootpath + '/Landsat_Ori_TIFF/'
        self.keydic_path = self.rootpath + '/Landsat_key_dic/'

        if not os.path.exists(self.orifile_path) or not os.path.exists(self.keydic_path):
            self.caution_msg_box('Please manually input the original file path and the key dictionary path!')
            self.update_rt_related_button(False)
        else:
            self.levelfactor = 'root'
            self.basic_information_retrieval()
            self.all_available_date()

    def update_ori_file_path(self):
        try:
            path.exists(self.rootpath)
            path_temp = self.rootpath
        except:
            path_temp = 'C:\\'
        self.orifile_path = QFileDialog.getExistingDirectory(self, "Please manually update the original tiff file path", path_temp)

        if len(Landsat_main_v1.file_filter(self.orifile_path, ['.TIF'])) == 0:
            self.caution_msg_box('Please double check the original tif file path')
            self.update_rt_related_button(False)
        else:
            self.all_available_date()

    def update_key_dic_path(self):
        try:
            path.exists(self.rootpath)
            path_temp = self.rootpath
        except:
            path_temp = 'C:\\'
        self.orifile_path = QFileDialog.getExistingDirectory(self, "Please manually update the key dictionary path", path_temp)
        self.levelfactor = 'dic_path'
        self.basic_information_retrieval()

    def update_fundamental_dic(self):
        try:
            path.exists(self.keydic_path)
            path_temp = self.keydic_path
        except:
            path_temp = 'C:\\'
        dic_path_temp = QFileDialog.getOpenFileUrl(self, "Please select the original file path", path_temp, "numpy files (*.npy)")
        try:
            self.fundamental_dic = np.load(dic_path_temp[0].url().replace('file:///', ''), allow_pickle=True).item()
        except:
            self.caution_msg_box('Cannot find the fundamental dictionary')
            self.update_rt_related_button(False)
        finally:
            self.levelfactor = 'dic'
            self.basic_information_retrieval()

    def all_available_date(self):
        all_ori_filename = os.listdir(self.orifile_path)
        try:
            self.all_date = [i[17:25] for i in all_ori_filename if 'MTL.txt' in i]
        except:
            self.caution_msg_box('No MTL file founded')
            self.update_rt_related_button(False)

        if self.all_date != []:
            self.manual_input_date_Edit.setText(self.all_date[0])

    def query_information(self):
        try:
            self.vi_list = self.fundamental_dic['all_vi']
            self.sa_list = self.fundamental_dic['study_area']
            self.shpfile_path = self.fundamental_dic['shpfile_path']
        except:
            self.caution_msg_box('Key attributes missing in the key dictionary')
            self.default_sa_related_factors()
            self.update_sa_related_button(False)

    def basic_information_retrieval(self):
        if self.levelfactor == 'root':
            try:
                self.fundamental_dic = np.load(self.rootpath + '/Landsat_key_dic/fundamental_information_dic.npy', allow_pickle=True).item()
            except:
                self.caution_msg_box('Cannot find the fundamental dictionary')
                self.update_rt_related_button(False)
            self.query_information()

        elif self.levelfactor == 'dic_path':
            try:
                self.fundamental_dic = np.load(self.keydic_path + 'fundamental_information_dic.npy', allow_pickle=True).item()
            except:
                self.caution_msg_box('Cannot find the fundamental dictionary')
                self.update_rt_related_button(False)
            self.query_information()

        elif self.levelfactor == 'dic':
            self.query_information()
        self.enable_initialisation()

    def enable_initialisation(self):
        if self.fundamental_dic is None:
            self.Initialization_box.setEnabled(False)
        else:
            try:
                # self.sa_combobox.
                sa_allitems = [self.sa_combobox.itemText(i) for i in range(self.sa_combobox.count())]
                vi_allitems = [self.vi_combobox.itemText(i) for i in range(self.vi_combobox.count())]
                sa_additems = [i for i in self.sa_list if i not in sa_allitems]
                vi_additems = [i for i in self.vi_list if i not in vi_allitems]
                if sa_additems != []:
                    self.sa_combobox.addItems(self.sa_list)
                    self.sa_current = self.sa_combobox.currentText()
                    self.get_shp()
                if vi_additems != []:
                    self.vi_combobox.addItems(self.vi_list)
                    self.vi_current = self.sa_combobox.currentText()
                self.Initialization_box.setEnabled(True)
            except:
                self.caution_msg_box('Unknown error occurred!')

    def get_shp(self):
        if self.shpfile_path != '' and self.sa_current != []:
            try:
                if len(Landsat_main_v1.file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) == 1:
                    self.shpfile = Landsat_main_v1.file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])[0]
                elif len(Landsat_main_v1.file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) > 1:
                    self.caution_msg_box('Unknown error occurred!')
                elif len(Landsat_main_v1.file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) == 0:
                    self.caution_msg_box('There has no required shape file! Please manually input')
            except:
                self.caution_msg_box('Unknown error occurred!')
        else:
            self.caution_msg_box('Lack essential information to retrieve the shp of study area, manual input or try again!')

    def update_shpfile_path(self):
        try:
            os.path.exists(self.shpfile_path)
            path_temp = self.rootpath
        except:
            path_temp = 'C:\\'
        self.shpfile = QFileDialog.getOpenFileUrl(self, "Please select the shape file of " + str(self.sa_current), path_temp, "shp files (*.shp)")
        self.show_demo()

    def update_shp_infor(self):
        self.sa_current = self.sa_combobox.currentText()
        self.get_shp()
        self.show_demo()
        self.default_sa_related_factors()

        if self.sa_demo._scene.items() != []:
            self.update_sa_related_button(True)
        else:
            self.update_sa_related_button(False)

    # Section 5 Output data
    def update_output_button(self):
        if self.phenology_information_view.phenology_scene.items() != []:
            self.output_fig_button.setEnabled(True)
            self.output_visual_path = self.rootpath + '//visualisation_output//'
            Landsat_main_v1.create_folder(self.output_visual_path)
        else:
            self.output_fig_button.setEnabled(False)

        if self.data_output_factor:
            self.output_data_button.setEnabled(True)
            self.output_visual_path = self.rootpath + '//visualisation_output//'
            Landsat_main_v1.create_folder(self.output_visual_path)
        else:
            self.output_data_button.setEnabled(False)

    def output_figure(self):
        if self.phenology_information_view.phenology_scene.items() != []:
            if self.ra_domain == 'Entire SA':
                try:
                    shutil.copyfile(self.output_phe_fig_path, self.output_visual_path + self.sa_current + '_' + self.vi_current + '_' + self.phenology_view_factor + '_sa.png')
                    self.caution_msg_box('Successfully generate!')
                except:
                    self.caution_msg_box('Unknown error occurred during the output!')
            elif self.ra_domain == 'Pixel':
                try:
                    shutil.copyfile(self.output_phe_fig_path, self.output_visual_path + self.sa_current + '_' + self.vi_current + '_' + self.phenology_view_factor + '_' + str(self.demo_pos_x) + '_' + str(self.demo_pos_y) + '.png')
                    self.caution_msg_box('Successfully generate!')
                except:
                    self.caution_msg_box('Unknown error occurred during the output!')

    def output_data(self):
        if self.data_output_factor:
            self.output = pd.DataFrame()

    # Section 6 Update and default the button
    def default_rt_related_factors(self):
        self.manual_input_date_Edit.setText('20000000')
        self.sa_combobox.setCurrentText('')
        self.default_sa_related_factors()

    def default_sa_related_factors(self):
        self.vi_combobox.setCurrentText('None')
        self.inundated_combobox.setCurrentText('None')
        self.gamma_para = 1.52
        self.gamma_para_spinbox.setValue(1.52)
        self.update_vi_related_button(False)

    def default_vi_related_factors(self):
        self.vi_combobox.setCurrentText('None')
        self.inundated_combobox.setCurrentText('None')
        self.inundated_process_factor = False
        self.doy_line_box.setText('2000000')
        self.time_span_end.clear()
        self.time_span_begin.clear()
        self.time_span_end.addItem('0000')
        self.time_span_begin.addItem('0000')
        self.time_span_begin.setCurrentText('0000')
        self.time_span_end.setCurrentText('0000')
        self.phenology_view_box.setCurrentText('Annual')
        self.ra_domain_box.setCurrentText('Entire SA')
        self.regression_box.setCurrentText('None')
        self.x_coordinate.setText('0')
        self.y_coordinate.setText('0')
        self.data_output_factor = False

    def update_rt_related_button(self, bool_factor):
        if bool_factor:
            pass
        else:
            self.update_sa_related_button(False)
            self.Initialization_box.setEnabled(False)
            self.default_rt_related_factors()

    def update_sa_related_button(self, bool_factor):
        if bool_factor:
            self.gamma_para_spinbox.setEnabled(True)
            self.gamma_correction_button.setEnabled(True)
            self.VI_para_box.setEnabled(True)
            self.phenology_information_view.phenology_scene.clear()
            self.update_vi_sa_dic()
            self.update_doy_list()
            self.update_inundated_item()
            self.update_display_button()
            self.update_rescale_button()
        else:
            self.gamma_para_spinbox.setEnabled(False)
            self.gamma_correction_button.setEnabled(False)
            self.VI_para_box.setEnabled(False)
            self.update_display_button()
            self.update_rescale_button()
            self.update_vi_related_button(False)
            self.default_sa_related_factors()

    def update_vi_related_button(self, bool_factor):
        if bool_factor:
            self.data_output_factor = False
            self.update_doy_list()
            self.phenology_view_box.setEnabled(True)
            self.ra_domain_box.setEnabled(True)
            self.regression_box.setEnabled(True)
            self.generate_button.setEnabled(True)
            self.phenology_factor_box.setEnabled(True)
            self.update_display_button()
            self.update_rescale_button()
            self.update_output_button()
        else:
            self.data_output_factor = False
            self.default_vi_related_factors()
            self.phenology_view_box.setEnabled(False)
            self.ra_domain_box.setEnabled(False)
            self.regression_box.setEnabled(False)
            self.generate_button.setEnabled(False)
            self.phenology_factor_box.setEnabled(False)
            self.update_doy_list()
            self.update_display_button()
            self.update_rescale_button()
            self.update_output_button()

    def update_inundated_item(self):
        self.inundated_combobox.clear()
        self.inundated_combobox.addItem('None')
        if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current)])) == 0:
            self.inundated_combobox.setEnable(False)
            self.inundated_combobox.setCurrentText('None')
            self.inundated_para = self.inundated_combobox.currentText()
        else:
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current), 'local'])) != 0:
                self.inundated_combobox.addItem('local')
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current), 'global'])) != 0:
                self.inundated_combobox.addItem('global')
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current), 'survey'])) != 0:
                self.inundated_combobox.addItem('survey')
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current), 'final'])) != 0:
                self.inundated_combobox.addItem('final')
            self.inundated_combobox.setCurrentText('None')
            self.inundated_para = self.inundated_combobox.currentText()

    def update_doy_list(self):
        self.year_list = []
        if self.sdc_dic != {}:
            try:
                self.doy_list = np.load(self.sdc_dic[self.vi_current + '_path'] + 'doy.npy', allow_pickle=True).astype(int).tolist()
            except:
                self.caution_msg_box('Unknown error occurred during doy list update!')
                self.doy_line_box.setEnabled(False)
                self.time_span_begin.setEnabled(False)
                self.time_span_end.setEnabled(False)

            if self.doy_list == []:
                self.caution_msg_box('Void doy list detected!')
                self.doy_line_box.setEnabled(False)
                self.time_span_begin.setEnabled(False)
                self.time_span_end.setEnabled(False)
            else:
                self.doy_line_box.setEnabled(True)
                self.time_span_begin.setEnabled(True)
                self.time_span_end.setEnabled(True)
                self.year_list = [str(int(self.doy_list[i]) // 1000) for i in range(len(self.doy_list)) if str(int(self.doy_list[i]) // 1000) not in self.year_list]
                self.year_list = np.unique(np.array(self.year_list)).tolist()
                self.doy_line_box.setText(str(self.doy_list[0]))
                self.time_span_begin.clear()
                self.time_span_end.clear()
                self.time_span_begin.addItems(self.year_list)
                self.time_span_end.addItems(self.year_list)
                self.time_span_begin.setCurrentText(min(self.year_list))
                self.time_span_end.setCurrentText(max(self.year_list))
                self.doy = str(self.doy_list[0])
                self.update_display_button()
        else:
            self.doy_line_box.setEnabled(False)
            self.time_span_begin.setEnabled(False)
            self.time_span_end.setEnabled(False)

    def caution_msg_box(self, text_temp):
        message = QMessageBox()
        message.setWindowTitle('Caution!')
        message.setText(text_temp)
        message.exec_()


def main():
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # os.environ["QT_SCALE_FACTOR"] = "2"
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    np.seterr(divide='ignore', invalid='ignore')
    app = QApplication(sys.argv)
    # app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    initialise_window = InitialisingScreen()
    # window = Visualisation_gui()
    sys.exit(app.exec_())


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
        return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x)+b2 * np.sin(2 * w * x)


def r_square():
    pass


if __name__ == '__main__':
    main()
