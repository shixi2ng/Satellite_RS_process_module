import sys
import PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import pyqtgraph as pg
import numpy as np
import pandas
import os
import gdal
import cv2
import random
import Landsat_main_v1


counter = 0


class ProcessingScreen(QMainWindow):
    def __init__(self):
        super(ProcessingScreen, self).__init__()
        uic.loadUi('ui\\processing.ui', self)

        self.setWindowFlag(PyQt5.QtCore.Qt.FramelessWindowHint)
        self.setAttribute(PyQt5.QtCore.Qt.WA_TranslucentBackground)

        self.show()


class InitialisingScreen(QMainWindow):
    def __init__(self):
        super(InitialisingScreen, self).__init__()
        uic.loadUi('ui\\initialising.ui', self)
        # Main

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
        self.timer.start(19)

        PyQt5.QtCore.QTimer.singleShot(1000, lambda: self.initlabel.setText('Loading data.'))
        PyQt5.QtCore.QTimer.singleShot(1300, lambda: self.initlabel.setText('Loading data..'))
        PyQt5.QtCore.QTimer.singleShot(1600, lambda: self.initlabel.setText('Loading data...'))
        PyQt5.QtCore.QTimer.singleShot(1900, lambda: self.initlabel.setText('Done'))
        self.show()

    def initialise(self):
        global counter
        self.progressBar.setValue(counter)
        if counter > 100:
            self.timer.stop()
            self.main = Visulisation_gui()
            self.close()
        counter += 1


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


class Demo_Viewer(QGraphicsView):
    photoClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(Demo_Viewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = Demo_scene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def clear_scene(self):
        self._scene.clear()
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
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
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
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
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(Demo_Viewer, self).mousePressEvent(event)


class Visulisation_gui(QMainWindow):

    def __init__(self):
        super(Visulisation_gui, self).__init__()
        uic.loadUi('ui\\visulisation_backup.ui', self)
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

        # Phenology-related attributes
        self.sdc_dic = {}
        self.vi_temp_dic = np.array([])
        self.doy_list = []
        self.year_list = []
        self.inundated_para = ''
        self.inundated_dic = {}
        self.doy = np.nan
        self.ori_vi_image = np.array([]).astype(np.float64)
        self.vi_demo_image = np.array([])
        self.phenology_resize_factor = np.nan
        self.phenology_scene = QGraphicsScene()
        self.inundated_map = np.array([])
        self.rescale_factor = False
        # Phenology factor
        self.phenology_view_factor = ''
        self.ra_domain = ''
        self.regression_type = ''
        self.begin_year = ''
        self.end_year = ''
        self.vi_sa_dic = np.array([])

        # demo_related attributes
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
        self.demo_pos_x = np.nan
        self.demo_pos_y = np.nan

        # self.sa_demo.photoClicked.connect(self.photoClicked)

        self.update_root_path()

        self.show()
        self.enable_intialisation()
        # close
        self.actionclose.triggered.connect(exit)
        # input data connection
        self.actionUpdate_the_root_path.triggered.connect(self.update_root_path)
        self.actionUpdate_Key_dictionary_path.triggered.connect(self.update_key_dic_path)
        self.actionUpdate_Original_file_path.triggered.connect(self.update_ori_file_path)
        self.actionUpdate_fundamental_dic.triggered.connect(self.update_fundatmental_dic)
        self.actionUpdate_shpfile_path.triggered.connect(self.update_shpfile_path)
        # generate output connection
        self.try_Button.clicked.connect(self.update_demo_image)
        self.manual_input_date_Edit.returnPressed.connect(self.update_demo_image)
        self.random_input_Button.clicked.connect(self.random_show_demo)
        self.sa_combobox.activated.connect(self.update_shp_infor)
        self.gamma_correction_button.clicked.connect(self.gamma_correction)
        self.gamma_para_spinbox.valueChanged.connect(self.gamma_para_update)
        self.vi_combobox.currentTextChanged.connect(self.update_vi_sa_dic)
        self.inundated_combobox.currentTextChanged.connect(self.update_inundated_combobox_dic)
        self.doy_line_box.returnPressed.connect(self.doy_update)
        self.display_button.clicked.connect(self.update_vi_display)
        self.rescale_button.clicked.connect(self.rescale)
        self.time_span_begin.currentTextChanged.connect(self.update_time_span_begin)
        self.time_span_begin.currentTextChanged.connect(self.update_phenology_factor)
        self.time_span_end.currentTextChanged.connect(self.update_time_span_end)
        self.time_span_end.currentTextChanged.connect(self.update_phenology_factor)
        self.generate_button.clicked.connect(self.update_phenology_inform_figure)
        self.ra_domain_box.currentTextChanged.connect(self.update_phenology_factor)
        self.ra_domain_box.currentTextChanged.connect(self.update_phenology_factor)
        self.regression_box.currentTextChanged.connect(self.update_phenology_factor)

    def photoClicked(self, pos):
        if self.sa_demo.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.demo_pos_x = pos.x()
            self.demo_pos_y = pos.y()

    def update_ra_domain(self):
        self.demoscene.ra_f = self.ra_domain

    def update_phenology_factor(self):
        self.phenology_view_factor = self.phenology_view_box.currentText()
        self.ra_domain = self.ra_domain_box.currentText()
        if self.ra_domain == 'Entire SA':
            self.vi_sa_dic = np.nansum(self.vi_temp_dic, axis=2)
        elif self.ra_domain == 'Pixel':
            self.vi_sa_dic = np.array([])
            self.sa_demo.toggleDragMode()
        self.regression_type = self.regression_box.currentText()
        self.vi_current = self.vi_combobox.currentText()
        # self.update_ra_domain()

    def update_phenology_inform_figure(self):
        self.update_phenology_factor()

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

    def update_vi_phenology(self):
        pass

    def doy_update(self):
        self.doy = self.doy_line_box.text()
        if len(self.doy) == 7:
            try:
                self.doy = int(self.doy)
            except:
                self.caution_msg_box('Please input the doy in the format of YYYYDOY!')
        elif(len(self.doy)) == 8:
            try:
                self.doy = int(Landsat_main_v1.date2doy(self.doy))
            except:
                self.caution_msg_box('Please input the doy in the format of YYYYDOY!')
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
        if np.isnan(float(self.doy)) or self.vi_current == 'None' or self.vi_temp_dic.shape[0] == 0:
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
                    temp_ds = gdal.Open(Landsat_main_v1.file_filter(self.inundated_dic[self.inundated_para + '_' + self.sa_current], [str(self.doy)])[0])
                    self.inundated_map = temp_ds.GetRasterBand(1).ReadAsArray()
                    if self.inundated_map.shape[0] != 0:
                        if self.inundated_map.shape[0] == vi_temp.shape[0] and self.inundated_map.shape[1] == vi_temp.shape[1]:
                            vi_temp[self.inundated_map > 0] = np.nan
                    else:
                        vi_temp = vi_temp
                except:
                    self.caution_msg_box('Unknown Error occurred during inundated map input!')

            if np.sum(~np.isnan(vi_temp)) == 0:
                if self.doy != str(self.doy_list[0]):
                    self.caution_msg_box('This is a void image, Please re-input the date!')
                else:
                    return
            else:
                self.ori_vi_image = np.stack([vi_temp, vi_temp, vi_temp], axis=2).astype(np.float64)
                phenology_window_height = self.phenology_information_view.geometry().height()
                phenology_window_width = self.phenology_information_view.geometry().width()
                self.phenology_resize_factor = min((phenology_window_height - 1)/self.ori_vi_image.shape[0], (phenology_window_width - 1)/self.ori_vi_image.shape[1])
                if self.ori_vi_image.dtype != np.uint16:
                    self.vi_demo_image = cv2.resize(self.ori_vi_image.astype(np.uint16), (int(self.phenology_resize_factor * self.ori_vi_image.shape[1]) - 1, int(self.phenology_resize_factor * self.ori_vi_image.shape[0]) - 1))
                    cv2.imwrite(self.sa_demo_folder + 'phenology_temp.png', self.vi_demo_image)
                else:
                    self.vi_demo_image = cv2.resize(self.ori_vi_image, (int(self.phenology_resize_factor * self.ori_vi_image.shape[1]) - 1, int(self.phenology_resize_factor * self.ori_vi_image.shape[0]) - 1))
                    cv2.imwrite(self.sa_demo_folder + 'phenology_temp.png', self.vi_demo_image)
                if self.rescale_factor:
                    self.rescale()
                else:
                    self.phenology_scene.clear()
                    self.phenology_scene.addPixmap(QPixmap(self.sa_demo_folder + 'phenology_temp.png'))
                    self.phenology_information_view.setScene(self.phenology_scene)
                    self.rescale_factor = False
                    self.update_rescale_button()

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
        self.phenology_scene.clear()
        self.phenology_scene.addPixmap(QPixmap(self.sa_demo_folder + 'phenology_temp_rescaled.png'))
        self.phenology_information_view.setScene(self.phenology_scene)
        self.rescale_factor = True
        self.update_rescale_button()

    def default_factors(self):
        self.vi_combobox.setCurrentText('None')
        self.inundated_combobox.setCurrentText('None')
        self.doy_line_box.setText('2000000')
        self.time_span_begin.setCurrentText('0000')
        self.time_span_end.setCurrentText('0000')
        self.time_span_begin.setEnabled(False)
        self.time_span_end.setEnabled(False)
        self.doy_line_box.setEnabled(False)
        self.display_button.setEnabled(False)
        self.rescale_button.setEnabled(False)

    def update_inundated_combobox_dic(self):
        self.inundated_para = self.inundated_combobox.currentText()
        if self.inundated_para != 'None' and self.inundated_para != '':
            try:
                self.inundated_dic = np.load(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', self.inundated_para, self.sa_current], and_or_factor='and')[0], allow_pickle=True).item()
            except:
                self.caution_msg_box('Unknown error during updated inundated dic')
        else:
            self.inundated_dic = {}
        self.update_vi_display()

    def update_vi_sa_dic(self):
        self.vi_current = self.vi_combobox.currentText()
        if self.vi_current != 'None':
            try:
                if len(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')) == 1:
                    self.sdc_dic = np.load(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')[0], allow_pickle=True).item()
                    try:
                        self.vi_temp_dic = np.load(self.sdc_dic[self.vi_current + '_path'] + self.vi_current + '_sequenced_datacube.npy', allow_pickle=True)
                        self.update_doy_list()
                        self.update_display_button()
                    except:
                        self.caution_msg_box('Please generate the ' + self.vi_current + ' sequenced datacube of the ' + self.sa_current + ' before visualization!')
                        self.default_factors()
                elif len(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')) == 0:
                    self.caution_msg_box('Please generate the sdc datacube of the ' + self.sa_current + ' before visualization!')
                    self.default_factors()
                else:
                    self.caution_msg_box('There are more than two sdc dic in the key dictionary folder!')
                    self.default_factors()
            except:
                self.caution_msg_box('Unknown error occurred during update_vi_sa_dic')
                self.default_factors()

    def update_vi_parameter_box(self, bool_factor):
        if bool_factor:
            self.gamma_para_spinbox.setEnabled(True)
            self.gamma_correction_button.setEnabled(True)
            self.VI_para_box.setEnabled(True)
            self.phenology_view_box.setEnabled(True)
            self.ra_domain_box.setEnabled(True)
            self.regression_box.setEnabled(True)
            self.generate_button.setEnabled(True)
            self.phenology_factor_box.setEnabled(True)
            self.update_doy_list()
            self.update_display_button()
            self.update_rescale_button()
            self.update_inundated_item()
            self.phenology_scene.clear()
            self.default_factors()
        else:
            self.gamma_para_spinbox.setEnabled(False)
            self.gamma_correction_button.setEnabled(False)
            self.VI_para_box.setEnabled(False)
            self.phenology_factor_box.setEnabled(False)

    def update_inundated_item(self):
        self.inundated_combobox.clear()
        self.inundated_combobox.addItem('None')
        if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current)])) == 0:
            self.inundated_combobox.setEnable(False)
        else:
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current), 'local'])) != 0:
                self.inundated_combobox.addItem('local')
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current), 'global'])) != 0:
                self.inundated_combobox.addItem('global')
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current), 'surv'])) != 0:
                self.inundated_combobox.addItem('surveyed')
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current), 'fina'])) != 0:
                self.inundated_combobox.addItem('final')
            self.inundated_combobox.setCurrentText('None')
            self.inundated_para = self.inundated_combobox.currentText()

    def update_doy_list(self):
        self.year_list = []
        if self.sdc_dic != {}:
            try:
                self.doy_list = self.sdc_dic['doy']
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

    def update_rescale_button(self):
        if not self.display_button.isEnabled:
            self.rescale_button.setEnabled(False)
        else:
            if not self.rescale_factor:
                self.rescale_button.setEnabled(True)
            else:
                self.rescale_button.setEnabled(False)

    def update_display_button(self):
        if self.phenology_scene.items() == []:
            self.display_button.setEnabled(True)
        else:
            if self.rescale_factor is True:
                self.display_button.setEnabled(False)
            else:
                if float(self.doy) == np.nan:
                    self.display_button.setEnabled(False)
                else:
                    self.display_button.setEnabled(True)

    def random_show_demo(self):
        i = random.randint(0, int(len(self.all_date)))
        self.manual_input_date_Edit.setText(self.all_date[i])
        self.update_demo_image()

    def show_demo(self):
        if len(Landsat_main_v1.file_filter(self.orifile_path, ['.TIF'])) == 0:
            self.caution_msg_box('Please double check the original tif file path')

        if not os.path.exists(self.shpfile):
            self.caution_msg_box('The shapefile doesnot exists, please manually input!')
            self.update_vi_parameter_box(False)
        else:
            if self.date != '':
                if len(self.date) == 8:
                    if len(Landsat_main_v1.file_filter(self.orifile_path, [self.date, '.TIF'], and_or_factor='and')) == 0:
                        self.caution_msg_box('This is not a valid date! Try again!')
                        self.update_vi_parameter_box(False)
                    else:
                        self.sa_demo_folder = self.rootpath + '/Landsat_phenology_demo/' + str(self.sa_current) + '/'
                        Landsat_main_v1.create_folder(self.sa_demo_folder)
                        ori_file = Landsat_main_v1.file_filter(self.orifile_path, [self.date, '.TIF'], and_or_factor='and')
                        if 'LC08' in ori_file[0]:
                            self.rgb_dic = {'r': 'B4', 'g': 'B3', 'b': 'B2'}
                        elif 'LE07' in ori_file[0] or 'LT05' in ori_file[0]:
                            self.rgb_dic = {'r': 'B3', 'g': 'B2', 'b': 'B1'}
                        else:
                            self.caution_msg_box('Unkown error occured!')
                            self.update_vi_parameter_box(False)
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
                        self.update_vi_parameter_box(True)
                else:
                    self.caution_msg_box('Please manual input the date in YYYYMMDD format!')
                    self.update_vi_parameter_box(False)
            else:
                self.caution_msg_box('Please manual input the date of the demo!')
                self.update_vi_parameter_box(False)

    def update_demo_image(self):
        self.date = self.manual_input_date_Edit.text()
        self.show_demo()

    def gamma_para_update(self):
        self.gamma_para = self.gamma_para_spinbox.value()
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

    def all_avaliable_date(self):
        all_ori_filename = os.listdir(self.orifile_path)
        try:
            self.all_date = [i[17:25] for i in all_ori_filename if 'MTL.txt' in i]
        except:
            self.caution_msg_box('No MTL file founded')

        if self.all_date != []:
            self.manual_input_date_Edit.setText(self.all_date[0])

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
        self.update_inundated_item()
        self.update_vi_display()
        self.update_display_button()
        if self.demoscene.items() != []:
            self.update_vi_parameter_box(True)
        else:
            self.update_vi_parameter_box(False)

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
        else:
            self.levelfactor = 'root'
            self.basic_information_retreival()
            self.all_avaliable_date()

    def update_ori_file_path(self):
        try:
            path.exists(self.rootpath)
            path_temp = self.rootpath
        except:
            path_temp = 'C:\\'
        self.orifile_path = QFileDialog.getExistingDirectory(self, "Please manually update the original tiff file path", path_temp)

        if len(Landsat_main_v1.file_filter(self.orifile_path, ['.TIF'])) == 0:
            self.caution_msg_box('Please double check the original tif file path')
        else:
            self.all_avaliable_date()

    def update_key_dic_path(self):
        try:
            path.exists(self.rootpath)
            path_temp = self.rootpath
        except:
            path_temp = 'C:\\'
        self.orifile_path = QFileDialog.getExistingDirectory(self, "Please manually update the key dictionary path", path_temp)
        self.levelfactor = 'dic_path'
        self.basic_information_retreival()

    def update_fundatmental_dic(self):
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
        finally:
            self.levelfactor = 'dic'
            self.basic_information_retreival()

    def query_information(self):
        try:
            self.vi_list = self.fundamental_dic['all_vi']
            self.sa_list = self.fundamental_dic['study_area']
            self.shpfile_path = self.fundamental_dic['shpfile_path']
        except:
            self.caution_msg_box('Key attributes missing in the key dictionary')

    def caution_msg_box(self, text_temp):
        message = QMessageBox()
        message.setWindowTitle('Caution!')
        message.setText(text_temp)
        message.exec_()

    def basic_information_retreival(self):
        if self.levelfactor == 'root':
            try:
                self.fundamental_dic = np.load(self.rootpath + '/Landsat_key_dic/fundamental_information_dic.npy', allow_pickle=True).item()
            except:
                self.caution_msg_box('Cannot find the fundamental dictionary')
            self.query_information()

        elif self.levelfactor == 'dic_path':
            try:
                self.fundamental_dic = np.load(self.keydic_path + 'fundamental_information_dic.npy', allow_pickle=True).item()
            except:
                self.caution_msg_box('Cannot find the fundamental dictionary')
            self.query_information()

        elif self.levelfactor == 'dic':
            self.query_information()
        self.enable_intialisation()

    def enable_intialisation(self):
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
                self.caution_msg_box('Unknown error occured!')

    def get_shp(self):
        if self.shpfile_path != '' and self.sa_current != []:
            try:
                if len(Landsat_main_v1.file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) == 1:
                    self.shpfile = Landsat_main_v1.file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])[0]
                elif len(Landsat_main_v1.file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) > 1:
                    self.caution_msg_box('Unknown error occured!')
                elif len(Landsat_main_v1.file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) == 0:
                    self.caution_msg_box('There has no requried shape file! Please manually input')
            except:
                self.caution_msg_box('Unknown error occured!')
        else:
            self.caution_msg_box('Lack essential information to retrieve the shp of study area, manual input or try again!')


def main():
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    np.seterr(divide='ignore', invalid='ignore')
    app = QApplication(sys.argv)
    initialise_window = InitialisingScreen()
    # window = Visulisation_gui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
