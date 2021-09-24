import sys
import PyQt5.QtCore
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


class Visulisation_gui(QMainWindow):

    def __init__(self):
        super(Visulisation_gui, self).__init__()
        uic.loadUi('ui\\visulisation.ui', self)
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
        # demo_related attributes
        self.gamma_para = self.gamma_para_spinbox.value()
        self.demoscene = QGraphicsScene()
        # Phenology-related attributes
        self.sdc_dic = {}
        self.vi_temp_dic = {}
        self.doy_temp_dic = {}

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
        self.try_Button.clicked.connect(self.input_date)
        self.random_input_Button.clicked.connect(self.random_show_demo)
        self.sa_combobox.activated.connect(self.update_shp_infor)
        self.gamma_correction_button.clicked.connect(self.gamma_correction)
        self.gamma_para_spinbox.valueChanged.connect(self.gamma_para_update)
        self.vi_combobox.activated.connect(self.input_vi_file)

    def input_vi_file(self):
        self.vi_current = self.vi_combobox.currentText()
        if self.vi_current != 'None':
            try:
                if len(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')) == 1:
                    self.sdc_dic = np.load(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')[0], allow_pickle=True).item()
                    self.vi_combobox.setcurrentText('None')
                    try:
                        np.load(self.sdc_dic[self.vi_current + '_path'] + self.vi_current + '_sequenced_datacube.npy', allow_pickle=True)
                    except:
                        self.caution_msg_box('Please generate the ' + self.vi_current + ' sequenced datacube of the ' + self.sa_current + ' before visualization!')
                        self.vi_combobox.setcurrentText('None')
                elif len(Landsat_main_v1.file_filter(self.keydic_path, ['.npy', 'sdc', self.sa_current], and_or_factor='and')) == 0:
                    self.caution_msg_box('Please generate the sdc datacube of the ' + self.sa_current + ' before visualization!')
                    self.vi_combobox.setcurrentText('None')
                else:
                    self.caution_msg_box('There are more than two sdc dic in the key dictionary folder!')
                    self.vi_combobox.setcurrentText('None')
            except:
                self.caution_msg_box('Unknown error occurred during input_vi_file')
                self.vi_combobox.setcurrentText('None')

    def update_paramemter_box(self, bool_factor):
        if bool_factor:
            self.gamma_para_spinbox.setEnabled(True)
            self.gamma_correction_button.setEnabled(True)
            self.VI_para_box.setEnabled(True)
            if len(Landsat_main_v1.file_filter(self.keydic_path, ['inundat', str(self.sa_current)])) == 0:
                self.inundated_combobox.setEnable(False)
        else:
            self.gamma_para_spinbox.setEnabled(False)
            self.gamma_correction_button.setEnabled(False)
            self.VI_para_box.setEnabled(False)

    def random_show_demo(self):
        i = random.randint(0, int(len(self.all_date)))
        self.manual_input_date_Edit.setText(self.all_date[i])
        self.input_date()

    def show_demo(self):
        if len(Landsat_main_v1.file_filter(self.orifile_path, ['.TIF'])) == 0:
            self.caution_msg_box('Please double check the original tif file path')

        if not os.path.exists(self.shpfile):
            self.caution_msg_box('The shapefile doesnot exists, please manually input!')
            self.update_paramemter_box(False)
        else:
            if self.date != '':
                if len(self.date) == 8:
                    if len(Landsat_main_v1.file_filter(self.orifile_path, [self.date, '.TIF'], and_or_factor='and')) == 0:
                        self.caution_msg_box('This is not a valid date! Try again!')
                        self.update_paramemter_box(False)
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
                            self.update_paramemter_box(False)
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
                        self.demoscene.clear()
                        self.demoscene.addPixmap(QPixmap(self.sa_demo_folder + 'temp.png'))
                        self.sa_demo.setScene(self.demoscene)
                        self.update_paramemter_box(True)
                else:
                    self.caution_msg_box('Please manual input the date in YYYYMMDD format!')
                    self.update_paramemter_box(False)
            else:
                self.caution_msg_box('Please manual input the date of the demo!')
                self.update_paramemter_box(False)

    def input_date(self):
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
        self.demoscene.clear()
        self.demoscene.addPixmap(QPixmap(self.sa_demo_folder + 'temp_gamma.png'))
        self.sa_demo.setScene(self.demoscene)
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
    PyQt5.QtCore.QCoreApplication.setAttribute(PyQt5.QtCore.Qt.AA_EnableHighDpiScaling)
    np.seterr(divide='ignore', invalid='ignore')
    app = QApplication(sys.argv)
    window = Visulisation_gui()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
