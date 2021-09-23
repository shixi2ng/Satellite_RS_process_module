import PyQt5.QtCore
from PyQt5.QtWidgets import *
from PyQt5 import uic
import numpy as np
import pandas
import os
import gdal


def file_filter(file_path_temp, containing_word_list, subfolder_detection=False, and_or_factor=None, exclude_word_list=[]):
    if and_or_factor is None:
        and_or_factor = 'or'
    elif and_or_factor not in ['and', 'or']:
        print("Caution the and or should exactly be string as 'and' or 'or'")
        sys.exit(-1)

    if and_or_factor == 'or':
        file_list = os.listdir(file_path_temp)
        filter_list = []
        for file in file_list:
            if os.path.isdir(file_path_temp + file) and subfolder_detection:
                filter_list_temp = file_filter(file_path_temp + file + '\\', containing_word_list, subfolder_detection=True, and_or_factor=and_or_factor)
                if filter_list_temp != []:
                    filter_list.append(filter_list_temp)
            else:
                for containing_word in containing_word_list:
                    if containing_word in file_path_temp + file:
                        if exclude_word == []:
                            filter_list.append(file_path_temp + file)
                        else:
                            exclude_factor = False
                            for exclude_word in exclude_word_list:
                                if exclude_word in file_path_temp + file:
                                    exclude_factor = True
                                    break
                            if not exclude_factor:
                                filter_list.append(file_path_temp + file)
                        break
        return filter_list
    elif and_or_factor == 'and':
        file_list = os.listdir(file_path_temp)
        filter_list = []
        for file in file_list:
            file_factor = True
            if os.path.isdir(file_path_temp + file) and subfolder_detection:
                filter_list_temp = file_filter(file_path_temp + file + '\\', containing_word_list,
                                               subfolder_detection=True, and_or_factor=and_or_factor)
                if filter_list_temp != []:
                    filter_list.append(filter_list_temp)
            else:
                for containing_word in containing_word_list:
                    if containing_word not in file_path_temp + file:
                        file_factor = False
                        break
                for exclude_word in exclude_word_list:
                    if exclude_word in file_path_temp + file:
                        file_factor = False
                        break
                if file_factor:
                    filter_list.append(file_path_temp + file)
        return filter_list


class Visulisation_gui(QMainWindow):

    def __init__(self):
        super(Visulisation_gui, self).__init__()
        uic.loadUi('ui\\visulisation.ui', self)
        self.rootpath = []
        self.orifile_path = []
        self.keydic_path = []
        self.shpfile_path = []
        self.levelfactor = None
        self.fundamental_dic = None
        self.vi_list = None
        self.sa_list = None
        self.sa_current = self.sa_combobox.currentText()
        self.shpfile = ''
        self.vi_current = self.sa_combobox.currentText()
        self.update_root_path()

        self.show()
        self.enable_intialisation()
        # close
        self.actionclose.triggered.connect(exit)
        # input data connection
        self.actionUpdate_the_root_path.triggered.connect(self.update_root_path)
        self.actionUpdate_Key_dictionary_path.triggered.connect(self.update_key_dic_path)
        self.actionUpdate_fundamental_dic.triggered.connect(self.update_fundatmental_dic)
        self.actionUpdate_shpfile_path.triggered.connect(self.update_shpfile_path)
        # generate output connection
        self.try_Button.clicked.connect(self.show_demo)
        self.random_input_Button.clicked.connect(self.random_show_demo)
        self.sa_combobox.activated.connect(self.update_shp_infor)

    def show_demo(self):
        if len(file_filter(self.orifile_path, ['.tif'])) == 0:
            self.caution_msg_box('Please double check the ori file path')

        if not os.path.exists(self.shpfile):
            self.caution_msg_box('The shapefile doesnot exists, please manually input!')
        else:
            if self.manual_input_date_Edit.text() != '':
                if len(file_filter(self.orifile_path, [self.manual_input_date_Edit.text(), '.tif'], and_or_factor='and')) == 0:
                    self.caution_msg_box('This is not a valid date! Try again!')
                else:
                    ori_file = file_filter(self.orifile_path, [self.manual_input_date_Edit.text(), '.tif'], and_or_factor='and')
                    if 'LC08' in ori_file[0]:
                        rgb_dic = {'r': 'B4', 'g': 'B3', 'b': 'B2'}
                    elif 'LE07' in ori_file[0] or 'LT05' in ori_file[0]:
                        rgb_dic = {'r': 'B3', 'g': 'B2', 'b': 'B1'}
                    else:
                        self.caution_msg_box('Please manual input the date of the demo!')
                        return

                    self.Parameter_box.setEnabled(True)
            else:
                self.caution_msg_box('Please manual input the date of the demo!')

    def random_show_demo(self):
        pass

    def update_shpfile_path(self):
        try:
            path.exists(self.shpfile_path)
            path_temp = self.rootpath
        except:
            path_temp = 'C:\\'
        self.shpfile = QFileDialog.getOpenFileUrl(self, "Please select the shape file of " + str(self.sa_current), path_temp, "shp files (*.shp)")

    def update_shp_infor(self):
        self.sa_current = self.sa_combobox.currentText()
        self.get_shp()

    def update_root_path(self):
        self.rootpath = QFileDialog.getExistingDirectory(self, "Please select the root path", "E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\")
        self.orifile_path = self.rootpath + '/Landsat_Ori_TIFF/'
        self.keydic_path = self.rootpath + '/Landsat_key_dic/'

        if not os.path.exists(self.orifile_path) or not os.path.exists(self.keydic_path):
            self.caution_msg_box('Please manually input the original file path and the key dictionary path!')
        else:
            self.levelfactor = 'root'
            self.basic_information_retreival()

    def update_key_dic_path(self):
        try:
            path.exists(self.rootpath)
            path_temp = self.rootpath
        except:
            path_temp = 'C:\\'
        self.orifile_path = QFileDialog.getExistingDirectory(self, "Please select the original file path", path_temp)
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
                if len(file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) == 1:
                    self.shpfile = file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])[0]
                elif len(file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) > 1:
                    self.caution_msg_box('Unknown error occured!')
                elif len(file_filter(self.shpfile_path, [self.sa_current, '.shp'], and_or_factor='and', exclude_word_list=['.xml'])) == 0:
                    self.caution_msg_box('There has no requried shape file! Please manually input')
            except:
                self.caution_msg_box('Unknown error occured!')
        else:
            self.caution_msg_box('Lack essential information to retrieve the shp of study area, manual input or try again!')

def main():
    PyQt5.QtCore.QCoreApplication.setAttribute(PyQt5.QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication([])
    window = Visulisation_gui()
    app.exec_()


if __name__ == '__main__':
    main()
