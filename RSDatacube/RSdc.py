import basic_function as bf
from basic_function import Path
import os
import json
import time
import numpy as np
from NDsm import NDSparseMatrix
import datetime
import copy


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x)+b2 * np.sin(2 * w * x)


class Denv_dc(object):

    ####################################################################################################
    # Denv dc represents "Daily Environment Datacube"
    # It normally contains data like daily temperature and daily radiation, etc
    # And stack into a 3-D datacube type.
    # Currently, it was integrated into the NCEI and MODIS FPAR toolbox as a output datatype.
    ####################################################################################################

    def __init__(self, Denv_dc_filepath, work_env=None, autofill=True):

        # Check the phemetric path
        self.Denv_dc_filepath = bf.Path(Denv_dc_filepath).path_name

        # Init key var
        self.ROI_name, self.ROI, self.ROI_tif, self.ROI_array = None, None, None, None
        self.index, self.Datatype, self.coordinate_system = None, None, None
        self.dc_group_list, self.tiles, self.oritif_folder = None, None, None
        self.sdc_factor, self.sparse_matrix, self.size_control_factor, self.huge_matrix = False, False, False, False
        self.Denv_factor, self.timescale, self.timerange = False, None, None
        self.compete_doy_list = []

        # Check work env
        if work_env is not None:
            self._work_env = Path(work_env).path_name
        else:
            self._work_env = Path(os.path.dirname(os.path.dirname(self.Denv_dc_filepath))).path_name
        self.root_path = Path(os.path.dirname(os.path.dirname(self._work_env))).path_name

        # Define the basic var name
        self._fund_factor = ('ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'sdc_factor',
                             'coordinate_system', 'oritif_folder', 'ROI_tif', 'sparse_matrix',
                             'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles', 'timescale', 'timerange')

        # Read the metadata file
        metadata_file = bf.file_filter(self.Denv_dc_filepath, ['metadata.json'])
        if len(metadata_file) == 0:
            raise ValueError('There has no valid sdc or the metadata file of the sdc was missing!')
        elif len(metadata_file) > 1:
            raise ValueError('There has more than one metadata file in the dir')
        else:
            try:
                # Load json metadata
                with open(metadata_file[0]) as js_temp:
                    dc_metadata = json.load(js_temp)

                if not isinstance(dc_metadata, dict):
                    raise Exception('Please make sure the metadata file is a dictionary constructed in python!')
                else:
                    # Determine whether this datacube is Denv or not
                    if 'Denv_factor' not in dc_metadata.keys():
                        raise TypeError('The Denv factor was lost or it is not a Denv datacube!')
                    elif dc_metadata['Denv_factor'] is False:
                        raise TypeError(f'{self.Denv_dc_filepath} is not a Denv datacube!')
                    elif dc_metadata['Denv_factor'] is True:
                        self.Denv_factor = dc_metadata['Denv_factor']
                    else:
                        raise TypeError('The Denv factor was under wrong type!')

                    for dic_name in self._fund_factor:
                        if dic_name not in dc_metadata.keys():
                            raise Exception(f'The {dic_name} is not in the dc metadata, double check!')
                        else:
                            self.__dict__[dic_name] = dc_metadata[dic_name]

            except:
                raise Exception('Something went wrong when reading the metadata!')

        # Check the timescale and timerange is consistency or not
        if self.timescale == 'year' and len(str(self.timerange)) != 4:
            raise Exception('The annual time range should be in YYYY format')
        elif self.timescale == 'month' and len(str(self.timerange)) != 6:
            raise Exception('The monthly time range should be in YYYYMM format')
        elif self.timescale == 'all' and len(str(self.timerange)) != '.TIF':
            raise Exception('The all time range should be in .TIF format')
        elif self.timescale is None or self.timerange is None:
            raise Exception('The timescale and timerange para is not properly assigned!')

        start_time = time.time()
        print(f'Start loading the Denv dc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        # Read doy or date file of the Datacube
        try:
            if self.sdc_factor is True:
                # Read doylist
                doy_file = bf.file_filter(self.Denv_dc_filepath, ['doy.npy'])
                if len(doy_file) == 0:
                    raise ValueError('There has no valid doy file or file was missing!')
                elif len(doy_file) > 1:
                    raise ValueError('There has more than one doy file in the dc dir')
                else:
                    sdc_doylist = np.load(doy_file[0], allow_pickle=True)
                    self.sdc_doylist = [int(sdc_doy) for sdc_doy in sdc_doylist]
            else:
                raise TypeError('Please input as a sdc')
        except:
            raise Exception('Something went wrong when reading the doy and date list!')
        self.year_domain = set([int(np.floor(temp)) for temp in self.sdc_doylist])

        # Compete doy list
        if self.timescale == 'year':
            date_start = datetime.date(year=self.timerange, month=1, day=1).toordinal()
            date_end = datetime.date(year=self.timerange + 1, month=1, day=1).toordinal()
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_start, date_end)]
            self.compete_doy_list = bf.date2doy(compete_doy_list)
        elif self.timescale == 'month':
            year_temp = int(np.floor(self.timerange/100))
            month_temp = int(np.mod(self.timerange, 100))
            date_start = datetime.date(year=year_temp, month=month_temp, day=1).toordinal()
            date_end = datetime.date(year=year_temp, month=month_temp + 1, day=1).toordinal()
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_start, date_end)]
            self.compete_doy_list = bf.date2doy(compete_doy_list)
        elif self.timescale == 'all':
            date_min, date_max = bf.doy2date(min(self.sdc_doylist)), bf.doy2date(max(self.sdc_doylist))
            date_min = datetime.date(year=int(np.floor(date_min/1000)), month=1, day=1).toordinal() + np.mod(date_min, 1000) - 1
            date_max = datetime.date(year=int(np.floor(date_max/1000)), month=1, day=1).toordinal() + np.mod(date_max, 1000)
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_min, date_max)]
            self.compete_doy_list = bf.date2doy(compete_doy_list)

        # Read the Denv datacube
        try:
            if self.sparse_matrix and self.huge_matrix:
                if os.path.exists(self.Denv_dc_filepath + f'{self.index}_Denv_datacube\\'):
                    self.dc = NDSparseMatrix().load(self.Denv_dc_filepath + f'{self.index}_Denv_datacube\\')
                else:
                    raise Exception('Please double check the code if the sparse huge matrix is generated properly')
            elif not self.huge_matrix:
                self.dc_filename = bf.file_filter(self.Denv_dc_filepath, ['Denv_datacube.npy'])
                if len(self.dc_filename) == 0:
                    raise ValueError('There has no valid dc or the dc was missing!')
                elif len(self.dc_filename) > 1:
                    raise ValueError('There has more than one date file in the dc dir')
                else:
                    self.dc = np.load(self.dc_filename[0], allow_pickle=True)
            elif self.huge_matrix and not self.sparse_matrix:
                self.dc_filename = bf.file_filter(self.Denv_dc_filepath, ['Denv_datacube', '.npy'], and_or_factor='and')
        except:
            raise Exception('Something went wrong when reading the datacube!')

        if autofill is True and len(self.compete_doy_list) > len(self.sdc_doylist):
            self._autofill_Denv_DC()
        elif len(self.compete_doy_list) < len(self.sdc_doylist):
            raise Exception('Code has issues in the Denc autofill procedure!')

        # Size calculation and shape definition
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]
        if self.dc_ZSize != len(self.sdc_doylist):
            raise TypeError('The Denv datacube is not consistent with the doy list')

        print(f'Finish loading the Denv dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def __sizeof__(self):
        return self.dc.__sizeof__() + self.sdc_doylist.__sizeof__()

    def _autofill_Denv_DC(self):
        for date_temp in self.compete_doy_list:
            if date_temp not in self.sdc_doylist:
                if date_temp == self.compete_doy_list[0]:
                    date_merge = self.compete_doy_list[1]
                    if self.sparse_matrix:
                        self.dc.add_layer(self.dc.SM_group[date_merge], date_temp, 0)
                    else:
                        self.dc = np.insert(self.dc, 0, values=self.dc[:,:,0], axis=2)
                    self.sdc_doylist.insert(0, date_temp)
                elif date_temp == self.compete_doy_list[-1]:
                    date_merge = self.compete_doy_list[-2]
                    if self.sparse_matrix:
                        self.dc.add_layer(self.dc.SM_group[date_merge], date_temp, -1)
                    else:
                        self.dc = np.insert(self.dc, 0, values=self.dc[:,:,-1], axis=2)
                    self.sdc_doylist.insert(-1, date_temp)
                else:
                    date_beg, date_end, _beg, _end = None, None, None, None
                    for _ in range(1, 30):
                        ordinal_date = datetime.date(year=int(np.floor(bf.doy2date(date_temp)/10000)), month=int(np.floor(np.mod(bf.doy2date(date_temp), 10000)/100)), day=int(np.mod(bf.doy2date(date_temp), 100))).toordinal()
                        if date_beg is None:
                            date_out = bf.date2doy(int(datetime.date.fromordinal(ordinal_date - _).strftime('%Y%m%d')))
                            date_beg = date_out if date_out in self.sdc_doylist else None
                            _beg = _ if date_out in self.sdc_doylist else None

                        if date_end is None:
                            date_out = bf.date2doy(int(datetime.date.fromordinal(ordinal_date + _).strftime('%Y%m%d')))
                            date_end = date_out if date_out in self.sdc_doylist else None
                            _end = _ if date_out in self.sdc_doylist else None

                        if date_end is not None and date_beg is not None:
                            break

                    if self.sparse_matrix:
                        type_temp = type(self.dc.SM_group[date_beg])
                        array_beg = self.dc.SM_group[date_beg].toarray()
                        array_end = self.dc.SM_group[date_end].toarray()
                        dtype_temp = array_end.dtype
                        array_beg = array_beg.astype(np.float32)
                        array_end = array_end.astype(np.float32)
                        array_out = array_beg + (array_end - array_beg) * _beg / (_beg + _end)
                        array_out = array_out.astype(dtype_temp)
                        array_out = type_temp(array_out)
                        self.dc.add_layer(array_out, date_temp, self.compete_doy_list.index(date_temp))
                    else:
                        array_beg = self.dc[:, :, date_beg]
                        array_end = self.dc[:, :, date_end]
                        array_beg = array_beg.astype(np.float32)
                        array_end = array_end.astype(np.float32)
                        array_out = array_beg + (array_end - array_beg) * _beg / (_beg + _end)
                        self.dc = np.insert(self.dc, self.compete_doy_list.index(date_temp), values=array_out.reshape([array_out.shape[0], array_out.shape[1], 1]), axis=2)
                    self.sdc_doylist.insert(self.compete_doy_list.index(date_temp), date_temp)

        if self.sdc_doylist != self.compete_doy_list:
            raise Exception('Error occurred during the autofill for the Denv DC!')

        self.save(self.Denv_dc_filepath)
        self.__init__(self.Denv_dc_filepath)

    def save(self, output_path: str):

        start_time = time.time()
        print(f'Start saving the sdc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        if not os.path.exists(output_path):
            bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name

        metadata_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI,
                        'ROI_array': self.ROI_array, 'ROI_tif': self.ROI_tif, 'sdc_factor': self.sdc_factor,
                        'coordinate_system': self.coordinate_system, 'sparse_matrix': self.sparse_matrix, 'huge_matrix': self.huge_matrix,
                        'size_control_factor': self.size_control_factor, 'oritif_folder': self.oritif_folder, 'dc_group_list': self.dc_group_list,
                        'tiles': self.tiles, 'timescale': self.timescale, 'timerange': self.timerange, 'Denv_factor': self.Denv_factor}
        doy = self.sdc_doylist
        np.save(f'{output_path}doy.npy', doy)
        with open(f'{output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_Denv_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_Denv_datacube.npy', self.dc)

        print(f'Finish saving the sdc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')


class Phemetric_dc(object):

    ####################################################################################################
    # Phemetric_dc represents "Phenological Metric Datacube"
    # It normally contains data like phenological parameters derived from the curve fitting, etc
    # And stack into a 3-D datacube type.
    # Currently, it was integrated into the Sentinel_dcs as a output data for phenological generation.
    ####################################################################################################

    def __init__(self, phemetric_filepath, work_env=None):

        # Check the phemetric path
        self.Phemetric_dc_filepath = bf.Path(phemetric_filepath).path_name

        # Init key var
        self.ROI_name, self.ROI, self.ROI_tif = None, None, None
        self.index, self.Datatype, self.coordinate_system = None, None, None
        self.dc_group_list, self.tiles = None, None
        self.sdc_factor, self.sparse_matrix, self.size_control_factor, self.huge_matrix = False, False, False, False
        self.Phemetric_factor, self.pheyear = False, None
        self.curfit_dic = {}

        # Init protected var
        self._support_pheme_list = ['SOS', 'EOS', 'trough_vi', 'peak_vi', 'peak_doy', 'GR', 'DR', 'DR2']

        # Check work env
        if work_env is not None:
            self._work_env = Path(work_env).path_name
        else:
            self._work_env = Path(os.path.dirname(os.path.dirname(self.Phemetric_dc_filepath))).path_name
        self.root_path = Path(os.path.dirname(os.path.dirname(self._work_env))).path_name

        # Define the basic var name
        self._fund_factor = ('ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'curfit_dic', 'pheyear',
                             'coordinate_system', 'oritif_folder', 'ROI_tif', 'sparse_matrix',
                             'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles')

        # Read the metadata file
        metadata_file = bf.file_filter(self.Phemetric_dc_filepath, ['metadata.json'])
        if len(metadata_file) == 0:
            raise ValueError('There has no valid sdc or the metadata file of the sdc was missing!')
        elif len(metadata_file) > 1:
            raise ValueError('There has more than one metadata file in the dir')
        else:
            try:
                with open(metadata_file[0]) as js_temp:
                    dc_metadata = json.load(js_temp)

                if not isinstance(dc_metadata, dict):
                    raise Exception('Please make sure the metadata file is a dictionary constructed in python!')
                else:
                    # Determine whether this datacube is Phemetric or not
                    if 'Phemetric_factor' not in dc_metadata.keys():
                        raise TypeError('The Phemetric_factor was lost or it is not a Phemetric datacube!')
                    elif dc_metadata['Phemetric_factor'] is False:
                        raise TypeError(f'{self.Phemetric_dc_filepath} is not a Phemetric datacube!')
                    elif dc_metadata['Phemetric_factor'] is True:
                        self.Phemetric_factor = dc_metadata['Phemetric_factor']
                    else:
                        raise TypeError('The Phemetric factor was under wrong type!')

                    for dic_name in self._fund_factor:
                        if dic_name not in dc_metadata.keys():
                            raise Exception(f'The {dic_name} is not in the dc metadata, double check!')
                        else:
                            self.__dict__[dic_name] = dc_metadata[dic_name]
            except:
                raise Exception('Something went wrong when reading the metadata!')

        start_time = time.time()
        print(f'Start loading the Phemetric datacube of {str(self.pheyear)} \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        # Read paraname file of the Phemetric datacube
        try:
            if self.Phemetric_factor is True:
                # Read paraname
                paraname_file = bf.file_filter(self.Phemetric_dc_filepath, ['paraname.npy'])
                if len(paraname_file) == 0:
                    raise ValueError('There has no paraname file or file was missing!')
                elif len(paraname_file) > 1:
                    raise ValueError('There has more than one paraname file in the Phemetric datacube dir')
                else:
                    paraname_list = np.load(paraname_file[0], allow_pickle=True)
                    self.paraname_list = [paraname for paraname in paraname_list]
            else:
                raise TypeError('Please input as a Phemetric datacube')
        except:
            raise Exception('Something went wrong when reading the paraname list!')

        # Read func dic
        try:
            if self.sparse_matrix and self.huge_matrix:
                if os.path.exists(self.Phemetric_dc_filepath + f'{self.index}_Phemetric_datacube\\'):
                    self.dc = NDSparseMatrix().load(self.Phemetric_dc_filepath + f'{self.index}_Phemetric_datacube\\')
                else:
                    raise Exception('Please double check the code if the sparse huge matrix is generated properly')
            elif not self.huge_matrix:
                self.dc_filename = bf.file_filter(self.Phemetric_dc_filepath, ['Phemetric_datacube.npy'])
                if len(self.dc_filename) == 0:
                    raise ValueError('There has no valid Phemetric datacube or the dc was missing!')
                elif len(self.dc_filename) > 1:
                    raise ValueError('There has more than one data file in the dc dir')
                else:
                    self.dc = np.load(self.dc_filename[0], allow_pickle=True)
            elif self.huge_matrix and not self.sparse_matrix:
                self.dc_filename = bf.file_filter(self.Phemetric_dc_filepath, ['sequenced_datacube', '.npy'], and_or_factor='and')
        except:
            raise Exception('Something went wrong when reading the Phemetric datacube!')

        # Drop duplicate layers
        self._drop_duplicate_layers()

        # Size calculation and shape definition
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]
        if self.dc_ZSize != len(self.paraname_list):
            raise TypeError('The Phemetric datacube is not consistent with the paraname file')

        print(f'Finish loading the Phemetric datacube of {str(self.pheyear)} \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def _drop_duplicate_layers(self):
        for _ in self.paraname_list:
            if len([t for t in self.paraname_list if t == _]) != 1:
                pos = [tt for tt in range(len(self.paraname_list)) if self.paraname_list[tt] == _]
                if self.sparse_matrix:
                    self.dc.SM_namelist.pop(pos[-1])
                    self.paraname_list.pop(pos[-1])
                    self.dc._update_size_para()
                else:
                    self.dc = np.delete(self.dc, pos[-1], axis=2)
                    self.paraname_list.pop(pos[-1])

    def __sizeof__(self):
        return self.dc.__sizeof__() + self.paraname_list.__sizeof__()

    def calculate_phemetrics(self, pheme_list: list, save2phemedc = True):

        pheme_list_temp = copy.copy(pheme_list)
        for pheme_temp in pheme_list:
            if pheme_temp not in self._support_pheme_list:
                raise ValueError(f'The {pheme_temp} is not supported')
            elif f'{self.pheyear}_{pheme_temp}' in self.paraname_list:
                pheme_list_temp.remove(pheme_temp)
        pheme_list = pheme_list_temp

        if self.curfit_dic['CFM'] == 'SPL':
            for pheme_temp in pheme_list:
                if pheme_temp == 'SOS':
                    if isinstance(self.dc, NDSparseMatrix):
                        self._add_layer(self.dc.SM_group[f'{str(self.pheyear)}_para_2'], 'SOS')
                    else:
                        self._add_layer(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_2')], 'SOS')
                elif pheme_temp == 'EOS':
                    if isinstance(self.dc, NDSparseMatrix):
                        self._add_layer(self.dc.SM_group[f'{str(self.pheyear)}_para_4'], 'EOS')
                    else:
                        self._add_layer(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_4')], 'EOS')
                elif pheme_temp == 'trough_vi':
                    if isinstance(self.dc, NDSparseMatrix):
                        self._add_layer(self.dc.SM_group[f'{str(self.pheyear)}_para_0'], 'trough_vi')
                    else:
                        self._add_layer(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_0')], 'trough_vi')
                elif pheme_temp == 'peak_vi':
                    para_dic = {}
                    for para_num in range(self.curfit_dic['para_num']):
                        if isinstance(self.dc, NDSparseMatrix):
                            para_dic[para_num] = copy.copy(self.dc.SM_group[f'{str(self.pheyear)}_para_{str(para_num)}'].toarray())
                        else:
                            para_dic[para_num] = copy.copy(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_{str(para_num)}')])

                    try:
                        peak_vi_array = np.zeros([para_dic[0].shape[0], para_dic[0].shape[1], 365])
                        for _ in range(peak_vi_array.shape[2]):
                            peak_vi_array[:, :, _] = _ + 1
                        peak_vi_array[para_dic[0] == 0] = np.nan
                        peak_vi_array = np.nanmax(seven_para_logistic_function(peak_vi_array, para_dic[0][y_temp, x_temp], para_dic[1][y_temp, x_temp], para_dic[2][y_temp, x_temp], para_dic[3][y_temp, x_temp], para_dic[4][y_temp, x_temp], para_dic[5][y_temp, x_temp], para_dic[6][y_temp, x_temp]), axis=2)
                        peak_vi_array[np.isnan(peak_vi_array)] = 0
                    except MemoryError:
                        peak_vi_array = copy.copy(para_dic[0])
                        peak_vi_array[peak_vi_array != 0] = -1
                        for y_temp in range(para_dic[0].shape[0]):
                            for x_temp in range(para_dic[0].shape[1]):
                                if peak_vi_array[y_temp, x_temp] == -1:
                                    peak_vi_array[y_temp, x_temp] = np.max(seven_para_logistic_function(np.linspace(1, 365, 365), para_dic[0][y_temp, x_temp], para_dic[1][y_temp, x_temp], para_dic[2][y_temp, x_temp], para_dic[3][y_temp, x_temp], para_dic[4][y_temp, x_temp], para_dic[5][y_temp, x_temp], para_dic[6][y_temp, x_temp]))
                        peak_vi_array[peak_vi_array == -1] = 0

                    self._add_layer(peak_vi_array, 'peak_vi')

                elif pheme_temp == 'peak_doy':
                    para_dic = {}
                    for para_num in range(self.curfit_dic['para_num']):
                        if isinstance(self.dc, NDSparseMatrix):
                            para_dic[para_num] = copy.copy(self.dc.SM_group[f'{str(self.pheyear)}_para_{str(para_num)}'].toarray())
                        else:
                            para_dic[para_num] = copy.copy(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_{str(para_num)}')])

                    peak_doy_array = copy.copy(para_dic[0])
                    peak_doy_array[peak_doy_array != 0] = -1
                    for y_temp in range(para_dic[0].shape[0]):
                        for x_temp in range(para_dic[0].shape[1]):
                            if peak_doy_array[y_temp, x_temp] == -1:
                                peak_doy_array[y_temp, x_temp] = np.argmax(seven_para_logistic_function(np.linspace(1, 365, 365), para_dic[0][y_temp, x_temp], para_dic[1][y_temp, x_temp], para_dic[2][y_temp, x_temp], para_dic[3][y_temp, x_temp], para_dic[4][y_temp, x_temp], para_dic[5][y_temp, x_temp], para_dic[6][y_temp, x_temp])) + 1
                    peak_doy_array[peak_doy_array == -1] = 0

                    self._add_layer(peak_doy_array, 'peak_doy')

                elif pheme_temp == 'GR':
                    if isinstance(self.dc, NDSparseMatrix):
                        self._add_layer(self.dc.SM_group[f'{str(self.pheyear)}_para_3'], 'GR')
                    else:
                        self._add_layer(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_3')], 'GR')
                elif pheme_temp == 'DR':
                    if isinstance(self.dc, NDSparseMatrix):
                        self._add_layer(self.dc.SM_group[f'{str(self.pheyear)}_para_5'], 'DR')
                    else:
                        self._add_layer(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_5')], 'DR')
                elif pheme_temp == 'DR2':
                    if isinstance(self.dc, NDSparseMatrix):
                        self._add_layer(self.dc.SM_group[f'{str(self.pheyear)}_para_6'], 'DR2')
                    else:
                        self._add_layer(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_6')], 'DR2')

        else:
            pass

        if save2phemedc:
            self.save(self.Phemetric_dc_filepath)
            self.__init__(self.Phemetric_dc_filepath)
        else:
            # Size calculation and shape definition
            self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]
            if self.dc_ZSize != len(self.paraname_list) or self.dc_ZSize != self.curfit_dic['para_num']:
                raise TypeError('The Phemetric datacube is not consistent with the paraname file')

    def _add_layer(self, array, layer_name: str):

        # Process the layer name
        if not isinstance(layer_name, str):
            raise TypeError('Please input the adding layer name as a string！')
        elif not layer_name.startswith(str(self.pheyear)):
            layer_name = str(self.pheyear) + '_' + layer_name

        if self.sparse_matrix:
            sparse_type = type(self.dc.SM_group[self.dc.SM_namelist[0]])
            if not isinstance(array, sparse_type):
                try:
                    array = type(self.dc.SM_group[self.dc.SM_namelist[0]])(array)
                except:
                    raise Exception(f'The adding layer {layer_name} cannot be converted to the data type')
            try:
                self.dc.add_layer(array, layer_name, self.dc.shape[2])
                self.paraname_list.append(layer_name)
            except:
                raise Exception('Some error occurred during the add layer within a phemetric dc')
        else:
            try:
                self.dc = np.concatenate((self.dc, array.reshape(array.shape[0], array.shape[1], 1)), axis=2)
                self.paraname_list.append(layer_name)
            except:
                raise Exception('Some error occurred during the add layer within a phemetric dc')

    def save(self, output_path: str):
        start_time = time.time()
        print(f'Start saving the Phemetric datacube of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        if not os.path.exists(output_path):
            bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name

        metadata_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI,
                        'ROI_array': self.ROI_array, 'ROI_tif': self.ROI_tif, 'sdc_factor': self.sdc_factor,
                        'coordinate_system': self.coordinate_system, 'sparse_matrix': self.sparse_matrix, 'huge_matrix': self.huge_matrix,
                        'size_control_factor': self.size_control_factor, 'oritif_folder': self.oritif_folder, 'dc_group_list': self.dc_group_list, 'tiles': self.tiles,
                        'pheyear': self.pheyear, 'curfit_dic': self.curfit_dic, 'Phemetric_factor': self.Phemetric_factor}

        paraname = self.paraname_list
        np.save(f'{output_path}paraname.npy', paraname)
        with open(f'{output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_Phemetric_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_Phemetric_datacube.npy', self.dc)

        print(f'Finish saving the Phemetric datacube of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')