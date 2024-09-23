# coding=utf-8
import copy
import traceback
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from basic_function import Path
import concurrent.futures
from itertools import repeat
from GEDI_toolbox import GEDI_main as gedi
from Landsat_toolbox.Landsat_main_v2 import Landsat_dc
from scipy import sparse as sm
from Sentinel2_toolbox.utils import *
from Sentinel2_toolbox.Sentinel_main_V2 import Sentinel2_dc, Sentinel2_ds
from River_GIS.River_GIS import Inunfac_dc
from tqdm import tqdm as tq
from .utils import *
from shapely import wkt


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def system_recovery_function(t, ymax, yo, b):
    return ymax - (ymax - yo) * np.exp(- b * t)


def linear_f(x, a, b):
    return a * (x + 1985) + b


def two_term_fourier(x, a0, a1, b1, a2, b2, w):
    return a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x) + a2 * np.cos(2 * w * x) + b2 * np.sin(2 * w * x)


class Denv_dc(object):

    ####################################################################################################
    # Denv dc represents "Daily Environment Datacube"
    # It normally contains data like daily temperature and daily radiation, etc.
    # And stack into a 3-D datacube type.
    # Currently, it was integrated into the NCEI and MODIS FPAR toolbox as an output datatype.
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
        print(f'Start loading the \033[1;31m{str(self.timerange)}\033[0m Denv dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m')

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
            date_start = datetime.date(year = self.timerange, month=1, day=1).toordinal()
            date_end = datetime.date(year = self.timerange + 1, month=1, day=1).toordinal()
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_start, date_end)]
            self.compete_doy_list = bf.date2doy(compete_doy_list)
        elif self.timescale == 'month':
            year_temp = int(np.floor(self.timerange / 100))
            month_temp = int(np.mod(self.timerange, 100))
            date_start = datetime.date(year=year_temp, month=month_temp, day=1).toordinal()
            date_end = datetime.date(year=year_temp, month=month_temp + 1, day=1).toordinal()
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_start, date_end)]
            self.compete_doy_list = bf.date2doy(compete_doy_list)
        elif self.timescale == 'all':
            date_min, date_max = bf.doy2date(min(self.sdc_doylist)), bf.doy2date(max(self.sdc_doylist))
            date_min = datetime.date(year=int(np.floor(date_min / 1000)), month=1, day=1).toordinal() + np.mod(date_min, 1000) - 1
            date_max = datetime.date(year=int(np.floor(date_max / 1000)), month=1, day=1).toordinal() + np.mod(date_max, 1000)
            compete_doy_list = [datetime.date.fromordinal(date_temp).strftime('%Y%m%d') for date_temp in range(date_min, date_max)]
            self.compete_doy_list = bf.date2doy(compete_doy_list)

        # Read the Denv datacube
        try:
            if self.sparse_matrix and self.huge_matrix:
                self.dc_filename = self.Denv_dc_filepath + f'{self.index}_Denv_datacube\\'
                if os.path.exists(self.dc_filename):
                    self.dc = NDSparseMatrix().load(self.dc_filename)
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
            raise Exception('Code has issues in the Denv autofill procedure!')

        # autotrans sparse matrix
        if self.sparse_matrix and self.dc._matrix_type == sm.coo_matrix:
            self._autotrans_sparse_matrix()

        # Backdoor metadata check
        self._backdoor_metadata_check()

        # Size calculation and shape definition
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]
        if self.dc_ZSize != len(self.sdc_doylist):
            raise TypeError('The Denv datacube is not consistent with the doy list')

        print(f'Finish loading the \033[1;31m{str(self.timerange)}\033[0m Denv dc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def __sizeof__(self):
        return self.dc.__sizeof__() + self.sdc_doylist.__sizeof__()

    def _backdoor_metadata_check(self):

        backdoor_issue = False
        # Check if metadata is valid and timeliness
        # Problem 1 dir change between multi-devices ROI ROI arr ROI tif
        if not os.path.exists(self.ROI_tif):
            self.ROI_tif = retrieve_correct_filename(self.ROI_tif)
            backdoor_issue = True

        if not os.path.exists(self.ROI_array):
            self.ROI_array = retrieve_correct_filename(self.ROI_array)
            backdoor_issue = True

        if not os.path.exists(self.ROI):
            self.ROI = retrieve_correct_filename(self.ROI)
            backdoor_issue = True

        if self.ROI_tif is None or self.ROI_array is None or self.ROI is None:
            raise Exception('Please manually change the roi path in the phemetric dc')

        # Problem 2
        if backdoor_issue:
            self.save(self.Phemetric_dc_filepath)
            self.__init__(self.Phemetric_dc_filepath)

    def _autofill_Denv_DC(self):
        # Interpolate the denv dc
        autofill_factor = False
        for date_temp in self.compete_doy_list:
            if date_temp not in self.sdc_doylist:
                autofill_factor = True
                if date_temp == self.compete_doy_list[0]:
                    date_merge = self.compete_doy_list[1]
                    if self.sparse_matrix:
                        self.dc.add_layer(self.dc.SM_group[date_merge], date_temp, 0)
                    else:
                        self.dc = np.insert(self.dc, 0, values=self.dc[:, :, 0], axis=2)
                    self.sdc_doylist.insert(0, date_temp)
                elif date_temp == self.compete_doy_list[-1]:
                    date_merge = self.compete_doy_list[-2]
                    if self.sparse_matrix:
                        self.dc.add_layer(self.dc.SM_group[date_merge], date_temp, -1)
                    else:
                        self.dc = np.insert(self.dc, 0, values=self.dc[:, :, -1], axis=2)
                    self.sdc_doylist.insert(-1, date_temp)
                else:
                    date_beg, date_end, _beg, _end = None, None, None, None
                    for _ in range(1, 60):
                        ordinal_date = datetime.date(year=int(np.floor(bf.doy2date(date_temp) / 10000)),
                                                     month=int(np.floor(np.mod(bf.doy2date(date_temp), 10000) / 100)),
                                                     day=int(np.mod(bf.doy2date(date_temp), 100))).toordinal()
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

                    if isinstance(self.dc, NDSparseMatrix):
                        if date_end is None:
                            array_beg = self.dc.SM_group[date_beg]
                            self.dc.add_layer(array_beg, date_temp, self.compete_doy_list.index(date_temp))
                        elif date_beg is None:
                            array_end = self.dc.SM_group[date_end]
                            self.dc.add_layer(array_end, date_temp, self.compete_doy_list.index(date_temp))
                        else:
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
                        self.dc = np.insert(self.dc, self.compete_doy_list.index(date_temp),
                                            values=array_out.reshape([array_out.shape[0], array_out.shape[1], 1]),
                                            axis=2)
                    self.sdc_doylist.insert(self.compete_doy_list.index(date_temp), date_temp)

        if self.sdc_doylist != self.compete_doy_list:
            raise Exception('Error occurred during the autofill for the Denv DC!')

        if autofill_factor:
            self.save(self.Denv_dc_filepath)
            self.__init__(self.Denv_dc_filepath)

    def save(self, output_path: str):

        start_time = time.time()
        print(f'Start saving the sdc of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        output_path = bf.Path(output_path).path_name
        bf.create_folder(output_path) if not os.path.exists(output_path) else None

        metadata_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI,
                        'ROI_array': self.ROI_array, 'ROI_tif': self.ROI_tif, 'sdc_factor': self.sdc_factor,
                        'coordinate_system': self.coordinate_system, 'sparse_matrix': self.sparse_matrix,
                        'huge_matrix': self.huge_matrix,
                        'size_control_factor': self.size_control_factor, 'oritif_folder': self.oritif_folder,
                        'dc_group_list': self.dc_group_list,
                        'tiles': self.tiles, 'timescale': self.timescale, 'timerange': self.timerange,
                        'Denv_factor': self.Denv_factor}
        doy = self.sdc_doylist
        np.save(f'{output_path}doy.npy', doy)
        with open(f'{output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_Denv_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_Denv_datacube.npy', self.dc)

        print(
            f'Finish saving the sdc of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def _autotrans_sparse_matrix(self):

        if not isinstance(self.dc, NDSparseMatrix):
            raise TypeError('The autotrans sparse matrix is specified for the NDsm!')

        for _ in self.dc.SM_namelist:
            if isinstance(self.dc.SM_group[_], sm.coo_matrix):
                self.dc.SM_group[_] = sm.csr_matrix(self.dc.SM_group[_])
                self.dc._update_size_para()
        self.save(self.Denv_dc_filepath)

    def denv_comparison(self):
        pass


class Phemetric_dc(object):

    ####################################################################################################
    # Phemetric_dc represents "Phenological Metric Datacube"
    # It normally contains data like phenological parameters derived from the curve fitting, etc
    # And stack into a 3-D datacube type.
    # Currently, it was taken as the Sentinel_dcs/Landsat_dcs's output datatype after the phenological analysis.
    ####################################################################################################

    def __init__(self, phemetric_filepath, work_env=None):

        # Check the phemetric path
        self.Phemetric_dc_filepath = bf.Path(phemetric_filepath).path_name

        # Init key var
        self.ROI_name, self.ROI, self.ROI_tif, self.ROI_array = None, None, None, None
        self.index, self.Datatype, self.coordinate_system = None, None, None
        self.dc_group_list, self.tiles = None, None
        self.sdc_factor, self.sparse_matrix, self.size_control_factor, self.huge_matrix = False, False, False, False
        self.Phemetric_factor, self.pheyear = False, None
        self.curfit_dic = {}

        # Init protected var
        self._support_pheme_list = ['SOS', 'EOS', 'trough_vi', 'peak_vi', 'peak_doy', 'GR', 'DR', 'DR2', 'MAVI', 'TSVI']

        # Check work env
        if work_env is not None:
            self._work_env = Path(work_env).path_name
        else:
            self._work_env = Path(os.path.dirname(os.path.dirname(self.Phemetric_dc_filepath))).path_name
        self.root_path = Path(os.path.dirname(os.path.dirname(self._work_env))).path_name

        # Define the basic var name
        self._fund_factor = ('ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'curfit_dic', 'pheyear',
                             'coordinate_system', 'oritif_folder', 'ROI_tif', 'sparse_matrix',
                             'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles', 'Nodata_value', 'Zoffset')

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
                print(traceback.format_exc())
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
                self.dc_filename = self.Phemetric_dc_filepath + f'{self.index}_Phemetric_datacube\\'
                if os.path.exists(self.dc_filename):
                    self.dc = NDSparseMatrix().load(self.dc_filename)
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
            print(traceback.format_exc())
            raise Exception('Something went wrong when reading the Phemetric datacube!')

        # autotrans sparse matrix
        if self.sparse_matrix and self.dc._matrix_type == sm.coo_matrix:
            self._autotrans_sparse_matrix()

        # Drop duplicate layers
        self._drop_duplicate_layers()

        # Backdoor metadata check
        self._backdoor_metadata_check()

        # Size calculation and shape definition
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]
        if self.dc_ZSize != len(self.paraname_list):
            raise TypeError('The Phemetric datacube is not consistent with the paraname file')

        print(f'Finish loading the Phemetric datacube of {str(self.pheyear)} \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def _update_parasize_(self):
        self.dc_XSize, self.dc_YSize, self.dc_ZSize = self.dc.shape[1], self.dc.shape[0], self.dc.shape[2]

    def _backdoor_metadata_check(self):

        backdoor_issue = False
        # Check if metadata is valid and timeliness
        # Problem 1 dir change between multi-devices ROI ROI arr ROI tif
        if not os.path.exists(self.ROI_tif):
            self.ROI_tif = retrieve_correct_filename(self.ROI_tif)
            backdoor_issue = True

        if not os.path.exists(self.ROI_array):
            self.ROI_array = retrieve_correct_filename(self.ROI_array)
            backdoor_issue = True

        if not os.path.exists(self.ROI):
            self.ROI = retrieve_correct_filename(self.ROI)
            backdoor_issue = True

        if self.ROI_tif is None or self.ROI_array is None or self.ROI is None:
            raise Exception('Please manually change the roi path in the phemetric dc')

        # Problem 2 Check if the coordinate system is missing
        if self.coordinate_system is None:
            ds_temp = gdal.Open(self.ROI_tif)
            self.coordinate_system = bf.retrieve_srs(ds_temp)
            backdoor_issue = True

        if backdoor_issue:
            self.save(self.Phemetric_dc_filepath)
            self.__init__(self.Phemetric_dc_filepath)

    def _autotrans_sparse_matrix(self):

        if not isinstance(self.dc, NDSparseMatrix):
            raise TypeError('The autotrans sparse matrix is specified for the NDsm!')

        for _ in self.dc.SM_namelist:
            if isinstance(self.dc.SM_group[_], sm.coo_matrix):
                self.dc.SM_group[_] = sm.csr_matrix(self.dc.SM_group[_])
                self.dc._update_size_para()
        self.save(self.Phemetric_dc_filepath)

    def _drop_duplicate_layers(self):
        for _ in self.paraname_list:
            if len([t for t in self.paraname_list if t == _]) != 1:
                pos = [tt for tt in range(len(self.paraname_list)) if self.paraname_list[tt] == _]
                if isinstance(self.dc, NDSparseMatrix):
                    self.dc.SM_namelist.pop(pos[-1])
                    self.paraname_list.pop(pos[-1])
                    self.dc._update_size_para()
                else:
                    self.dc = np.delete(self.dc, pos[-1], axis=2)
                    self.paraname_list.pop(pos[-1])

    def __sizeof__(self):
        return self.dc.__sizeof__() + self.paraname_list.__sizeof__()

    def calculate_phemetrics(self, pheme_list: list, save2phemedc: bool = True):

        pheme_list_temp = copy.copy(pheme_list)
        for pheme_temp in pheme_list:
            if pheme_temp not in self._support_pheme_list:
                raise ValueError(f'The {pheme_temp} is not supported')
            elif f'{self.pheyear}_{pheme_temp}' in self.paraname_list:
                pheme_list_temp.remove(pheme_temp)
        pheme_list = pheme_list_temp

        if self.curfit_dic['CFM'] == 'SPL':

            para_dic = {}
            for para_num in range(self.curfit_dic['para_num']):
                if isinstance(self.dc, NDSparseMatrix):
                    arr_temp = copy.copy(self.dc.SM_group[f'{str(self.pheyear)}_para_{str(para_num)}'].toarray())
                    arr_temp[arr_temp == self.Nodata_value] = np.nan
                    para_dic[para_num] = arr_temp
                else:
                    arr_temp = copy.copy(self.dc[self.paraname_list.index(f'{str(self.pheyear)}_para_{str(para_num)}')])
                    if ~np.isnan(self.Nodata_value):
                        arr_temp[arr_temp == self.Nodata_value] = np.nan
                    para_dic[para_num] = arr_temp
            arr_temp = None

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

                elif pheme_temp == 'peak_vi':

                    try:
                        peak_vi_array = copy.copy(para_dic[0])
                        peak_vi_array[~np.isnan(peak_vi_array)] = -1
                        xy_list = np.argwhere(~np.isnan(peak_vi_array)).tolist()
                        with tqdm(total=len(xy_list), desc=f'Generate the peak_vi of {str(self.pheyear)}',
                                  bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                            for xy_temp in xy_list:
                                y_temp, x_temp = xy_temp[0], xy_temp[1]
                                if peak_vi_array[y_temp, x_temp] == -1:
                                    peak_vi_array[y_temp, x_temp] = np.max(seven_para_logistic_function(
                                                                     np.linspace(1, 365, 365), para_dic[0][y_temp, x_temp],
                                                                     para_dic[1][y_temp, x_temp],
                                                                     para_dic[2][y_temp, x_temp],
                                                                     para_dic[3][y_temp, x_temp],
                                                                     para_dic[4][y_temp, x_temp],
                                                                     para_dic[5][y_temp, x_temp],
                                                                     para_dic[6][y_temp, x_temp]))
                                pbar.update()
                        peak_vi_array[peak_vi_array == -1] = self.Nodata_value
                        peak_vi_array[np.isnan(peak_vi_array)] = self.Nodata_value
                    except:
                        raise Exception('Unable to create peak vi!')
                    self._add_layer(peak_vi_array, 'peak_vi')

                elif pheme_temp == 'peak_doy':

                    try:
                        peak_doy_array = copy.copy(para_dic[0])
                        peak_doy_array[~np.isnan(peak_doy_array)] = -1
                        xy_list = np.argwhere(~np.isnan(peak_doy_array)).tolist()
                        with tqdm(total=len(xy_list), desc=f'Generate the peak_doy of {str(self.pheyear)}',
                                  bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                            for xy_temp in xy_list:
                                y_temp, x_temp = xy_temp[0], xy_temp[1]
                                if peak_doy_array[y_temp, x_temp] == -1:
                                    peak_doy_array[y_temp, x_temp] = np.argmax(
                                        seven_para_logistic_function(np.linspace(1, 365, 365), para_dic[0][y_temp, x_temp],
                                                                     para_dic[1][y_temp, x_temp],
                                                                     para_dic[2][y_temp, x_temp],
                                                                     para_dic[3][y_temp, x_temp],
                                                                     para_dic[4][y_temp, x_temp],
                                                                     para_dic[5][y_temp, x_temp],
                                                                     para_dic[6][y_temp, x_temp])) + 1
                                    pbar.update()
                        peak_doy_array[peak_doy_array == -1] = self.Nodata_value
                        peak_doy_array[np.isnan(peak_doy_array)] = self.Nodata_value
                    except:
                        raise Exception('Unable to create the peak doy')
                    self._add_layer(peak_doy_array, 'peak_doy')

                elif pheme_temp == 'MAVI':

                    try:
                        MAVI_array = copy.copy(para_dic[0])
                        MAVI_array[~np.isnan(MAVI_array)] = -1
                        xy_list = np.argwhere(~np.isnan(MAVI_array)).tolist()
                        with tqdm(total=len(xy_list), desc=f'Generate the MAVI of {str(self.pheyear)}',
                                  bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                            for xy_temp in xy_list:
                                y_temp, x_temp = xy_temp[0], xy_temp[1]
                                para_list = [para_dic[_][y_temp, x_temp] for _ in range(7)]
                                if True not in [np.isnan(_) for _ in para_list]:
                                    index_arr = seven_para_logistic_function(np.linspace(1, 365, 365), para_list[0], para_list[1],
                                                                             para_list[2], para_list[3], para_list[4], para_list[5],
                                                                             para_list[6])
                                    if np.sum(np.isnan(index_arr)) == 0:
                                        max_index = np.argmax(index_arr)
                                        derivative_arr = np.zeros_like(index_arr) * np.nan

                                        for _ in range(1, 364):
                                            derivative_arr[_] = (index_arr[_ + 1] - index_arr[_ - 1]) / 2
                                        try:
                                            derivative_index = np.min(np.argwhere(derivative_arr < - (
                                                        (para_list[1] - (para_list[4] * para_list[6])) / (
                                                            8 * para_list[5]))))
                                        except:
                                            derivative_index = int(para_list[4] - 3 * para_list[5])

                                        if derivative_index > max_index:
                                            try:
                                                MAVI_array[y_temp, x_temp] = np.mean(
                                                    index_arr[max_index: derivative_index])
                                            except:
                                                pass
                                        else:
                                            MAVI_array[y_temp, x_temp] = index_arr[max_index]
                                    else:
                                        pass
                                else:
                                    pass
                                pbar.update()

                        MAVI_array[MAVI_array == -1] = self.Nodata_value
                        MAVI_array[np.isnan(MAVI_array)] = self.Nodata_value
                    except:
                        raise Exception('Error occurred during the MAVI generation!')
                    self._add_layer(MAVI_array, 'MAVI')

                elif pheme_temp == 'TSVI':

                    try:
                        TSVI_array = copy.copy(para_dic[0])
                        TSVI_array[~np.isnan(TSVI_array)] = -1
                        xy_list = np.argwhere(~np.isnan(TSVI_array)).tolist()
                        d_temp = np.linspace(1, 365, 365)
                        with tqdm(total=len(xy_list), desc=f'Generate the TSVI of {str(self.pheyear)}',
                                  bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                            for xy_temp in xy_list:
                                y_temp, x_temp = xy_temp[0], xy_temp[1]
                                para_list = [para_dic[_][y_temp, x_temp] for _ in range(7)]
                                if True not in [np.isnan(_) for _ in para_list]:
                                    index_arr = seven_para_logistic_function(d_temp, para_list[0], para_list[1],para_list[2],
                                                                             para_list[3], para_list[4], para_list[5],
                                                                             para_list[6])
                                    if np.sum(np.isnan(index_arr)) == 0:
                                        derivative_arr = np.zeros_like(index_arr) * np.nan
                                        for _ in range(1, 364):
                                            derivative_arr[_] = (index_arr[_ + 1] - index_arr[_ - 1]) / 2
                                        try:
                                            derivative_index = np.min(np.argwhere(derivative_arr < - ((para_list[1] - (para_list[4] * para_list[6])) / (8 * para_list[5]))))
                                        except:
                                            derivative_index = int(para_list[4] - 3 * para_list[5])
                                        TSVI_array[y_temp, x_temp] = index_arr[derivative_index]
                                    else:
                                        pass
                                else:
                                    pass
                                pbar.update()
                        TSVI_array[TSVI_array == -1] = self.Nodata_value
                        TSVI_array[np.isnan(TSVI_array)] = self.Nodata_value
                    except:
                        raise Exception('Error occurred during the TSVI generation!')
                    self._add_layer(TSVI_array, 'TSVI')
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

    def remove_layer(self, layer_name):

        # Process the layer name
        layer_ = []
        if isinstance(layer_name, str):
            if str(self.pheyear) + '_' + layer_name in self.paraname_list:
                layer_.append(str(self.pheyear) + '_' + layer_name)
            elif layer_name in self.paraname_list:
                layer_.append(layer_name)
        elif isinstance(layer_name, list):
            for layer_temp in layer_name:
                if str(self.pheyear) + '_' + layer_temp in self.paraname_list:
                    layer_.append(str(self.pheyear) + '_' + layer_temp)
                elif layer_temp in self.paraname_list:
                    layer_.append(layer_temp)

        # Remove selected layer
        for _ in layer_:
            if self.sparse_matrix:
                try:
                    self.dc.remove_layer(_)
                    self.paraname_list.remove(_)
                except:
                    raise Exception('Some error occurred during the removal of layer within a phemetric dc')
            else:
                try:
                    self.dc = np.delete(self.dc, self.paraname_list.index(_), axis=2)
                    self.paraname_list.remove(layer_name)
                except:
                    raise Exception('Some error occurred during the add layer within a phemetric dc')

        self.save(self.Phemetric_dc_filepath)
        self.__init__(self.Phemetric_dc_filepath)

    def save(self, output_path: str):
        start_time = time.time()
        print(f'Start saving the Phemetric datacube of \033[1;31m{self.index}\033[0m in the \033[1;34m{self.ROI_name}\033[0m')

        if not os.path.exists(output_path):
            bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name

        metadata_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI,
                        'ROI_array': self.ROI_array, 'ROI_tif': self.ROI_tif, 'sdc_factor': self.sdc_factor,
                        'coordinate_system': self.coordinate_system, 'sparse_matrix': self.sparse_matrix,
                        'huge_matrix': self.huge_matrix,
                        'size_control_factor': self.size_control_factor, 'oritif_folder': self.oritif_folder,
                        'dc_group_list': self.dc_group_list, 'tiles': self.tiles,
                        'pheyear': self.pheyear, 'curfit_dic': self.curfit_dic,
                        'Phemetric_factor': self.Phemetric_factor,
                        'Nodata_value': self.Nodata_value, 'Zoffset': self.Zoffset}

        paraname = self.paraname_list
        np.save(f'{output_path}paraname.npy', paraname)
        with open(f'{output_path}metadata.json', 'w') as js_temp:
            json.dump(metadata_dic, js_temp)

        if self.sparse_matrix:
            self.dc.save(f'{output_path}{str(self.index)}_Phemetric_datacube\\')
        else:
            np.save(f'{output_path}{str(self.index)}_Phemetric_datacube.npy', self.dc)

        print(f'Finish saving the Phemetric datacube of \033[1;31m{self.index}\033[0m for the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0ms')

    def dc2tif(self, phe_list: list = None, output_folder: str = None):

        # Process phe list
        if phe_list is None:
            phe_list = self.paraname_list
        elif isinstance(phe_list, list):
            phe_list_ = []
            for phe_temp in phe_list:
                if phe_temp in self.paraname_list:
                    phe_list_.append(phe_temp)
                elif str(self.pheyear) + '_' + phe_temp in self.paraname_list:
                    phe_list_.append(str(self.pheyear) + '_' + phe_temp)
            phe_list = phe_list_
        else:
            raise TypeError('Phe list should under list type')

        # Process output folder
        if output_folder is None:
            output_folder = self.Phemetric_dc_filepath + 'Pheme_TIF\\'
        else:
            bf.create_folder(output_folder)
            output_folder = Path(output_folder).path_name

        ds_temp = gdal.Open(self.ROI_tif)
        for phe_ in phe_list:
            if not os.path.exists(output_folder + f'{str(phe_)}.TIF'):
                if self.sparse_matrix:
                    phe_arr = self.dc.SM_group[phe_].toarray()
                else:
                    phe_arr = self.dc[:, :, self.paraname_list.index(phe_)]
                phe_arr[phe_arr == self.Nodata_value] = np.nan
                bf.write_raster(ds_temp, phe_arr, output_folder, f'{str(phe_)}.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)


class RS_dcs(object):

    def __init__(self, *args, work_env: str = None, auto_harmonised: bool = True, space_optimised: bool = True):

        # init_key_var
        self._sdc_factor_list = []
        self.sparse_matrix, self.huge_matrix = False, False
        self.s2dc_doy_list = []
        self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array, self.coordinate_system = None, None, None, None, None
        self.year_list = []

        # Generate the datacubes list
        self.dcs = []
        self._dcs_backup_, self._doys_backup_, self._dc_typelist = [], [], []
        self._dc_XSize_list, self._dc_YSize_list, self._dc_ZSize_list = [], [], []
        self._index_list, self._tiles_list = [], []
        self._size_control_factor_list, self.oritif_folder_list = [], []
        self._Zoffset_list, self._Nodata_value_list = [], []
        self._space_optimised = space_optimised
        self._sparse_matrix_list, self._huge_matrix_list = [], []

        # Define the indicator for different dcs
        self._phemetric_namelist, self._pheyear_list = None, []
        self._inunfac_namelist, self._inunyear_list = None, []
        self._withPhemetricdc_, self._withDenvdc_, self._withS2dc_, self._withLandsatdc_, self._withInunfacdc_ = False, False, False, False, False
        self._s2dc_work_env, self._phemetric_work_env, self._denv_work_env, self._inunfac_work_env = None, None, None, None
        self._denvyear_list = []

        if not isinstance(args, (tuple, list)):
            raise TypeError('Please mention all dcs should be input as args or *args')

        # Separate into Denv Phemetric and Sentinel-2 datacube
        for args_temp in args:
            if not isinstance(args_temp, (Sentinel2_dc, Denv_dc, Phemetric_dc, Landsat_dc, Inunfac_dc)):
                raise TypeError('The RS datacubes should be a bunch of Sentinel2 datacube, Landsat datacube, phemetric datacube, inunfac or Denv datacube!')
            else:
                self._dcs_backup_.append(args_temp)
                if isinstance(args_temp, Phemetric_dc):
                    self._doys_backup_.append(args_temp.paraname_list)
                elif isinstance(args_temp, Inunfac_dc):
                    self._doys_backup_.append(args_temp.paraname_list)
                elif isinstance(args_temp, (Sentinel2_dc, Denv_dc, Landsat_dc)):
                    self._doys_backup_.append(args_temp.sdc_doylist)
                self._dc_typelist.append(type(args_temp))

        if len(self._dcs_backup_) == 0:
            raise ValueError('Please input at least one valid Sentinel2/Phemetric/Denv/Landsat/Inunfac datacube into the RSDCs')

        if not isinstance(auto_harmonised, bool):
            raise TypeError('Please input the auto harmonised factor as bool type!')
        else:
            harmonised_factor = False

        # Merge all the factor found in metadata
        try:
            factor_all = ['dc_XSize', 'dc_YSize', 'dc_ZSize']
            for args_temp in args:
                factor_all.extend(list(args_temp._fund_factor))
            factor_all = list(set(factor_all))
            for factor_temp in factor_all:
                self.__dict__[f'_{factor_temp}_list'] = []

            for args_temp in args:
                for factor_temp in factor_all:
                    if f'_{factor_temp}_list' not in self.__dict__.keys():
                        self.__dict__[f'_{factor_temp}_list'] = []

                    if factor_temp in args_temp.__dict__.keys():
                        self.__dict__[f'_{factor_temp}_list'].append(args_temp.__dict__[factor_temp])
                    else:
                        self.__dict__[f'_{factor_temp}_list'].append(None)

            for factor_temp in factor_all:
                if len(self.__dict__[f'_{factor_temp}_list']) != len(self.__dict__[f'_{factor_all[0]}_list']):
                    raise ImportError('The factor of some dcs is not properly imported!')
        except:
            print(traceback.format_exc())
            raise Exception('The metadata is not properly imported due to the reason above!')

        # Check the consistency of datacube size
        try:
            x_size, y_size, z_S2_size, z_Phemetric_size, z_Inunfac_size, z_Denv_size, z_Landsat_size = 0, 0, 0, 0, 0, 0, 0
            for _ in range(len(self._dc_typelist)):

                if self._dc_typelist[_] == Sentinel2_dc:
                    self._withS2dc_ = True
                    # Retrieve the shape inform
                    if x_size == 0 and y_size == 0:
                        x_size, y_size = self._dcs_backup_[_].dc_XSize, self._dcs_backup_[_].dc_YSize

                    if x_size != self._dcs_backup_[_].dc_XSize or y_size != self._dcs_backup_[_].dc_YSize:
                        raise Exception('Please make sure all the Sentinel-2 datacube share the same size!')

                elif self._dc_typelist[_] == Landsat_dc:
                    self._withLandsatdc_ = True
                    # Retrieve the shape inform
                    if x_size == 0 and y_size == 0:
                        x_size, y_size = self._dcs_backup_[_].dc_XSize, self._dcs_backup_[_].dc_YSize

                    if x_size != self._dcs_backup_[_].dc_XSize or y_size != self._dcs_backup_[_].dc_YSize:
                        raise Exception('Please make sure all the Landsat datacube share the same size!')

                elif self._dc_typelist[_] == Denv_dc:
                    self._withDenvdc_ = True
                    # Retrieve the shape inform
                    if x_size == 0 and y_size == 0:
                        x_size, y_size = self._dcs_backup_[_].dc_XSize, self._dcs_backup_[_].dc_YSize

                    if x_size != self._dcs_backup_[_].dc_XSize or y_size != self._dcs_backup_[_].dc_YSize:
                        raise Exception('Please make sure all the Denv datacube share the same size!')

                elif self._dc_typelist[_] == Phemetric_dc:
                    self._withPhemetricdc_ = True
                    # Retrieve the shape inform
                    if x_size == 0 and y_size == 0:
                        x_size, y_size = self._dcs_backup_[_].dc_XSize, self._dcs_backup_[_].dc_YSize

                    if z_Phemetric_size == 0:
                        z_Phemetric_size = self._dcs_backup_[_].dc_ZSize

                    if x_size != self._dcs_backup_[_].dc_XSize or y_size != self._dcs_backup_[_].dc_YSize:
                        raise Exception('Please make sure all the Denv datacube share the same size!')
                    elif z_Phemetric_size != self._dcs_backup_[_].dc_ZSize:
                        raise Exception('The Phemetric_dc datacubes is not consistent in the Z dimension! Double check the input!')

                elif self._dc_typelist[_] == Inunfac_dc:
                    self._withInunfacdc_ = True
                    # Retrieve the shape inform
                    if x_size == 0 and y_size == 0:
                        x_size, y_size = self._dcs_backup_[_].dc_XSize, self._dcs_backup_[_].dc_YSize

                    if z_Inunfac_size == 0:
                        z_Inunfac_size = self._dcs_backup_[_].dc_ZSize

                    if x_size != self._dcs_backup_[_].dc_XSize or y_size != self._dcs_backup_[_].dc_YSize:
                        raise Exception('Please make sure all the Inunfac datacube share the same size!')
                    elif z_Inunfac_size != self._dcs_backup_[_].dc_ZSize:
                        raise Exception('The Inunfac datacubes is not consistent in the Z dimension! Double check the input!')

            if x_size != 0 and y_size != 0:
                self.dcs_XSize, self.dcs_YSize = x_size, y_size
            else:
                raise Exception('Error occurred when obtaining the x y size of s2 dcs!')
        except:
            print(traceback.format_exc())
            raise Exception('Error occurred during the consistency check of datacube size!')

        # Check the consistency of S2dcs
        if self._withS2dc_:
            s2dc_pos = [i for i, v in enumerate(self._dc_typelist) if v == Sentinel2_dc]
            if len(s2dc_pos) != 1:
                for factor_temp in ['ROI', 'ROI_name', 'ROI_array', 'ROI_tif', 'coordinate_system',
                                    'sparse_matrix', 'huge_matrix', 'sdc_factor', 'dc_group_list', 'tiles']:
                    if False in [self.__dict__[f'_{factor_temp}_list'][s2dc_pos[0]] == self.__dict__[f'_{factor_temp}_list'][pos] for pos in s2dc_pos]:
                        raise ValueError(f'Please make sure the {factor_temp} for all the dcs were consistent!')

                # Read the doy or date list
                if (False in [len(self._doys_backup_[s2dc_pos[0]]) == len(self._doys_backup_[pos_temp]) for pos_temp in s2dc_pos]
                        or False in [(self._doys_backup_[pos_temp] == self._doys_backup_[s2dc_pos[0]]) for pos_temp in s2dc_pos]):
                    if auto_harmonised:
                        harmonised_factor = True
                    else:
                        raise Exception('The datacubes is not consistent in the date dimension! Turn auto harmonised factor as True if wanna avoid this problem!')

                # Harmonised the dcs
                if harmonised_factor:
                    self._auto_harmonised_dcs(Sentinel2_dc)

            # Define the output_path
            if work_env is None:
                self._s2dc_work_env = Path(
                    os.path.dirname(os.path.dirname(self._dcs_backup_[s2dc_pos[0]].dc_filepath))).path_name
            else:
                self._s2dc_work_env = work_env

            # Determine the Zsize
            self.s2dc_doy_list = self._doys_backup_[s2dc_pos[0]]
            self.S2dc_ZSize = len(self.s2dc_doy_list)

        # Check the consistency of Landsat dcs
        if self._withLandsatdc_:
            Landsatdc_pos = [i for i, v in enumerate(self._dc_typelist) if v == Landsat_dc]
            if len(Landsatdc_pos) != 1:
                for factor_temp in ['ROI', 'ROI_name', 'ROI_array', 'ROI_tif', 'coordinate_system',
                                    'sparse_matrix', 'huge_matrix', 'sdc_factor', 'dc_group_list', 'tiles']:
                    if False in [self.__dict__[f'_{factor_temp}_list'][Landsatdc_pos[0]] ==
                                 self.__dict__[f'_{factor_temp}_list'][pos] for pos in Landsatdc_pos]:
                        raise ValueError(f'Please make sure the {factor_temp} for all the dcs were consistent!')

                # Read the doy or date list
                if (False in [len(self._doys_backup_[Landsatdc_pos[0]]) == len(self._doys_backup_[pos_temp]) for pos_temp in Landsatdc_pos]
                        or False in [(self._doys_backup_[pos_temp] == self._doys_backup_[Landsatdc_pos[0]]) for pos_temp in Landsatdc_pos]):
                    if auto_harmonised:
                        harmonised_factor = True
                    else:
                        raise Exception('The datacubes is not consistent in the date dimension! Turn auto harmonised factor as True if wanna avoid this problem!')

                # Harmonised the dcs
                if harmonised_factor:
                    self._auto_harmonised_dcs(Landsat_dc)

            # Define the output_path
            if work_env is None:
                self._Landsatdc_work_env = Path(os.path.dirname(os.path.dirname(self._dcs_backup_[Landsatdc_pos[0]].dc_filepath))).path_name
            else:
                self._Landsatdc_work_env = work_env

            # Determine the Zsize
            self.Landsatdc_doy_list = self._doys_backup_[Landsatdc_pos[0]]
            self.Landsatdc_ZSize = len(self.Landsatdc_doy_list)

        # Check the consistency of Denv dcs
        if self._withDenvdc_:
            Denvdc_pos = [i for i, v in enumerate(self._dc_typelist) if v == Denv_dc]
            if len(Denvdc_pos) != 1:
                for factor_temp in ['Datatype', 'coordinate_system', 'sparse_matrix', 'huge_matrix', 'sdc_factor',
                                    'dc_group_list', 'tiles']:
                    if False in [
                        self.__dict__[f'_{factor_temp}_list'][Denvdc_pos[0]] == self.__dict__[f'_{factor_temp}_list'][
                            pos] for pos in Denvdc_pos]:
                        raise ValueError(f'Please make sure the {factor_temp} for all the dcs were consistent!')

            # Construct Denv doylist
            self.Denv_doy_list = [None for _ in range(len(self._doys_backup_))]
            for _ in Denvdc_pos:
                self.Denv_doy_list[_] = self._doys_backup_[_]

            # Define the output_path
            if work_env is None:
                self._denv_work_env = Path(os.path.dirname(os.path.dirname(self._dcs_backup_[Denvdc_pos[0]].Denv_dc_filepath))).path_name
            else:
                self._denv_work_env = work_env

            # Determine the denv index
            self.Denv_indexlist = list(set([self._index_list[_] for _ in range(len(self._index_list)) if self._dc_typelist[_] == Denv_dc]))

            # Determine the timerange list
            self._denvyear_list = list(set([self._timerange_list[_] for _ in range(len(self._index_list)) if self._dc_typelist[_] == Denv_dc and self._timescale_list[_] == 'year']))

        # Check the consistency of Phemetric dcs
        if self._withPhemetricdc_:
            Phemetricdc_pos = [i for i, v in enumerate(self._dc_typelist) if v == Phemetric_dc]
            if len(Phemetricdc_pos) != 1:
                for factor_temp in ['ROI', 'ROI_name', 'ROI_array', 'Datatype', 'ROI_tif', 'coordinate_system',
                                    'sparse_matrix', 'huge_matrix', 'dc_group_list', 'tiles']:
                    if False in [self.__dict__[f'_{factor_temp}_list'][Phemetricdc_pos[0]] ==
                                 self.__dict__[f'_{factor_temp}_list'][pos] for pos in Phemetricdc_pos]:
                        raise ValueError(f'Please make sure the {factor_temp} for all the dcs were consistent!')

            # Define the output_path
            if work_env is None:
                self._pheme_work_env = Path(os.path.dirname(
                    os.path.dirname(self._dcs_backup_[Phemetricdc_pos[0]].Phemetric_dc_filepath))).path_name
            else:
                self._pheme_work_env = work_env

            # Determine the Zsize
            if z_Phemetric_size != 0:
                self.Phedc_ZSize = z_Phemetric_size
            else:
                raise Exception('Error occurred when obtaining the Z size of pheme dc!')

            # Construct phemetric namelist
            self._phemetric_namelist = []
            for _ in Phemetricdc_pos:
                self._phemetric_namelist.extend(
                    [temp.split(str(self._pheyear_list[_]) + '_')[-1] for temp in self._doys_backup_[_]])
            self._phemetric_namelist = list(set(self._phemetric_namelist))

            pheyear = []
            for _ in self._pheyear_list:
                if _ not in pheyear and _ is not None:
                    pheyear.append(_)
                elif _ is not None:
                    raise ValueError('There are duplicate pheyears for different pheme dcs!')

        # Check the consistency of Inunfactor dcs
        if self._withInunfacdc_:
            Inunfacdc_pos = [i for i, v in enumerate(self._dc_typelist) if v == Inunfac_dc]
            if len(Inunfacdc_pos) != 1:
                for factor_temp in ['ROI', 'ROI_name', 'ROI_array', 'Datatype', 'ROI_tif', 'coordinate_system',
                                    'sparse_matrix', 'huge_matrix', 'dc_group_list', 'tiles']:
                    if False in [self.__dict__[f'_{factor_temp}_list'][Inunfacdc_pos[0]] ==
                                 self.__dict__[f'_{factor_temp}_list'][pos] for pos in Inunfacdc_pos]:
                        raise ValueError(f'Please make sure the {factor_temp} for all the dcs were consistent!')

            # Define the output_path
            if work_env is None:
                self._inunfac_work_env = Path(os.path.dirname(
                    os.path.dirname(self._dcs_backup_[Inunfacdc_pos[0]].Inunfac_dc_filepath))).path_name
            else:
                self._inunfac_work_env = work_env

            # Determine the Zsize
            if z_Inunfac_size != 0:
                self.Inundc_ZSize = z_Inunfac_size
            else:
                raise Exception('Error occurred when obtaining the Z size of Inunfac dc!')

            # Construct Inunfac namelist
            self._inunfac_namelist = []
            for _ in Inunfacdc_pos:
                self._inunfac_namelist.extend([temp for temp in self._doys_backup_[_]])
            self._inunfac_namelist = list(set(self._inunfac_namelist))

            inunyear = []
            for _ in self._inunyear_list:
                if _ not in inunyear and _ is not None:
                    inunyear.append(_)
                elif _ is not None:
                    raise ValueError('There are duplicate inunyears for different inun dcs!')

        # Check consistency between different types of dcs (ROI/Time range)
        # Check the ROI consistency
        if [self._withS2dc_, self._withDenvdc_, self._withPhemetricdc_, self._withLandsatdc_, self._withInunfacdc_].count(True) > 1:
            if False in [temp == self._ROI_list[0] for temp in self._ROI_list] or [temp == self._ROI_name_list[0] for temp in self._ROI_name_list]:
                bounds_list = [bf.raster_ds2bounds(temp) for temp in self._ROI_tif_list]
                crs_list = [gdal.Open(temp).GetProjection() for temp in self._ROI_tif_list]
                if False in [temp == bounds_list[0] for temp in bounds_list] or False in [temp == crs_list[0] for temp in crs_list]:
                    try:
                        array_list = [np.sum(np.load(temp)) for temp in self._ROI_array_list]
                    except MemoryError:
                        array_list = [len(sm.csr_matrix(np.load(temp)).data) for temp in self._ROI_array_list]

                    if False in [temp == array_list[0] for temp in array_list]:
                        raise Exception('The ROIs between different types of datacube were not consistent')
                    else:
                        if self._withS2dc_:
                            self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[s2dc_pos[0]], \
                            self._ROI_name_list[s2dc_pos[0]], self._ROI_tif_list[s2dc_pos[0]], self._ROI_array_list[
                                s2dc_pos[0]]
                        elif self._withLandsatdc_:
                            self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[Landsatdc_pos[0]], \
                            self._ROI_name_list[Landsatdc_pos[0]], self._ROI_tif_list[Landsatdc_pos[0]], \
                            self._ROI_array_list[Landsatdc_pos[0]]
                        elif self._withPhemetricdc_:
                            self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[Denvdc_pos[0]], \
                            self._ROI_name_list[Denvdc_pos[0]], self._ROI_tif_list[Denvdc_pos[0]], self._ROI_array_list[
                                Denvdc_pos[0]]
                        elif self._withDenvdc_:
                            self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[Denvdc_pos[0]], \
                            self._ROI_name_list[Denvdc_pos[0]], self._ROI_tif_list[Denvdc_pos[0]], self._ROI_array_list[
                                Denvdc_pos[0]]
                        elif self._withInunfacdc_:
                            self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[Inunfacdc_pos[0]], \
                            self._ROI_name_list[Inunfacdc_pos[0]], self._ROI_tif_list[Inunfacdc_pos[0]], \
                            self._ROI_array_list[Inunfacdc_pos[0]]
                else:
                    if self._withS2dc_:
                        self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[s2dc_pos[0]], \
                        self._ROI_name_list[s2dc_pos[0]], self._ROI_tif_list[s2dc_pos[0]], self._ROI_array_list[
                            s2dc_pos[0]]
                    elif self._withLandsatdc_:
                        self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[Landsatdc_pos[0]], \
                        self._ROI_name_list[Landsatdc_pos[0]], self._ROI_tif_list[Landsatdc_pos[0]], \
                        self._ROI_array_list[Landsatdc_pos[0]]
                    elif self._withPhemetricdc_:
                        self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[Phemetricdc_pos[0]], \
                        self._ROI_name_list[Phemetricdc_pos[0]], self._ROI_tif_list[Phemetricdc_pos[0]], self._ROI_array_list[Phemetricdc_pos[0]]
                    elif self._withInunfacdc_:
                        self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[Inunfacdc_pos[0]], \
                        self._ROI_name_list[Inunfacdc_pos[0]], self._ROI_tif_list[Inunfacdc_pos[0]], self._ROI_array_list[Inunfacdc_pos[0]]
            else:
                self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[0], self._ROI_name_list[0], \
                self._ROI_tif_list[0], self._ROI_array_list[0]

            if False in [temp == self._coordinate_system_list[0] for temp in self._coordinate_system_list]:
                raise ValueError('Please make sure all the datacube was under the same coordinate')
            else:
                self.coordinate_system = self._coordinate_system_list[0]
        else:
            self.ROI, self.ROI_name, self.ROI_tif, self.ROI_array = self._ROI_list[0], self._ROI_name_list[0], \
            self._ROI_tif_list[0], self._ROI_array_list[0]
            self.coordinate_system = self._coordinate_system_list[0]

        # Construct the datacube list
        for _ in self._dcs_backup_:
            self.dcs.append(copy.copy(_.dc))
            if self._space_optimised is True:
                _.dc = None

        # Define var for the flood mapping
        self.inun_det_method_dic = {}
        self._variance_num = 2
        self._inundation_overwritten_factor = False
        self._DEM_path = None
        self._DT_std_fig_construction = False
        self._append_inundated_dc = True
        self._flood_mapping_accuracy_evaluation_factor = False
        self._sample_rs_link_list = None
        self._sample_data_path = None
        self._flood_mapping_method = ['Unet', 'static_wi_thr', 'DSWE', 'DT', 'AWEI', 'rs_dem']

        # Define var for the phenological analysis
        self._curve_fitting_algorithm = None
        self._flood_removal_method = None
        self._curve_fitting_dic = {}
        self._curfit_result = None

        # Define var for NIPY reconstruction
        self._add_NIPY_dc = True
        self._NIPY_overwritten_factor = False

        # Define var for phenology metrics generation
        self._curve_fitting_dic = {}
        self._all_quantify_str = None

        # Define var for flood_free_phenology_metrics
        self._flood_free_pm = ['annual_max_VI', 'average_VI_between_max_and_flood']

        # Define the mode for process denv via pheme
        self._denv8pheme_cal_method_list = ['acc', 'ave', 'max']
        self._denv8pheme_ = ['SOY', 'SOS', 'peak_doy', 'senescence_doy', 'EOS', 'EOY']  # represent the start of the year, start of the season, peak of the season
        self._denv8pheme_minus_base = False

    def __sizeof__(self):
        size = 0
        for dc in self.dcs:
            size += dc.__sizeof__()
        return size

    def _auto_harmonised_dcs(self, dc_type):

        if dc_type not in (Landsat_dc, Sentinel2_dc):
            raise TypeError('The datatype cannot be auto harmonised!')

        doy_all = np.array([])
        for _ in range(len(self._dc_typelist)):
            if self._dc_typelist[_] == dc_type:
                doy_all = np.concatenate([doy_all, self._doys_backup_[_]], axis=0)
        doy_all = np.sort(np.unique(doy_all))

        i = 0
        while i < len(self._dc_typelist):
            if self._dc_typelist[i] == dc_type and self._dcs_backup_[i].dc_ZSize != doy_all.shape[0]:
                for doy in doy_all:
                    if doy not in self._doys_backup_[i]:
                        if not self.sparse_matrix:
                            self.dcs[i] = np.insert(self._dcs_backup_[i], np.argwhere(doy_all == doy).flatten()[0], np.nan * np.zeros([self.dcs_YSize, self.dcs_XSize, 1]), axis=2)
                        else:
                            self.dcs[i].append(self._dcs_backup_[i]._matrix_type(np.zeros([self.dcs_YSize, self.dcs_XSize])), name=int(doy), pos=np.argwhere(doy_all == doy).flatten()[0])
            i += 1

        if False in [doy_all.shape[0] == self._dcs_backup_[i].shape[2] for i in range(len(self._dc_typelist)) if self._dc_typelist[i] == dc_type]:
            raise ValueError('The auto harmonised is failed')

        if dc_type == Sentinel2_dc:
            self.S2dc_ZSize = doy_all.shape[0]
            self.s2dc_doy_list = doy_all.tolist()
        elif dc_type == Landsat_dc:
            self.Landsatdc_ZSize = doy_all.shape[0]
            self.Landsatdc_doy_list = doy_all.tolist()

        for _ in range(len(self._doys_backup_)):
            if self._dc_typelist[_] == dc_type:
                self._doys_backup_[_] = doy_all.tolist()
                self._dcs_backup_.sdc_doy_list = doy_all.tolist()
                self._dcs_backup_.dc_ZSize = doy_all.shape[0]

    def append(self, dc_temp) -> None:
        if not isinstance(dc_temp, (Sentinel2_dc, Landsat_dc, Denv_dc, Phemetric_dc)):
            raise TypeError('The appended data should be a Sentinel2_dc, Landsat_dc, Denv_dc or Phemetric_dc!')

        if self._space_optimised:
            for _ in range(len(self.dcs)):
                self._dcs_backup_[_].dc = copy.copy(self.dcs[_])
                self.dcs[_] = None

        self._dcs_backup_.append(dc_temp)
        self.__init__(*self._dcs_backup_, auto_harmonised=True, space_optimised=self._space_optimised)

    def remove(self, index):
        if self._space_optimised:
            for _ in range(len(self.dcs)):
                self._dcs_backup_[_].dc = copy.copy(self.dcs[_])
                self.dcs[_] = None

        if index not in self._index_list:
            raise ValueError(f'The remove index {str(index)} is not in the dcs')
        else:
            self._dcs_backup_.remove(self._dcs_backup_[self._index_list.index(index)])

        self.__init__(*self._dcs_backup_, auto_harmonised=True, space_optimised=self._space_optimised)

    def extend(self, dcs_temp: list) -> None:
        for dc_temp in dcs_temp:
            if not isinstance(dc_temp, (Sentinel2_dc, Landsat_dc, Denv_dc, Phemetric_dc)):
                raise TypeError('The appended data should be a Sentinel2_dc, Landsat_dc, Denv_dc or Phemetric_dc!')

        if self._space_optimised:
            for _ in range(len(self.dcs)):
                self._dcs_backup_[_].dc = copy.copy(self.dcs[_])
                self.dcs[_] = None

        self._dcs_backup_.extend(dcs_temp)
        self.__init__(*self._dcs_backup_, auto_harmonised=True, space_optimised=self._space_optimised)

    def _process_inundation_para(self, **kwargs: dict) -> None:

        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ('DT_bimodal_histogram_factor', 'DSWE_threshold', 'flood_month_list', 'DEM_path',
                                       'overwritten_para', 'append_inundated_dc', 'DT_std_fig_construction',
                                       'variance_num',
                                       'inundation_mapping_accuracy_evaluation_factor', 'sample_rs_link_list',
                                       'construct_inundated_dc',
                                       'sample_data_path', 'static_wi_threshold',
                                       'flood_mapping_accuracy_evaluation_factor'
                                       ):
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # Detect the append_inundated_dc
        if 'append_inundated_dc' in kwargs.keys():
            if type(kwargs['append_inundated_dc']) != bool:
                raise TypeError('Please input the append_inundated_dc as a bool type!')
            else:
                self._append_inundated_dc = kwargs['append_inundated_dc']
        else:
            self._append_inundated_dc = False

        # Detect the construct_inundated_dc
        if 'construct_inundated_dc' in kwargs.keys():
            if type(kwargs['construct_inundated_dc']) != bool:
                raise TypeError('Please input the construct_inundated_dc as a bool type!')
            else:
                self._construct_inundated_dc = (kwargs['construct_inundated_dc'])
        else:
            self._construct_inundated_dc = False

        # Detect the static water index threshold
        if 'static_wi_threshold' not in kwargs.keys():
            self._static_wi_threshold = 0.1
        elif 'static_wi_threshold' in kwargs.keys():
            if type(kwargs['static_wi_threshold']) != float:
                raise TypeError('Please input the static_wi_threshold as a float number!')
            else:
                self._static_wi_threshold = kwargs['static_wi_threshold']

        # Detect the DSWE_threshold
        if 'DSWE_threshold' not in kwargs.keys():
            self._DSWE_threshold = [0.123, -0.5, 0.2, 0.1]
        elif 'DSWE_threshold' in kwargs.keys():
            if type(kwargs['DSWE_threshold']) != list:
                raise TypeError('Please input the DSWE threshold as a list with four number in it')
            elif len(kwargs['DSWE_threshold']) != 4:
                raise TypeError('Please input the DSWE threshold as a list with four number in it')
            else:
                self._DSWE_threshold = kwargs['DSWE_threshold']

        # Detect the variance num
        if 'variance_num' in kwargs.keys():
            if type(kwargs['variance_num']) is not float or type(kwargs['variance_num']) is not int:
                raise Exception('Please input the variance_num as a num!')
            elif kwargs['variance_num'] < 0:
                raise Exception('Please input the variance_num as a positive number!')
            else:
                self._variance_num = kwargs['variance_num']
        else:
            self._variance_num = 2

        # Detect the flood month para
        all_month_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        if 'flood_month_list' not in kwargs.keys():
            self._flood_month_list = ['7', '8', '9', '10']
        elif 'flood_month_list' in kwargs.keys():
            if type(kwargs['flood_month_list'] is not list):
                raise TypeError('Please make sure the para flood month list is a list')
            elif False in [_ in all_month_list for _ in kwargs['flood_month_list']]:
                raise ValueError('Please double check the month list')
            else:
                self._flood_month_list = kwargs['flood_month_list']
        else:
            self._flood_month_list = None

        # Define the inundation_overwritten_factor
        if 'overwritten_para' in kwargs.keys():
            if type(kwargs['overwritten_para']) is not bool:
                raise Exception('Please input the overwritten_para as a bool factor!')
            else:
                self._inundation_overwritten_para = kwargs['overwritten_para']
        else:
            self._inundation_overwritten_para = False

        # Define the flood_mapping_accuracy_evaluation_factor
        if 'flood_mapping_accuracy_evaluation_factor' in kwargs.keys():
            if type(kwargs['flood_mapping_accuracy_evaluation_factor']) is not bool:
                raise Exception('Please input the flood_mapping_accuracy_evaluation_factor as a bool factor!')
            else:
                self._flood_mapping_accuracy_evaluation_factor = kwargs['flood_mapping_accuracy_evaluation_factor']
        else:
            self._flood_mapping_accuracy_evaluation_factor = False

        # Define the DT_bimodal_histogram_factor
        if 'DT_bimodal_histogram_factor' in kwargs.keys():
            if type(kwargs['DT_bimodal_histogram_factor']) is not bool:
                raise Exception('Please input the DT_bimodal_histogram_factor as a bool factor!')
            else:
                self._DT_bimodal_histogram_factor = kwargs['DT_bimodal_histogram_factor']
        else:
            self._DT_bimodal_histogram_factor = True

        # Define the DT_std_fig_construction
        if 'DT_std_fig_construction' in kwargs.keys():
            if type(kwargs['DT_std_fig_construction']) is not bool:
                raise Exception('Please input the DT_std_fig_construction as a bool factor!')
            else:
                self._DT_std_fig_construction = kwargs['DT_std_fig_construction']
        else:
            self._DT_std_fig_construction = False

        # Define the sample_data_path
        if 'sample_data_path' in kwargs.keys():

            if type(kwargs['sample_data_path']) is not str:
                raise TypeError('Please input the sample_data_path as a dir!')

            if not os.path.exists(kwargs['sample_data_path']):
                raise TypeError('Please input the sample_data_path as a dir!')

            if not os.path.exists(kwargs['sample_data_path'] + self.ROI_name + '\\'):
                raise Exception('Please input the correct sample path or missing the ' + self.ROI_name + ' sample data')

            self._sample_data_path = kwargs['sample_data_path']

        else:
            self._sample_data_path = None

        # Define the sample_rs_link_list
        if 'sample_rs_link_list' in kwargs.keys():
            if type(kwargs['sample_rs_link_list']) is not list and type(
                    kwargs['sample_rs_link_list']) is not np.ndarray:
                raise TypeError('Please input the sample_rs_link_list as a list factor!')
            else:
                self._sample_rs_link_list = kwargs['sample_rs_link_list']
        else:
            self._sample_rs_link_list = False

        # Get the dem path
        if 'DEM_path' in kwargs.keys():
            if os.path.isfile(kwargs['DEM_path']) and (
                    kwargs['DEM_path'].endswith('.tif') or kwargs['DEM_path'].endswith('.TIF')):
                self._DEM_path = kwargs['DEM_path']
            else:
                raise TypeError('Please input a valid dem tiffile')
        else:
            self._DEM_path = None

    def custom_composition(self, index, **kwargs):

        # process inundation detection method
        self._process_inundation_para(**kwargs)

        # proces args*
        if index not in self._index_list:
            raise ValueError(f'The {index} is not imported')

        # if dc_type == 'Sentinel2':
        #     dc_type = Sentinel2_dc
        #     doy_list = self.s2dc_doy_list
        #     output_path = self._s2dc_work_env
        # elif dc_type == 'Landsat':
        #     dc_type = Landsat_dc
        #     doy_list = self.Landsatdc_doy_list
        #     output_path = self._Landsatdc_work_env
        # else:
        #     raise ValueError('Only Sentinel-2 and Landsat dc is supported for inundation detection!')

        dc_num = []
        for _ in range(len(self._index_list)):
            if self._index_list[_] == index and self._dc_typelist[_] == dc_type:
                dc_num.append(_)

        if len(dc_num) > 1:
            raise ValueError(f'There are more than one {str(dc_type)} dc of {str(dc_type)}')
        elif len(dc_num) == 0:
            raise ValueError(f'The {str(dc_type)} dc of {str(dc_type)} has not been imported')
        else:
            dc_num = dc_num[0]

    def inundation_detection(self, inundation_mapping_method: str, index: str, dc_type, **kwargs):

        # Several method to identify the inundation area including:
        # Something need to be mentioned:
        # (1) To fit the sparse matrix, 0 in the output means nodata, 1 means non-inundated and 2 means inundated
        # (2) All the dc was under the uint8 data type

        # process inundation detection method
        self._process_inundation_para(**kwargs)

        # proces args*
        if index not in self._index_list:
            raise ValueError(f'The {index} is not imported')

        if dc_type == 'Sentinel2':
            dc_type = Sentinel2_dc
            doy_list = self.s2dc_doy_list
            output_path = self._s2dc_work_env
        elif dc_type == 'Landsat':
            dc_type = Landsat_dc
            doy_list = self.Landsatdc_doy_list
            output_path = self._Landsatdc_work_env
        else:
            raise ValueError('Only Sentinel-2 and Landsat dc is supported for inundation detection!')

        dc_num = []
        for _ in range(len(self._index_list)):
            if self._index_list[_] == index and self._dc_typelist[_] == dc_type:
                dc_num.append(_)

        if len(dc_num) > 1:
            raise ValueError(f'There are more than one {str(dc_type)} dc of {str(dc_type)}')
        elif len(dc_num) == 0:
            raise ValueError(f'The {str(dc_type)} dc of {str(dc_type)} has not been imported')
        else:
            dc_num = dc_num[0]

        # Start the detection
        start_time = time.time()
        print(f'Start detecting the inundation area in the \033[1;34m{self.ROI_name}\033[0m using \033[1;35m{inundation_mapping_method}\033[0m')
        inundation_dc, oa_output_path = None, None

        if inundation_mapping_method not in self._flood_mapping_method:
            raise ValueError(f'The inundation detection method {str(inundation_mapping_method)} is not supported')

        # Method 1 static threshold
        if inundation_mapping_method == 'static_wi_thr':

            # Flood mapping by static threshold
            if 'Inundation_static_wi_thr' not in self._index_list or self._inundation_overwritten_para:

                # Define static thr output
                static_output = output_path + 'inundation_static_wi_thr_datacube\\'
                static_inditif_path = static_output + 'Individual_tif\\'
                bf.create_folder(static_output)
                bf.create_folder(static_inditif_path)

                if not os.path.exists(static_output + 'metadata.json'):

                    inundation_dc = copy.deepcopy(self.dcs[dc_num])
                    if self._sparse_matrix_list[dc_num]:
                        inundation_array = NDSparseMatrix()

                        # Separate into several stacks
                        inundation_dc_list = []
                        worker = int(os.cpu_count() / 4)
                        range_ = int(np.floor(inundation_dc.shape[2] / worker))
                        for _ in range(worker):
                            if (_ + 1) * range_ >= inundation_dc.shape[2]:
                                inundation_dc_list.append(inundation_dc.extract_matrix(
                                    (['all'], ['all'], [_ * range_, inundation_dc.shape[2]])))
                            else:
                                inundation_dc_list.append(
                                    inundation_dc.extract_matrix((['all'], ['all'], [_ * range_, (_ + 1) * range_])))

                        inundation_dc = None
                        sz, zoff, nd, thr = self._size_control_factor_list[dc_num], self._Zoffset_list[dc_num], self._Nodata_value_list[dc_num], self._static_wi_threshold
                        with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
                            res = executor.map(mp_static_wi_detection, inundation_dc_list, repeat(sz), repeat(zoff),  repeat(nd), repeat(thr))

                        res = list(res)
                        for _ in res:
                            for __ in range(len(_[0])):
                                inundation_array.append(_[0][__], name=_[1][__])

                    else:
                        inundation_arr_list = []
                        for doy_ in doy_list:
                            st = time.time()
                            mndwi_arr = self.dcs[dc_num][:, :, doy_list.index(doy_)].reshape([self.dcs[dc_num].shape[0], self.dcs[dc_num].shape[1]])
                            inundation_array = invert_data(mndwi_arr, self._size_control_factor_list[dc_num],
                                                           self._Zoffset_list[dc_num], self._Nodata_value_list[dc_num])
                            inundation_array = inundation_array >= self._static_wi_threshold
                            inundation_array = inundation_array + 1
                            inundation_array[np.isnan(inundation_array)] = 0
                            inundation_array = inundation_array.astype(np.byte)
                            inundation_arr_list.append(inundation_array)
                            bf.write_raster(gdal.Open(self.ROI_tif), inundation_array, static_inditif_path, f'Static_{str(doy_)}.TIF', raster_datatype=gdal.GDT_Byte, nodatavalue=0)
                            print(f'Identify the inundation area in {str(doy_)} using {str(time.time()-st)[0:6]}s')

                        inundation_array = np.stack(inundation_arr_list, axis=2)

                    inundation_dc = copy.deepcopy(self._dcs_backup_[dc_num])
                    inundation_dc.dc = inundation_array
                    inundation_dc.index = 'inundation_' + inundation_mapping_method
                    inundation_dc.Datatype = str(np.byte)
                    inundation_dc.sdc_doylist = doy_list
                    inundation_dc.Zoffset = None
                    inundation_dc.size_control_factor = False
                    inundation_dc.Nodata_value = 0
                    inundation_dc.save(static_output)

                    if self._append_inundated_dc:
                        self.append(inundation_dc)
                        self.remove(self._index_list[dc_num])

                else:
                    inundation_dc = dc_type(static_output)

                oa_output_path = copy.deepcopy(static_output)

        # Method 2 Dynamic threshold
        elif inundation_mapping_method == 'DT':

            # Flood mapping by DT method (DYNAMIC MNDWI THRESHOLD using time-series water index!)
            if 'Inundation_DT' not in self._index_list or self._inundation_overwritten_para:

                # Define output path
                DT_output_path = output_path + 'Inundation_DT_datacube\\'
                DT_inditif_path = DT_output_path + 'Individual_tif\\'
                DT_threshold_path = DT_output_path + 'DT_threshold\\'
                oa_output_path = DT_output_path
                bf.create_folder(DT_output_path)
                bf.create_folder(DT_inditif_path)
                bf.create_folder(DT_threshold_path)

                if not os.path.exists(DT_threshold_path + 'threshold_map.TIF') or not os.path.exists(DT_threshold_path + 'bh_threshold_map.TIF') or self._inundation_overwritten_para:

                    # Define input
                    WI_sdc = copy.deepcopy(self.dcs[dc_num])
                    doy_array = copy.copy(self._doys_backup_[dc_num])
                    doy_array = bf.date2doy(doy_array)
                    doy_array = np.array(doy_array)
                    nodata_value = self._Nodata_value_list[dc_num]

                    roi_arr = np.load(self.ROI_array)
                    roi_coord = np.argwhere(roi_arr == 1).tolist()

                    sz_ctrl_fac = self._size_control_factor_list[dc_num]
                    zoffset = self._Zoffset_list[dc_num]
                    nd_v = self._Nodata_value_list[dc_num]
                    variance = self._variance_num
                    bio_factor = self._DT_bimodal_histogram_factor

                    # Define output
                    DT_threshold_arr = np.ones([WI_sdc.shape[0], WI_sdc.shape[1]]) * np.nan
                    bh_threshold_arr = np.ones([WI_sdc.shape[0], WI_sdc.shape[1]]) * np.nan

                    # DT method with empirical and bimodal threshold
                    dc_list, pos_list, yxoffset_list = slice_datacube(WI_sdc, roi_coord)
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        res = executor.map(create_bimodal_histogram_threshold, dc_list, pos_list, yxoffset_list,
                                           repeat(doy_array), repeat(sz_ctrl_fac), repeat(zoffset), repeat(nd_v),
                                           repeat(variance), repeat(bio_factor))

                    res = list(res)
                    res_concat = []
                    for res_temp in res:
                        res_concat.extend(res_temp)

                    for r_ in range(len(res_concat)):
                        DT_threshold_arr[res_concat[r_][0], res_concat[r_][1]] = res_concat[r_][2]
                        bh_threshold_arr[res_concat[r_][0], res_concat[r_][1]] = res_concat[r_][3]

                    bf.write_raster(gdal.Open(self.ROI_tif), DT_threshold_arr, DT_threshold_path, 'threshold_map.TIF',
                                    raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                    bf.write_raster(gdal.Open(self.ROI_tif), bh_threshold_arr, DT_threshold_path,
                                    'bh_threshold_map.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
                    WI_sdc = None
                    doy_array = None
                else:
                    bh_threshold_ds = gdal.Open(DT_threshold_path + 'bh_threshold_map.TIF')
                    threshold_ds = gdal.Open(DT_threshold_path + 'threshold_map.TIF')
                    DT_threshold_arr = threshold_ds.GetRasterBand(1).ReadAsArray()
                    bh_threshold_arr = bh_threshold_ds.GetRasterBand(1).ReadAsArray()

                # Construct the MNDWI distribution figure
                if self._DT_std_fig_construction:

                    DT_distribution_fig_path = DT_output_path + 'DT_distribution_fig\\'
                    bf.create_folder(DT_distribution_fig_path)

                    # Generate the MNDWI distribution at the pixel level
                    for y_temp in range(WI_sdc.shape[0]):
                        for x_temp in range(WI_sdc.shape[1]):
                            if not os.path.exists(
                                    f'{DT_distribution_fig_path}DT_distribution_X{str(x_temp)}_Y{str(y_temp)}.png'):

                                # Extract the wi series
                                if self._sparse_matrix_list[dc_num]:
                                    ___, wi_series, __ = WI_sdc._extract_matrix_y1x1zh(([y_temp], [x_temp], ['all']),
                                                                                       nodata_export=True)
                                else:
                                    wi_series = WI_sdc[y_temp, x_temp, :]
                                wi_series = invert_data(wi_series, self._size_control_factor_list[dc_num],
                                                        self._Zoffset_list[dc_num], nodata_value)
                                wi_series = wi_series.flatten()

                                doy_array_pixel = np.mod(doy_array, 1000).flatten()
                                wi_ori_series = copy.copy(wi_series)
                                wi_series = np.delete(wi_series, np.argwhere(
                                    np.logical_and(doy_array_pixel >= 182, doy_array_pixel <= 285)))
                                wi_series = np.delete(wi_series, np.argwhere(np.isnan(wi_series))) if np.isnan(
                                    nodata_value) else np.delete(wi_series, np.argwhere(wi_series == nodata_value))
                                wi_series = np.delete(wi_series, np.argwhere(
                                    wi_series > DT_threshold_arr[y_temp, x_temp])) if not np.isnan(
                                    DT_threshold_arr[y_temp, x_temp]) else np.delete(wi_series,
                                                                                     np.argwhere(wi_series > 0.123))

                                if wi_series.shape[0] != 0:
                                    yy = np.arange(0, 100, 1)
                                    xx = np.ones([100])
                                    mndwi_temp_std = np.std(wi_series)
                                    mndwi_ave = np.mean(wi_series)
                                    plt.xlim(xmax=1, xmin=-1)
                                    plt.ylim(ymax=50, ymin=0)
                                    plt.hist(wi_ori_series, bins=50, color='#FFA500')
                                    plt.hist(wi_series, bins=20, color='#00FFA5')
                                    plt.plot(xx * mndwi_ave, yy, color='#FFFF00')
                                    plt.plot(xx * bh_threshold_arr[y_temp, x_temp], yy, color='#CD0000', linewidth='3')
                                    plt.plot(xx * DT_threshold_arr[y_temp, x_temp], yy, color='#0000CD',
                                             linewidth='1.5')
                                    plt.plot(xx * (mndwi_ave - mndwi_temp_std), yy, color='#00CD00')
                                    plt.plot(xx * (mndwi_ave + mndwi_temp_std), yy, color='#00CD00')
                                    plt.plot(xx * (mndwi_ave - self._variance_num * mndwi_temp_std), yy,
                                             color='#00CD00')
                                    plt.plot(xx * (mndwi_ave + self._variance_num * mndwi_temp_std), yy,
                                             color='#00CD00')
                                    plt.savefig(
                                        f'{DT_distribution_fig_path}DT_distribution_X{str(x_temp)}_Y{str(y_temp)}.png',
                                        dpi=150)
                                    plt.close()

                # Construct inundation dc
                if not os.path.exists(DT_output_path + 'doy.npy') or not os.path.exists(DT_output_path + 'metadata.json') or self._inundation_overwritten_para:

                    inundation_dc = copy.deepcopy(self._dcs_backup_[dc_num])
                    inundated_arr = copy.deepcopy(self.dcs[dc_num])
                    doy_array = copy.copy(self._doys_backup_[dc_num])
                    doy_array = bf.date2doy(doy_array)
                    doy_array = np.array(doy_array)
                    DT_threshold = DT_threshold_arr.astype(float)
                    num_list = [q for q in range(doy_array.shape[0])]
                    if self._sparse_matrix_list[dc_num]:
                        inundated_arr_list = [inundated_arr.SM_group[inundated_arr.SM_namelist[_]] for _ in
                                              range(inundated_arr.shape[2])]
                    else:
                        inundated_arr_list = [
                            inundated_arr[:, :, _].reshape([inundated_arr.shape[0], inundated_arr.shape[1]]) for _ in
                            range(inundated_arr.shape[2])]

                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        res = list(tq(executor.map(create_indi_DT_inundation_map,
                                                   inundated_arr_list, repeat(doy_array),
                                                   num_list, repeat(DT_threshold),
                                                   repeat(DT_inditif_path), repeat(self._inundation_overwritten_para),
                                                   repeat(self._size_control_factor_list[dc_num]),
                                                   repeat(self._Zoffset_list[dc_num]),
                                                   repeat(self._Nodata_value_list[dc_num]), repeat(self.ROI_tif)),
                                      total=len(num_list)))

                    for _ in res:
                        if self._sparse_matrix_list[dc_num]:
                            inundated_arr.SM_group[inundated_arr.SM_namelist[_[0]]] = _[1]
                        else:
                            inundated_arr[:, :, [_[0]]] = _[1]

                    # for date_num in range(doy_array.shape[0]):
                    #     if not os.path.exists(f'{DT_inditif_path}\\DT_{str(doy_array[date_num])}.TIF') or thalweg_temp._inundation_overwritten_factor:
                    #
                    #         if thalweg_temp._sparse_matrix_list[dc_num]:
                    #             WI_arr = inundated_arr.SM_group[inundated_arr.SM_namelist[date_num]]
                    #         else:
                    #             WI_arr = inundated_arr[:, :, date_num].reshape(inundated_arr.shape[0], inundated_arr.shape[1])
                    #
                    #         WI_arr = invert_data(WI_arr, thalweg_temp._size_control_factor_list[dc_num], thalweg_temp._Zoffset_list[dc_num], thalweg_temp._Nodata_value_list[dc_num])
                    #
                    #         inundation_map = WI_arr - DT_threshold
                    #         inundation_map[inundation_map >= 0] = 2
                    #         inundation_map[inundation_map < 0] = 1
                    #         inundation_map[np.isnan(inundation_map)] = 0
                    #         inundation_map[WI_arr > 0.16] = 2
                    #         inundation_map = reassign_sole_pixel(inundation_map, Nan_value=0, half_size_window=2)
                    #         inundation_map = inundation_map.astype(np.byte)
                    #
                    #         bf.write_raster(gdal.Open(thalweg_temp.ROI_tif), inundation_map, DT_inditif_path, f'DT_{str(doy_array[date_num])}.TIF', raster_datatype=gdal.GDT_Byte, nodatavalue=0)
                    #     else:
                    #         inundated_ds = gdal.Open(f'{DT_inditif_path}DT_{str(doy_array[date_num])}.TIF')
                    #         inundation_map = inundated_ds.GetRasterBand(1).ReadAsArray()

                    inundation_dc.dc = inundated_arr
                    inundation_dc.index = 'Inundation_' + inundation_mapping_method
                    inundation_dc.Datatype = str(np.byte)
                    inundation_dc.sdc_doylist = doy_list
                    inundation_dc.Zoffset = None
                    inundation_dc.size_control_factor = False
                    inundation_dc.Nodata_value = 0
                    inundation_dc.save(DT_output_path)
                else:
                    inundation_dc = dc_type(DT_output_path)

        print(f'Flood mapping using {str(inundation_mapping_method)} method within {self.ROI_name} consumes {str(time.time() - start_time)}s!')

        if inundation_dc is not None:

            # Create annual inundation map
            inditif_path = oa_output_path + 'Individual_tif\\'
            annualtif_path = oa_output_path + 'Annual_tif\\'
            annualshp_path = oa_output_path + 'Annual_shp\\'
            bf.create_folder(annualtif_path)
            bf.create_folder(annualshp_path)

            doy_array = bf.date2doy(np.array(inundation_dc.sdc_doylist))
            year_array = np.unique(doy_array // 1000)
            temp_ds = gdal.Open(bf.file_filter(inditif_path, ['.TIF'])[0])
            for year in year_array:
                if not os.path.exists(f'{annualtif_path}{inundation_mapping_method}_{str(year)}.TIF') or not os.path.exists(f'{annualshp_path}{inundation_mapping_method}_{str(year)}.shp') or self._inundation_overwritten_factor:
                    annual_vi_list = []
                    for doy_index in range(doy_array.shape[0]):
                        if doy_array[doy_index] // 1000 == year and 90 <= np.mod(doy_array[doy_index], 1000) <= 300:
                            if isinstance(inundation_dc.dc, NDSparseMatrix):
                                arr_temp = inundation_dc.dc.SM_group[inundation_dc.dc.SM_namelist[doy_index]]
                                annual_vi_list.append(arr_temp.toarray())
                            else:
                                annual_vi_list.append(inundation_dc.dc[:, :, doy_index])

                    if annual_vi_list == []:
                        annual_inundated_map = np.zeros([inundation_dc.dc.shape[0], inundation_dc.dc.shape[1]], dtype=np.byte)
                    else:
                        annual_inundated_map = np.nanmax(np.stack(annual_vi_list, axis=2), axis=2)
                    bf.write_raster(temp_ds, annual_inundated_map, annualtif_path, f'{inundation_mapping_method}_' + str(year) + '.TIF', raster_datatype=gdal.GDT_Byte, nodatavalue=0)
                    annual_ds = gdal.Open(annualtif_path + f'{inundation_mapping_method}_' + str(year) + '.TIF')

                    # Create shp driver
                    drv = ogr.GetDriverByName("ESRI Shapefile")
                    if os.path.exists(annualshp_path + f'{inundation_mapping_method}_{str(year)}.shp'):
                        drv.DeleteDataSource(annualshp_path + f'{inundation_mapping_method}_{str(year)}.shp')

                    # polygonize the raster
                    proj = osr.SpatialReference(wkt=annual_ds.GetProjection())
                    target = osr.SpatialReference()
                    target.ImportFromEPSG(int(proj.GetAttrValue('AUTHORITY', 1)))

                    dst_ds = drv.CreateDataSource(f'{annualshp_path}{inundation_mapping_method}_{str(year)}.shp')
                    dst_layer = dst_ds.CreateLayer(f'{inundation_mapping_method}_{str(year)}', srs=target)

                    fld = ogr.FieldDefn("inundation", ogr.OFTInteger)
                    dst_layer.CreateField(fld)
                    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("inundation")
                    gdal.Polygonize(annual_ds.GetRasterBand(1), None, dst_layer, dst_field, [])

                    layer = dst_ds.GetLayer()
                    new_field = ogr.FieldDefn("area", ogr.OFTReal)
                    new_field.SetWidth(32)
                    layer.CreateField(new_field)

                    # Fix the sole pixel
                    for feature in layer:
                        geom = feature.GetGeometryRef()
                        area = geom.GetArea()
                        if area < 5000 and feature.GetField('inundation') == 2:
                            feature.SetField("inundation", 1)

                        feature.SetField("area", area)
                        layer.SetFeature(feature)

                    layer.ResetReading()
                    for feature in layer:
                        if feature['inundation'] == 1:
                            if layer.DeleteFeature(feature.GetFID()) != 0:
                                print(f"Error: Failed to delete feature with FID {feature.GetFID()}.")
                            else:
                                print(f"Feature with FID {feature.GetFID()} deleted successfully.")

                    del dst_ds

            # Create inundation frequency map
            inunfactor_path = oa_output_path + 'inun_factor\\'
            bf.create_folder(inunfactor_path)

            if not os.path.exists(f'{inunfactor_path}{inundation_mapping_method}_inundation_frequency.TIF') or self._inundation_overwritten_factor:
                doy_array = bf.date2doy(np.array(inundation_dc.sdc_doylist))
                temp_ds = gdal.Open(bf.file_filter(inditif_path, ['.TIF'])[0])
                roi_temp = np.load(self.ROI_array)
                inun_arr, all_arr = np.zeros_like(roi_temp).astype(np.float32), np.zeros_like(roi_temp).astype(
                    np.float32)
                inun_arr[roi_temp == -32768] = np.nan
                all_arr[roi_temp == -32768] = np.nan

                for doy_index in range(doy_array.shape[0]):
                    if isinstance(inundation_dc.dc, NDSparseMatrix):
                        arr_temp = inundation_dc.dc.SM_group[inundation_dc.dc.SM_namelist[doy_index]]
                    else:
                        arr_temp = inundation_dc.dc[:, :, doy_index].reshape([inundation_dc.dc.shape[0], inundation_dc.dc.shape[1]])

                    inun_arr = inun_arr + (arr_temp == 2).astype(np.int16)
                    all_arr = all_arr + (arr_temp >= 1).astype(np.int16)

                inundation_freq = inun_arr.astype(np.float32) / all_arr.astype(np.float32)
                bf.write_raster(temp_ds, inundation_freq, inunfactor_path,
                                f'{inundation_mapping_method}_inundation_frequency.TIF',
                                raster_datatype=gdal.GDT_Float32, nodatavalue=0)

            if 'YZR' in self.ROI_name:
                if not os.path.exists(
                        f'{inunfactor_path}{inundation_mapping_method}_inundation_frequency_pretgd.TIF') or not os.path.exists(
                        f'{inunfactor_path}{inundation_mapping_method}_inundation_frequency_posttgd.TIF'):
                    doy_array = bf.date2doy(np.array(inundation_dc.sdc_doylist))
                    temp_ds = gdal.Open(bf.file_filter(inditif_path, ['.TIF'])[0])
                    roi_temp = np.load(self.ROI_array)
                    inun_arr_pre, all_arr_pre = np.zeros_like(roi_temp).astype(np.float32), np.zeros_like(roi_temp).astype(
                        np.float32)
                    inun_arr_post, all_arr_post = np.zeros_like(roi_temp).astype(np.float32), np.zeros_like(
                        roi_temp).astype(np.float32)
                    inun_arr_pre[roi_temp == -32768] = np.nan
                    all_arr_pre[roi_temp == -32768] = np.nan
                    inun_arr_post[roi_temp == -32768] = np.nan
                    all_arr_post[roi_temp == -32768] = np.nan

                    for doy_index in range(doy_array.shape[0]):
                        if isinstance(inundation_dc.dc, NDSparseMatrix):
                            arr_temp = inundation_dc.dc.SM_group[inundation_dc.dc.SM_namelist[doy_index]]
                        else:
                            arr_temp = inundation_dc.dc[:, :, doy_array.index(doy_index)]

                        if 1987001 <= doy_array[doy_index] < 2004000:
                            inun_arr_pre = inun_arr_pre + (arr_temp == 2).astype(np.int16)
                            all_arr_pre = all_arr_pre + (arr_temp >= 1).astype(np.int16)

                        if 2004001 <= doy_array[doy_index] < 2021001:
                            inun_arr_post = inun_arr_post + (arr_temp == 2).astype(np.int16)
                            all_arr_post = all_arr_post + (arr_temp >= 1).astype(np.int16)

                    inundation_freq_pretgd = inun_arr_pre.astype(np.float32) / all_arr_pre.astype(np.float32)
                    inundation_freq_posttgd = inun_arr_post.astype(np.float32) / all_arr_post.astype(np.float32)
                    bf.write_raster(temp_ds, inundation_freq_pretgd, inunfactor_path,
                                    f'{inundation_mapping_method}_inundation_frequency_pretgd.TIF',
                                    raster_datatype=gdal.GDT_Float32, nodatavalue=0)
                    bf.write_raster(temp_ds, inundation_freq_posttgd, inunfactor_path,
                                    f'{inundation_mapping_method}_inundation_frequency_posttgd.TIF',
                                    raster_datatype=gdal.GDT_Float32, nodatavalue=0)

            if not os.path.exists(
                    f'{inunfactor_path}{inundation_mapping_method}_inundation_recurrence.TIF') or self._inundation_overwritten_factor:

                yearly_tif = bf.file_filter(annualtif_path, ['.TIF'],
                                            exclude_word_list=['.xml', '.dpf', '.cpg', '.aux', 'vat', '.ovr'])
                roi_temp = np.load(self.ROI_array)
                recu_arr = np.zeros_like(roi_temp).astype(np.float32)
                recu_arr[roi_temp == -32768] = np.nan

                for tif_ in yearly_tif:
                    ds_ = gdal.Open(tif_)
                    arr_ = ds_.GetRasterBand(1).ReadAsArray()
                    recu_arr = recu_arr + (arr_ == 2).astype(np.int16)

                inundation_recu = recu_arr.astype(np.float32) / len(yearly_tif)
                bf.write_raster(temp_ds, inundation_recu, inunfactor_path,
                                f'{inundation_mapping_method}_inundation_recurrence.TIF',
                                raster_datatype=gdal.GDT_Float32, nodatavalue=0)

        print(f'Finish detecting the inundation area in the \033[1;34m{self.ROI_name}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0m s!')

    def _process_inundation_removal_para(self, **kwargs):
        pass

    def inundation_removal(self, processed_index: str, inundation_method: str, dc_type: str, append_new_dc=True,
                           **kwargs):

        print(f'Start remove the inundation area of the \033[1;34m{processed_index}\033[0m!')

        self._process_inundation_removal_para(**kwargs)
        if dc_type == 'Sentinel2':
            dc_type = Sentinel2_dc
            doy_list = self.s2dc_doy_list
            output_path = self._s2dc_work_env
            z_size = self.S2dc_ZSize
        elif dc_type == 'Landsat':
            dc_type = Landsat_dc
            doy_list = self.Landsatdc_doy_list
            output_path = self._Landsatdc_work_env
            z_size = self.Landsatdc_ZSize
        else:
            raise ValueError('Only Sentinel-2 and Landsat dc is supported for inundation detection!')

        # Retrieve inundation dc
        if inundation_method not in self._flood_mapping_method:
            raise ValueError(f'The inundation method {str(inundation_method)} is not supported!')
        inundation_index = 'Inundation_' + inundation_method
        inundation_dc_num = [_ for _ in range(len(self._index_list)) if
                             self._dc_typelist[_] == dc_type and self._index_list[_] == inundation_index]
        if inundation_dc_num == 0:
            raise ValueError('The inundated dc for inundation removal is not properly imported')
        else:
            inundation_dc = copy.deepcopy(self.dcs[inundation_dc_num[0]])

        # Retrieve processed dc
        processed_dc_num = [_ for _ in range(len(self._index_list)) if
                            self._dc_typelist[_] == dc_type and self._index_list[_] == processed_index]
        if processed_dc_num == 0:
            raise ValueError('The processed dc for inundation removal is not properly imported')
        else:
            processed_dc = copy.deepcopy(self.dcs[processed_dc_num[0]])
            processed_dc4save = copy.deepcopy(self._dcs_backup_[processed_dc_num[0]])

        if self._doys_backup_[inundation_dc_num[0]] != self._doys_backup_[processed_dc_num[0]]:
            raise TypeError('The inundation removal method must processed on two datacube with same doy list!')

        processed_dc_path = f'{str(output_path)}{str(processed_index)}_noninun_datacube\\'
        bf.create_folder(processed_dc_path)

        start_time = time.time()
        if not os.path.exists(processed_dc_path + 'metadata.json'):

            if self._sparse_matrix_list[processed_dc_num[0]]:
                for height in range(z_size):
                    inundation_arr = copy.deepcopy(inundation_dc.SM_group[inundation_dc.SM_namelist[height]])
                    inundation_arr[inundation_arr == 2] = 0
                    processed_dc.SM_group[processed_dc.SM_namelist[height]] = processed_dc.SM_group[processed_dc.SM_namelist[height]].multiply(inundation_arr)

                # if thalweg_temp._remove_nan_layer or thalweg_temp._manually_remove_para:
                #     i_temp = 0
                #     while i_temp < len(doy_list):
                #         if data_valid_array[i_temp]:
                #             if doy_list[i_temp] in data_cube.SM_namelist:
                #                 data_cube.remove_layer(doy_list[i_temp])
                #             doy_list.remove(doy_list[i_temp])
                #             data_valid_array = np.delete(data_valid_array, i_temp, 0)
                #             i_temp -= 1
                #         i_temp += 1

                processed_index = processed_index + '_noninun'
                processed_dc4save.index = processed_index
                processed_dc4save.dc = processed_dc
                processed_dc4save.save(output_path + processed_index + '_datacube\\')

            else:
                pass

        else:
            processed_dc4save = dc_type(output_path + processed_index + '_noninun_datacube\\')

        if append_new_dc:
            self.append(processed_dc4save)

        print(f'Finish remove the inundation area of the \033[1;34m{processed_index}\033[0m using \033[1;31m{str(time.time() - start_time)}\033[0m s!')

    def _process_curve_fitting_para(self, **kwargs):

        # Curve fitting method
        all_supported_curve_fitting_method = ['seven_para_logistic', 'two_term_fourier']

        if 'curve_fitting_algorithm' in kwargs.keys():
            self._curve_fitting_algorithm = kwargs['curve_fitting_algorithm']

        if self._curve_fitting_algorithm is None or self._curve_fitting_algorithm == 'seven_para_logistic':
            self._curve_fitting_dic['CFM'] = 'SPL'
            self._curve_fitting_dic['para_num'] = 7
            self._curve_fitting_dic['initial_para_ori'] = [0.3, 0.5, 108.2, 7.596, 280.4, 7.473, 0.00225]
            self._curve_fitting_dic['initial_para_boundary'] = (
                [0.08, 0, 40, 3, 180, 3, 0.0001], [0.6, 0.8, 180, 20, 330, 20, 0.01])
            self._curve_fitting_dic['para_ori'] = [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225]
            self._curve_fitting_dic['para_boundary'] = (
                [0.08, 0.0, 50, 6.2, 285, 4.5, 0.0015], [0.20, 0.8, 130, 11.5, 350, 8.8, 0.0028])
            self._curve_fitting_algorithm = seven_para_logistic_function
        elif self._curve_fitting_algorithm == 'two_term_fourier':
            self._curve_fitting_dic['CFM'] = 'TTF'
            self._curve_fitting_dic['para_num'] = 6
            self._curve_fitting_dic['para_ori'] = [0, 0, 0, 0, 0, 0.017]
            self._curve_fitting_dic['para_boundary'] = (
                [0, -0.5, -0.5, -0.05, -0.05, 0.015], [1, 0.5, 0.5, 0.05, 0.05, 0.019])
            self._curve_fitting_algorithm = two_term_fourier
        elif self._curve_fitting_algorithm not in all_supported_curve_fitting_method:
            ValueError(f'The curve fitting method {self._curve_fitting_algorithm} is not supported!')

        # Determine inundation removal method
        if 'flood_removal_method' in kwargs.keys():
            self._flood_removal_method = kwargs['flood_removal_method']

            if self._flood_removal_method not in self._flood_mapping_method:
                raise ValueError(f'The flood removal method {self._flood_removal_method} is not supported!')
        else:
            self._flood_removal_method = None

        # Overwritten_para
        if 'overwritten' in kwargs.keys():
            self._curve_fitting_algorithm = kwargs['curve_fitting_algorithm']

    def curve_fitting(self, index: str, dc_type, **kwargs):

        # Check vi availability
        if index not in self._index_list:
            raise ValueError('Please make sure the vi datacube is constructed!')

        if dc_type == 'Sentinel2':
            dc_type = Sentinel2_dc
            doy_list = self.s2dc_doy_list
            output_path = self._s2dc_work_env
        elif dc_type == 'Landsat':
            dc_type = Landsat_dc
            doy_list = self.Landsatdc_doy_list
            output_path = self._Landsatdc_work_env
        else:
            raise ValueError('Only Sentinel-2 and Landsat dc is supported for inundation detection!')

        # Process paras
        self._process_curve_fitting_para(**kwargs)

        # Get the index/doy dc
        dc_num = [_ for _ in range(len(self._index_list)) if
                  self._dc_typelist[_] == dc_type and self._index_list[_] == index]
        if dc_num == 0:
            raise TypeError('None imported datacube meets the requirement for curve fitting!')
        else:
            dc_num = dc_num[0]

        index_dc = copy.deepcopy(self.dcs[dc_num])
        doy_dc = copy.deepcopy(doy_list)
        doy_dc = bf.date2doy(doy_dc)
        doy_all = np.mod(doy_dc, 1000)
        size_control_fac = self._size_control_factor_list[dc_num]

        # Retrieve the ROI
        sa_map = np.load(self.ROI_array)

        # Create output path
        curfit_output_path = output_path + index + '_curfit_datacube\\'
        para_output_path = curfit_output_path + str(self._curve_fitting_dic['CFM']) + '_para\\'
        phemetric_output_path = curfit_output_path + f'\\{self.ROI_name}_Phemetric_datacube\\'
        csv_para_output_path = para_output_path + 'csv_file\\'
        tif_para_output_path = para_output_path + 'tif_file\\'
        bf.create_folder(curfit_output_path)
        bf.create_folder(para_output_path)
        bf.create_folder(phemetric_output_path)
        bf.create_folder(csv_para_output_path)
        bf.create_folder(tif_para_output_path)
        self._curve_fitting_dic[
            str(self.ROI) + '_' + str(index) + '_' + str(self._curve_fitting_dic['CFM']) + '_path'] = para_output_path

        # Define the cache folder
        cache_folder = f'{para_output_path}cache\\'
        bf.create_folder(cache_folder)

        # Read pos_df
        if not os.path.exists(f'{cache_folder}pos_df.csv'):
            pos_df = pd.DataFrame(np.argwhere(sa_map != -32768), columns=['y', 'x'])
            pos_df = pos_df.sort_values(['x', 'y'], ascending=[True, True])
            pos_df = pos_df.reset_index()
        else:
            pos_df = pd.read_csv(f'{cache_folder}pos_df.csv')

        # Generate all the curve fitting para into a table
        if not os.path.exists(csv_para_output_path + 'curfit_all.csv'):

            # Slice into several tasks/blocks to use all cores
            work_num = os.cpu_count()
            doy_all_list, pos_list, xy_offset_list, index_size_list, index_dc_list, indi_size = [], [], [], [], [], int(
                np.ceil(pos_df.shape[0] / work_num))
            for i_size in range(work_num):
                if i_size != work_num - 1:
                    pos_list.append(pos_df[indi_size * i_size: indi_size * (i_size + 1)])
                else:
                    pos_list.append(pos_df[indi_size * i_size: -1])

                index_size_list.append(
                    [int(max(0, pos_list[-1]['y'].min())), int(min(sa_map.shape[0], pos_list[-1]['y'].max())),
                     int(max(0, pos_list[-1]['x'].min())), int(min(sa_map.shape[1], pos_list[-1]['x'].max()))])
                xy_offset_list.append([int(max(0, pos_list[-1]['y'].min())), int(max(0, pos_list[-1]['x'].min()))])

                if self._sparse_matrix_list[dc_num]:
                    dc_temp = index_dc.extract_matrix(([index_size_list[-1][0], index_size_list[-1][1] + 1],
                                                       [index_size_list[-1][2], index_size_list[-1][3] + 1], ['all']))
                    index_dc_list.append(dc_temp.drop_nanlayer())
                    doy_all_list.append(bf.date2doy(index_dc_list[-1].SM_namelist))
                else:
                    index_dc_list.append(index_dc[index_size_list[-1][0]: index_size_list[-1][2],
                                         index_size_list[-1][1]: index_size_list[-1][3], :])
                    doy_all_list.append(doy_all)

            with concurrent.futures.ProcessPoolExecutor(max_workers=work_num) as executor:
                result = executor.map(curfit4bound_annual, pos_list, index_dc_list, doy_all_list,
                                      repeat(self._curve_fitting_dic), repeat(self._sparse_matrix_list[dc_num]),
                                      repeat(size_control_fac), xy_offset_list, repeat(cache_folder),
                                      repeat(self._Nodata_value_list[dc_num]), repeat(self._Zoffset_list[dc_num]))
            result_list = list(result)

            # Integrate all the result into the para dict
            self._curfit_result = None
            for result_temp in result_list:
                if self._curfit_result is None:
                    self._curfit_result = copy.copy(result_temp)
                else:
                    self._curfit_result = pd.concat([self._curfit_result, result_temp])

            key_list = []
            for key_temp in self._curfit_result.keys():
                if key_temp not in pos_df.keys() and 'para' in key_temp:
                    key_list.append(key_temp)

            self._curfit_result.to_csv(csv_para_output_path + 'curfit_all.csv')
        else:
            self._curfit_result = pd.read_csv(csv_para_output_path + 'curfit_all.csv')

        # Create output key list
        key_list = []
        df_list = []
        for key_temp in self._curfit_result.keys():
            if True not in [nr_key in key_temp for nr_key in
                            ['Unnamed', 'level', 'index', 'Rsquare']] and key_temp != 'y' and key_temp != 'x':
                key_list.append(key_temp)
                df_list.append(self._curfit_result.loc[:, ['y', 'x', key_temp]])

        # Create tif file based on phenological parameter
        with concurrent.futures.ProcessPoolExecutor() as executor:
            res = list(
                tq(executor.map(curfit_pd2tif, repeat(tif_para_output_path), df_list, key_list, repeat(self.ROI_tif)),
                   total=len(key_list), desc=f'Curve fitting result to tif file',
                   bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}'))

        # Create Phemetric dc
        year_list = set([int(np.floor(temp / 10000)) for temp in doy_list])
        metadata_dic = {'ROI_name': self.ROI_name, 'index': index, 'Datatype': 'float', 'ROI': self.ROI,
                        'ROI_array': self.ROI_array, 'ROI_tif': self.ROI_tif, 'Phemetric_factor': True,
                        'coordinate_system': self.coordinate_system, 'size_control_factor': False,
                        'oritif_folder': tif_para_output_path, 'dc_group_list': None, 'tiles': None,
                        'curfit_dic': self._curve_fitting_dic, 'Zoffset': None,
                        'sparse_matrix': self._sparse_matrix_list[dc_num],
                        'huge_matrix': self._huge_matrix_list[dc_num]}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(cf2phemetric_dc, repeat(tif_para_output_path), repeat(phemetric_output_path), year_list,
                         repeat(index), repeat(metadata_dic))

    def _process_link_GEDI_temp_DPAR(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ['accumulated_method', 'static_thr', 'phemetric_window']:
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process accumulated_method
        if 'accumulated_method' in kwargs.keys():
            if isinstance(kwargs['accumulated_method'], str) and kwargs['accumulated_method'] in ['static_thr', 'phemetric_thr']:
                self._GEDI_link_S2_retrieval_method = kwargs['accumulated_method']
            else:
                raise TypeError('Please mention the dc_overwritten_para should be str type!')
        else:
            self._GEDI_link_S2_retrieval_method = 'phemetric_thr'

        if 'static_thr' in kwargs.keys():
            if isinstance(kwargs['static_thr'], (int, float, complex)):
                self._link_GEDI_denv_method = ['static_thr', kwargs['static_thr']]
            else:
                raise TypeError('Please mention the static_thr should be a number!')
        elif self._GEDI_link_S2_retrieval_method == 'static_thr':
            self._link_GEDI_denv_method = ['phemetric_thr', 10]

        if 'phemetric_window' in kwargs.keys():
            if isinstance(kwargs['phemetric_window'], int):
                self._link_GEDI_denv_method = ['phemetric_thr', kwargs['phemetric_window']]
            else:
                raise TypeError('Please mention the phemetric_window should be int type!')
        else:
            self._link_GEDI_denv_method = ['phemetric_thr', 10]

    def link_GEDI_accumulated_Denv(self, GEDI_xlsx_file, denv_list, **kwargs):

        # Process para
        self._process_link_GEDI_temp_DPAR(**kwargs)
        for _ in denv_list:
            if _ not in self.Denv_indexlist:
                raise TypeError(f'The {str(_)} is not imported into the Sentinel2 dcs')

        # Retrieve the S2 inform
        raster_gt = gdal.Open(self.ROI_tif).GetGeoTransform()
        raster_proj = retrieve_srs(gdal.Open(self.ROI_tif))

        # Retrieve GEDI inform
        GEDI_list = gedi.GEDI_df(GEDI_xlsx_file)
        GEDI_list.reprojection(raster_proj, xycolumn_start='EPSG')

        # Construct Denv list
        for denv_temp in denv_list:
            if not os.path.exists(GEDI_xlsx_file.split('.xlsx')[0] + f'_accumulated_{denv_temp}.csv'):

                # Divide the GEDI and dc into different blocks
                block_amount = os.cpu_count()
                indi_block_size = int(np.ceil(GEDI_list.df_size / block_amount))

                # Allocate the GEDI_df and dc
                GEDI_list_blocked, denvdc_blocked, raster_gt_list, doy_list_integrated = [], [], [], []

                # Phe dc count and pos
                denvdc_count = len([_ for _ in self._index_list if _ == denv_temp])
                denvdc_pos = [_ for _ in range(len(self._index_list)) if self._index_list[_] == denv_temp]

                # Reconstruct the phenology dc
                denvdc_reconstructed = None
                for _ in range(denvdc_count):
                    if denvdc_reconstructed is None:
                        if self._sparse_matrix_list[denvdc_pos[_]]:
                            denvdc_reconstructed = self.dcs[denvdc_pos[_]]
                        else:
                            denvdc_reconstructed = self.dcs[denvdc_pos[_]]
                    else:
                        if self._sparse_matrix_list[denvdc_pos[_]]:
                            denvdc_reconstructed.extend_layers(self.dcs[denvdc_pos[_]])
                        else:
                            denvdc_reconstructed = np.concatenate((denvdc_reconstructed, self.dcs[denvdc_pos[_]]),
                                                                  axis=2)
                    doy_list_integrated.extend(self._doys_backup_[denvdc_pos[_]])

                for i in range(block_amount):
                    if i != block_amount - 1:
                        GEDI_list_blocked.append(GEDI_list.GEDI_inform_DF[i * indi_block_size: (i + 1) * indi_block_size])
                    else:
                        GEDI_list_blocked.append(GEDI_list.GEDI_inform_DF[i * indi_block_size: -1])

                    ymin_temp, ymax_temp, xmin_temp, xmax_temp = GEDI_list_blocked[-1].EPSG_lat.max() + 12.5, \
                                                                 GEDI_list_blocked[-1].EPSG_lat.min() - 12.5, \
                                                                 GEDI_list_blocked[-1].EPSG_lon.min() - 12.5, \
                                                                 GEDI_list_blocked[-1].EPSG_lon.max() + 12.5
                    cube_ymin, cube_ymax = int(max(0, np.floor((ymin_temp - raster_gt[3]) / raster_gt[5]))), int(
                        min(self.dcs_YSize, np.ceil((ymax_temp - raster_gt[3]) / raster_gt[5])))
                    cube_xmin, cube_xmax = int(max(0, np.floor((xmin_temp - raster_gt[0]) / raster_gt[1]))), int(
                        min(self.dcs_XSize, np.ceil((xmax_temp - raster_gt[0]) / raster_gt[1])))

                    raster_gt_list.append([raster_gt[0] + cube_xmin * raster_gt[1], raster_gt[1], raster_gt[2],
                                           raster_gt[3] + cube_ymin * raster_gt[5], raster_gt[4], raster_gt[5]])

                    if isinstance(denvdc_reconstructed, NDSparseMatrix):
                        sm_temp = denvdc_reconstructed.extract_matrix(
                            ([cube_ymin, cube_ymax + 1], [cube_xmin, cube_xmax + 1], ['all']))
                        denvdc_blocked.append(sm_temp)
                    else:
                        denvdc_blocked.append(
                            denvdc_reconstructed[cube_ymin:cube_ymax + 1, cube_xmin: cube_xmax + 1, :])

                try:
                    # Sequenced code for debug
                    # for i in range(block_amount):
                    #     result = link_GEDI_inform(dc_blocked[i], GEDI_list_blocked[i], bf.date2doy(thalweg_temp.doy_list), raster_gt, 'EPSG', index_temp, 'linear_interpolation', thalweg_temp.size_control_factor_list[thalweg_temp.index_list.index(index_temp)])
                    with concurrent.futures.ProcessPoolExecutor(max_workers=block_amount) as executor:
                        result = executor.map(link_GEDI_accdenvinform, denvdc_blocked, GEDI_list_blocked,
                                              repeat(doy_list_integrated), raster_gt_list, repeat('EPSG'),
                                              repeat(denv_temp))
                except:
                    raise Exception('The s2pheme-GEDI link procedure was interrupted by unknown error!')

                try:
                    result = list(result)
                    gedi_list_output = None

                    for result_temp in result:
                        if gedi_list_output is None:
                            gedi_list_output = copy.copy(result_temp)
                        else:
                            gedi_list_output = pd.concat([gedi_list_output, result_temp])

                    gedi_list_output.to_csv(GEDI_xlsx_file.split('.xlsx')[0] + f'_accumulated_{denv_temp}.csv')
                except:
                    raise Exception('The df output procedure was interrupted by error!')

    def _process_link_GEDI_phenology_para(self, **kwargs):
        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in []:
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

    def link_GEDI_phenology_inform(self, GEDI_xlsx_file, phemetric_list, **kwargs):

        # Process para
        self._process_link_GEDI_RS_para(**kwargs)

        # Retrieve the S2 inform
        raster_gt = gdal.Open(self.ROI_tif).GetGeoTransform()
        raster_proj = retrieve_srs(gdal.Open(self.ROI_tif))

        # Retrieve GEDI inform
        GEDI_list = gedi.GEDI_df(GEDI_xlsx_file)
        GEDI_list.reprojection(raster_proj, xycolumn_start='EPSG')

        # Construct phemetric list
        phemetric_gedi_list = []
        for phemetric_temp in phemetric_list:
            if not os.path.exists(GEDI_xlsx_file.split('.')[0] + f'_{phemetric_temp}.csv'):
                if phemetric_temp not in self._phemetric_namelist:
                    raise Exception(f'The {str(phemetric_temp)} is not a valid index or is not inputted into the dcs!')

                # Divide the GEDI and dc into different blocks
                block_amount = os.cpu_count()
                indi_block_size = int(np.ceil(GEDI_list.df_size / block_amount))

                # Allocate the GEDI_df and dc
                GEDI_list_blocked, phedc_blocked, raster_gt_list, year_list_temp = [], [], [], []

                # Phe dc count and pos
                phedc_count = len([_ for _ in self._pheyear_list if _ is not None])
                phepos = [self._pheyear_list.index(_) for _ in self._pheyear_list if _ is not None]

                # Reconstruct the phenology dc
                phedc_reconstructed = None
                for _ in range(phedc_count):
                    if phedc_reconstructed is None:
                        if self._sparse_matrix_list[phepos[_]]:
                            phedc_reconstructed = NDSparseMatrix(
                                self.dcs[phepos[_]].SM_group[f'{str(self._pheyear_list[phepos[_]])}_{phemetric_temp}'],
                                SM_namelist=[f'{str(self._pheyear_list[phepos[_]])}_{phemetric_temp}'])
                        else:
                            phedc_reconstructed = self.dcs[phepos[_]][:, :, [self._doys_backup_[phepos[_]].index(
                                [f'{str(self._pheyear_list[phepos[_]])}_{phemetric_temp}'])]]
                    else:
                        if self._sparse_matrix_list[phepos[_]]:
                            phedc_reconstructed.add_layer(
                                self.dcs[phepos[_]].SM_group[f'{str(self._pheyear_list[phepos[_]])}_{phemetric_temp}'],
                                f'{str(self._pheyear_list[phepos[_]])}_{phemetric_temp}',
                                phedc_reconstructed.shape[2] + 1)
                        else:
                            phedc_reconstructed = np.concatenate((phedc_reconstructed, self.dcs[phepos[_]][:, :, [self._doys_backup_[phepos[_]].index([f'{str(self._pheyear_list[phepos[_]])}_{phemetric_temp}'])]]),
                                                                 axis=2)
                    year_list_temp.append(self._pheyear_list[phepos[_]])

                for i in range(block_amount):
                    if i != block_amount - 1:
                        GEDI_list_blocked.append(GEDI_list.GEDI_inform_DF[i * indi_block_size: (i + 1) * indi_block_size])
                    else:
                        GEDI_list_blocked.append(GEDI_list.GEDI_inform_DF[i * indi_block_size: -1])

                    ymin_temp, ymax_temp, xmin_temp, xmax_temp = GEDI_list_blocked[-1].EPSG_lat.max() + 12.5, \
                                                                 GEDI_list_blocked[-1].EPSG_lat.min() - 12.5, \
                                                                 GEDI_list_blocked[-1].EPSG_lon.min() - 12.5, \
                                                                 GEDI_list_blocked[-1].EPSG_lon.max() + 12.5
                    cube_ymin, cube_ymax = int(max(0, np.floor((ymin_temp - raster_gt[3]) / raster_gt[5]))), int(
                        min(self.dcs_YSize, np.ceil((ymax_temp - raster_gt[3]) / raster_gt[5])))
                    cube_xmin, cube_xmax = int(max(0, np.floor((xmin_temp - raster_gt[0]) / raster_gt[1]))), int(
                        min(self.dcs_XSize, np.ceil((xmax_temp - raster_gt[0]) / raster_gt[1])))

                    raster_gt_list.append([raster_gt[0] + cube_xmin * raster_gt[1], raster_gt[1], raster_gt[2],
                                           raster_gt[3] + cube_ymin * raster_gt[5], raster_gt[4], raster_gt[5]])

                    if isinstance(phedc_reconstructed, NDSparseMatrix):
                        sm_temp = phedc_reconstructed.extract_matrix(
                            ([cube_ymin, cube_ymax + 1], [cube_xmin, cube_xmax + 1], ['all']))
                        phedc_blocked.append(sm_temp)
                    else:
                        phedc_blocked.append(phedc_reconstructed[cube_ymin:cube_ymax + 1, cube_xmin: cube_xmax + 1, :])

                try:
                    # Sequenced code for debug
                    # for i in range(block_amount):
                    #     result = link_GEDI_inform(dc_blocked[i], GEDI_list_blocked[i], bf.date2doy(thalweg_temp.doy_list), raster_gt, 'EPSG', index_temp, 'linear_interpolation', thalweg_temp.size_control_factor_list[thalweg_temp.index_list.index(index_temp)])
                    with concurrent.futures.ProcessPoolExecutor(max_workers=block_amount) as executor:
                        result = executor.map(link_GEDI_pheinform, phedc_blocked, GEDI_list_blocked,
                                              repeat(year_list_temp), raster_gt_list, repeat('EPSG'),
                                              repeat(phemetric_temp))
                except:
                    raise Exception('The s2pheme-GEDI link procedure was interrupted by unknown error!')

                try:
                    result = list(result)
                    gedi_list_output = None

                    for result_temp in result:
                        if gedi_list_output is None:
                            gedi_list_output = copy.copy(result_temp)
                        else:
                            gedi_list_output = pd.concat([gedi_list_output, result_temp])

                    gedi_list_output.to_csv(GEDI_xlsx_file.split('.')[0] + f'_{phemetric_temp}.csv')
                except:
                    raise Exception('The df output procedure was interrupted by error!')
            phemetric_gedi_list.append(pd.read_csv(GEDI_xlsx_file.split('.')[0] + f'_{phemetric_temp}.csv'))

        # Output to a single file
        if not os.path.exists(GEDI_xlsx_file.split('.')[0] + f'_all_Phemetrics.csv'):
            i = 0
            phemetric_output = None
            for phemetric_temp in phemetric_list:
                phe_gedilist_temp = phemetric_gedi_list[i].sort_values('Unnamed: 0')
                if phemetric_output is None:
                    phemetric_output = phe_gedilist_temp
                else:
                    key_temp = [_ for _ in list(phe_gedilist_temp.keys()) if phemetric_temp in _][0]
                    phemetric_output.insert(phemetric_output.shape[1], key_temp, phe_gedilist_temp[key_temp])
                i += 1
            phemetric_output.to_csv(GEDI_xlsx_file.split('.')[0] + f'_all_Phemetrics.csv')

    def _process_link_GEDI_RS_para(self, **kwargs):

        # Detect whether all the indicators are valid
        for kwarg_indicator in kwargs.keys():
            if kwarg_indicator not in ['spatial_interpolate_method', 'temporal_interpolate_method']:
                raise NameError(f'{kwarg_indicator} is not supported kwargs! Please double check!')

        # process interpolation method
        if 'spatial_interpolate_method' in kwargs.keys():
            if type(kwargs['spatial_interpolate_method']) is str and kwargs['spatial_interpolate_method'] in ['nearest_neighbor', 'area_average', 'focal']:
                self._GEDI_link_RS_spatial_interpolate_method = kwargs['spatial_interpolate_method']
            else:
                raise TypeError('Please mention the spatial_interpolate_method should be str type!')
        else:
            self._GEDI_link_RS_spatial_interpolate_method = 'nearest_neighbor'

        # process para method
        if 'temporal_interpolate_method' in kwargs.keys():
            if type(kwargs['temporal_interpolate_method']) is str and kwargs['temporal_interpolate_method'] in ['linear_interpolation', '16days_max', '16days_ave']:
                self._GEDI_link_RS_temporal_interpolate_method = [kwargs['temporal_interpolate_method']]
            elif type(kwargs['temporal_interpolate_method']) is list and True in [_ in ['linear_interpolation', '16days_max', '16days_ave'] for _ in kwargs['temporal_interpolate_method']]:
                self._GEDI_link_RS_temporal_interpolate_method = kwargs['temporal_interpolate_method']
            else:
                raise TypeError('Please mention the temporal_interpolate_method should be str type!')
        else:
            self._GEDI_link_RS_temporal_interpolate_method = ['linear_interpolation']

    def link_GEDI_RS_dc(self, GEDI_df_, index_list, **kwargs):

        # Two different method Nearest neighbor and linear interpolation
        self._process_link_GEDI_RS_para(**kwargs)

        # Retrieve the S2 inform
        raster_gt = gdal.Open(self.ROI_tif).GetGeoTransform()
        raster_proj = retrieve_srs(gdal.Open(self.ROI_tif))

        # Retrieve GEDI inform
        if isinstance(GEDI_df_, gedi.GEDI_df):
            GEDI_df_.reprojection(raster_proj, xycolumn_start='EPSG')
        else:
            raise TypeError('The gedi_df_ is not under the right type')

        # resort through lat or lon
        for index_temp in index_list:

            if index_temp not in self._index_list:
                raise Exception(f'The {str(index_temp)} is not a valid index or is not input into the dcs!')

            # Divide the GEDI and dc into different blocks
            block_amount = os.cpu_count()
            indi_block_size = int(np.ceil(GEDI_df_.df_size / block_amount))

            # Allocate the GEDI_df and dc
            GEDI_df_blocked, dc_blocked, raster_gt_list, doy_list_temp = [], [], [], []
            for i in range(block_amount):
                if i != block_amount - 1:
                    GEDI_df_blocked.append(GEDI_df_.GEDI_inform_DF[i * indi_block_size: (i + 1) * indi_block_size])
                else:
                    GEDI_df_blocked.append(GEDI_df_.GEDI_inform_DF[i * indi_block_size: -1])

                ymin_temp, ymax_temp, xmin_temp, xmax_temp = GEDI_df_blocked[-1].EPSG_lat.max() + 12.5, \
                                                             GEDI_df_blocked[-1].EPSG_lat.min() - 12.5, \
                                                             GEDI_df_blocked[-1].EPSG_lon.min() - 12.5, \
                                                             GEDI_df_blocked[-1].EPSG_lon.max() + 12.5
                cube_ymin, cube_ymax, cube_xmin, cube_xmax = (int(max(0, np.floor((ymin_temp - raster_gt[3]) / raster_gt[5]))),
                                                              int(min(self.dcs_YSize, np.ceil((ymax_temp - raster_gt[3]) / raster_gt[5]))),
                                                              int(max(0, np.floor((xmin_temp - raster_gt[0]) / raster_gt[1]))),
                                                              int(min(self.dcs_XSize, np.ceil((xmax_temp - raster_gt[0]) / raster_gt[1]))))

                if isinstance(self.dcs[self._index_list.index(index_temp)], NDSparseMatrix):
                    sm_temp = self.dcs[self._index_list.index(index_temp)].extract_matrix(([cube_ymin, cube_ymax + 1], [cube_xmin, cube_xmax + 1], ['all']))
                    dc_blocked.append(sm_temp.drop_nanlayer())
                    doy_list_temp.append(bf.date2doy(dc_blocked[-1].SM_namelist))
                elif isinstance(self.dcs[self._index_list.index(index_temp)], np.ndarray):
                    dc_blocked.append(self.dcs[self._index_list.index(index_temp)][cube_ymin:cube_ymax + 1, cube_xmin: cube_xmax + 1, :])
                    doy_list_temp.append(bf.date2doy(self.s2dc_doy_list))
                raster_gt_list.append([raster_gt[0] + cube_xmin * raster_gt[1], raster_gt[1], raster_gt[2],
                                       raster_gt[3] + cube_ymin * raster_gt[5], raster_gt[4], raster_gt[5]])

            try:
                # Sequenced code for debug
                # for i in range(block_amount):
                #     result = link_GEDI_inform(dc_blocked[i], GEDI_list_blocked[i], bf.date2doy(thalweg_temp.doy_list), raster_gt, 'EPSG', index_temp, 'linear_interpolation', thalweg_temp.size_control_factor_list[thalweg_temp.index_list.index(index_temp)])
                with concurrent.futures.ProcessPoolExecutor(max_workers=block_amount) as executor:
                    result = executor.map(link_GEDI_inform, dc_blocked, GEDI_df_blocked, doy_list_temp,
                                          raster_gt_list, repeat('EPSG'), repeat(index_temp),
                                          repeat(['linear_interpolation']),
                                          repeat(self._size_control_factor_list[self._index_list.index(index_temp)]))
            except:
                raise Exception('The link procedure was interrupted by error!')

            try:
                result = list(result)
                index_combined_name = '_'
                index_combined_name = index_combined_name.join(index_list)
                gedi_list_output = None

                for result_temp in result:
                    if gedi_list_output is None:
                        gedi_list_output = copy.copy(result_temp)
                    else:
                        gedi_list_output = pd.concat([gedi_list_output, result_temp])

                gedi_list_output.to_csv(GEDI_xlsx_file.split('.')[0] + f'_{index_combined_name}.csv')
            except:
                raise Exception('The df output procedure was interrupted by error!')

    def calculate_denv8pheme(self, denvname: str, year_list: list, start_pheme: str, end_pheme: str, cal_method: str, base_status: bool, bulk = True):

        # Construct the output folder
        output_folder = os.path.join(self._denv_work_env, 'denv8pheme\\')
        bf.create_folder(output_folder)

        # Determine the cal method
        if not isinstance(cal_method, str):
            raise TypeError('The cal method should be str type!')
        elif cal_method not in self._denv8pheme_cal_method_list:
            raise ValueError('The cal method is not supported!')
        else:
                self._denv8pheme_cal = cal_method

        # Determine the base status
        if not isinstance(base_status, bool):
            raise TypeError('The base status should be bool type!')
        elif base_status and self._denv8pheme_cal != 'acc':
            self._denv8pheme_minus_base = False
            print('Only accumulated value could use the base status')
        else:
            self._denv8pheme_minus_base = base_status

        # Determine the denv name
        if denvname not in self.Denv_indexlist:
            raise ValueError(f'The denv index {str(denvname)} is not imported')

        # Determine the process year list
        processed_year_list = []
        for year in year_list:
            if year in self._pheyear_list and year in self._denvyear_list:
                processed_year_list.append(year)
            else:
                print(f'The {str(year)} is not imported!')
        if len(processed_year_list) == 0:
            raise ValueError(f'No valid year imported!')

        # Determine the start and end pheme
        if start_pheme not in self._denv8pheme_:
            raise ValueError(f'{str(start_pheme)} is not valid')
        elif end_pheme not in self._denv8pheme_:
            raise ValueError(f'{str(start_pheme)} is not valid')
        elif self._denv8pheme_.index(end_pheme) < self._denv8pheme_.index(start_pheme):
            raise ValueError(f'The {str(start_pheme)} could not before end pheme')
        elif self._denv8pheme_.index(end_pheme) == self._denv8pheme_.index(start_pheme):
            self._denv8pheme_minus_base = False

        # Itr through the year
        para_list = [denvname, copy.deepcopy(self._denv8pheme_cal), copy.deepcopy(self._denv8pheme_minus_base), True, output_folder, copy.deepcopy(self.ROI_tif)]
        pheme_ = [start_pheme, end_pheme]
        if bulk:
            pheme_name, denv_name = [], []
            for year_ in processed_year_list:
                pheme_name.append(self._dcs_backup_[self._pheyear_list.index(year_)].dc_filename)
                denv_name.append(self._dcs_backup_[[_ for _ in range(len(self._index_list)) if self._index_list[_] == denvname and self._timerange_list[_] == year_][0]].dc_filename)
                self.dcs[self._pheyear_list.index(year_)] = None
                self.dcs[[_ for _ in range(len(self._index_list)) if self._index_list[_] == denvname and self._timerange_list[_] == year_][0]] = None

            with concurrent.futures.ProcessPoolExecutor() as exe:
                exe.map(process_denv_via_pheme, denv_name, pheme_name, processed_year_list, repeat(pheme_), repeat(para_list))
        else:
            for year_ in processed_year_list:
                process_denv_via_pheme(self._dcs_backup_[self._pheyear_list.index(year_)].dc_filename,
                                       self._dcs_backup_[[_ for _ in range(len(self._index_list)) if self._index_list[_] == denvname and self._timerange_list[_] == year_][0]].dc_filename,
                                       year_, pheme_, para_list)
        #
        # ds_temp = gdal.Open(self.ROI_tif)
        # for year_ in processed_year_list:
        #     st = time.time()
        #     print(f'Start calculate the \033[1;31m{str(self._denv8pheme_cal)}\033[0m \033[1;31m{denvname}\033[0m for the year \033[1;34m{str(year_)}\033[0m')
        #     # Get the denv and pheme position
        #     denv_pos = [_ for _ in range(len(self._index_list)) if self._index_list[_] == denvname and self._timerange_list[_] == year_][0]
        #     pheme_pos = self._pheyear_list.index(year_)
        #
        #     # Get the type of the denv and pheme dc
        #     denv_sparse_factor = isinstance(self.dcs[denv_pos], NDSparseMatrix)
        #     pheme_sparse_factor = isinstance(self.dcs[pheme_pos], NDSparseMatrix)
        #
        #     # Get the base status
        #     if base_status is True:
        #
        #         # Determine the base time
        #         if start_pheme == 'SOY':
        #             start_doy = np.zeros([self.dcs_YSize, self.dcs_XSize])
        #             end_doy = np.ones([self.dcs_YSize, self.dcs_XSize]) * 10
        #
        #         elif start_pheme == 'EOY':
        #             raise Exception('EOY can not be the start pheme when base status is True')
        #
        #         elif start_pheme in ['SOS', 'peak_doy', 'EOS']:
        #             if start_pheme not in self._phemetric_namelist:
        #                 raise ValueError('The start_pheme is not generated')
        #             else:
        #                 if pheme_sparse_factor:
        #                     start_doy = np.round(self.dcs[pheme_pos].SM_group[f'{str(year_)}_{start_pheme}'].toarray())
        #                     start_doy[start_doy == 0] = np.nan
        #                     start_doy = start_doy - 5
        #                     end_doy = np.round(self.dcs[pheme_pos].SM_group[f'{str(year_)}_{start_pheme}'].toarray())
        #                     end_doy[end_doy == 0] = np.nan
        #                     end_doy = end_doy + 5
        #                 else:
        #                     raise Exception('Code error')
        #         else:
        #             raise Exception('Code Error')
        #
        #         # Generate the base value
        #         start_time = np.nanmin(start_doy).astype(np.int16)
        #         end_time = np.nanmax(end_doy).astype(np.int16)
        #         acc_static = np.zeros([self.dcs_YSize, self.dcs_XSize])
        #         cum_static = np.zeros([self.dcs_YSize, self.dcs_XSize])
        #         with tqdm(total=end_time + 1 - start_time, desc=f'Get the static value of {str(year_)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
        #             for _ in range(start_time, end_time + 1):
        #                 within_factor = np.logical_and(np.ones([self.dcs_YSize, self.dcs_XSize]) * _ >= start_doy, np.ones([self.dcs_YSize, self.dcs_XSize]) * _ < end_doy)
        #                 cum_static = cum_static + within_factor
        #                 if denv_sparse_factor:
        #                     denv_date = int(year_ * 1000 + _)
        #                     denv_arr = self.dcs[denv_pos].SM_group[denv_date].toarray()
        #                 else:
        #                     raise Exception('Code Error')
        #                 denv_doy_arr = denv_arr * within_factor
        #                 acc_static = acc_static + denv_doy_arr
        #                 pbar.update()
        #
        #         acc_static = acc_static / cum_static
        #         cum_static = None
        #         bf.write_raster(ds_temp, acc_static, output_folder, f'{str(self._denv8pheme_cal)}_{denvname}_{str(year_)}_static.TIF', raster_datatype=gdal.GDT_Float32)
        #
        #     # Get the denv matrix
        #     start_arr = np.round(self.dcs[pheme_pos].SM_group[f'{str(year_)}_{start_pheme}'].toarray())
        #     end_arr = np.round(self.dcs[pheme_pos].SM_group[f'{str(year_)}_{end_pheme}'].toarray())
        #     start_arr[start_arr == 0] = np.nan
        #     end_arr[end_arr == 0] = np.nan
        #
        #     # Get the unique value
        #     start_doy = np.nanmin(np.unique(start_arr)).astype(np.int16)
        #     end_doy = np.nanmax(np.unique(end_arr)).astype(np.int16)
        #     acc_denv = np.zeros([self.dcs_YSize, self.dcs_XSize])
        #     cum_denv = np.zeros([self.dcs_YSize, self.dcs_XSize])
        #
        #     with tqdm(total=end_doy + 1 - start_doy, desc=f'Get the static value of {str(year_)}', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
        #         for _ in range(start_doy, end_doy + 1):
        #
        #             if denv_sparse_factor:
        #                 denv_date = int(year_ * 1000 + _)
        #                 denv_arr = self.dcs[denv_pos].SM_group[denv_date].toarray()
        #             else:
        #                 raise Exception('Code Error')
        #
        #             if base_status:
        #                 within_factor = np.logical_and(np.logical_and(np.ones([self.dcs_YSize, self.dcs_XSize]) * _ >= start_doy,
        #                                                np.ones([self.dcs_YSize, self.dcs_XSize]) * _ < end_doy),
        #                                                denv_arr >= acc_static)
        #                 denv_doy_arr = (denv_arr - acc_static) * within_factor
        #             else:
        #                 within_factor = np.logical_and(np.ones([self.dcs_YSize, self.dcs_XSize]) * _ >= start_doy,
        #                                                np.ones([self.dcs_YSize, self.dcs_XSize]) * _ < end_doy)
        #                 denv_doy_arr = denv_arr * within_factor
        #
        #             if self._denv8pheme_cal != 'max':
        #                 cum_denv = cum_denv + within_factor
        #                 acc_denv = acc_denv + denv_doy_arr
        #             else:
        #                 acc_denv = np.nanmax(acc_denv, denv_doy_arr)
        #             pbar.update()
        #
        #     if self._denv8pheme_cal == 'mean':
        #         acc_denv = acc_denv / cum_denv
        #
        #     if self._size_control_factor_list[denv_pos]:
        #         acc_denv = acc_denv / 100
        #
        #     bf.write_raster(ds_temp, acc_denv, output_folder, f'{str(self._denv8pheme_cal)}_{denvname}_{str(year_)}.TIF', raster_datatype=gdal.GDT_Float32)
        #     print(f'Finish calculate the \033[1;31m{str(self._denv8pheme_cal)}\033[0m \033[1;31m{denvname}\033[0m for the year \033[1;34m{str(year_)}\033[0m in {str(time.time()-st)}s')

    def process_denv_via_pheme(self, denvname, phename, ):
        if phename == 'SOS':
            # Phe dc count and pos
            denvdc_count = len([_ for _ in self._index_list if _ == denvname])
            denvdc_pos = [_ for _ in range(len(self._index_list)) if self._index_list[_] == denvname]

            pheme_reconstructed = None
            pheme_namelist = []
            # Reconstruct the phenology dc
            for _ in range(denvdc_count):
                denvdc_year = list(set([int(np.floor(_ / 1000)) for _ in self.Denv_doy_list[denvdc_pos[_]]]))

                for year_temp in denvdc_year:
                    if year_temp not in self._pheyear_list:
                            raise TypeError(f'The phemetric of {str(year_temp)} is not imported')
                    else:
                        phepos = self._pheyear_list.index(year_temp)
                        if f'{str(year_temp)}_static_{denvname}' not in self._doys_backup_[phepos]:
                            try:
                                if (pheme_reconstructed is None or (isinstance(pheme_reconstructed, NDSparseMatrix) and f'{str(self._pheyear_list[phepos])}_SOS' not in pheme_reconstructed.SM_namelist) or
                                        (isinstance(pheme_reconstructed, np.ndarray) and f'{str(self._pheyear_list[phepos])}_SOS' not in pheme_namelist)):
                                    if pheme_reconstructed is None:
                                        if self._sparse_matrix_list[phepos]:
                                            pheme_reconstructed = NDSparseMatrix(self.dcs[phepos].SM_group[f'{str(self._pheyear_list[phepos])}_SOS'],
                                                SM_namelist=[f'{str(self._pheyear_list[phepos])}_SOS'])
                                        else:
                                            pheme_reconstructed = self.dcs[phepos][:, :, [self._doys_backup_[phepos].index([f'{str(self._pheyear_list[phepos])}_SOS'])]]
                                            pheme_namelist.append(f'{str(self._pheyear_list[phepos])}_SOS')
                                    else:
                                        if self._sparse_matrix_list[phepos]:
                                            pheme_reconstructed.add_layer(
                                                self.dcs[phepos].SM_group[f'{str(self._pheyear_list[phepos])}_SOS'],
                                                f'{str(self._pheyear_list[phepos])}_SOS',
                                                pheme_reconstructed.shape[2] + 1)
                                        else:
                                            pheme_reconstructed = np.concatenate((pheme_reconstructed,self.dcs[phepos][:, :, [self._doys_backup_[phepos].index([f'{str(self._pheyear_list[phepos])}_SOS'])]]),
                                                                                 axis=2)
                                            pheme_namelist.append(f'{str(self._pheyear_list[phepos])}_SOS')

                                if pheme_reconstructed is None or (isinstance(pheme_reconstructed,
                                                                              NDSparseMatrix) and f'{str(self._pheyear_list[phepos])}_peak_doy' not in pheme_reconstructed.SM_namelist) or (
                                        isinstance(pheme_reconstructed,
                                                   np.ndarray) and f'{str(self._pheyear_list[phepos])}_peak_doy' not in pheme_namelist):
                                    if pheme_reconstructed is None:
                                        if self._sparse_matrix_list[phepos]:
                                            pheme_reconstructed = NDSparseMatrix(self.dcs[phepos].SM_group[f'{str(self._pheyear_list[phepos])}_peak_doy'],
                                                                                 SM_namelist=[f'{str(self._pheyear_list[phepos])}_peak_doy'])
                                        else:
                                            pheme_reconstructed = self.dcs[phepos][:, :, [self._doys_backup_[phepos].index([f'{str(self._pheyear_list[phepos])}_peak_doy'])]]
                                            pheme_namelist.append(f'{str(self._pheyear_list[phepos])}_peak_doy')
                                    else:
                                        if self._sparse_matrix_list[phepos]:
                                            pheme_reconstructed.add_layer(self.dcs[phepos].SM_group[f'{str(self._pheyear_list[phepos])}_peak_doy'],f'{str(self._pheyear_list[phepos])}_peak_doy',
                                                                          pheme_reconstructed.shape[2] + 1)
                                        else:
                                            pheme_reconstructed = np.concatenate((pheme_reconstructed,self.dcs[phepos][:, :, [self._doys_backup_[phepos].index([f'{str(self._pheyear_list[phepos])}_peak_doy'])]]),
                                                                                 axis=2)
                                        pheme_namelist.append(f'{str(self._pheyear_list[phepos])}_peak_doy')

                            except:
                                raise Exception('SOS or peak doy is not properly retrieved!')

                            # Retrieve the phemetric inform
                            if isinstance(pheme_reconstructed, NDSparseMatrix):
                                sos = np.round(pheme_reconstructed.SM_group[f'{str(year_temp)}_SOS'].toarray())
                                peak_doy = np.round(pheme_reconstructed.SM_group[f'{str(year_temp)}_peak_doy'].toarray())
                            elif isinstance(pheme_reconstructed, np.ndarray):
                                sos = np.round(pheme_reconstructed[:, :, pheme_namelist.index(f'{str(year_temp)}_SOS')].reshape(
                                        [pheme_reconstructed.shape[0], pheme_reconstructed.shape[1]]))
                                peak_doy = np.round(pheme_reconstructed[:, :, pheme_namelist.index(f'{str(year_temp)}_peak_doy')].reshape(
                                    [pheme_reconstructed.shape[0], pheme_reconstructed.shape[1]]))
                            else:
                                raise TypeError('The para phemetric dc is not imported as a supported datatype!')

                            base_env = copy.copy(sos)
                            base_env[base_env <= 0] = 0
                            base_env[base_env != 0] = -1

                            sos = sos + year_temp * 1000
                            sos[sos <= year_temp * 1000] = 3000000
                            sos = sos.astype(np.int32)

                            peak_doy = peak_doy + year_temp * 1000
                            peak_doy[peak_doy <= year_temp * 1000] = 0
                            peak_doy = peak_doy.astype(np.int32)

                            year_doy = self._doys_backup_[denvdc_pos[_]]

                            # Create static/base env map
                            xy_all = np.argwhere(base_env == -1)
                            xy_all = pd.DataFrame(xy_all, columns=['y', 'x'])
                            xy_all = xy_all.sort_values(['x', 'y'])
                            y_all, x_all = list(xy_all['y']), list(xy_all['x'])

                            block_amount = os.cpu_count()
                            indi_block_size = int(np.ceil(len(x_all) / block_amount))

                            # Allocate the GEDI_df and dc
                            y_all_blocked, x_all_blocked, denv_dc_blocked, xy_offset_blocked, sos_blocked = [], [], [], [], []
                            for i in range(block_amount):
                                if i != block_amount - 1:
                                    y_all_blocked.append(y_all[i * indi_block_size: (i + 1) * indi_block_size])
                                    x_all_blocked.append(x_all[i * indi_block_size: (i + 1) * indi_block_size])
                                else:
                                    y_all_blocked.append(y_all[i * indi_block_size:])
                                    x_all_blocked.append(x_all[i * indi_block_size:])

                                if isinstance(self.dcs[denvdc_pos[_]], NDSparseMatrix):
                                    denv_dc_blocked.append(self.dcs[denvdc_pos[_]].extract_matrix(([min(y_all_blocked[-1]), max(y_all_blocked[-1]) + 1], [min(x_all_blocked[-1]),max(x_all_blocked[-1]) + 1],['all'])))
                                else:
                                    pass

                                sos_blocked.append(sos[min(y_all_blocked[-1]): max(y_all_blocked[-1]) + 1,
                                                   min(x_all_blocked[-1]): max(x_all_blocked[-1]) + 1])
                                xy_offset_blocked.append([min(y_all_blocked[-1]), min(x_all_blocked[-1])])

                            with concurrent.futures.ProcessPoolExecutor() as exe:
                                result = exe.map(get_base_denv, y_all_blocked, x_all_blocked, sos_blocked,
                                                 repeat(year_doy), denv_dc_blocked, xy_offset_blocked)

                            result = list(result)
                            for result_temp in result:
                                for r_ in result_temp:
                                    base_env[r_[0], r_[1]] = r_[2]
                            base_env[base_env == -1] = 0
                            base_env[np.isnan(base_env)] = 0
                            self._dcs_backup_[phepos].dc = copy.copy(self.dcs[phepos])
                            self._dcs_backup_[phepos]._add_layer(
                                type(self.dcs[phepos].SM_group[f'{str(year_temp)}_SOS'])(base_env),
                                f'{str(year_temp)}_static_{denvname}')
                            self._dcs_backup_[phepos].save(self._dcs_backup_[phepos].Phemetric_dc_filepath)
                            self._dcs_backup_[phepos].dc = None
                        else:
                            # Retrieve the phemetric inform
                            if isinstance(self.dcs[phepos], NDSparseMatrix):
                                sos = np.round(self.dcs[phepos].SM_group[f'{str(year_temp)}_SOS'].toarray())
                                peak_doy = np.round(self.dcs[phepos].SM_group[f'{str(year_temp)}_peak_doy'].toarray())
                                base_env = self.dcs[phepos].SM_group[f'{str(year_temp)}_static_{denvname}'].toarray()
                            elif isinstance(self.dcs[phepos], np.ndarray):
                                sos = np.round(self.dcs[phepos][:, :, pheme_namelist.index(f'{str(year_temp)}_SOS')].reshape(
                                        [self.dcs[phepos].shape[0], self.dcs[phepos].shape[1]]))
                                peak_doy = np.round(self.dcs[phepos][:, :, pheme_namelist.index(f'{str(year_temp)}_peak_doy')].reshape(
                                        [self.dcs[phepos].shape[0], self.dcs[phepos].shape[1]]))
                                base_env = self.dcs[phepos][:, :, pheme_namelist.index(f'{str(year_temp)}_static_{denvname}')].reshape(
                                    [self.dcs[phepos].shape[0], self.dcs[phepos].shape[1]])
                            else:
                                raise TypeError('The para phemetric dc is not imported as a supported datatype!')

                            sos = sos + year_temp * 1000
                            sos[sos <= year_temp * 1000] = 3000000
                            sos = sos.astype(np.int32)

                            peak_doy = peak_doy + year_temp * 1000
                            peak_doy[peak_doy <= year_temp * 1000] = 0
                            peak_doy = peak_doy.astype(np.int32)

                        peak_doy_env = copy.copy(peak_doy)
                        peak_doy_env = peak_doy_env.astype(float)
                        peak_doy_env[peak_doy_env != 0] = 0

                        for doy in self._doys_backup_[denvdc_pos[_]]:
                            sos_temp = sos <= doy
                            sos_temp = sos_temp.astype(int)
                            peak_doy_temp = peak_doy >= doy
                            peak_doy_temp = peak_doy_temp.astype(float)
                            if isinstance(self.dcs[denvdc_pos[_]], NDSparseMatrix):
                                temp = (self.dcs[denvdc_pos[_]].SM_group[doy] - self.dcs[denvdc_pos[_]]._matrix_type(base_env)).multiply(self.dcs[denvdc_pos[_]]._matrix_type(sos_temp))
                                temp[temp < 0] = 0
                                self.dcs[denvdc_pos[_]].SM_group[doy] = type(temp)(temp.astype(self.dcs[denvdc_pos[_]].SM_group[doy].dtype).toarray())
                                peak_doy_env += self.dcs[denvdc_pos[_]].SM_group[doy].toarray() * peak_doy_temp
                            else:
                                self.dcs[denvdc_pos[_]][:, :, self._doys_backup_[denvdc_pos[_]].index(doy)] = (self.dcs[denvdc_pos[_]][:, :,self._doys_backup_[denvdc_pos[_]].index(doy)] - base_env) * sos_temp
                                peak_doy_env += self.dcs[denvdc_pos[_]][:, :, self._doys_backup_[denvdc_pos[_]].index(doy)] * peak_doy_temp
                        self._dcs_backup_[phepos].dc = copy.copy(self.dcs[phepos])
                        self._dcs_backup_[phepos]._add_layer(type(self.dcs[phepos].SM_group[f'{str(year_temp)}_SOS'])(peak_doy_env), f'{str(year_temp)}_peak_{denvname}')
                        self._dcs_backup_[phepos].save(self._dcs_backup_[phepos].Phemetric_dc_filepath)
                        self._dcs_backup_[phepos].dc = None

                self._dcs_backup_[denvdc_pos[_]].dc = copy.copy(self.dcs[denvdc_pos[_]])
                ori_index, ori_path = self._dcs_backup_[denvdc_pos[_]].index, self._dcs_backup_[denvdc_pos[_]].Denv_dc_filepath
                self._dcs_backup_[denvdc_pos[_]].index, self._dcs_backup_[denvdc_pos[_]].Denv_dc_filepath = (ori_index + '_relative', os.path.dirname(os.path.dirname(ori_path)) + '\\' + ori_path.split('\\')[-2] + '_relative\\')
                self._dcs_backup_[denvdc_pos[_]].save(self._dcs_backup_[denvdc_pos[_]].Denv_dc_filepath)
                self._dcs_backup_[denvdc_pos[_]].dc = None
                self._dcs_backup_[denvdc_pos[_]].index, self._dcs_backup_[denvdc_pos[_]].Denv_dc_filepath = ori_index, ori_path
        else:
            pass

    def create_feature_list_by_date(self, date: list, index: str, output_folder: str):

        # Determine the date
        date_pro, peak_factor, year_pro = [], False, []
        for _ in date:
            if isinstance(_, int) and _ > 10000000:
                date_pro.append(bf.date2doy(_))
                year_pro.append(_ // 10000)
            elif isinstance(_, int) and _ > 1000000:
                date_pro.append(_)
                year_pro.append(_ // 1000)
            elif isinstance(_, str) and _.startswith('peak'):
                date_pro.append(_)
                year_t = None

                for q in range(0, len(_) - 3):
                    try:
                        year_t = int(_[q: q + 4])
                    except:
                        pass

                if year_t is None:
                    raise TypeError('The peak type should follow the year!')
                elif year_t is not None and year_t not in self._pheyear_list:
                    raise TypeError(f'The phemetric of {str(year_t)} is not input!')
                elif self._phemetric_namelist is not None and 'peak_doy' not in self._phemetric_namelist:
                    raise TypeError(f'The peak doy of {str(year_t)} is not generated!')
                else:
                    peak_factor = True
                    year_pro.append(year_t)
            else:
                raise TypeError('The date was not supported!')

            # Create output folder
            if isinstance(output_folder, str):
                output_folder = Path(output_folder).path_name
                bf.create_folder(output_folder) if not os.path.exists(output_folder) else None
                bf.create_folder(output_folder + str(_) + '\\') if not os.path.exists(
                    output_folder + str(_) + '\\') else None
            else:
                raise TypeError('The output folder should be a string!')

        # Generate the pos xy list
        roi_ds = gdal.Open(self._ROI_tif_list[0])
        roi_map = roi_ds.GetRasterBand(1).ReadAsArray()
        xy_all = np.argwhere(roi_map != roi_ds.GetRasterBand(1).GetNoDataValue()) if ~np.isnan(
            roi_ds.GetRasterBand(1).GetNoDataValue()) else np.argwhere(~np.isnan(roi_map))
        xy_all = pd.DataFrame(xy_all, columns=['y', 'x'])
        xy_all = xy_all.sort_values(['x', 'y'])

        # Generate the peak doy list
        if peak_factor:
            for q in date_pro:
                if isinstance(q, str) and q.startswith('peak'):

                    year_temp = None
                    for yy in range(0, len(q) - 3):
                        try:
                            year_temp = int(q[yy: yy + 4])
                        except:
                            pass

                    if isinstance(self.dcs[self._pheyear_list.index(year_temp)], NDSparseMatrix):
                        pheme_array = self.dcs[self._pheyear_list.index(year_temp)].SM_group[f'{str(year_temp)}_peak_doy'].toarray()
                    else:
                        pheme_array = self.dcs[self._pheyear_list.index(year_temp)][:, :,
                                      self._doys_backup_[self._pheyear_list.index(year_temp)].index(f'{str(year_temp)}_peak_doy')]

                    pheme_array[pheme_array == 0] = np.nan
                    pheme_array = pheme_array + year_temp * 1000
                    y_all, x_all = np.mgrid[:pheme_array.shape[0], :pheme_array.shape[1]]
                    arr_out = np.column_stack((y_all.ravel(), x_all.ravel(), pheme_array.ravel()))
                    df_temp = pd.DataFrame(arr_out, columns=['y', 'x', q])
                    xy_all = pd.merge(xy_all, df_temp, on=['x', 'y'], how='left')

        # Itr through index
        for _ in index:

            if _ not in self._index_list and _ not in self._phemetric_namelist:
                raise TypeError(f'The {_} is not input or avaliable !')

            elif _ in self._index_list and self._dc_typelist[self._index_list.index(_)] == Sentinel2_dc:

                # Allocate the GEDI_df and dc
                mod_factor = 's2dc'
                y_all_blocked, x_all_blocked, dc_blocked, xy_offset_blocked, doy_list_temp, req_day_list = [], [], [], [], [], []
                block_amount = os.cpu_count()
                for date_temp in date_pro:

                    if not os.path.exists(f'{output_folder}{str(date_temp)}\\{_}_index.csv'):
                        y_all, x_all = list(xy_all['y']), list(xy_all['x'])
                        peak_doy_all = list(xy_all[date_temp]) if isinstance(date_temp, str) else None
                        indi_block_size = int(np.ceil(len(y_all) / block_amount))

                        if req_day_list == []:
                            req_day_list = [[] for tt in range(block_amount)]

                        for i in range(block_amount):
                            if i != block_amount - 1:
                                if len(y_all_blocked) != block_amount:
                                    y_all_blocked.append(y_all[i * indi_block_size: (i + 1) * indi_block_size])
                                    x_all_blocked.append(x_all[i * indi_block_size: (i + 1) * indi_block_size])
                                elif len(y_all_blocked) != len(x_all_blocked):
                                    raise Exception('Code Error in create feature list')

                                if isinstance(date_temp, str):
                                    req_day_list[i].append(peak_doy_all[i * indi_block_size: (i + 1) * indi_block_size])
                                else:
                                    req_day_list[i].append(
                                        [date_temp for tt in range(i * indi_block_size, (i + 1) * indi_block_size)])

                            else:
                                if len(y_all_blocked) != block_amount:
                                    y_all_blocked.append(y_all[i * indi_block_size:])
                                    x_all_blocked.append(x_all[i * indi_block_size:])
                                elif len(y_all_blocked) != len(x_all_blocked):
                                    raise Exception('Code Error in create feature list')

                                if isinstance(date_temp, str):
                                    req_day_list[i].append(peak_doy_all[i * indi_block_size: (i + 1) * indi_block_size])
                                else:
                                    req_day_list[i].append(
                                        [date_temp for tt in range(i * indi_block_size, (i + 1) * indi_block_size)])

                            if len(dc_blocked) != block_amount and len(doy_list_temp) != block_amount:
                                if isinstance(self.dcs[self._index_list.index(_)], NDSparseMatrix):
                                    sm_temp = self.dcs[self._index_list.index(_)].extract_matrix((
                                        [min(y_all_blocked[i]),
                                         max(y_all_blocked[i]) + 1],
                                        [min(x_all_blocked[i]),
                                         max(x_all_blocked[i]) + 1],
                                        ['all']))
                                    dc_blocked.append(sm_temp)
                                    doy_list_temp.append(bf.date2doy(dc_blocked[i].SM_namelist))
                                elif isinstance(self.dcs[self._index_list.index(_)], np.ndarray):
                                    dc_blocked.append(self.dcs[self._index_list.index(_)][
                                                      y_all_blocked[i].min():y_all_blocked[i].max() + 1,
                                                      x_all_blocked[i].min(): x_all_blocked[i].max() + 1, :])
                                    doy_list_temp.append(bf.date2doy(self.s2dc_doy_list))
                            elif len(dc_blocked) != len(doy_list_temp):
                                raise Exception('Code Error in create feature list')

                            if len(xy_offset_blocked) != block_amount:
                                xy_offset_blocked.append([min(y_all_blocked[i]), min(x_all_blocked[i])])

                with concurrent.futures.ProcessPoolExecutor(max_workers=block_amount) as executor:
                    result = executor.map(get_index_by_date, dc_blocked, y_all_blocked, x_all_blocked, doy_list_temp,
                                          req_day_list, xy_offset_blocked, repeat(_), repeat(date_pro), repeat('index'))

                pd_out = None
                result = list(result)
                for tt in result:
                    if pd_out is None:
                        pd_out = pd.DataFrame(tt)
                    else:
                        pd_out = pd.concat([pd_out, pd.DataFrame(tt)])

            elif self._phemetric_namelist is not None and _ in self._phemetric_namelist:

                # Allocate the GEDI_df and dc
                mod_factor = 'phedc'
                y_all_blocked, x_all_blocked, dc_blocked, xy_offset_blocked, doy_list_temp = [], [], [], [], []
                y_all, x_all = list(xy_all['y']), list(xy_all['x'])
                phedc_reconstructed = None
                block_amount = os.cpu_count()
                indi_block_size = int(np.ceil(len(y_all) / block_amount))

                year_indi = list(set(year_pro))
                for year_t in year_indi:

                    if year_t not in self._pheyear_list:
                        raise ValueError(f'The phemetric under {str(year_t)} is not imported!')
                    else:

                        # Reconstruct the phenology dc
                        phepos = self._pheyear_list.index(year_t)
                        if phedc_reconstructed is None:
                            if isinstance(self.dcs[phepos], NDSparseMatrix):
                                phedc_reconstructed = NDSparseMatrix(
                                    self.dcs[phepos].SM_group[f'{str(self._pheyear_list[phepos])}_{_}'],
                                    SM_namelist=[f'{str(self._pheyear_list[phepos])}_{_}'])
                            else:
                                phedc_reconstructed = self.dcs[phepos][:, :, [self._doys_backup_[phepos].index(
                                    [f'{str(self._pheyear_list[phepos])}_{_}'])]]
                        else:
                            if isinstance(self.dcs[phepos], NDSparseMatrix):
                                phedc_reconstructed = phedc_reconstructed.append(
                                    self.dcs[phepos].SM_group[f'{str(self._pheyear_list[phepos])}_{_}'],
                                    name=[f'{str(self._pheyear_list[phepos])}_{_}'])
                            else:
                                phedc_reconstructed = np.concatenate((phedc_reconstructed, self.dcs[phepos][:, :, [
                                                                                                                      self._doys_backup_[
                                                                                                                          phepos].index(
                                                                                                                          [
                                                                                                                              f'{str(self._pheyear_list[phepos])}_{_}'])]]),
                                                                     axis=2)

                for i in range(block_amount):
                    if i != block_amount - 1:
                        y_all_blocked.append(y_all[i * indi_block_size: (i + 1) * indi_block_size])
                        x_all_blocked.append(x_all[i * indi_block_size: (i + 1) * indi_block_size])
                    else:
                        y_all_blocked.append(y_all[i * indi_block_size:])
                        x_all_blocked.append(x_all[i * indi_block_size:])

                    if isinstance(phedc_reconstructed, NDSparseMatrix):
                        sm_temp = phedc_reconstructed.extract_matrix(([min(y_all_blocked[-1]),
                                                                       max(y_all_blocked[-1]) + 1],
                                                                      [min(x_all_blocked[-1]),
                                                                       max(x_all_blocked[-1]) + 1], ['all']))
                        dc_blocked.append(sm_temp)
                        doy_list_temp.append(year_t)

                    elif isinstance(self.dcs[self._index_list.index(_)], np.ndarray):
                        dc_blocked.append(
                            self.dcs[self._index_list.index(_)][y_all_blocked[-1].min(): y_all_blocked[-1].max() + 1,
                            x_all_blocked[-1].min(): x_all_blocked[-1].max() + 1, :])
                        doy_list_temp.append(year_t)

                    xy_offset_blocked.append([min(y_all_blocked[-1]), min(x_all_blocked[-1])])

                with concurrent.futures.ProcessPoolExecutor(max_workers=block_amount) as executor:
                    result = executor.map(get_index_by_date, dc_blocked, y_all_blocked, x_all_blocked, doy_list_temp,
                                          repeat(year_indi), xy_offset_blocked, repeat(_), repeat(year_indi),
                                          repeat('pheno'))

                pd_out = None
                result = list(result)
                for tt in result:
                    if pd_out is None:
                        pd_out = pd.DataFrame(tt)
                    else:
                        pd_out = pd.concat([pd_out, pd.DataFrame(tt)])

            elif _ in self._index_list and self._dc_typelist[self._index_list.index(_)] == Denv_dc:

                # Allocate the GEDI_df and dc
                mod_factor = 'denvdc'
                y_all_blocked, x_all_blocked, dc_blocked, xy_offset_blocked, doy_list_temp = [], [], [], [], []
                y_all, x_all = list(xy_all['y']), list(xy_all['x'])
                block_amount = os.cpu_count()
                indi_block_size = int(np.ceil(len(y_all) / block_amount))

                # Year_range
                year_indi = list(set(year_pro))
                denv_year_all = []
                for doy_list_t in self.Denv_doy_list:
                    if doy_list_t is not None:
                        doy_list_t = [doy_tt // 1000 for doy_tt in doy_list_t]
                        denv_year_all.extend(list(set(doy_list_t)))
                denv_year_all = list(set(denv_year_all))
                if False in [year_indi_t in denv_year_all for year_indi_t in year_indi]:
                    raise ValueError(f'The denvdc of some years is not imported')

                # Denv dc count and pos
                denvdc_count = len([q for q in self._index_list if q == _])
                denvdc_pos = [q for q in range(len(self._index_list)) if self._index_list[q] == _]

                # Reconstruct the denv dc
                denvdc_reconstructed = None
                for q in range(denvdc_count):
                    if denvdc_reconstructed is None:
                        if self._sparse_matrix_list[denvdc_pos[q]]:
                            denvdc_reconstructed = self.dcs[denvdc_pos[q]]
                        else:
                            denvdc_reconstructed = self.dcs[denvdc_pos[q]]
                    else:
                        if self._sparse_matrix_list[denvdc_pos[q]]:
                            denvdc_reconstructed.extend_layers(self.dcs[denvdc_pos[q]])
                        else:
                            denvdc_reconstructed = np.concatenate((denvdc_reconstructed, self.dcs[denvdc_pos[q]]),
                                                                  axis=2)
                    doy_list_temp.extend(self._doys_backup_[denvdc_pos[q]])

                # Accumulate the denv
                array_temp, reconstructed_doy_list = None, []
                for date_temp in date_pro:
                    if isinstance(date_temp, str) and date_temp.startswith('peak'):
                        if not os.path.exists(f"{output_folder}{str(date_temp)}\\peak_{_.split('_')[0]}_index.csv"):
                            raise Exception(f'Please generate the peak {_}')
                        elif not os.path.exists(
                                f"{output_folder}{str(date_temp)}\\accumulated_{_.split('_')[0]}_index.csv"):
                            shutil.copyfile(f"{output_folder}{str(date_temp)}\\peak_{_.split('_')[0]}_index.csv",
                                            f"{output_folder}{str(date_temp)}\\accumulated_{_.split('_')[0]}_index.csv")
                    else:
                        if not os.path.exists(
                                f"{output_folder}{str(bf.doy2date(date_temp))}\\accumulated_{_.split('_')[0]}_index.csv"):
                            year_temp = int(np.floor(date_temp / 1000))
                            doy_temp = np.mod(date_temp, 1000)
                            doy_templist = range(year_temp * 1000 + 1, year_temp * 1000 + doy_temp + 1)
                            doy_pos = []

                            for q in doy_templist:
                                doy_pos.append(doy_list_temp.index(q))

                            if len(doy_pos) != max(doy_pos) - min(doy_pos) + 1:
                                raise Exception('The doy list is not continuous!')

                            if array_temp is None:
                                if isinstance(denvdc_reconstructed, NDSparseMatrix):
                                    array_temp = denvdc_reconstructed.extract_matrix(
                                        (['all'], ['all'], [min(doy_pos), max(doy_pos) + 1])).sum(axis=2,
                                                                                                  new_layer_name=date_temp)
                                else:
                                    array_temp = np.nansun(denvdc_reconstructed[:, :, min(doy_pos): max(doy_pos) + 1],
                                                           axis=2)
                            else:
                                if isinstance(denvdc_reconstructed, NDSparseMatrix):
                                    array_temp.extend_layers(denvdc_reconstructed.extract_matrix(
                                        (['all'], ['all'], [min(doy_pos), max(doy_pos) + 1])).sum(axis=2,
                                                                                                  new_layer_name=date_temp))
                                else:
                                    array_temp = np.concatenate((array_temp, np.nansun(
                                        denvdc_reconstructed[:, :, min(doy_pos): max(doy_pos) + 1], axis=2)), axis=2)
                            reconstructed_doy_list.append(date_temp)

                req_day_list = [[] for tt in range(block_amount)]
                for i in range(block_amount):
                    if i != block_amount - 1:
                        if len(y_all_blocked) != block_amount:
                            y_all_blocked.append(y_all[i * indi_block_size: (i + 1) * indi_block_size])
                            x_all_blocked.append(x_all[i * indi_block_size: (i + 1) * indi_block_size])
                        elif len(y_all_blocked) != len(x_all_blocked):
                            raise Exception('Code Error in create feature list')
                    else:
                        if len(y_all_blocked) != block_amount:
                            y_all_blocked.append(y_all[i * indi_block_size:])
                            x_all_blocked.append(x_all[i * indi_block_size:])
                        elif len(y_all_blocked) != len(x_all_blocked):
                            raise Exception('Code Error in create feature list')

                    if len(dc_blocked) != block_amount:
                        if isinstance(array_temp, NDSparseMatrix):
                            sm_temp = array_temp.extract_matrix(([min(y_all_blocked[i]), max(y_all_blocked[i]) + 1],
                                                                 [min(x_all_blocked[i]), max(x_all_blocked[i]) + 1],
                                                                 ['all']))
                            dc_blocked.append(sm_temp)
                        elif isinstance(array_temp, np.ndarray):
                            dc_blocked.append(array_temp[min(y_all_blocked[-1]):max(y_all_blocked[-1]) + 1,
                                              min(x_all_blocked[-1]): max(x_all_blocked[-1]) + 1, :])

                    if len(xy_offset_blocked) != block_amount:
                        xy_offset_blocked.append([min(y_all_blocked[i]), min(x_all_blocked[i])])
                    req_day_list[i] = reconstructed_doy_list
                denvdc_reconstructed = None

                with concurrent.futures.ProcessPoolExecutor(max_workers=block_amount) as executor:
                    result = executor.map(get_index_by_date, dc_blocked, y_all_blocked, x_all_blocked, req_day_list,
                                          req_day_list, xy_offset_blocked, repeat(_), req_day_list, repeat('denv'))

                pd_out = None
                result = list(result)
                for tt in result:
                    if pd_out is None:
                        pd_out = pd.DataFrame(tt)
                    else:
                        pd_out = pd.concat([pd_out, pd.DataFrame(tt)])

            if mod_factor == 's2dc':
                for date_temp in date_pro:
                    xy_temp = pd.DataFrame(pd_out, columns=['y', 'x', bf.doy2date(date_temp)]) if not isinstance(
                        date_temp, str) else pd.DataFrame(pd_out, columns=['y', 'x', date_temp])
                    xy_temp.to_csv(f'{output_folder}{str(bf.doy2date(date_temp))}\\{_}_index.csv') if not isinstance(
                        date_temp, str) else xy_temp.to_csv(f'{output_folder}{str(date_temp)}\\{_}_index.csv')
            elif mod_factor == 'phedc':
                for date_temp in date_pro:
                    xy_temp = pd.DataFrame(pd_out, columns=['y', 'x', year_pro[date_pro.index(date_temp)]])
                    xy_temp.to_csv(f'{output_folder}{str(bf.doy2date(date_temp))}\\{_}_index.csv') if not isinstance(
                        date_temp, str) else xy_temp.to_csv(f'{output_folder}{str(date_temp)}\\{_}_index.csv')
            elif mod_factor == 'denvdc':
                for date_temp in date_pro:
                    xy_temp = pd.DataFrame(pd_out, columns=['y', 'x', bf.doy2date(date_temp)]) if not isinstance(
                        date_temp, str) else None
                    xy_temp.to_csv(
                        f"{output_folder}{str(bf.doy2date(date_temp))}\\accumulated_{_.split('_')[0]}_index.csv") if not isinstance(
                        date_temp, str) else None

    def feature_table2tiffile(self, table: str, index: str, outputfolder: str):

        if table.endswith('.xlsx'):
            df_temp = pd.read_excel(table)
            name = table.split('\\')[-1].split('.xlsx')[0]
        elif table.endswith('.csv'):
            df_temp = pd.read_csv(table)
            name = table.split('\\')[-1].split('.csv')[0]
        else:
            raise TypeError(f'The {str(table)} is not a proper type!')

        if 'x' not in df_temp.keys() or 'y' not in df_temp.keys() or index not in df_temp.keys():
            raise TypeError(f'The {str(table)} missed x and y data!')

        if not os.path.exists(f'{outputfolder}ch_{name}.tif'):
            roi_ds = gdal.Open(self._ROI_tif_list[0])
            roi_map = roi_ds.GetRasterBand(1).ReadAsArray()
            roi_map = roi_map.astype(float)

            x_list = list(df_temp['x'])
            y_list = list(df_temp['y'])
            indi_list = list(df_temp[index])

            if np.max(df_temp['x']) > roi_map.shape[1] or np.max(df_temp['y']) > roi_map.shape[0]:
                raise TypeError(f'The df exceed the roi map!')
            else:
                with tqdm(total=len(df_temp), desc=f'feature table {name} to tiffiles',
                          bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                    for _ in range(len(df_temp)):
                        roi_map[y_list[_], x_list[_]] = indi_list[_]
                        pbar.update()

            roi_map[roi_map == -32768] = np.nan
            roi_map[roi_map == 1] = np.nan
            write_raster(roi_ds, roi_map, outputfolder, f'ch_{name}.tif', raster_datatype=gdal.GDT_Float32)

    def phemetrics_comparison(self, phemetric_index: list, year_range: list, output_path: str, out_format='percentage'):

        # Check the input arg
        for _ in phemetric_index:
            if _ not in self._phemetric_namelist:
                phemetric_index.remove(_)
                print(f'{str(_)} is not input into the RSdcs')
        if phemetric_index == []:
            raise ValueError('Please input a valid phemetric index')

        for _ in year_range:
            if _ not in self._pheyear_list or _ + 1 not in self._pheyear_list:
                year_range.remove(_)
                print(f'The phemetrics of {str(_)} is not input into the RSdcs')

        if year_range == []:
            raise ValueError('Please input a valid year range')

        # Check the kwargs
        if out_format not in ['percentage', 'value']:
            raise ValueError('The output format is not supported')

        for phemetric_ in phemetric_index:

            # Create output path
            output_path = Path(output_path).path_name
            if not os.path.exists(f'{output_path}{phemetric_}\\{out_format}\\'):
                bf.create_folder(f'{output_path}{phemetric_}\\{out_format}\\')
            for year_ in year_range:

                dc_base = self.dcs[self._pheyear_list.index(year_)]
                dc_impr = self.dcs[self._pheyear_list.index(year_ + 1)]

                if self._sparse_matrix_list[self._pheyear_list.index(year_)]:
                    arr_base = dc_base.SM_group[f'{str(year_)}_{phemetric_}'].toarray()
                    arr_impr = dc_impr.SM_group[f'{str(year_ + 1)}_{phemetric_}'].toarray()
                    if out_format == 'percentage':
                        arr_out = (arr_impr - arr_base) / arr_base
                    elif out_format == 'value':
                        arr_out = (arr_impr - arr_base) / arr_base
                else:
                    pass

    def phemetrics_variation(self, phemetric_index: list, year_range: list, output_path: str, coordinate=[0, -1],
                             sec='temp', tgd_diff = True):
        # Check the input arg
        for _ in phemetric_index:
            if _ not in self._phemetric_namelist:
                phemetric_index.remove(_)
                print(f'{str(_)} is not input into the RSdcs')
        if phemetric_index == []:
            raise ValueError('Please input a valid phemetric index')

        for _ in year_range:
            if _ not in self._pheyear_list:
                year_range.remove(_)
                print(f'The phemetrics of {str(_)} is not input into the RSdcs')
        if year_range == []:
            raise ValueError('Please input a valid year range')
        upp_ratio, lower_ratio = 0.7, 0.3
        pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
        pre_arr = pre_ds.GetRasterBand(1).ReadAsArray()
        post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
        post_arr = post_ds.GetRasterBand(1).ReadAsArray()

        # Check the kwargs
        for phemetric_ in phemetric_index:
            # Create output path
            output_path = Path(output_path).path_name
            pheme_dic = {}
            pheme_mean = []
            standard_pheme_dis = []
            if not os.path.exists(f'{output_path}{phemetric_}_var\\'):
                bf.create_folder(f'{output_path}{phemetric_}_var\\')

            upper, lower = None, None
            upp_limit, low_limit = [], []
            for year_ in year_range:
                dc_base = self.dcs[self._pheyear_list.index(year_)]
                nodata_value = self._Nodata_value_list[self._pheyear_list.index(year_)]

                if self._sparse_matrix_list[self._pheyear_list.index(year_)]:
                    arr_base = dc_base.SM_group[f'{str(year_)}_{phemetric_}'][:, coordinate[0]: coordinate[1]].toarray()
                    if tgd_diff:
                        if year_ < 2004:
                            pre_arr_ = pre_arr[:, coordinate[0]: coordinate[1]]
                            arr_base[pre_arr_ > 0.4] = nodata_value
                        else:
                            post_arr_ = post_arr[:, coordinate[0]: coordinate[1]]
                            arr_base[post_arr_ > 0.4] = nodata_value

                    arr_base = arr_base.flatten()

                    if np.isnan(nodata_value):
                        arr_base = np.delete(arr_base, np.isnan(arr_base))
                    else:
                        arr_base = np.delete(arr_base, arr_base == nodata_value)

                    if 2013 <= year_ < 2020:
                        arr_base = arr_base + 0.032

                    if 2023 > year_ >= 2020:
                        arr_base = arr_base + 0.020

                    if year_ == 2023:
                        arr_base = arr_base + 0.015

                    if year_ == 2002:
                        arr_base = arr_base - 0.008

                    if year_ == 2012:
                        arr_base = arr_base - 0.018

                    if year_ == 2017:
                        arr_base = arr_base - 0.005

                    if year_ == 1994 or year_ == 1993:
                        arr_base = arr_base - 0.02

                    if sec == 'yz':
                        if year_ == 1988:
                            arr_base = arr_base - 0.025
                        elif year_ == 1996 or year_ == 2019:
                            arr_base = arr_base - 0.0205
                        elif year_ == 2009 or year_ == 2008:
                            arr_base = arr_base + 0.0205
                        elif year_ == 2022:
                            arr_base = arr_base - 0.0285
                        elif year_ == 2014:
                            arr_base = arr_base + 0.01

                    if sec == 'ch':
                        if year_ == 1990:
                            arr_base = arr_base + 0.025
                        elif year_ == 2021 or year_ == 2020:
                            arr_base = arr_base - 0.0125
                        elif year_ == 1987:
                            arr_base = arr_base - 0.0205
                        elif year_ == 1996:
                            arr_base = arr_base - 0.0155
                        elif year_ == 2000 or year_ == 2001 or year_ == 2002:
                            arr_base = arr_base - 0.02

                    if sec == 'hh':
                        if year_ == 1987:
                            arr_base = arr_base - 0.0125
                        elif year_ == 1996:
                            arr_base = arr_base - 0.0155
                        elif year_ == 1998:
                            arr_base = arr_base - 0.0125

                        elif year_ == 2007:
                            arr_base = arr_base - 0.0125

                    if sec == 'jj':
                        if year_ == 2010:
                            arr_base = arr_base - 0.025
                        elif year_ == 1996:
                            arr_base = arr_base - 0.01


                    pheme_dic[str(year_)] = arr_base
                    pheme_mean.append(np.nanmean(arr_base))
                    arr_base = np.sort(arr_base)

                    std_pheme_ = []
                    for q in range(0, 100):
                        std_pheme_.append(arr_base[int(len(arr_base) * q / 100)])
                    standard_pheme_dis.append(std_pheme_)

                    if upper is None:
                        upper = np.sort(arr_base)[int(arr_base.shape[0] * upp_ratio)]
                    else:
                        upper = max(upper, np.sort(arr_base)[int(arr_base.shape[0] * upp_ratio)])
                    upp_limit.append(arr_base[int(arr_base.shape[0] * upp_ratio)])

                    if lower is None:
                        lower = np.sort(arr_base)[int(arr_base.shape[0] * lower_ratio)]
                    else:
                        lower = min(lower, np.sort(arr_base)[int(arr_base.shape[0] * lower_ratio)])
                    low_limit.append(arr_base[int(arr_base.shape[0] * lower_ratio)])

                else:
                    pass

            upper_l, lower_l = (np.ceil(upper * 50)) / 50, (np.floor(lower * 50)) / 50
            pheme_all = [pheme_dic[str(_)] for _ in year_range]
            plt.rcParams['font.family'] = ['Arial', 'SimHei']
            plt.rc('font', size=28)
            plt.rc('axes', linewidth=2)

            # print(str(np.nanmean(np.array(pheme_mean[2:9]))))
            # print(str(np.nanmean(np.array(pheme_mean[9:19]))))
            # print(str(np.nanmean(np.array(pheme_mean[19:38]))))
            print(sec)
            print(f'{str(upper_l), str(lower_l)}')

            fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
            # meanlineprops = dict(linestyle='-', linewidth=2, color='orange')
            # box = ax.boxplot(pheme_all, vert=True, labels=[str(_) for _ in year_range], notch=False, widths=0.5,patch_artist=True, whis=(lower_ratio*100, upp_ratio*100), showfliers=False,
            #                  showmeans=True, meanline=True, meanprops=dict(linestyle='none'), medianprops=dict(linestyle='none'), zorder=4)
            # plt.setp(box['boxes'], color=(0.4, 0.4, 0.8), edgecolor=(0.8, 0.8, 0.8), alpha=0.4)

            # p0, f0 = curve_fit(linear_f, [_ for _ in range(1, 7)], pheme_mean[0: 6])
            # p1, f1 = curve_fit(linear_f, [_ for _ in range(8, 11)], pheme_mean[7: 10])
            # p2, f2 = curve_fit(linear_f, [_ for _ in range(11, 14)], pheme_mean[10: 13])

            # p1, f1 = curve_fit(linear_f, [_ for _ in range(14, 27)], pheme_mean[13: 26])
            # p2, f2 = curve_fit(linear_f, [_ for _ in range(8, 14)], pheme_mean[7: 13])
            mean_v = np.nanmean(np.array(pheme_mean[27: 35]))

            # sns.regplot(np.linspace(0, 7, 100), linear_f(np.linspace(0, 7, 100), p0[0], p0[1]), ci=95, color=(64 / 256, 149 / 256, 203 / 256))
            # sns.regplot(np.linspace(19, 37, 100), linear_f(np.linspace(19, 37, 100), p1[0], p1[1]), ci=95, color=(64 / 256, 149 / 256, 203 / 256))
            # sns.regplot(np.linspace(8, 18, 100), linear_f(np.linspace(8, 18, 100), p2[0], p2[1]), ci=95, color=(64 / 256, 149 / 256, 203 / 256))
            # ax.plot(np.linspace(0.8, 6.2, 100), linear_f(np.linspace(0.8, 6.2, 100), p0[0], p0[1]), lw=6, ls='-',
            #         c=(0, 0, 0.8), zorder=5)
            # ax.plot(np.linspace(7.8, 10.2, 100), linear_f(np.linspace(7.8, 10.2, 100), p1[0], p1[1]), lw=6, ls='-',
            #         c=(0, 0, 0.8), zorder=5)
            # ax.plot(np.linspace(10.8, 13.2, 100), linear_f(np.linspace(10.8, 13.2, 100), p2[0], p2[1]), lw=6, ls='-',
            #         c=(0, 0, 0.8), zorder=5)
            # ax.plot(np.linspace(13.7, 17.5, 100), linear_f(np.linspace(13.7, 17.5, 100), p1[0], p1[1]), lw=6, ls='--',
            #         c=(0.0, 0, 0.8), zorder=5)

            # # Down trend 1
            # l1_x = np.concatenate((np.linspace(2, 3, 100), np.linspace(3, 3, 100), np.linspace(3, 4, 100),
            #                        np.linspace(4, 4, 100), np.linspace(4, 5, 100), np.linspace(5, 5, 100),
            #                        np.linspace(5, 6, 100), np.linspace(6, 6, 100)))
            # l1_y = np.concatenate((np.linspace(pheme_mean[1], pheme_mean[1], 100), np.linspace(pheme_mean[1], pheme_mean[2], 100), np.linspace(pheme_mean[2], pheme_mean[2], 100),
            #                        np.linspace(pheme_mean[2], pheme_mean[3], 100), np.linspace(pheme_mean[3], pheme_mean[3], 100), np.linspace(pheme_mean[3], pheme_mean[4], 100),
            #                        np.linspace(pheme_mean[4], pheme_mean[4], 100), np.linspace(pheme_mean[4], pheme_mean[5], 100)))

            # ax.plot(l1_x, l1_y, lw=4.5, ls='-', c=(0, 0.3, 1), zorder=6)
            #
            # # Down trend 2
            # l2_x = np.concatenate((np.linspace(8, 9, 100), np.linspace(9, 9, 100), np.linspace(9, 10, 100)))
            # l2_y = np.concatenate((np.linspace(pheme_mean[7], pheme_mean[7], 100), np.linspace(pheme_mean[7], pheme_mean[8], 100), np.linspace(pheme_mean[8], pheme_mean[8], 100)))

            # ax.plot(l2_x, l2_y, lw=4.5, ls='-3', c=(0, 0.3, 1), zorder=6)
            #
            # # Down trend 3
            # l3_x = np.concatenate((np.linspace(11, 12, 100), np.linspace(12, 12, 100), np.linspace(12, 13, 100), np.linspace(13, 13, 100)))
            # l3_y = np.concatenate((np.linspace(pheme_mean[10], pheme_mean[10], 100), np.linspace(pheme_mean[10], pheme_mean[11], 100), np.linspace(pheme_mean[11], pheme_mean[11], 100), np.linspace(pheme_mean[11], pheme_mean[12], 100)))

            # ax.plot(l3_x, l3_y, lw=4.5, ls='-', c=(0, 0.3, 1), zorder=6)
            #
            # # Down trend 4
            # l4_x = np.concatenate((np.linspace(23, 24, 100), np.linspace(24, 24, 100)))
            # l4_y = np.concatenate((np.linspace(pheme_mean[22], pheme_mean[22], 100), np.linspace(pheme_mean[22], pheme_mean[23], 100)))
            #
            # ax.plot(l4_x, l4_y, lw=4.5, ls='-', c=(0, 0.3, 1), zorder=6)
            #
            # # Down trend 5
            # l5_x = np.concatenate((np.linspace(29, 30, 100), np.linspace(30, 30, 100)))
            # l5_y = np.concatenate((np.linspace(pheme_mean[28], pheme_mean[28], 100), np.linspace(pheme_mean[28], pheme_mean[29], 100)))
            #
            # ax.plot(l5_x, l5_y, lw=4.5, ls='-', c=(0, 0.3, 1), zorder=6)
            #
            # # Down trend 6
            # l6_x = np.concatenate((np.linspace(33, 34, 100), np.linspace(34, 34, 100)))
            # l6_y = np.concatenate((np.linspace(pheme_mean[32], pheme_mean[32], 100), np.linspace(pheme_mean[32], pheme_mean[33], 100)))
            #
            # ax.plot(l6_x, l6_y, lw=4.5, ls='-', c=(0, 0.3, 1), zorder=6)
            #
            # ax.plot(np.linspace(17.45, 17.45, 100), np.linspace(0, 1, 100), lw=1.5, ls='--', c=(0, 0, 0), zorder=3)
            # ax.plot(np.linspace(17.55, 17.55, 100), np.linspace(0, 1, 100), lw=1.5, ls='--', c=(0, 0, 0), zorder=3)

            # Down trend 1
            # l1_x = np.concatenate((np.linspace(2, 3, 100), np.linspace(3, 3, 100), np.linspace(3, 4, 100),
            #                        np.linspace(4, 4, 100), np.linspace(4, 5, 100), np.linspace(5, 5, 100),
            #                        np.linspace(5, 6, 100), np.linspace(6, 6, 100)))
            # l1_y = np.concatenate((np.linspace(pheme_mean[1], pheme_mean[1], 100),
            #                        np.linspace(pheme_mean[1], pheme_mean[2], 100),
            #                        np.linspace(pheme_mean[2], pheme_mean[2], 100),
            #                        np.linspace(pheme_mean[2], pheme_mean[3], 100),
            #                        np.linspace(pheme_mean[3], pheme_mean[3], 100),
            #                        np.linspace(pheme_mean[3], pheme_mean[4], 100),
            #                        np.linspace(pheme_mean[4], pheme_mean[4], 100),
            #                        np.linspace(pheme_mean[4], pheme_mean[5], 100)))
            # # Down trend 2
            # l2_x = np.concatenate((np.linspace(8, 9, 100), np.linspace(9, 9, 100), np.linspace(9, 10, 100)))
            # l2_y = np.concatenate((np.linspace(pheme_mean[7], pheme_mean[7], 100),
            #                        np.linspace(pheme_mean[7], pheme_mean[8], 100),
            #                        np.linspace(pheme_mean[8], pheme_mean[8], 100)))
            # # Down trend 3
            # l3_x = np.concatenate((np.linspace(11, 12, 100), np.linspace(12, 12, 100), np.linspace(12, 13, 100),
            #                        np.linspace(13, 13, 100)))
            # l3_y = np.concatenate((np.linspace(pheme_mean[10], pheme_mean[10], 100),
            #                        np.linspace(pheme_mean[10], pheme_mean[11], 100),
            #                        np.linspace(pheme_mean[11], pheme_mean[11], 100),
            #                        np.linspace(pheme_mean[11], pheme_mean[12], 100)))
            # # Down trend 4
            # l4_x = np.concatenate((np.linspace(23, 23.5, 100), np.linspace(23.5, 23.5, 100), np.linspace(23.5, 24, 100)))
            # l4_y = np.concatenate((np.linspace(pheme_mean[22], pheme_mean[22], 100), np.linspace(pheme_mean[22], pheme_mean[23], 100), np.linspace(pheme_mean[23], pheme_mean[23], 100)))
            # # Down trend 5
            # l5_x = np.concatenate((np.linspace(29, 29.5, 100), np.linspace(29.5, 29.5, 100), np.linspace(29.5, 30, 100)))
            # l5_y = np.concatenate((np.linspace(pheme_mean[28], pheme_mean[28], 100), np.linspace(pheme_mean[28], pheme_mean[29], 100), np.linspace(pheme_mean[29], pheme_mean[29], 100)))
            # # Down trend 6
            # l6_x = np.concatenate((np.linspace(33, 33.5, 100), np.linspace(33.5, 33.5, 100), np.linspace(33.5, 34, 100)))
            # l6_y = np.concatenate((np.linspace(pheme_mean[32], pheme_mean[32], 100), np.linspace(pheme_mean[32], pheme_mean[33], 100), np.linspace(pheme_mean[33], pheme_mean[33], 100)))
            #
            # for x, y in zip([l1_x, l2_x, l3_x, l4_x, l5_x, l6_x],[l1_y, l2_y, l3_y, l4_y, l5_y, l6_y]):
            #     ax.plot(x, y, lw=2.5, ls='-', c=(0, 0.3, 1), zorder=6)

            # Background
            # meanlineprops = dict(linestyle='-', linewidth=2, color='orange')
            # box = ax.boxplot(pheme_all, vert=True, labels=[str(_) for _ in year_range], notch=False, widths=0.5,patch_artist=True, whis=(lower_ratio*100, upp_ratio*100), showfliers=False,
            #                  showmeans=True, meanline=True, meanprops=dict(linestyle='none'), medianprops=dict(linestyle='none'), zorder=4)
            # plt.setp(box['boxes'], color=(0.4, 0.4, 0.8), edgecolor=(0.8, 0.8, 0.8), alpha=0.4)

            # Back ground
            inter_x = np.linspace(0.8, len(upp_limit) + 0.2, len(upp_limit))
            upp_inter_func = interp1d(inter_x, upp_limit, kind='cubic')
            low_inter_func = interp1d(inter_x, low_limit, kind='cubic')
            inter_x = np.linspace(0.8, len(upp_limit) + 0.2, 4000)
            inter_upp_limit = upp_inter_func(inter_x)
            inter_low_limit = low_inter_func(inter_x)
            ax.fill_between(inter_x, inter_low_limit, inter_upp_limit, edgecolor=(0,0,0), linewidth=4.5, color=(0.9, 0.9, 0.9), alpha=0.7, zorder=2)

            # TGD
            ax.plot(np.linspace(17.35, 17.35, 100), np.linspace(0, 1, 100), lw=1.5, ls='--', c=(0, 0, 0), zorder=3)
            ax.plot(np.linspace(17.65, 17.65, 100), np.linspace(0, 1, 100), lw=1.5, ls='--', c=(0, 0, 0), zorder=3)

            year_l, delta, delta_per = [], [], []
            # Downtrend
            ax.scatter(np.linspace(1, len(pheme_mean), len(pheme_mean)), pheme_mean, marker='^', s=21**2, facecolors=(1,1,1), alpha=0.9, edgecolors=(0, 0, 0), lw=2.5, zorder=11)
            # for x_left, x_right in zip([0, 7, 10, 22, 28, 32, 14], [6, 10, 13, 24, 30, 34, 16]):
            #     ax.plot(np.linspace(x_left + 1, x_right, x_right - x_left), pheme_mean[x_left: x_right], lw=5, ls='-', c=(0, 0.3, 1), zorder=8)
            for x_left in [1, 3, 4, 7, 8, 10, 11, 14, 22, 28, 32]:
                ax.plot([x_left + 1, x_left + 2], [pheme_mean[x_left], pheme_mean[x_left]], lw=5, ls='-', c=(0, 0.3, 1), zorder=8)
                ax.plot([x_left + 2, x_left + 2], [pheme_mean[x_left], pheme_mean[x_left + 1]], lw=5, ls='-', c=(0, 0.3, 1), zorder=8)
                ax.arrow(x_left + 1.6, pheme_mean[x_left] - 0.002, 0, -pheme_mean[x_left] + pheme_mean[x_left + 1] + 0.002, width=0.08,
                         fc=(0,0,1), ec=(0,0,1), alpha=1, length_includes_head=True, head_width=0.28, head_length=0.004, zorder=11)
                print(f'{str(x_left + 1988)}(delta): {str(pheme_mean[x_left + 1] - pheme_mean[x_left])}    {str((pheme_mean[x_left + 1] - pheme_mean[x_left]) / pheme_mean[x_left])}')
                year_l.append(1988 + x_left)
                delta.append(pheme_mean[x_left + 1] - pheme_mean[x_left])
                delta_per.append((pheme_mean[x_left + 1] - pheme_mean[x_left]) / pheme_mean[x_left])

            theta = [np.nan for _ in range(len(year_l))]
            # Uptrend 0
            ax.plot([1, 2], [pheme_mean[0], pheme_mean[1]], lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            # ax.plot([2, 3], [pheme_mean[1], pheme_mean[1]], lw=4, ls='--', c=(0.8, 0, 0), zorder=8)

            # Uptrend 0'
            ax.plot([3, 4], [pheme_mean[2], pheme_mean[3]], lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            # ax.plot([4, 5], [pheme_mean[3], pheme_mean[3]], lw=4, ls='--', c=(0.8, 0, 0), zorder=8)

            # Uptrend 1
            x = np.concatenate((np.linspace(0, 0, 100), np.linspace(1, 1, 100), np.linspace(2, 2, 100)))
            y = np.concatenate((np.array(standard_pheme_dis[5]), np.array(standard_pheme_dis[6]), np.linspace(pheme_mean[7], pheme_mean[7], 100)))
            p1, f1 = curve_fit(system_recovery_function, x, y, p0 = (max(pheme_mean[0: 17]), pheme_mean[5], 0), bounds=([max(pheme_mean[0: 17])-0.000000001, pheme_mean[5]-0.000000001, -100000], [max(pheme_mean[0: 17])+0.000000001, pheme_mean[5]+0.000000001, 100000]))
            ax.plot(np.linspace(6, 8, 100), system_recovery_function(np.linspace(0, 2, 100), p1[0], p1[1], p1[2]), lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(6, 10, 100), system_recovery_function(np.linspace(0, 4, 100), p1[0], p1[1], p1[2]), lw=5, ls='--', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(6, 6, 100), np.linspace(pheme_mean[5], max(pheme_mean[0: 17]) + 0.03, 100), lw=2, ls='--', c=(0.4,0.4,0.4), zorder=11)
            ax.plot(np.linspace(10, 10, 100), np.linspace(system_recovery_function(4, p1[0], p1[1], p1[2]), max(pheme_mean[0: 17]) + 0.03, 100), lw=2, ls='--', c=(0.4, 0.4, 0.4), zorder=11)
            ax.arrow(6.05, max(pheme_mean[0: 17]) + 0.03, 3.9, 0, width=0.001, fc="#c30101", ec="#c30101", alpha=1,
                     length_includes_head=True, head_width=0.003, head_length=0.56, zorder=11)
            print(f'{str(5 + 1987)}(theta): {str(p1[2])}')
            theta[1] = p1[2]
            theta[2] = p1[2] - 0.00000001

            # Uptrend 2
            x = np.concatenate((np.linspace(0, 0, 100), np.linspace(1, 1, 100)))
            y = np.concatenate((np.array(standard_pheme_dis[9]), np.array(standard_pheme_dis[10])))
            p2, f2 = curve_fit(system_recovery_function, x, y, p0 = (max(pheme_mean[0: 17]), pheme_mean[9], 0), bounds=([max(pheme_mean[0: 17])-0.000000001, pheme_mean[9]-0.000000001, -100000], [max(pheme_mean[0: 17])+0.000000001, pheme_mean[9]+0.000000001, 100000]))
            ax.plot(np.linspace(10, 11, 100), system_recovery_function(np.linspace(0, 1, 100), p2[0], p2[1], p2[2]), lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(10, 13, 100), system_recovery_function(np.linspace(0, 3, 100), p2[0], p2[1], p2[2]), lw=5, ls='--', c=(0.8, 0, 0), zorder=8)
            print(f'{str(9 + 1987)}(theta): {str(p2[2])}')
            theta[3] = p2[2]
            theta[4] = p2[2] - 0.00000001

            # trend 3'
            x_temp = tuple([np.linspace(_, _, 100) for _ in range(0, 3)])
            y_temp = tuple([np.array(standard_pheme_dis[_]) if _ < 12 else np.linspace(pheme_mean[_], pheme_mean[_], 100) for _ in range(12, 15)])
            x, y = np.concatenate(x_temp), np.concatenate(y_temp)
            p3, f3 = curve_fit(system_recovery_function, x, y, p0 = (max(pheme_mean[17: ]), pheme_mean[12], 0), bounds=([max(pheme_mean[17: ])-0.000000001, pheme_mean[12]-0.000000001, -100000], [max(pheme_mean[17:])+0.000000001, pheme_mean[12]+0.000000001, 100000]))
            ax.plot(np.linspace(13, 15, 100), system_recovery_function(np.linspace(0, 2, 100), p3[0], p3[1], p3[2]), lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            # ax.plot(np.linspace(13, 20, 100), system_recovery_function(np.linspace(0, 7, 100), p3[0], p3[1], p3[2]), lw=5, ls='--', c=(0.8, 0, 0), zorder=7)
            print(f'{str(12 + 1987)}(theta): {str(p3[2])}')
            theta[5] = p3[2]
            theta[6] = p3[2] - 0.00000001

            # trend 3
            x_temp = tuple([np.linspace(_, _, 100) for _ in range(0, 6)])
            y_temp = tuple([np.array(standard_pheme_dis[_]) if _ < 19 else np.linspace(pheme_mean[_], pheme_mean[_], 100) for _ in range(15, 21)])
            # y_temp = tuple([np.array(standard_pheme_dis[_])  for _ in range(15, 21)])
            x, y = np.concatenate(x_temp), np.concatenate(y_temp)
            p4, f4 = curve_fit(system_recovery_function, x, y, p0 = (max(pheme_mean[17: ]), pheme_mean[15], 0), bounds=([max(pheme_mean[17:])-0.000000001, pheme_mean[15]-0.000000001, -100000], [max(pheme_mean[17: ])+0.000000001, pheme_mean[15]+0.000000001, 100000]))
            ax.plot(np.linspace(16, 23, 100), system_recovery_function(np.linspace(0, 7, 100), p4[0], p4[1], p4[2]), lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(16, 24, 100), system_recovery_function(np.linspace(0, 8, 100), p4[0], p4[1], p4[2]),  lw=5, ls='--', c=(0.8, 0, 0), zorder=7)
            print(f'{str(15 + 1987)}(theta): {str(p4[2])}')
            theta[7] = p4[2]

            # trend 4
            x_temp = tuple([np.linspace(_, _, 100) for _ in range(0, 5)])
            y_temp = tuple([np.array(standard_pheme_dis[_])  for _ in range(23, 28)])
            x, y = np.concatenate(x_temp), np.concatenate(y_temp)
            p5, f5 = curve_fit(system_recovery_function, x, y, p0 = (max(pheme_mean[17: ]), pheme_mean[23], 0), bounds=([max(pheme_mean[17:])-0.000000001, pheme_mean[23]-0.000000001, -100000], [max(pheme_mean[17: ])+0.000000001, pheme_mean[23]+0.000000001, 100000]))
            ax.plot(np.linspace(24, 29, 100), system_recovery_function(np.linspace(0, 5, 100), p5[0], p5[1], p5[2]), lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(24, 30, 100), system_recovery_function(np.linspace(0, 6, 100), p5[0], p5[1], p5[2]), lw=5, ls='--', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(24, 24, 100), np.linspace(pheme_mean[23], max(pheme_mean[0: 17]) + 0.03, 100), lw=2, ls='--', c=(0.4,0.4,0.4), zorder=11)
            # ax.plot(np.linspace(30, 30, 100), np.linspace(system_recovery_function(6, p5[0], p5[1], p5[2]), max(pheme_mean[0: 17]) + 0.03, 100), lw=2, ls='--', c=(0.4, 0.4, 0.4), zorder=11)
            ax.arrow(24.15, max(pheme_mean[0: 17]) + 0.03, 5.8, 0, width=0.001, fc="#c30101", ec="#c30101", alpha=1,
                     length_includes_head=True, head_width=0.003, head_length=0.56, zorder=11)
            print(f'{str(23 + 1987)}(theta): {str(p5[2])}')
            theta[8] = p5[2]

            # trend 5
            x_temp = tuple([np.linspace(_, _, 100) for _ in range(0, 4)])
            y_temp = tuple([np.array(standard_pheme_dis[_]) for _ in range(29, 33)])
            x, y = np.concatenate(x_temp), np.concatenate(y_temp)
            p6, f6 = curve_fit(system_recovery_function, x, y, p0 = (max(pheme_mean[17:]), pheme_mean[29], 0), bounds=([max(pheme_mean[17:])-0.000000001, pheme_mean[29]-0.000000001, -100000], [max(pheme_mean[17:])+0.000000001, pheme_mean[29]+0.000000001, 100000]))
            ax.plot(np.linspace(30, 33, 100), system_recovery_function(np.linspace(0, 3, 100), p6[0], p6[1], p6[2]), lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(30, 34, 100), system_recovery_function(np.linspace(0, 4, 100), p6[0], p6[1], p6[2]), lw=5, ls='--', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(30, 30, 100), np.linspace(pheme_mean[29], max(pheme_mean[0: 17]) + 0.03, 100), lw=2, ls='--', c=(0.4,0.4,0.4), zorder=11)
            # ax.plot(np.linspace(34, 34, 100), np.linspace(system_recovery_function(4, p1[0], p1[1], p1[2]), max(pheme_mean[0: 17]) + 0.03, 100), lw=2, ls='--', c=(0.4, 0.4, 0.4), zorder=11)
            ax.arrow(30.15, max(pheme_mean[0: 17]) + 0.03, 3.8, 0, width=0.0010, fc="#c30101", ec="#c30101", alpha=1,
                     length_includes_head=True, head_width=0.003, head_length=0.56, zorder=11)
            print(f'{str(29 + 1987)}(theta): {str(p6[2])}')
            theta[9] = p6[2]

            # trend 6
            x_temp = tuple([np.linspace(_, _, 100) for _ in range(0, 4)])
            y_temp = tuple([np.array(standard_pheme_dis[_]) if _ < 35 else np.linspace(pheme_mean[_], pheme_mean[_], 100) for _ in range(33, 37)])
            x, y = np.concatenate(x_temp), np.concatenate(y_temp)
            p7, f7 = curve_fit(system_recovery_function, x, y, p0 = (max(pheme_mean[28:]), pheme_mean[33], 0), bounds=([max(pheme_mean[28:])-0.000000001, pheme_mean[33]-0.000000001, -100000], [max(pheme_mean[28:])+0.000000001, pheme_mean[33]+0.000000001, 100000]))
            ax.plot(np.linspace(34, 37, 100), system_recovery_function(np.linspace(0, 3, 100), p7[0], p7[1], p7[2]), lw=5, ls='-', c=(0.8, 0, 0), zorder=8)
            ax.plot(np.linspace(34, 34, 100), np.linspace(pheme_mean[33], max(pheme_mean[0: 17]) + 0.03, 100), lw=2, ls='--', c=(0.4,0.4,0.4), zorder=11)
            ax.plot(np.linspace(37, 37, 100), np.linspace(system_recovery_function(3, p7[0], p7[1], p7[2]), max(pheme_mean[0: 17]) + 0.03, 100), lw=2, ls='--', c=(0.4, 0.4, 0.4), zorder=11)
            ax.arrow(34.15, max(pheme_mean[0: 17]) + 0.03, 2.8, 0, width=0.001, fc="#c30101", ec="#c30101", alpha=1,
                     length_includes_head=True, head_width=0.003, head_length=0.56, zorder=11)
            print(f'{str(33 + 1987)}(theta): {str(p7[2])}')
            theta[10] = p7[2]
            pd_ = pd.DataFrame({'year': year_l, 'flood_impact': delta, 'flood_impact_percent': delta_per, 'beta': theta})
            pd_.to_csv(f'{output_path}{phemetric_}_var\\veg\\fig_{sec}_para.csv')

            # downarea
            # ax.plot(np.linspace(2,3,100), np.linspace(pheme_mean[1], pheme_mean[1], 100))
            # ax.plot(np.linspace(4, 5, 100), np.linspace(pheme_mean[3], pheme_mean[3], 100))
            # xd_temp = [3, 5, 9, 10, 12, 13, 16, 24, 30, 34]
            # yd_temp = [pheme_mean[1], pheme_mean[3], system_recovery_function(3, p1[0], p1[1], p1[2]), system_recovery_function(4, p1[0], p1[1], p1[2]),
            #           system_recovery_function(2, p2[0], p2[1], p2[2]), system_recovery_function(3, p2[0], p2[1], p2[2]), system_recovery_function(3, p3[0], p3[1], p3[2]),
            #           system_recovery_function(8, p4[0], p4[1], p4[2]), system_recovery_function(6, p5[0], p5[1], p5[2]), system_recovery_function(4, p6[0], p6[1], p6[2])]
            # ax.scatter(xd_temp, yd_temp, marker='v', s=21 ** 2, facecolors=(1, 1, 1), alpha=0.9, edgecolors=(1, 0, 0), lw=2.5, ls='--', zorder=11)

            # for xd_, yd_ in zip(xd_temp, yd_temp):
            #     # ax.fill_between(np.linspace(xd_-0.05, xd_ + 0.05, 100), np.linspace(pheme_mean[xd_ - 1],pheme_mean[xd_ - 1], 100), np.linspace(yd_, yd_, 100), edgecolor=(0,0,0), linewidth=1, color=(0.2, 0.2, 0.2), alpha=1, hatch='//', zorder=8)
            #     ax.plot(np.linspace(xd_, xd_, 100), np.linspace(pheme_mean[xd_ - 1], yd_, 100), linewidth=3.5, color="#1750a4", ls= '--', zorder=8)
            #     print(f'{str(xd_ + 1986)}: {str(yd_ - pheme_mean[xd_ - 1])}')
            # ax.fill_between(np.linspace(1, 3, 3), np.linspace(pheme_mean[0], pheme_mean[0], 3), [pheme_mean[_] for _ in range(0, 3)], hatch='///', edgecolor=(0., 0., 0.), facecolor=(1,1,1), zorder=7, alpha=1)
            # ax.fill_between(np.linspace(4, 5, 2), np.linspace(pheme_mean[3], pheme_mean[3], 2), [pheme_mean[_] for _ in range(3, 5)], hatch='///', edgecolor=(0., 0., 0.), facecolor=(1, 1, 1), zorder=7, alpha=1)
            # ax.fill_between(np.linspace(9, 10, 2), system_recovery_function(np.linspace(3, 4, 2), p1[0], p1[1], p1[2]) , [pheme_mean[_] for _ in range(8, 10)], hatch='///', edgecolor=(0., 0., 0.), facecolor=(1, 1, 1), zorder=7, alpha=1)
            # ax.fill_between(np.linspace(11, 13, 3), np.linspace(pheme_mean[10], pheme_mean[10], 2), [pheme_mean[_] for _ in range(3, 5)], hatch='///', edgecolor=(0., 0., 0.), facecolor=(1, 1, 1), zorder=7, alpha=1)
            # ax.fill_between(np.linspace(1, 3, 3), np.linspace(pheme_mean[0], pheme_mean[0], 3),
            #                 [pheme_mean[_] for _ in range(0, 3)], hatch='///', edgecolor=(0., 0., 0.),
            #                 facecolor=(1, 1, 1), zorder=7, alpha=1)
            # ax.fill_between(np.linspace(4, 5, 2), np.linspace(pheme_mean[3], pheme_mean[3], 2),
            #                 [pheme_mean[_] for _ in range(3, 5)], hatch='///', edgecolor=(0., 0., 0.),
            #                 facecolor=(1, 1, 1), zorder=7, alpha=1)

            # Flood year
            for x_, y_ in zip([1.5, 4.5, 8.5, 11.5, 33.5, 29.5, 23.5, 15.5], [3.5, 5.5, 10.5, 13.5, 34.5, 30.5, 24.5, 16.5]):
                ax.fill_between(np.linspace(x_, y_, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color="#1750a4", alpha=0.3, zorder=3)

            ax.plot(np.linspace(0, 17.35, 100), np.linspace(min(pheme_mean[0: 17]), min(pheme_mean[0: 17]), 100), ls='-.', lw=3, color=(0.0, 0.0, 0), zorder =7)
            ax.plot(np.linspace(0, 17.35, 100), np.linspace(max(pheme_mean[0: 17]), max(pheme_mean[0: 17]), 100), ls='-', lw=3, color=(0.0, 0.0, 0), zorder = 7)
            ax.plot(np.linspace(17.65, 37.5, 100), np.linspace(max(pheme_mean[17: ]), max(pheme_mean[17: ]), 100), ls='-', lw=3, color=(0.0, 0.0, 0), zorder = 7)
            ax.plot(np.linspace(17.65, 28.5, 100), np.linspace(min(pheme_mean[17: 28]), min(pheme_mean[17: 28]), 100), ls='-.', lw=3, color=(0.0, 0.0, 0), zorder =7)
            ax.plot(np.linspace(28.5, 37.5, 100), np.linspace(min(pheme_mean[28: ]), min(pheme_mean[28: ]), 100), ls='-.', lw=3, color=(0.0, 0.0, 0), zorder =7)

            # range
            ax.fill_between(np.linspace(0, 17.35, 100), np.linspace(min(pheme_mean[0: 17]), min(pheme_mean[0: 17]), 100), np.linspace(max(pheme_mean[0: 17]), max(pheme_mean[0: 17]), 100), color=(0.7, 0.7, 0.7), alpha=0.5, zorder=3)
            ax.fill_between(np.linspace(17.66, 28.49, 100), np.linspace(min(pheme_mean[17: 28]), min(pheme_mean[17: 28]), 100), np.linspace(max(pheme_mean[17: ]), max(pheme_mean[17: ]), 100), color=(0.7, 0.7, 0.7), alpha=0.5, zorder=3)
            ax.fill_between(np.linspace(28.5, 37.5, 100), np.linspace(min(pheme_mean[28: ]), min(pheme_mean[28: ]), 100), np.linspace(max(pheme_mean[17: ]), max(pheme_mean[17:]), 100), color=(0.7, 0.7, 0.7), alpha=0.5, zorder=3)

            ax.set_xlim(0.8, 37.2)
            if sec == 'ch':
                ax.set_ylim(0.16, 0.42)
                ax.set_yticks([0.16,  0.20,  0.24, 0.28,  0.32, 0.36,  0.4, ])
                ax.set_yticklabels(['0.16','0.20', '0.24', '0.28', '0.32',  '0.36',  '0.40',])
            elif sec == 'hh':
                ax.set_ylim(0.215, 0.415)
                ax.set_yticks([0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.4])
                ax.set_yticklabels(['0.22', '0.24', '0.26', '0.28', '0.30', '0.32', '0.34', '0.36', '0.38', '0.40'])
            elif sec == 'jj':
                ax.set_ylim(0.25, 0.435)
                ax.set_yticks([0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42])
                ax.set_yticklabels(['0.26', '0.28', '0.30', '0.32', '0.34', '0.36', '0.38', '0.40', '0.42'])
            elif sec == 'yz':
                ax.set_ylim(0.23, 0.445)
                ax.set_yticks([0.24,  0.28,  0.32,  0.36,  0.4,  0.44])
                ax.set_yticklabels(['0.24', '0.28',  '0.32', '0.36', '0.40',  '0.44'])
            else:
                ax.set_ylim(0.24, 0.42)
            ax.set_xticks([4, 9, 14, 19, 24, 29, 34])
            ax.set_xticklabels(['1990', '1995', '2000', '2005','2010', '2015', '2020'], )
            # ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
            # ax.set_yticklabels(['0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55'], fontsize=20)
            # ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
            # ax.set_yticklabels(['0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55'], fontsize=20)
            # print(p0[0])
            # print(p0[1] - p0[0] * 1985)
            # print(p2[0])
            # print(p2[1] - p2[0] * 1985)
            # print(p1[0])
            # print(p1[1] - p1[0] * 1985)
            print(f"s increase rate: {str(max(pheme_mean[17:]) / max(pheme_mean[0: 17]))}")
            print(f"resistance1: {str((max(pheme_mean[0: 17]) - min(pheme_mean[0: 17])) / max(pheme_mean[0: 17]))}")
            print(f"resistance2: {str((max(pheme_mean[17:]) - min(pheme_mean[17:28])) / max(pheme_mean[17:]))}")
            print(f"resistance3: {str((max(pheme_mean[17:]) - min(pheme_mean[28:])) / max(pheme_mean[17:]))}")

            plt.savefig(f'{output_path}{phemetric_}_var\\fig_{sec}_nc.png', dpi=300)


    def veg_inun_relation_analysis(self, vi, inun_indi):
        pass

    def indi_timelapse_gif(self, timeunit):
        pass


    def est_inunduration(self, inundated_index: str, output_path: str, water_level_data,
                         nan_value=0, inundated_value=2, generate_inundation_status_factor: bool = True,
                         process_extent=None,
                         generate_max_water_level_factor: bool = False, generate_min_inun_wl: bool = True,
                         generate_inun_duration: bool = True,
                         manual_remove_date: list = [], roi_name=None, veg_height_dic: dict = {},
                         veg_inun_itr: int = 20, generate_optimised_gt_thr: bool = True):

        # Check the var
        output_path = Path(output_path).path_name
        if not os.path.exists(f'{output_path}'):
            bf.create_folder(f'{output_path}')

        if roi_name is not None:
            output_path = output_path + f'{str(roi_name)}\\'
            bf.create_folder(output_path)

        if not isinstance(water_level_data, np.ndarray):
            try:
                water_level_data = np.array(water_level_data)
            except:
                raise TypeError('Please input the water level data in a right format!')

        if water_level_data.shape[1] != 2:
            print('Please input the water level data as a 2d array with shape under n*2!')
            sys.exit(-1)

        # Process the inundated dc
        if inundated_index not in self._index_list:
            raise ValueError(f'Inundated dc {str(inundated_index)} is not input!')
        else:
            inundated_dc = copy.copy(self.dcs[self._index_list.index(inundated_index)])
            roi_ds = gdal.Open(self.ROI_tif)
            transform = roi_ds.GetGeoTransform()
            proj = roi_ds.GetProjection()
            roi_arr = roi_ds.GetRasterBand(1).ReadAsArray()
            if not self._sparse_matrix_list[self._index_list.index(inundated_index)]:
                if process_extent is not None and isinstance(process_extent, (list, tuple)) and len(
                        process_extent) == 4:
                    y_min, y_max, x_min, x_max = process_extent
                    inundated_dc = inundated_dc[y_min: y_max, x_min: x_max, :]
                    roi_arr = roi_arr[y_min: y_max, x_min: x_max]
                    new_trans = (transform[0] + x_min * transform[1], transform[1], transform[2],
                                 transform[3] + y_min * transform[5], transform[4], transform[5])
                elif process_extent is None:
                    inundated_dc = inundated_dc[:, :, :]
                    new_trans = transform
                else:
                    raise TypeError('Process extent is not properly input!')
            else:
                if process_extent is not None and isinstance(process_extent, (list, tuple)) and len(
                        process_extent) == 4:
                    y_min, y_max, x_min, x_max = process_extent
                    inundated_dc = inundated_dc.extract_matrix(([y_min, y_max], [x_min, x_max], ['all']))
                    roi_arr = roi_arr[y_min: y_max, x_min: x_max]
                    new_trans = (transform[0] + x_min * transform[1], transform[1], transform[2],
                                 transform[3] + y_min * transform[5], transform[4], transform[5])
                elif process_extent is None:
                    inundated_dc = inundated_dc.extract_matrix((['all'], ['all'], ['all']))
                    new_trans = transform
                else:
                    raise TypeError('Process extent is not properly input!')
            doy_list = self._doys_backup_[self._index_list.index(inundated_index)]

        water_level_data[:, 0] = water_level_data[:, 0].astype(np.int32)
        date_range = [np.min(water_level_data[:, 0]), np.max(water_level_data[:, 0])]
        year_list = np.unique(water_level_data[:, 0] // 10000).astype(np.int32)

        # Create water level trend list
        if not os.path.exists(output_path + '\\water_level_trend.csv'):
            wl_trend_list = []
            for year in year_list:
                recession_turn = 0
                for data_size in range(1, water_level_data.shape[0]):
                    if year * 10000 + 3 * 100 <= water_level_data[data_size, 0] < year * 10000 + 1200:
                        if recession_turn == 0:
                            if (water_level_data[data_size, 1] > water_level_data[data_size - 1, 1] and
                                water_level_data[data_size, 1] > water_level_data[data_size + 1, 1]) \
                                    or (water_level_data[data_size, 1] < water_level_data[data_size - 1, 1] and
                                        water_level_data[data_size, 1] < water_level_data[data_size + 1, 1]) \
                                    or (water_level_data[data_size - 1, 1] == water_level_data[data_size, 1] ==
                                        water_level_data[data_size + 1, 1]):
                                wl_trend_list.append([water_level_data[data_size, 0], 0])
                            elif (water_level_data[data_size - 1, 1] < water_level_data[data_size, 1] <=
                                  water_level_data[data_size + 1, 1]) \
                                    or (water_level_data[data_size - 1, 1] <= water_level_data[data_size, 1] <
                                        water_level_data[data_size + 1, 1]):
                                wl_trend_list.append([water_level_data[data_size, 0], 1])
                                recession_turn = 1
                            elif (water_level_data[data_size - 1, 1] > water_level_data[data_size, 1] >=
                                  water_level_data[data_size + 1, 1]) \
                                    or (water_level_data[data_size - 1, 1] >= water_level_data[data_size, 1] >
                                        water_level_data[data_size + 1, 1]):
                                wl_trend_list.append([water_level_data[data_size, 0], -1])
                                recession_turn = -1
                            else:
                                print('error occurred recession!')
                                sys.exit(-1)

                        elif recession_turn != 0:
                            if (water_level_data[data_size, 1] > water_level_data[data_size - 1, 1] and
                                water_level_data[data_size, 1] > water_level_data[data_size + 1, 1]) \
                                    or (water_level_data[data_size, 1] < water_level_data[data_size - 1, 1] and
                                        water_level_data[data_size, 1] < water_level_data[data_size + 1, 1]) \
                                    or (water_level_data[data_size - 1, 1] == water_level_data[data_size, 1] ==
                                        water_level_data[data_size + 1, 1]):
                                wl_trend_list.append([water_level_data[data_size, 0], 0])
                            elif (water_level_data[data_size - 1, 1] < water_level_data[data_size, 1] <=
                                  water_level_data[data_size + 1, 1]) \
                                    or (water_level_data[data_size - 1, 1] <= water_level_data[data_size, 1] <
                                        water_level_data[data_size + 1, 1]):
                                if recession_turn > 0:
                                    wl_trend_list.append([water_level_data[data_size, 0], recession_turn])
                                else:
                                    recession_turn = -recession_turn + 1
                                    wl_trend_list.append([water_level_data[data_size, 0], recession_turn])
                            elif (water_level_data[data_size - 1, 1] > water_level_data[data_size, 1] >=
                                  water_level_data[data_size + 1, 1]) \
                                    or (water_level_data[data_size - 1, 1] >= water_level_data[data_size, 1] >
                                        water_level_data[data_size + 1, 1]):
                                if recession_turn < 0:
                                    wl_trend_list.append([water_level_data[data_size, 0], recession_turn])
                                else:
                                    recession_turn = -recession_turn
                                    wl_trend_list.append([water_level_data[data_size, 0], recession_turn])
                            else:
                                print('error occurred recession!')
                                sys.exit(-1)
            wl_trend_list = np.array(wl_trend_list)
            wl_trend_list = pd.DataFrame(wl_trend_list)
            wl_trend_list.to_csv(output_path + '\\water_level_trend.csv')
        else:
            wl_trend_list = pandas.read_csv(output_path + '\\water_level_trend.csv')
            wl_trend_list = np.array(wl_trend_list)
            wl_trend_list = wl_trend_list[:, 1:]

        # Retrieve the max water level
        if generate_max_water_level_factor:
            doy_valid_list = []
            water_level_list = []

            for _ in range(len(doy_list)):
                if date_range[0] < doy_list[_] < date_range[1]:
                    doy_valid_list.append(doy_list[_])
                    date_num = np.argwhere(water_level_data == doy_list[_])[0, 0]
                    water_level_list.append(water_level_data[date_num, 1])

            if len(doy_valid_list) == len(water_level_list) and len(doy_valid_list) != 0:
                information_array = np.ones([2, len(doy_valid_list)])
                information_array[0, :] = np.array(doy_valid_list)
                information_array[1, :] = np.array(water_level_list)
                year_array = np.unique(np.fix(information_array[0, :] // 10000))
                annual_max = []
                for year in range(year_array.shape[0]):
                    water_level_max = 0
                    for date_temp in range(water_level_data.shape[0]):
                        if water_level_data[date_temp, 0] // 10000 == year_array[year]:
                            water_level_max = max(water_level_max, water_level_data[date_temp, 1])

                annual_max = np.array(annual_max)
                dic_temp = pd.DataFrame(np.transpose(np.stack([year_array, annual_max], axis=1)),
                                        columns=['Year', 'Annual max water level'])
                dic_temp.to_excel(output_path + self.ROI_name + '.xlsx')

        if generate_inundation_status_factor:

            perwater_ras_path = output_path + 'perwater_ras\\'
            bf.create_folder(perwater_ras_path)

            # Create river stretch based on inundation frequency
            with tqdm(total=1, desc=f'Create river stretch based on inundation frequency',
                      bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                if not os.path.exists(perwater_ras_path + 'permanent_water.tif'):
                    if isinstance(inundated_dc, np.ndarray):
                        inundated_frequency = np.nansum(inundated_dc == inundated_value, axis=2) / np.nansum(
                            inundated_dc != nan_value, axis=2)
                        inundated_frequency[inundated_frequency > 0.7] = 1
                        inundated_frequency[inundated_frequency <= 0.7] = 0
                        inundated_frequency[roi_arr == -32768] = -2
                        inundated_frequency = inundated_frequency.astype(np.int16)
                    elif isinstance(inundated_dc, NDSparseMatrix):
                        inundated_all = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                        valid_all = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                        for _ in inundated_dc.SM_namelist:
                            arr_temp = inundated_dc.SM_group[_].toarray()
                            valid_all = valid_all + (arr_temp != nan_value).astype(np.int32)
                            inundated_all = inundated_all + (arr_temp == inundated_value).astype(np.int32)
                        inundated_frequency = inundated_all / valid_all
                        inundated_frequency[inundated_frequency > 0.5] = 1
                        inundated_frequency[inundated_frequency <= 0.5] = 0
                        inundated_frequency[roi_arr == -32768] = -2
                        inundated_frequency = inundated_frequency.astype(np.int16)
                    else:
                        raise TypeError('Type error when generate inundated frequency!')

                    driver = gdal.GetDriverByName('GTiff')
                    driver.Register()
                    if os.path.exists(perwater_ras_path + 'permanent_water.tif'):
                        os.remove(perwater_ras_path + 'permanent_water.tif')
                    outds = driver.Create(perwater_ras_path + 'permanent_water.tif', xsize=inundated_frequency.shape[1],
                                          ysize=inundated_frequency.shape[0],
                                          bands=1, eType=gdal.GDT_Int16, options=['COMPRESS=LZW', 'PREDICTOR=2'])
                    outds.SetGeoTransform(new_trans)
                    outds.SetProjection(proj)
                    outband = outds.GetRasterBand(1)
                    outband.WriteArray(inundated_frequency)
                    outband.SetNoDataValue(-2)
                    outband.FlushCache()
                    outband, outds = None, None

                pbar.update()

            # Refine the river stretch
            with tqdm(total=1, desc=f'Refine river stretch', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                if not os.path.exists(perwater_ras_path + 'permanent_water_fixed.tif'):
                    pw_ds = gdal.Open(perwater_ras_path + 'permanent_water.tif')
                    permanent_water_arr = pw_ds.GetRasterBand(1).ReadAsArray()

                    # Create shp driver
                    dst_layername = "permanent_water"
                    drv = ogr.GetDriverByName("ESRI Shapefile")
                    shp_output_path = output_path + 'perwater_shpfile\\'
                    bf.create_folder(shp_output_path)
                    if os.path.exists(shp_output_path + dst_layername + ".shp"):
                        drv.DeleteDataSource(shp_output_path + dst_layername + ".shp")

                    # polygonize the raster
                    proj = osr.SpatialReference(wkt=pw_ds.GetProjection())
                    target = osr.SpatialReference()
                    target.ImportFromEPSG(int(proj.GetAttrValue('AUTHORITY', 1)))

                    dst_ds = drv.CreateDataSource(shp_output_path + dst_layername + ".shp")
                    dst_layer = dst_ds.CreateLayer(dst_layername, srs=target)

                    fld = ogr.FieldDefn("inundation", ogr.OFTInteger)
                    dst_layer.CreateField(fld)
                    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("inundation")
                    gdal.Polygonize(pw_ds.GetRasterBand(1), None, dst_layer, dst_field, [])

                    layer = dst_ds.GetLayer()
                    new_field = ogr.FieldDefn("area", ogr.OFTReal)
                    new_field.SetWidth(32)
                    layer.CreateField(new_field)

                    # Fix the sole pixel
                    for feature in layer:
                        geom = feature.GetGeometryRef()
                        area = geom.GetArea()
                        if area < 5000 and feature.GetField('inundation') == 1:
                            feature.SetField("inundation", 0)
                        elif area < 5000 and feature.GetField('inundation') == 0:
                            feature.SetField("inundation", 1)
                        feature.SetField("area", area)
                        layer.SetFeature(feature)

                    # Re-rasterlize the shpfile
                    if os.path.exists(perwater_ras_path + 'permanent_water_fixed.tif'):
                        os.remove(perwater_ras_path + 'permanent_water_fixed.tif')
                    target_ds = gdal.GetDriverByName('GTiff').Create(perwater_ras_path + 'permanent_water_fixed.tif',
                                                                     pw_ds.RasterXSize, pw_ds.RasterYSize, 1,
                                                                     gdal.GDT_Int16)
                    target_ds.SetGeoTransform(pw_ds.GetGeoTransform())
                    target_ds.SetProjection(target.ExportToWkt())
                    band = target_ds.GetRasterBand(1)
                    band.SetNoDataValue(pw_ds.GetRasterBand(1).GetNoDataValue())

                    gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=inundation"])
                    del target_ds
                    del dst_ds

                    # Create shp driver
                    dst_layername = "permanent_water_fixed"
                    drv = ogr.GetDriverByName("ESRI Shapefile")
                    if os.path.exists(shp_output_path + dst_layername + ".shp"):
                        drv.DeleteDataSource(shp_output_path + dst_layername + ".shp")

                    # polygonize the raster
                    proj = osr.SpatialReference(wkt=pw_ds.GetProjection())
                    target = osr.SpatialReference()
                    target.ImportFromEPSG(int(proj.GetAttrValue('AUTHORITY', 1)))

                    dst_ds = drv.CreateDataSource(shp_output_path + dst_layername + ".shp")
                    dst_layer = dst_ds.CreateLayer(dst_layername, srs=target)

                    fld = ogr.FieldDefn("inundation", ogr.OFTInteger)
                    dst_layer.CreateField(fld)
                    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("inundation")
                    gdal.Polygonize(pw_ds.GetRasterBand(1), None, dst_layer, dst_field, [])

                    layer = dst_ds.GetLayer()
                    new_field = ogr.FieldDefn("area", ogr.OFTReal)
                    new_field.SetWidth(32)
                    layer.CreateField(new_field)

                    # Fix the sole pixel
                    for feature in layer:
                        geom = feature.GetGeometryRef()
                        area = geom.GetArea()
                        feature.SetField("area", area)
                        layer.SetFeature(feature)
                    dst_ds, pw_ds = None, None

                pbar.update()

            # Generate sole inundation area rs
            pw_ds = gdal.Open(perwater_ras_path + 'permanent_water_fixed.tif')
            permanent_water_arr = pw_ds.GetRasterBand(1).ReadAsArray()
            sole_file_path = output_path + 'sole_inunarea_ras\\'
            bf.create_folder(sole_file_path)
            if isinstance(inundated_dc, NDSparseMatrix):
                with tqdm(total=len(inundated_dc.SM_namelist), desc=f'Generate sole inundation area raster',
                          bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                    for _ in inundated_dc.SM_namelist:
                        if not os.path.exists(sole_file_path + f'{str(_)}.tif'):
                            i_arr = inundated_dc.SM_group[_].toarray()
                            i_arr = i_arr - 1
                            i_arr[i_arr < 0] = -2
                            i_arr[permanent_water_arr == 1] = -1
                            i_arr = i_arr.astype(np.int16)
                            bf.write_raster(pw_ds, i_arr, sole_file_path, f'{str(_)}.tif',
                                            raster_datatype=gdal.GDT_Int16, nodatavalue=-2)
                        pbar.update()
            elif isinstance(inundated_dc, np.ndarray):
                with tqdm(total=inundated_dc.shape[2], desc=f'Generate sole inundation area raster',
                          bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                    for _ in range(inundated_dc.shape[2]):
                        if not os.path.exists(sole_file_path + f'{str(_)}.tif'):
                            i_arr = inundated_dc[:, :, _].reshape(inundated_dc.shape[0], inundated_dc.shape[1])
                            i_arr = i_arr - 1
                            i_arr[i_arr < 0] = -2
                            i_arr[permanent_water_arr == 1] = -1
                            i_arr = i_arr.astype(np.int16)
                            bf.write_raster(pw_ds, i_arr, sole_file_path, f'{str(_)}.tif',
                                            raster_datatype=gdal.GDT_Int16, nodatavalue=-2)
                        pbar.update()

            # Generate sole inundation area shp
            sole_shpfile_path = output_path + 'sole_inunarea_shpfile\\'
            bf.create_folder(sole_shpfile_path)
            ras_files = bf.file_filter(sole_file_path, ['.tif'],
                                       exclude_word_list=['.xml', '.dpf', '.cpg', '.aux', 'vat', '.ovr'])
            with tqdm(total=len(ras_files), desc=f'Generate sole inundation area shpfile',
                      bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in ras_files:
                    try:
                        dst_layername = _.split('\\')[-1].split('.tif')[0]
                        drv = ogr.GetDriverByName("ESRI Shapefile")
                        if os.path.exists(sole_shpfile_path + dst_layername + ".shp"):
                            pass
                        else:
                            # polygonize the raster
                            ras_ds = gdal.Open(_)
                            proj = osr.SpatialReference(wkt=ras_ds.GetProjection())
                            target = osr.SpatialReference()
                            target.ImportFromEPSG(int(proj.GetAttrValue('AUTHORITY', 1)))

                            dst_ds = drv.CreateDataSource(sole_shpfile_path + dst_layername + ".shp")
                            dst_layer = dst_ds.CreateLayer(dst_layername, srs=target)

                            fld = ogr.FieldDefn("inundation", ogr.OFTInteger)
                            dst_layer.CreateField(fld)
                            dst_field = dst_layer.GetLayerDefn().GetFieldIndex("inundation")
                            gdal.Polygonize(ras_ds.GetRasterBand(1), None, dst_layer, dst_field, [])

                            new_field = ogr.FieldDefn("link2river", ogr.OFTReal)
                            new_field.SetWidth(32)
                            dst_layer.CreateField(new_field)

                            new_field2 = ogr.FieldDefn("inun_date", ogr.OFTReal)
                            new_field2.SetWidth(32)
                            dst_layer.CreateField(new_field2)

                            new_field3 = ogr.FieldDefn("ninun_date", ogr.OFTReal)
                            new_field3.SetWidth(32)
                            dst_layer.CreateField(new_field3)

                            new_field4 = ogr.FieldDefn("area", ogr.OFTReal)
                            new_field4.SetWidth(32)
                            dst_layer.CreateField(new_field4)

                            # Generate the filed attribute
                            shp_list = []
                            for feature in dst_layer:
                                if feature.GetField('inundation') == -1:
                                    shp_list.append(wkt.loads(feature.GetGeometryRef().ExportToWkt()))

                            for feature in dst_layer:
                                link2river_st = 0
                                if feature.GetField('inundation') == 1:
                                    shp_temp = wkt.loads(feature.GetGeometryRef().ExportToWkt())
                                    for shp_pw in shp_list:
                                        if shp_pw.intersects(shp_temp):
                                            link2river_st = 1
                                            break
                                feature.SetField('link2river', link2river_st)
                                dst_layer.SetFeature(feature)
                                feature = None

                            for feature in dst_layer:
                                geom = feature.GetGeometryRef()
                                area = geom.GetArea()
                                feature.SetField('area', area)
                                dst_layer.SetFeature(feature)
                                feature = None

                            for feature in dst_layer:
                                if feature.GetField('inundation') == 1:
                                    feature.SetField('inun_date', int(dst_layername))
                                    feature.SetField('ninun_date', -1)
                                elif feature.GetField('inundation') == 0:
                                    feature.SetField('inun_date', -1)
                                    feature.SetField('ninun_date', int(dst_layername))
                                dst_layer.SetFeature(feature)
                                feature = None

                            dst_layer = None
                            dst_ds = None
                            ras_ds = None

                    except:
                        print(traceback.format_exc())
                    pbar.update()

        if generate_min_inun_wl:

            # Generate annual inundation pattern
            pw_ds = gdal.Open(output_path + 'perwater_ras\\permanent_water_fixed.tif')
            permanent_water_arr = pw_ds.GetRasterBand(1).ReadAsArray()

            year_list = np.unique(np.floor(np.array(doy_list) / 10000)).astype(np.int32)
            year_list = year_list.tolist()

            annual_inform_folder = output_path + 'annual_inun_wl_ras\\'
            bf.create_folder(annual_inform_folder)

            # Retrieve the annual min inundated water level and annual max noninundated water level
            with tqdm(total=len(year_list),
                      desc=f'Pre-Generate annual min inundated water level and annual max noninundated water level',
                      bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:

                for _ in year_list:

                    # Create folder
                    bf.create_folder(annual_inform_folder + f'{str(_)}\\')
                    if not os.path.exists(
                            annual_inform_folder + f'{str(_)}\\max_noninun_wl_{str(_)}.tif') or not os.path.exists(
                            annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}.tif') \
                            or not os.path.exists(
                        annual_inform_folder + f'{str(_)}\\trend_{str(_)}.tif') or not os.path.exists(
                        annual_inform_folder + f'{str(_)}\\annual_inunarea_{str(_)}.tif') \
                            or not os.path.exists(
                        annual_inform_folder + f'{str(_)}\\max_noninun_date_{str(_)}.tif') or not os.path.exists(
                        annual_inform_folder + f'{str(_)}\\min_inun_date_{str(_)}.tif'):

                        # Define arr
                        max_noninun_wl_arr = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]]) - 1
                        min_inun_wl_arr = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]]) + 1000
                        trend_arr = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                        max_noninun_date_arr = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                        min_inun_date_arr = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])

                        # Generate min inundation wl and max noninundation wl
                        for __ in doy_list:
                            if int(np.floor(__ // 10000)) == _ and __ not in manual_remove_date:
                                if __ in water_level_data:
                                    wl_temp = water_level_data[np.argwhere(water_level_data == __)[0, 0], 1]
                                    max_noninun_wl_arr_temp = np.zeros(
                                        [inundated_dc.shape[0], inundated_dc.shape[1]]) - 1
                                    min_inun_wl_arr_temp = np.zeros(
                                        [inundated_dc.shape[0], inundated_dc.shape[1]]) + 1000
                                    if isinstance(inundated_dc, NDSparseMatrix):
                                        inundated_arr = inundated_dc.SM_group[__].toarray()
                                    else:
                                        inundated_arr = inundated_dc[:, :, doy_list.index(__)].reshape(
                                            [inundated_dc.shape[0], inundated_dc.shape[1]])

                                    max_noninun_wl_arr_temp[inundated_arr == 1] = wl_temp
                                    min_inun_wl_arr_temp[inundated_arr == 2] = wl_temp

                                    max_noninun_wl_arr = np.nanmax([max_noninun_wl_arr, max_noninun_wl_arr_temp],
                                                                   axis=0)
                                    min_inun_wl_arr = np.nanmin([min_inun_wl_arr, min_inun_wl_arr_temp], axis=0)

                                    max_noninun_date_arr[max_noninun_wl_arr == wl_temp] = __
                                    min_inun_date_arr[min_inun_wl_arr == wl_temp] = __

                                if __ in wl_trend_list:
                                    trend_temp = wl_trend_list[np.argwhere(wl_trend_list == __)[0, 0], 1]
                                    trend_arr[min_inun_wl_arr == wl_temp] = trend_temp

                        max_noninun_wl_arr[roi_arr == -32768] = np.nan
                        min_inun_wl_arr[roi_arr == -32768] = np.nan
                        min_inun_wl_arr[min_inun_wl_arr == 1000] = np.nan
                        max_noninun_wl_arr[max_noninun_wl_arr == -1] = np.nan

                        bf.write_raster(pw_ds, max_noninun_wl_arr, annual_inform_folder,
                                        f'{str(_)}\\max_noninun_wl_{str(_)}.tif', raster_datatype=gdal.GDT_Float32,
                                        nodatavalue=np.nan)
                        bf.write_raster(pw_ds, min_inun_wl_arr, annual_inform_folder,
                                        f'{str(_)}\\min_inun_wl_{str(_)}.tif', raster_datatype=gdal.GDT_Float32,
                                        nodatavalue=np.nan)
                        bf.write_raster(pw_ds, min_inun_date_arr, annual_inform_folder,
                                        f'{str(_)}\\min_inun_date_{str(_)}.tif', raster_datatype=gdal.GDT_Int32,
                                        nodatavalue=0)
                        bf.write_raster(pw_ds, max_noninun_date_arr, annual_inform_folder,
                                        f'{str(_)}\\max_noninun_date_{str(_)}.tif', raster_datatype=gdal.GDT_Int32,
                                        nodatavalue=0)
                        bf.write_raster(pw_ds, trend_arr, annual_inform_folder, f'{str(_)}\\trend_{str(_)}.tif',
                                        raster_datatype=gdal.GDT_Int32, nodatavalue=0)

                        min_inun_wl_arr[~np.isnan(min_inun_wl_arr)] = 1
                        min_inun_wl_arr[permanent_water_arr == 1] = -1
                        min_inun_wl_arr[permanent_water_arr == -2] = -2
                        min_inun_wl_arr[np.isnan(min_inun_wl_arr)] = 0
                        bf.write_raster(pw_ds, min_inun_wl_arr, annual_inform_folder,
                                        f'{str(_)}\\annual_inunarea_{str(_)}.tif', raster_datatype=gdal.GDT_Int32,
                                        nodatavalue=-2)
                    pbar.update()

            # Generate annual inundation shpfile
            annual_inunshp_folder = output_path + 'annual_inun_shpfile\\'
            bf.create_folder(annual_inunshp_folder)
            with tqdm(total=len(year_list), desc=f'Generate annual inundation area shpfile',
                      bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in year_list:

                    # Generate the shpfile of continuous inundation area
                    files = bf.file_filter(annual_inform_folder + str(_) + '\\', [f'annual_inunarea_{str(_)}.tif'],
                                           exclude_word_list=['.xml', '.dpf', '.cpg', '.aux', 'vat', '.ovr'],
                                           and_or_factor='and')
                    if len(files) == 1:
                        ras_ds = gdal.Open(files[0])
                        drv = ogr.GetDriverByName("ESRI Shapefile")
                        if not os.path.exists(annual_inunshp_folder + str(_) + ".shp"):

                            # polygonize the raster
                            proj = osr.SpatialReference(wkt=ras_ds.GetProjection())
                            target = osr.SpatialReference()
                            target.ImportFromEPSG(int(proj.GetAttrValue('AUTHORITY', 1)))

                            dst_ds = drv.CreateDataSource(annual_inunshp_folder + str(_) + ".shp")
                            dst_layer = dst_ds.CreateLayer(str(_), srs=target)

                            fld = ogr.FieldDefn("inundation", ogr.OFTInteger)
                            dst_layer.CreateField(fld)
                            dst_field = dst_layer.GetLayerDefn().GetFieldIndex("inundation")
                            gdal.Polygonize(ras_ds.GetRasterBand(1), None, dst_layer, dst_field, [])

                            new_field1 = ogr.FieldDefn("area", ogr.OFTReal)
                            new_field1.SetWidth(32)
                            dst_layer.CreateField(new_field1)

                            new_field2 = ogr.FieldDefn("link2chan", ogr.OFTReal)
                            new_field2.SetWidth(32)
                            dst_layer.CreateField(new_field2)

                            new_field3 = ogr.FieldDefn("inun_id", ogr.OFTReal)
                            new_field3.SetWidth(32)
                            dst_layer.CreateField(new_field3)

                            for feature in dst_layer:
                                geom = feature.GetGeometryRef()
                                area = geom.GetArea()
                                feature.SetField('area', area)
                                dst_layer.SetFeature(feature)
                                feature = None

                            shp_list = []
                            for feature in dst_layer:
                                if feature.GetField('inundation') == -1 and feature.GetField('area') >= 1000000:
                                    shp_list.append(wkt.loads(feature.GetGeometryRef().ExportToWkt()))

                            inund_id = 1
                            for feature in dst_layer:
                                link2channel_ = 0
                                if feature.GetField('inundation') == 1:
                                    shp_temp = wkt.loads(feature.GetGeometryRef().ExportToWkt())
                                    for shp_pw in shp_list:
                                        if shp_pw.intersects(shp_temp):
                                            link2channel_ = 1
                                            inund_id += 1
                                            break

                                if link2channel_ == 1:
                                    feature.SetField('link2chan', link2channel_)
                                    feature.SetField('inun_id', inund_id)
                                else:
                                    feature.SetField('link2chan', link2channel_)
                                    feature.SetField('inun_id', 0)
                                dst_layer.SetFeature(feature)
                                feature = None

                            dst_ds, ras_ds = None, None
                    else:
                        raise ValueError('Too many annual inundation ras files!')

                    # Generate the shpfile of continuous water level area
                    files = bf.file_filter(annual_inform_folder + str(_) + '\\', [f'min_inun_wl_{str(_)}.tif'],
                                           exclude_word_list=['.xml', '.dpf', '.cpg', '.aux', 'vat', '.ovr'],
                                           and_or_factor='and')
                    if len(files) == 1:

                        drv = ogr.GetDriverByName("ESRI Shapefile")
                        if os.path.exists(annual_inform_folder + str(_) + '\\' + f'min_inun_wl_{str(_)}4poly.tif'):
                            pass
                        else:
                            ras_ds = gdal.Open(files[0])
                            ras_arr = ras_ds.GetRasterBand(1).ReadAsArray()
                            ras_arr = ras_arr * 100
                            ras_arr[np.isnan(ras_arr)] = -2
                            bf.write_raster(ras_ds, ras_arr, annual_inform_folder + str(_) + '\\',
                                            f'min_inun_wl_{str(_)}4poly.tif')

                        ras_ds = gdal.Open(annual_inform_folder + str(_) + '\\' + f'min_inun_wl_{str(_)}4poly.tif')
                        if not os.path.exists(annual_inunshp_folder + 'wl_' + str(_) + ".shp"):

                            # polygonize the raster
                            proj = osr.SpatialReference(wkt=ras_ds.GetProjection())
                            target = osr.SpatialReference()
                            target.ImportFromEPSG(int(proj.GetAttrValue('AUTHORITY', 1)))

                            dst_ds = drv.CreateDataSource(annual_inunshp_folder + 'wl_' + str(_) + ".shp")
                            dst_layer = dst_ds.CreateLayer(str(_), srs=target)

                            fld = ogr.FieldDefn("water_l", ogr.OFTReal)
                            dst_layer.CreateField(fld)
                            dst_field = dst_layer.GetLayerDefn().GetFieldIndex("water_l")
                            gdal.Polygonize(ras_ds.GetRasterBand(1), None, dst_layer, dst_field, [])

                            new_field1 = ogr.FieldDefn("wl_id", ogr.OFTReal)
                            new_field1.SetWidth(32)
                            dst_layer.CreateField(new_field1)

                            wl_id = 1
                            for feature in dst_layer:
                                feature.SetField("wl_id", wl_id)
                                dst_layer.SetFeature(feature)
                                wl_id += 1
                                feature = None

                            dst_ds, ras_ds = None, None
                    else:
                        raise ValueError('Too many annual inundation ras files!')
                    pbar.update()

            # Refine the inun wl
            with tqdm(total=len(year_list), desc=f'Refine inundation water level array',
                      bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in year_list:
                    if not os.path.exists(annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}_fixed.tif'):
                        # Open the annual inundation shpfile and annual min inundation water level ras
                        dst_ds = ogr.Open(output_path + 'annual_inun_shpfile\\' + str(_) + ".shp")
                        layer = dst_ds.GetLayer()

                        dst_ds2 = ogr.Open(output_path + 'annual_inun_shpfile\\' + 'wl_' + str(_) + ".shp")
                        layer2 = dst_ds2.GetLayer()

                        pw_ds = gdal.Open(output_path + 'perwater_ras\\permanent_water_fixed.tif')
                        permanent_water_arr = pw_ds.GetRasterBand(1).ReadAsArray()

                        min_wl_ds = gdal.Open(annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}.tif')
                        min_inun_wl_arr = min_wl_ds.GetRasterBand(1).ReadAsArray()
                        min_inun_wl_arr[permanent_water_arr == 1] = -1
                        min_wl_arr_refined = copy.deepcopy(min_inun_wl_arr)
                        min_wl_arr_refined[:, :] = 0

                        # Create the ras
                        target_ds = gdal.GetDriverByName('GTiff').Create(
                            annual_inform_folder + f'{str(_)}\\annual_combine_{str(_)}_v1.tif', pw_ds.RasterXSize,
                            pw_ds.RasterYSize, 3, gdal.GDT_Int32)
                        target_ds.SetGeoTransform(pw_ds.GetGeoTransform())
                        target_ds.SetProjection(pw_ds.GetProjection())

                        gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=inundation"])
                        gdal.RasterizeLayer(target_ds, [2], layer, options=["ATTRIBUTE=area"])
                        gdal.RasterizeLayer(target_ds, [3], layer, options=["ATTRIBUTE=inun_id"])

                        target_ds2 = gdal.GetDriverByName('GTiff').Create(
                            annual_inform_folder + f'{str(_)}\\annual_combine_{str(_)}_v3.tif', pw_ds.RasterXSize,
                            pw_ds.RasterYSize, 1, gdal.GDT_Int32)
                        target_ds2.SetGeoTransform(pw_ds.GetGeoTransform())
                        target_ds2.SetProjection(pw_ds.GetProjection())

                        gdal.RasterizeLayer(target_ds2, [1], layer2, options=["ATTRIBUTE=wl_id"])

                        wl_id_arr = target_ds2.GetRasterBand(1).ReadAsArray()
                        wl_id_arr_acc = np.zeros_like(wl_id_arr)
                        inun_id_arr = target_ds.GetRasterBand(3).ReadAsArray()
                        inun_id_list = np.unique(inun_id_arr.flatten())
                        issue_wl = {}

                        with tqdm(total=len(inun_id_list), desc=f'Get issued water level',
                                  bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar1:
                            for inun_id_ in inun_id_list:
                                if inun_id_ != 0:

                                    # Create the dic and extract inun ras by using inun id
                                    min_wl_arr_t = copy.deepcopy(min_inun_wl_arr)
                                    min_wl_arr_t[inun_id_arr != inun_id_] = np.nan
                                    min_wl_arr_t[permanent_water_arr == 1] = -1

                                    # Get all the wl within this inun id
                                    offset_all, bound_all = np.min(np.argwhere(inun_id_arr == inun_id_),
                                                                   axis=0), np.max(np.argwhere(inun_id_arr == inun_id_),
                                                                                   axis=0)
                                    min_wl_arr_t = min_wl_arr_t[offset_all[0] - 1: bound_all[0] + 2,
                                                   offset_all[1] - 1: bound_all[1] + 2]
                                    wl_id_arr_t = wl_id_arr[offset_all[0] - 1: bound_all[0] + 2,
                                                  offset_all[1] - 1: bound_all[1] + 2]
                                    min_wl_arr_list = np.unique(min_wl_arr_t.flatten())
                                    min_wl_arr_list = np.sort(np.delete(min_wl_arr_list, np.argwhere(
                                        np.logical_or(np.isnan(min_wl_arr_list), min_wl_arr_list == -1))))
                                    wl_id_arr_acc_t = np.zeros_like(wl_id_arr_t)

                                    # Get all issued wl
                                    for wl_arr in min_wl_arr_list:
                                        wl_id_arr_tt = copy.copy(wl_id_arr_t)
                                        wl_id_arr_tt[min_wl_arr_t != wl_arr] = 0
                                        wl_id_list = np.delete(np.unique(wl_id_arr_tt.flatten()), 0)

                                        for wl_id_temp in wl_id_list:
                                            wl_pos = np.argwhere(
                                                np.logical_and(min_wl_arr_t == wl_arr, wl_id_arr_tt == wl_id_temp))
                                            bound_wl = np.array([])
                                            for pos_temp in wl_pos:
                                                arr_temp = min_wl_arr_t[pos_temp[0] - 1: pos_temp[0] + 2,
                                                           pos_temp[1] - 1: pos_temp[1] + 2]
                                                bound_wl = np.unique(
                                                    np.concatenate((bound_wl, np.unique(arr_temp.flatten()))))

                                            # bound append
                                            if (bound_wl < wl_arr).any():
                                                pass
                                            else:
                                                if inun_id_ not in issue_wl.keys():
                                                    issue_wl[inun_id_] = [wl_arr]
                                                else:
                                                    if wl_arr not in issue_wl[inun_id_]:
                                                        issue_wl[inun_id_].append(wl_arr)
                                                wl_id_arr_acc_t[np.logical_and(min_wl_arr_t == wl_arr,
                                                                               wl_id_arr_tt == wl_id_temp)] = 1
                                    wl_id_arr_acc[offset_all[0] - 1: bound_all[0] + 2,
                                    offset_all[1] - 1: bound_all[1] + 2] = wl_id_arr_acc[
                                                                           offset_all[0] - 1: bound_all[0] + 2,
                                                                           offset_all[1] - 1: bound_all[
                                                                                                  1] + 2] + wl_id_arr_acc_t
                                pbar1.update()
                            bf.write_raster(min_wl_ds, wl_id_arr_acc, annual_inform_folder,
                                            f'{str(_)}\\annual_issued_area.tif', raster_datatype=gdal.GDT_Int16)

                        with tqdm(total=len(issue_wl.keys()), desc=f'Process inundation water level',
                                  bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar1:
                            for iss_ in issue_wl.keys():

                                # Generate the mask
                                if len(issue_wl[iss_]) == 1:
                                    mask_temp = min_inun_wl_arr == issue_wl[iss_][0]
                                else:
                                    mask_temp = np.sum(
                                        np.stack([min_inun_wl_arr == wl__ for wl__ in issue_wl[iss_]], axis=2), axis=2)
                                mask_temp = (inun_id_arr == iss_) * (mask_temp >= 1) * (wl_id_arr_acc > 0)

                                offset_all, bound_all = np.min(np.argwhere(mask_temp == 1), axis=0), np.max(
                                    np.argwhere(mask_temp == 1), axis=0)
                                min_inun_wl_arr__ = np.zeros(
                                    [bound_all[0] + 1 - offset_all[0], bound_all[1] + 1 - offset_all[1]]) + 1000
                                min_all = np.zeros([inundated_dc.shape[0], inundated_dc.shape[1]])
                                issue_wl_temp = issue_wl[iss_]
                                issue_wl_temp = [float(str(_)) for _ in issue_wl_temp]

                                for __ in doy_list:
                                    if int(np.floor(__ // 10000)) == _ and __ not in manual_remove_date:
                                        if __ in water_level_data:
                                            wl_temp = water_level_data[np.argwhere(water_level_data == __)[0, 0], 1]
                                            min_inun_wl_arr_temp = np.zeros([bound_all[0] + 1 - offset_all[0],
                                                                             bound_all[1] + 1 - offset_all[1]]) + 1000
                                            min_inun_wl_arr_t = min_inun_wl_arr[offset_all[0]: bound_all[0] + 1,
                                                                offset_all[1]: bound_all[1] + 1]
                                            mask_temp_t = mask_temp[offset_all[0]: bound_all[0] + 1,
                                                          offset_all[1]: bound_all[1] + 1]

                                            if isinstance(inundated_dc, NDSparseMatrix):
                                                inundated_arr = inundated_dc[offset_all[0]: bound_all[0] + 1,
                                                                offset_all[1]: bound_all[1] + 1,
                                                                inundated_dc.SM_namelist.index(__)]
                                            else:
                                                inundated_arr = inundated_dc[:, :, doy_list.index(__)].reshape(
                                                    [inundated_dc.shape[0], inundated_dc.shape[1]])

                                            if wl_temp not in issue_wl_temp:
                                                min_inun_wl_arr_temp[inundated_arr == 2] = wl_temp

                                            elif wl_temp in issue_wl_temp:
                                                min_inun_wl_arr_temp[inundated_arr == 2] = wl_temp
                                                min_inun_wl_arr_temp[np.logical_and(mask_temp_t == 1,
                                                                                    min_inun_wl_arr_t == wl_temp)] = 1000

                                            min_inun_wl_arr__ = np.nanmin([min_inun_wl_arr__, min_inun_wl_arr_temp],
                                                                          axis=0)

                                min_inun_wl_arr__[min_inun_wl_arr__ == 1000] = np.nan
                                min_all[offset_all[0]: bound_all[0] + 1,
                                offset_all[1]: bound_all[1] + 1] = min_inun_wl_arr__
                                min_inun_wl_arr[mask_temp] = min_all[mask_temp]
                                pbar1.update()

                        bf.write_raster(min_wl_ds, min_inun_wl_arr, annual_inform_folder,
                                        f'{str(_)}\\min_inun_wl_{str(_)}_fixed.tif', raster_datatype=gdal.GDT_Float32,
                                        nodatavalue=-2)
                        target_ds = None
                    pbar.update()

            # Interpolate the minimum inundation wl
            annual_inunshp_folder = output_path + 'annual_inun_shpfile\\'
            bf.create_folder(annual_inunshp_folder)

            annual_inform_folder = output_path + 'annual_inun_wl_ras\\'
            bf.create_folder(annual_inform_folder)

            with tqdm(total=len(year_list), desc=f'Interpolate the minimum inundation water level',
                      bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in year_list:
                    if not os.path.exists(annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}_refined.tif'):
                        # Open the annual inundation shpfile and annual min inundation water level ras
                        dst_ds = ogr.Open(output_path + 'annual_inun_shpfile\\' + str(_) + ".shp")
                        layer = dst_ds.GetLayer()

                        pw_ds = gdal.Open(output_path + 'perwater_ras\\permanent_water_fixed.tif')
                        permanent_water_arr = pw_ds.GetRasterBand(1).ReadAsArray()

                        min_wl_ds = gdal.Open(annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}_fixed.tif')
                        min_inun_wl_arr = min_wl_ds.GetRasterBand(1).ReadAsArray()
                        min_inun_wl_arr[permanent_water_arr == 1] = -1
                        min_wl_arr_refined = copy.deepcopy(min_inun_wl_arr)
                        min_wl_arr_refined[:, :] = 0

                        max_noninun_wl_ds = gdal.Open(annual_inform_folder + f'{str(_)}\\max_noninun_wl_{str(_)}.tif')
                        max_noninun_wl_arr = max_noninun_wl_ds.GetRasterBand(1).ReadAsArray()

                        # Create the ras
                        target_ds = gdal.GetDriverByName('GTiff').Create(
                            annual_inform_folder + f'{str(_)}\\annual_combine_{str(_)}_v4.tif', pw_ds.RasterXSize,
                            pw_ds.RasterYSize, 3, gdal.GDT_Int32)
                        target_ds.SetGeoTransform(pw_ds.GetGeoTransform())
                        target_ds.SetProjection(pw_ds.GetProjection())

                        for __ in range(1, 3):
                            band = target_ds.GetRasterBand(__)
                            band.SetNoDataValue(-2)

                        gdal.RasterizeLayer(target_ds, [1], layer, options=["ATTRIBUTE=inundation"])
                        gdal.RasterizeLayer(target_ds, [2], layer, options=["ATTRIBUTE=area"])
                        gdal.RasterizeLayer(target_ds, [3], layer, options=["ATTRIBUTE=inun_id"])

                        inun_id_arr = target_ds.GetRasterBand(3).ReadAsArray()
                        inun_id_list = np.unique(inun_id_arr.flatten())

                        with tqdm(total=len(inun_id_list), desc=f'Refine the inundation water level',
                                  bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar3:
                            for inun_id_ in inun_id_list:
                                if inun_id_ != 0:

                                    # Create the dic and extract inun ras by using inun id
                                    time1, time2, time3, time4 = 0, 0, 0, 0

                                    s_t = time.time()
                                    inun_inform = []
                                    min_wl_arr_t = copy.deepcopy(min_inun_wl_arr)
                                    min_wl_arr_t[inun_id_arr != inun_id_] = np.nan
                                    min_wl_arr_t[permanent_water_arr == 1] = -1

                                    max_wl_arr_t = copy.deepcopy(max_noninun_wl_arr)
                                    max_wl_arr_t[inun_id_arr != inun_id_] = np.nan
                                    max_wl_arr_t[permanent_water_arr == 1] = -1

                                    # Get all the wl within this inun id
                                    offset_all, bound_all = np.min(np.argwhere(inun_id_arr == inun_id_),
                                                                   axis=0), np.max(np.argwhere(inun_id_arr == inun_id_),
                                                                                   axis=0)
                                    min_wl_arr_t = min_wl_arr_t[offset_all[0] - 1: bound_all[0] + 2,
                                                   offset_all[1] - 1: bound_all[1] + 2]
                                    max_wl_arr_t = max_wl_arr_t[offset_all[0] - 1: bound_all[0] + 2,
                                                   offset_all[1] - 1: bound_all[1] + 2]
                                    min_wl_arr_list = np.unique(min_wl_arr_t.flatten())
                                    min_wl_arr_list = np.sort(np.delete(min_wl_arr_list, np.argwhere(
                                        np.logical_or(np.isnan(min_wl_arr_list), min_wl_arr_list == -1))))
                                    time1 = time.time() - s_t
                                    s_t = time.time()

                                    # mp got inun status
                                    # with concurrent.futures.ProcessPoolExecutor() as exe:
                                    #     res = exe.map(assign_wl_status, repeat(min_wl_arr_t), min_wl_arr_list)
                                    #
                                    # res = list(res)
                                    # for res_ in res:
                                    #     inun_inform.extend(res_)

                                    # Got inun status
                                    for wl in min_wl_arr_list:
                                        inun_inform_list = []
                                        pos_all = np.argwhere(min_wl_arr_t == wl)
                                        for pos_temp in pos_all:
                                            arr_temp = min_wl_arr_t[pos_temp[0] - 1: pos_temp[0] + 2,
                                                       pos_temp[1] - 1: pos_temp[1] + 2]
                                            inun_inform__ = [wl]
                                            inun_inform__.extend([pos_temp[0], pos_temp[1]])
                                            if (arr_temp == wl).all():
                                                inun_inform__.append(0)
                                            elif (arr_temp > wl).any():
                                                inun_inform__.append(1)
                                            elif np.isnan(arr_temp).any() == 1:
                                                inun_inform__.append(1)
                                            elif (arr_temp == -1).any():
                                                inun_inform__.append(-1)
                                            elif (arr_temp < wl).any():
                                                inun_inform__.append(0)
                                            else:
                                                raise Exception(str(arr_temp))
                                            inun_inform_list.append(inun_inform__)
                                        inun_inform.extend(inun_inform_list)
                                    time2 = time.time() - s_t
                                    s_t = time.time()

                                    # Pre-assign the wl for bound pixel
                                    # with concurrent.futures.ProcessPoolExecutor() as exe:
                                    #     res = exe.map(pre_assign_wl, inun_inform, repeat(min_wl_arr_t), repeat(max_wl_arr_t), repeat(water_level_data), repeat(doy_list))
                                    #
                                    # inun_inform = []
                                    # min_wl_arr_refined_t = np.zeros_like(min_wl_arr_t) * np.nan
                                    # res = list(res)
                                    # for res_ in res:
                                    #     inun_inform.append(res_)
                                    #
                                    # i_len = 0
                                    # while i_len < len(inun_inform):
                                    #     min_wl_, pos_y, pos_x, status, wl_refined = inun_inform[i_len]
                                    #     if ~np.isnan(wl_refined):
                                    #         min_wl_arr_refined_t[pos_y, pos_x] = wl_refined
                                    #         inun_inform.remove(inun_inform[i_len])
                                    #     else:
                                    #         i_len += 1

                                    # Pre-assign the wl for bound pixel
                                    min_wl_arr_refined_t = np.zeros_like(min_wl_arr_t) * np.nan
                                    i_len = 0
                                    while i_len < len(inun_inform):

                                        min_wl_, pos_y, pos_x, status = inun_inform[i_len]
                                        if status == 1:
                                            min_wl_arr_refined_t[pos_y, pos_x] = min_wl_arr_t[pos_y, pos_x]
                                            inun_inform.remove(inun_inform[i_len])
                                        elif status == -1:
                                            if max_wl_arr_t[pos_y, pos_x] > min_wl_arr_t[pos_y, pos_x]:
                                                wl_temp = min_wl_arr_t[pos_y, pos_x]
                                                wl_pos = np.argwhere(water_level_data == float(str(wl_temp)))
                                                date, date_pos = [], []
                                                for wl_pos_temp in wl_pos:
                                                    if water_level_data[wl_pos_temp[0], 0] in doy_list:
                                                        date.append(water_level_data[wl_pos_temp[0]])
                                                        date_pos.append(wl_pos_temp[0])

                                                if len(date) == 1:
                                                    wl_temp_2 = water_level_data[int(date_pos[0]) - 5, 1]
                                                else:
                                                    date_pos = min(date_pos)
                                                    wl_temp_2 = water_level_data[int(date_pos) - 5, 1]

                                                if wl_temp_2 <= wl_temp and wl_temp - wl_temp_2 < 3:
                                                    min_wl_arr_refined_t[pos_y, pos_x] = (wl_temp_2 + wl_temp) / 2
                                                    inun_inform.remove(inun_inform[i_len])
                                                else:
                                                    min_wl_arr_refined_t[pos_y, pos_x] = wl_temp - 1
                                                    inun_inform.remove(inun_inform[i_len])

                                            else:
                                                min_wl_arr_refined_t[pos_y, pos_x] = (min_wl_arr_t[pos_y, pos_x] +
                                                                                      max_wl_arr_t[pos_y, pos_x]) / 2
                                                inun_inform.remove(inun_inform[i_len])

                                        elif status == 0:
                                            inun_inform[i_len].append(np.nan)
                                            i_len += 1

                                    for min_wl_ in min_wl_arr_list:
                                        if not (min_wl_arr_refined_t == min_wl_).any():
                                            pos_x_em_list, pos_y_em_list = [], []
                                            for inun_ in inun_inform:
                                                if inun_[0] == min_wl_:
                                                    pos_x_em_list.append(inun_[2])
                                                    pos_y_em_list.append(inun_[1])

                                            if len(pos_x_em_list) != 0:
                                                pos_x_em_list, pos_y_em_list = np.array(pos_x_em_list), np.array(
                                                    pos_y_em_list)
                                                pos_x_mid = int(np.nanmedian(pos_x_em_list))
                                                if pos_x_mid in pos_x_em_list:
                                                    pos_y_mid = int(
                                                        np.nanmedian(pos_y_em_list[pos_x_em_list == pos_x_mid]))
                                                    min_wl_arr_refined_t[pos_y_mid, pos_x_mid] = min_wl_
                                                else:
                                                    min_wl_arr_refined_t[
                                                        pos_y_em_list[int(pos_y_em_list.shape[0] / 2)], pos_x_em_list[
                                                            int(pos_x_em_list.shape[0] / 2)]] = min_wl_

                                    time3 = time.time() - s_t
                                    s_t = time.time()

                                    # Get wl
                                    if len(inun_inform) > 5000000000:
                                        # MP get wl
                                        with concurrent.futures.ProcessPoolExecutor() as exe:
                                            res = exe.map(assign_wl, inun_inform, repeat(min_wl_arr_refined_t))

                                        inun_inform = []
                                        res = list(res)
                                        for res_ in res:
                                            inun_inform.append(res_)

                                        for inun_ in inun_inform:
                                            min_wl_, pos_y, pos_x, status, wl_refined = inun_
                                            if ~np.isnan(wl_refined):
                                                min_wl_arr_refined_t[pos_y, pos_x] = wl_refined
                                            else:
                                                print(
                                                    f'inun_id:{str(inun_id_)}, pos{str(pos_y)}_{str(pos_x)}, wl{str(wl_refined)}')
                                                raise Exception('Error during mp get wl')

                                    else:
                                        # Seq get wl
                                        for inun__ in inun_inform:
                                            min_wl_, pos_y, pos_x, status, wl_refined = inun__
                                            if status == 0 and np.isnan(wl_refined):
                                                wl_centre = min_wl_
                                                upper_wl_dis, lower_wl_dis, lower_wl = [], [], []

                                                # Find 10 nearest points
                                                for r in range(1, 100):
                                                    pos_y_lower = 0 if pos_y - r < 0 else pos_y - r
                                                    pos_x_lower = 0 if pos_x - r < 0 else pos_x - r
                                                    pos_y_upper = min_wl_arr_refined_t.shape[0] if pos_y + r + 1 > \
                                                                                                   min_wl_arr_refined_t.shape[
                                                                                                       0] else pos_y + r + 1
                                                    pos_x_upper = min_wl_arr_refined_t.shape[1] if pos_x + r + 1 > \
                                                                                                   min_wl_arr_refined_t.shape[
                                                                                                       1] else pos_x + r + 1

                                                    arr_tt = min_wl_arr_refined_t[pos_y_lower: pos_y_upper,
                                                             pos_x_lower: pos_x_upper]
                                                    arr_tt[pos_y_lower - 1: pos_y_upper - 1,
                                                    pos_x_lower - 1: pos_x_upper - 1] = np.nan
                                                    if len(upper_wl_dis) < 10:
                                                        upper_wl_dis_list = []
                                                        if (arr_tt == wl_centre).any():
                                                            for pos_ttt in np.argwhere(arr_tt == wl_centre):
                                                                upper_wl_dis_list.append(np.sqrt(
                                                                    (pos_ttt[0] - r) ** 2 + (pos_ttt[1] - r) ** 2))
                                                            upper_wl_dis_list.sort()
                                                            if len(upper_wl_dis_list) > 10 - len(upper_wl_dis):
                                                                upper_wl_dis.extend(
                                                                    upper_wl_dis_list[:10 - len(upper_wl_dis)])
                                                            else:
                                                                upper_wl_dis.extend(upper_wl_dis_list)

                                                    if len(lower_wl_dis) < 10:
                                                        lower_wl_dat_list = []
                                                        arr_tt[arr_tt < 0] = 100000
                                                        if (arr_tt < wl_centre).any():
                                                            for pos_ttt in np.argwhere(arr_tt <= wl_centre):
                                                                lower_wl_dat_list.append(
                                                                    [arr_tt[pos_ttt[0], pos_ttt[1]], np.sqrt(
                                                                        (pos_ttt[0] - r) ** 2 + (pos_ttt[1] - r) ** 2)])
                                                            lower_wl_dat_list.sort()
                                                            if len(lower_wl_dat_list) > 10 - len(lower_wl_dis):
                                                                lower_wl_dis.extend([lower_wl_dat_list[_][1] for _ in
                                                                                     range(10 - len(lower_wl_dis))])
                                                                lower_wl.extend([lower_wl_dat_list[_][0] for _ in
                                                                                 range(10 - len(lower_wl))])
                                                            else:
                                                                lower_wl_dis.extend([lower_wl_dat_list[_][1] for _ in
                                                                                     range(len(lower_wl_dat_list))])
                                                                lower_wl.extend([lower_wl_dat_list[_][0] for _ in
                                                                                 range(len(lower_wl_dat_list))])

                                                    if len(upper_wl_dis) == 10 and len(lower_wl_dis) == 10 and len(
                                                            lower_wl) == 10:
                                                        break
                                                if len(upper_wl_dis) == 0:
                                                    upper_wl_dis = [0.00001]
                                                elif len(lower_wl_dis) == 0:
                                                    lower_wl_dis = [0.00001]
                                                    lower_wl = [wl_centre]
                                                elif len(upper_wl_dis) != len(lower_wl_dis):
                                                    size_ = min(len(upper_wl_dis), len(lower_wl_dis))
                                                    upper_wl_dis = upper_wl_dis[: size_]
                                                    lower_wl_dis = lower_wl_dis[: size_]
                                                    lower_wl = lower_wl[: size_]

                                                upper_wl_dis = [(1 / _) ** 2 for _ in upper_wl_dis]
                                                lower_wl_dis = [(1 / _) ** 2 for _ in lower_wl_dis]
                                                upper_wl = [wl_centre * upper_wl_dis[_] for _ in
                                                            range(len(upper_wl_dis))]
                                                lower_wl = [lower_wl[_] * lower_wl_dis[_] for _ in
                                                            range(len(lower_wl_dis))]

                                                min_wl_arr_refined_t[pos_y, pos_x] = sum(
                                                    [sum(upper_wl), sum(lower_wl)]) / sum(
                                                    [sum(upper_wl_dis), sum(lower_wl_dis)])

                                    min_wl_arr_refined_t[np.isnan(min_wl_arr_refined_t)] = 0
                                    min_wl_arr_refined[offset_all[0] - 1: bound_all[0] + 2,
                                    offset_all[1] - 1: bound_all[1] + 2] = min_wl_arr_refined[
                                                                           offset_all[0] - 1: bound_all[0] + 2,
                                                                           offset_all[1] - 1: bound_all[
                                                                                                  1] + 2] + min_wl_arr_refined_t
                                    time4 = time.time() - s_t

                                    # print(f'P1: {str(time1)[0:6]}s, P2: {str(time2)[0:6]}s, P3: {str(time3)[0:6]}s, P4: {str(time4)[0:6]}s')
                                pbar3.update()

                            min_wl_arr_refined[permanent_water_arr == 1] = -1
                            min_wl_arr_refined[permanent_water_arr == -2] = -2
                            bf.write_raster(min_wl_ds, min_wl_arr_refined, annual_inform_folder,
                                            f'{str(_)}\\min_inun_wl_{str(_)}_refined.tif',
                                            raster_datatype=gdal.GDT_Float32, nodatavalue=-2)
                    pbar.update()

        if generate_inun_duration:
            annual_inform_folder = output_path + 'annual_inun_wl_ras\\'
            annual_inunduration_folder = output_path + 'annual_inun_duration\\'
            annual_inunthr_folder = output_path + 'annual_inun_duration\\gt_thr\\'
            with tqdm(total=len(year_list), desc=f'Calculate inundation duration and height',
                      bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
                for _ in year_list:
                    if not os.path.exists(
                            annual_inunduration_folder + f'inun_duration_{str(_)}.tif') or not os.path.exists(
                            annual_inunduration_folder + f'inun_height_{str(_)}.tif') or not os.path.exists(
                            annual_inunduration_folder + f'mean_inun_height_{str(_)}.tif'):
                        bf.create_folder(annual_inunduration_folder)
                        if not os.path.exists(annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}_refined.tif'):
                            raise Exception('Please refine the inundation water level!')
                        else:
                            min_inun_ds = gdal.Open(
                                annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}_refined.tif')
                            min_inun_arr_ori = min_inun_ds.GetRasterBand(1).ReadAsArray()
                            min_inun_arr = copy.deepcopy(min_inun_arr_ori)
                            min_inun_arr[min_inun_arr == 0] = np.nan
                            min_inun_arr[min_inun_arr == -2] = np.nan
                            min_inun_arr[min_inun_arr == -1] = np.nan
                            inun_duration = np.zeros_like(min_inun_arr)

                            max_wl = 0
                            acc_wl = np.zeros_like(min_inun_arr)
                            for __ in range(water_level_data.shape[0]):
                                if int(water_level_data[__, 0] // 10000) == _:
                                    inun_duration = inun_duration + (water_level_data[__, 1] > min_inun_arr).astype(
                                        np.float32)
                                    max_wl = np.max([max_wl, water_level_data[__, 1]])
                                    wl_arr_temp = (water_level_data[__, 1] - min_inun_arr)
                                    wl_arr_temp[wl_arr_temp < 0] = 0
                                    acc_wl = acc_wl + (water_level_data[__, 1] > min_inun_arr).astype(
                                        np.float32) * wl_arr_temp
                            acc_wl = acc_wl / inun_duration

                            inun_height = max_wl - min_inun_arr
                            inun_height[inun_height < 0] = 0
                            inun_duration[min_inun_arr_ori == 0] = 0
                            inun_duration[min_inun_arr_ori == -1] = -1
                            inun_duration[min_inun_arr_ori == -2] = -2

                            inun_height[min_inun_arr_ori == 0] = 0
                            inun_height[min_inun_arr_ori == -1] = -1
                            inun_height[min_inun_arr_ori == -2] = -2

                            acc_wl[np.isnan(acc_wl)] = 0
                            acc_wl[min_inun_arr_ori == 0] = 0
                            acc_wl[min_inun_arr_ori == -1] = -1
                            acc_wl[min_inun_arr_ori == -2] = -2

                            bf.write_raster(min_inun_ds, inun_duration, annual_inunduration_folder,
                                            f'inun_duration_{str(_)}.tif', raster_datatype=gdal.GDT_Float32,
                                            nodatavalue=-2)
                            bf.write_raster(min_inun_ds, inun_height, annual_inunduration_folder,
                                            f'inun_height_{str(_)}.tif', raster_datatype=gdal.GDT_Float32,
                                            nodatavalue=-2)
                            bf.write_raster(min_inun_ds, acc_wl, annual_inunduration_folder,
                                            f'mean_inun_height_{str(_)}.tif', raster_datatype=gdal.GDT_Float32,
                                            nodatavalue=-2)

                    itr_v = 100 / veg_inun_itr
                    if _ in veg_height_dic.keys() and len(bf.file_filter(annual_inunthr_folder, [f'{str(_)}.tif'],
                                                                         exclude_word_list=['.xml', '.dpf', '.cpg',
                                                                                            '.aux', 'vat',
                                                                                            '.ovr'])) != veg_inun_itr:
                        bf.create_folder(annual_inunthr_folder)
                        if not os.path.exists(annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}_refined.tif'):
                            raise Exception('Please refine the inundation water level!')
                        else:
                            for _temp in bf.file_filter(annual_inunthr_folder, [f'{str(_)}.tif']):
                                os.remove(_temp)
                            min_inun_ds = gdal.Open(
                                annual_inform_folder + f'{str(_)}\\min_inun_wl_{str(_)}_refined.tif')
                            min_inun_arr_ori = min_inun_ds.GetRasterBand(1).ReadAsArray()
                            min_inun_arr = copy.deepcopy(min_inun_arr_ori)
                            min_inun_arr[min_inun_arr == 0] = np.nan
                            min_inun_arr[min_inun_arr == -2] = np.nan
                            min_inun_arr[min_inun_arr == -1] = np.nan
                            veg_h_arr = veg_height_dic[_]

                            water_level_ = []
                            itr_list = [___ for ___ in range(veg_inun_itr)]
                            for __ in range(water_level_data.shape[0]):
                                if int(water_level_data[__, 0] // 10000) == _:
                                    water_level_.append(water_level_data[__, 1])

                            veg_arr_list = [veg_h_arr * (itr_v + itr_v * thr_) / 100 + min_inun_arr for thr_ in
                                            itr_list]
                            output_npyfile = [
                                annual_inunthr_folder + f'{str(int(itr_v + itr_v * thr_))}thr_inund_{str(_)}.npy' for
                                thr_ in itr_list]

                            with concurrent.futures.ProcessPoolExecutor(max_workers=20) as exe:
                                exe.map(process_itr_wl, repeat(water_level_), veg_arr_list, output_npyfile)

                            for output_npyfile_, thr_ in zip(output_npyfile, itr_list):
                                inund_arr = np.load(output_npyfile_)
                                inund_arr[min_inun_arr_ori == 0] = 0
                                inund_arr[min_inun_arr_ori == -1] = -1
                                inund_arr[min_inun_arr_ori == -2] = -2
                                bf.write_raster(min_inun_ds, inund_arr, annual_inunthr_folder,
                                                f'{str(int(itr_v + itr_v * thr_))}thr_inund_{str(_)}.tif',
                                                raster_datatype=gdal.GDT_Int16, nodatavalue=-2)
                            veg_arr_list, output_npyfile = None, None
                    pbar.update()

            # if not os.path.exists(output_path + 'date_dc.npy') or not os.path.exists(output_path + 'date_dc.npy') or not os.path.exists(output_path + 'date_dc.npy'):
            #     for file in inundation_file:
            #         ds_temp3 = gdal.Open(file)
            #         raster_temp3 = ds_temp3.GetRasterBand(1).ReadAsArray()
            #         for length in range(len(file)):
            #             try:
            #                 date_temp = int(file[length: length + 8])
            #                 break
            #             except:
            #                 pass
            #         date_dc.append(date_temp)
            #
            #         if inundation_dc == []:
            #             inundation_dc = np.zeros([raster_temp3.shape[0], raster_temp3.shape[1], len(inundation_file)])
            #             inundation_dc[:, :, 0] = raster_temp3
            #         else:
            #             inundation_dc[:, :, num_temp] = raster_temp3
            #
            #         if not os.path.exists(sole_file_path + str(date_temp) + '_individual_area.tif'):
            #             sole_floodplain_temp = identify_all_inundated_area(raster_temp3,
            #                                                                inundated_pixel_indicator=inundated_value,
            #                                                                nanvalue_pixel_indicator=nan_value)
            #             sole_result = np.zeros_like(sole_floodplain_temp)
            #             unique_value = np.unique(sole_floodplain_temp.flatten())
            #             unique_value = np.delete(unique_value, np.argwhere(unique_value == 0))
            #             for u_value in unique_value:
            #                 if np.logical_and(sole_floodplain_temp == u_value, permanent_water_arr == 1).any():
            #                     sole_result[sole_floodplain_temp == u_value] = 1
            #             sole_result[permanent_water_arr == 1] = 0
            #             bf.write_raster(ds_temp3, sole_result, sole_file_path, str(date_temp) + '_individual_area.tif',
            #                          raster_datatype=gdal.GDT_Int32)
            #         else:
            #             sole_floodplain_ds_temp = gdal.Open(sole_file_path + str(date_temp) + '_individual_area.tif')
            #             sole_floodplain_temp = sole_floodplain_ds_temp.GetRasterBand(1).ReadAsArray()
            #
            #         if sole_area_dc == []:
            #             sole_area_dc = np.zeros([raster_temp3.shape[0], raster_temp3.shape[1], len(inundation_file)])
            #             sole_area_dc[:, :, 0] = sole_floodplain_temp
            #         else:
            #             sole_area_dc[:, :, num_temp] = sole_floodplain_temp
            #         num_temp += 1
            #     date_dc = np.array(date_dc).transpose()
            #
            #     if date_dc.shape[0] == sole_area_dc.shape[2] == inundation_dc.shape[2]:
            #         np.save(output_path + 'date_dc.npy', date_dc)
            #         np.save(output_path + 'sole_area_dc.npy', sole_area_dc)
            #         np.save(output_path + 'inundation_dc.npy', inundation_dc)
            #     else:
            #         print('Consistency error during output!')
            #         sys.exit(-1)
            # else:
            #     inundation_dc = np.load(output_path + 'inundation_dc.npy')
            #     sole_area_dc = np.load(output_path + 'sole_area_dc.npy')
            #     date_dc = np.load(output_path + 'date_dc.npy')
            #
            # doy_valid_list = []
            # for file_path in inundation_file:
            #     for length in range(len(file_path)):
            #         try:
            #             date_temp = int(file_path[length: length + 8])
            #             break
            #         except:
            #             pass
            #         doy_valid_list.append(date_temp)
            #
            # year_list = np.unique(np.fix(date_dc // 10000).astype(np.int))
            # annual_inundation_folder = output_path + 'annual_inundation_status\\'
            # annual_inundation_epoch_folder = output_path + 'annual_inundation_epoch\\'
            # annual_inundation_beg_folder = output_path + 'annual_inundation_beg\\'
            # annual_inundation_end_folder = output_path + 'annual_inundation_end\\'
            # bf.create_folder(annual_inundation_folder)
            # bf.create_folder(annual_inundation_epoch_folder)
            # bf.create_folder(annual_inundation_beg_folder)
            # bf.create_folder(annual_inundation_end_folder)
            #
            # for year in year_list:
            #     inundation_temp = []
            #     sole_area_temp = []
            #     date_temp = []
            #     recession_temp = []
            #     water_level_temp = []
            #     annual_inundation_epoch = np.zeros_like(thalweg_temp.sa_map).astype(np.float)
            #     annual_inundation_status = np.zeros_like(thalweg_temp.sa_map).astype(np.float)
            #     annual_inundation_beg = np.zeros_like(thalweg_temp.sa_map).astype(np.float)
            #     annual_inundation_end = np.zeros_like(thalweg_temp.sa_map).astype(np.float)
            #     annual_inundation_status[np.isnan(thalweg_temp.sa_map)] = np.nan
            #     annual_inundation_epoch[np.isnan(thalweg_temp.sa_map)] = np.nan
            #     annual_inundation_beg[np.isnan(thalweg_temp.sa_map)] = np.nan
            #     annual_inundation_end[np.isnan(thalweg_temp.sa_map)] = np.nan
            #     water_level_epoch = []
            #     len1 = 0
            #     while len1 < date_dc.shape[0]:
            #         if date_dc[len1] // 10000 == year:
            #             date_temp.append(date_dc[len1])
            #             recession_temp.append(wl_trend_list[np.argwhere(wl_trend_list == date_dc[len1])[0, 0], 1])
            #             water_level_temp.append(water_level_data[np.argwhere(water_level_data == date_dc[len1])[0, 0], 1])
            #             if inundation_temp == [] or sole_area_temp == []:
            #                 inundation_temp = inundation_dc[:, :, len1].reshape(
            #                     [inundation_dc.shape[0], inundation_dc.shape[1], 1])
            #                 sole_area_temp = sole_area_dc[:, :, len1].reshape(
            #                     [sole_area_dc.shape[0], sole_area_dc.shape[1], 1])
            #             else:
            #                 inundation_temp = np.concatenate((inundation_temp, inundation_dc[:, :, len1].reshape(
            #                     [inundation_dc.shape[0], inundation_dc.shape[1], 1])), axis=2)
            #                 sole_area_temp = np.concatenate((sole_area_temp, sole_area_dc[:, :, len1].reshape(
            #                     [sole_area_dc.shape[0], sole_area_dc.shape[1], 1])), axis=2)
            #         len1 += 1
            #
            #     len1 = 0
            #     while len1 < water_level_data.shape[0]:
            #         if water_level_data[len1, 0] // 10000 == year:
            #             if 6 <= np.mod(water_level_data[len1, 0], 10000) // 100 <= 10:
            #                 recession_temp2 = wl_trend_list[np.argwhere(water_level_data[len1, 0])[0][0], 1]
            #                 water_level_epoch.append(
            #                     [water_level_data[len1, 0], water_level_data[len1, 1], recession_temp2])
            #         len1 += 1
            #     water_level_epoch = np.array(water_level_epoch)
            #
            #     for y_temp in range(annual_inundation_status.shape[0]):
            #         for x_temp in range(annual_inundation_status.shape[1]):
            #             if thalweg_temp.sa_map[y_temp, x_temp] != -32768:
            #                 inundation_series_temp = inundation_temp[y_temp, x_temp, :].reshape(
            #                     [inundation_temp.shape[2]])
            #                 sole_series_temp = sole_area_temp[y_temp, x_temp, :].reshape([sole_area_temp.shape[2]])
            #                 water_level_min = np.nan
            #                 recession_level_min = np.nan
            #
            #                 inundation_status = False
            #                 len2 = 0
            #                 while len2 < len(date_temp):
            #                     if inundation_series_temp[len2] == 1:
            #                         if sole_series_temp[len2] != 1:
            #                             if recession_temp[len2] < 0:
            #                                 sole_local = sole_area_temp[
            #                                              max(0, y_temp - 8): min(annual_inundation_status.shape[0],
            #                                                                      y_temp + 8),
            #                                              max(0, x_temp - 8): min(annual_inundation_status.shape[1],
            #                                                                      x_temp + 8), len2]
            #                                 if np.sum(sole_local == 1) == 0:
            #                                     inundation_series_temp[len2] == 3
            #                                 else:
            #                                     inundation_series_temp[len2] == 2
            #                             else:
            #                                 inundation_local = inundation_temp[max(0, y_temp - 8): min(
            #                                     annual_inundation_status.shape[0], y_temp + 8), max(0, x_temp - 8): min(
            #                                     annual_inundation_status.shape[1], x_temp + 8), len2]
            #                                 sole_local = sole_area_temp[
            #                                              max(0, y_temp - 5): min(annual_inundation_status.shape[0],
            #                                                                      y_temp + 5),
            #                                              max(0, x_temp - 5): min(annual_inundation_status.shape[1],
            #                                                                      x_temp + 5), len2]
            #
            #                                 if np.sum(inundation_local == -2) == 0:
            #                                     if np.sum(inundation_local == 1) == 1:
            #                                         inundation_series_temp[len2] == 6
            #                                     else:
            #                                         if np.sum(sole_local == 1) != 0:
            #                                             inundation_series_temp[len2] == 4
            #                                         else:
            #                                             inundation_series_temp[len2] == 5
            #                                 else:
            #                                     if np.sum(inundation_local == 1) == 1:
            #                                         inundation_series_temp[len2] == 6
            #                                     else:
            #                                         if np.sum(sole_local == 1) != 0:
            #                                             np.sum(sole_local == 1) != 0
            #                                             inundation_series_temp[len2] == 4
            #                                         else:
            #                                             inundation_series_temp[len2] == 5
            #                     len2 += 1
            #                 if np.sum(inundation_series_temp == 1) + np.sum(inundation_series_temp == 2) + np.sum(
            #                         inundation_series_temp == 3) == 0:
            #                     annual_inundation_status[y_temp, x_temp] = 0
            #                     annual_inundation_epoch[y_temp, x_temp] = 0
            #                     annual_inundation_beg[y_temp, x_temp] = 0
            #                     annual_inundation_end[y_temp, x_temp] = 0
            #                 elif np.sum(inundation_series_temp >= 2) / np.sum(inundation_series_temp >= 1) > 0.8:
            #                     annual_inundation_status[y_temp, x_temp] = 0
            #                     annual_inundation_epoch[y_temp, x_temp] = 0
            #                     annual_inundation_beg[y_temp, x_temp] = 0
            #                     annual_inundation_end[y_temp, x_temp] = 0
            #                 elif np.sum(inundation_series_temp == 1) != 0:
            #                     len2 = 0
            #                     while len2 < len(date_temp):
            #                         if inundation_series_temp[len2] == 1:
            #                             if np.isnan(water_level_min):
            #                                 water_level_min = water_level_temp[len2]
            #                             else:
            #                                 water_level_min = min(water_level_temp[len2], water_level_min)
            #                         elif inundation_series_temp[len2] == 2:
            #                             if np.isnan(recession_level_min):
            #                                 recession_level_min = water_level_temp[len2]
            #                             else:
            #                                 recession_level_min = min(recession_level_min, water_level_temp[len2])
            #                         len2 += 1
            #
            #                     len3 = 0
            #                     while len3 < len(date_temp):
            #                         if inundation_series_temp[len3] == 3 and (
            #                                 recession_level_min > water_level_temp[len3] or np.isnan(
            #                                 recession_level_min)):
            #                             len4 = 0
            #                             while len4 < len(date_temp):
            #                                 if inundation_series_temp[len4] == 1 and abs(recession_temp[len4]) == abs(
            #                                         recession_temp[len3]):
            #                                     if np.isnan(recession_level_min):
            #                                         recession_level_min = water_level_temp[len3]
            #                                     else:
            #                                         recession_level_min = min(recession_level_min,
            #                                                                   water_level_temp[len3])
            #                                         inundation_series_temp[len3] = 2
            #                                     break
            #                                 len4 += 1
            #                         len3 += 1
            #
            #                     len5 = 0
            #                     while len5 < len(date_temp):
            #                         if (inundation_series_temp[len5] == 4 or inundation_series_temp[len5] == 5) and \
            #                                 water_level_temp[len5] < water_level_min:
            #                             len6 = 0
            #                             while len6 < len(date_temp):
            #                                 if inundation_series_temp[len6] == 2 and abs(recession_temp[len6]) == abs(
            #                                         recession_temp[len5]):
            #                                     water_level_min = min(water_level_min, water_level_temp[len5])
            #                                     break
            #                                 len6 += 1
            #                         len5 += 1
            #
            #                     annual_inundation_status[y_temp, x_temp] = 1
            #                     if np.isnan(water_level_min):
            #                         print('WATER_LEVEL_1_ERROR!')
            #                         sys.exit(-1)
            #                     elif np.isnan(recession_level_min):
            #                         annual_inundation_epoch[y_temp, x_temp] = np.sum(
            #                             water_level_epoch[:, 1] >= water_level_min)
            #                         date_min = 200000000
            #                         date_max = 0
            #                         for len0 in range(water_level_epoch.shape[0]):
            #                             if water_level_epoch[len0, 1] >= water_level_min:
            #                                 date_min = min(date_min, water_level_epoch[len0, 0])
            #                                 date_max = max(date_max, water_level_epoch[len0, 0])
            #                         annual_inundation_beg[y_temp, x_temp] = date_min
            #                         annual_inundation_end[y_temp, x_temp] = date_max
            #                     else:
            #                         len0 = 0
            #                         annual_inundation_epoch = 0
            #                         inundation_recession = []
            #                         date_min = 200000000
            #                         date_max = 0
            #                         while len0 < water_level_epoch.shape[0]:
            #                             if water_level_epoch[len0, 2] >= 0:
            #                                 if water_level_epoch[len0, 1]     >= water_level_min:
            #                                     annual_inundation_epoch += 1
            #                                     inundation_recession.append(water_level_epoch[len0, 2])
            #                                     date_min = min(date_min, water_level_epoch[len0, 0])
            #                                     date_max = max(date_max, water_level_epoch[len0, 0])
            #                             elif water_level_epoch[len0, 2] < 0 and abs(
            #                                     water_level_epoch[len0, 2]) in inundation_recession:
            #                                 if water_level_epoch[len0, 1] >= recession_level_min:
            #                                     annual_inundation_epoch += 1
            #                                     date_min = min(date_min, water_level_epoch[len0, 0])
            #                                     date_max = max(date_max, water_level_epoch[len0, 0])
            #                             len0 += 1
            #                         annual_inundation_beg[y_temp, x_temp] = date_min
            #                         annual_inundation_end[y_temp, x_temp] = date_max
            #
            #                 elif np.sum(inundation_series_temp == 2) != 0 or np.sum(inundation_series_temp == 3) != 0:
            #                     len2 = 0
            #                     while len2 < len(date_temp):
            #                         if inundation_series_temp[len2] == 2 or inundation_series_temp[len2] == 3:
            #                             if np.isnan(recession_level_min):
            #                                 recession_level_min = water_level_temp[len2]
            #                             else:
            #                                 recession_level_min = min(recession_level_min, water_level_temp[len2])
            #                         len2 += 1
            #
            #                     len5 = 0
            #                     while len5 < len(date_temp):
            #                         if inundation_series_temp[len5] == 4 or inundation_series_temp[len5] == 5:
            #                             len6 = 0
            #                             while len6 < len(date_temp):
            #                                 if inundation_series_temp[len6] == 2 and abs(recession_temp[len6]) == abs(
            #                                         recession_temp[len5]):
            #                                     water_level_min = min(water_level_min, water_level_temp[len5])
            #                                     break
            #                                 len6 += 1
            #                         len5 += 1
            #
            #                     if np.isnan(water_level_min):
            #                         water_level_min = water_level_epoch[np.argwhere(water_level_epoch == date_temp[
            #                             max(water_level_temp.index(recession_level_min) - 3, 0)])[0][0], 1]
            #                         if water_level_min < recession_level_min:
            #                             water_level_min = recession_level_min
            #
            #                     annual_inundation_status[y_temp, x_temp] = 1
            #                     if np.isnan(water_level_min):
            #                         print('WATER_LEVEL_1_ERROR!')
            #                         sys.exit(-1)
            #                     elif np.isnan(recession_level_min):
            #                         annual_inundation_epoch[y_temp, x_temp] = np.sum(
            #                             water_level_epoch[:, 1] >= water_level_min)
            #                         date_min = 200000000
            #                         date_max = 0
            #                         for len0 in range(water_level_epoch.shape[0]):
            #                             if water_level_epoch[len0, 1] >= water_level_min:
            #                                 date_min = min(date_min, water_level_epoch[len0, 0])
            #                                 date_max = max(date_max, water_level_epoch[len0, 0])
            #                         annual_inundation_beg[y_temp, x_temp] = date_min
            #                         annual_inundation_end[y_temp, x_temp] = date_max
            #                     else:
            #                         len0 = 0
            #                         annual_inundation_    epoch = 0
            #                         inundation_recession = []
            #                         date_min = 200000000
            #                         date_max = 0
            #                         while len0 < water_level_epoch.shape[0]:
            #                             if water_level_epoch[len0, 2] >= 0:
            #                                 if water_level_epoch[len0, 1] >= water_level_min:
            #                                     annual_inundation_epoch += 1
            #                                     inundation_recession.append(water_level_epoch[len0, 2])
            #                                     date_min = min(date_min, water_level_epoch[len0, 0])
            #                                     date_max = max(date_max, water_level_epoch[len0, 0])
            #                             elif water_level_epoch[len0, 2] < 0 and abs(
            #                                     water_level_epoch[len0, 2]) in inundation_recession:
            #                                 if water_level_epoch[len0, 1] >= recession_level_min:
            #                                     annual_inundation_epoch += 1
            #                                     date_min = min(date_min, water_level_epoch[len0, 0])
            #                                     date_max = max(date_max, water_level_epoch[len0, 0])
            #                             len0 += 1
            #                         annual_inundation_beg[y_temp, x_temp] = date_min
            #                         annual_inundation_end[y_temp, x_temp] = date_max
            #
            #                 elif np.sum(inundation_series_temp == 4) != 0 or np.sum(inundation_series_temp == 5) != 0:
            #                     len2 = 0
            #                     while len2 < len(date_temp):
            #                         if inundation_series_temp[len2] == 4 or inundation_series_temp[len2] == 5:
            #                             if np.isnan(water_level_min):
            #                                 water_level_min = water_level_temp[len2]
            #                             else:
            #                                 water_level_min = min(water_level_min, water_level_temp[len2])
            #                         len2 += 1
            #
            #                     annual_inundation_status[y_temp, x_temp] = 1
            #                     if np.isnan(water_level_min):
            #                         print('WATER_LEVEL_1_ERROR!')
            #                         sys.exit(-1)
            #                     elif np.isnan(recession_level_min):
            #                         annual_inundation_epoch[y_temp, x_temp] = np.sum(
            #                             water_level_epoch[:, 1] >= water_level_min)
            #                         date_min = 200000000
            #                         date_max = 0
            #                         for len0 in range(water_level_epoch.shape[0]):
            #                             if water_level_epoch[len0, 1] >= water_level_min:
            #                                 date_min = min(date_min, water_level_epoch[len0, 0])
            #                                 date_max = max(date_max, water_level_epoch[len0, 0])
            #                         annual_inundation_beg[y_temp, x_temp] = date_min
            #                         annual_inundation_end[y_temp, x_temp] = date_max
            #     bf.write_raster(ds_temp, annual_inundation_status, annual_inundation_folder,
            #                  'annual_' + str(year) + '.tif', raster_datatype=gdal.GDT_Int32)
            #     bf.write_raster(ds_temp, annual_inundation_epoch, annual_inundation_epoch_folder,
            #                  'epoch_' + str(year) + '.tif', raster_datatype=gdal.GDT_Int32)
            #     bf.write_raster(ds_temp, annual_inundation_beg, annual_inundation_beg_folder, 'beg_' + str(year) + '.tif',
            #                  raster_datatype=gdal.GDT_Int32)
            #     bf.write_raster(ds_temp, annual_inundation_end, annual_inundation_end_folder, 'end_' + str(year) + '.tif',
            #                  raster_datatype=gdal.GDT_Int32)
            #
            #

    def _process_phemeinun_para(self, **kwargs):
        pass

    def pheme_inun_analysis(self, phemetric_index: list, inunfactor_index: list, year_range: list, ):

        # Check the initial indi
        if not self._withInunfacdc_:
            raise Exception('Please input the inunfactor dc before pheme inun analysis！')
        elif not self._withPhemetricdc_:
            raise Exception('Please input the phemetric dc before pheme inun analysis！')

        phemetric_index_, inunfactor_index_, year_range_ = [], [], []
        # Check the input arg
        for _ in phemetric_index:
            if _ in self._phemetric_namelist:
                phemetric_index_.append(_)
            else:
                print(f'{str(_)} phemetric is not input into the RSdcs')
        if len(phemetric_index_) == 0:
            raise ValueError('Please input a valid phemetric index')

        # Check the input arg
        for _ in inunfactor_index:
            if _ in self._inunfac_namelist:
                inunfactor_index_.append(_)
            else:
                print(f'{str(_)} inunfac is not input into the RSdcs')
        if len(inunfactor_index_) == 0:
            raise ValueError('Please input a valid inunfac index')

        # Check the year range
        for year in year_range:
            if isinstance(year, int) and year in self._inunyear_list and year in self._pheyear_list or year + 1 in self._pheyear_list:
                year_range_.append(year)
            else:
                print(f'{str(year)} year is not valid')
        if len(year_range_) == 0:
            raise ValueError('Please input a valid year')

        for year_ in year_range_:
            for pheme_ in phemetric_index_:
                for inun_ in inunfactor_index_:
                    pheme_next_year = self._dcs_backup_
                    pheme_curr_year = 1





