# coding=utf-8
import os.path
from scipy import sparse as sm
from itertools import repeat
import RSDatacube
from .utils import *
from NDsm import NDSparseMatrix
import concurrent.futures


class Phemetric_dc(object):

    ####################################################################################################
    # Phemetric_dc represents "Phenological Metric Datacube"
    # It normally contains data like phenological parameters derived from the curve fitting, etc
    # And stack into a 3-D datacube type.
    # Currently, it was taken as the Sentinel_dcs/Landsat_dcs's output datatype after the phenological analysis.
    ####################################################################################################

    def __init__(self, phemetric_filepath, work_env=None):

        # Check the phemetric path
        self.Phemetric_dc_filepath = Path(phemetric_filepath).path_name

        # Init key var
        self.ROI_name, self.ROI, self.ROI_tif, self.ROI_array = None, None, None, None
        self.index, self.Datatype, self.coordinate_system = None, None, None
        self.dc_group_list, self.tiles = None, None
        self.sdc_factor, self.sparse_matrix, self.size_control_factor, self.huge_matrix = False, False, False, False
        self.Phemetric_factor, self.pheyear = False, None
        self.curfit_dic = {}
        self.Nodata_value = None
        self.oritif_folder = None
        self.ori_dc_folder = None
        self.Zoffset = None

        # Init protected var
        self._support_pheme_list = ['SOS', 'EOS', 'trough_vi', 'peak_vi', 'peak_doy', 'GR', 'DR', 'DR2', 'MAVI', 'TSVI', 'MAVI_std', 'EOM']

        # Check work env
        if work_env is not None:
            self._work_env = Path(work_env).path_name
        else:
            self._work_env = Path(os.path.dirname(os.path.dirname(self.Phemetric_dc_filepath))).path_name
        self.root_path = Path(os.path.dirname(os.path.dirname(self._work_env))).path_name

        # Define the basic var name
        self._fund_factor = ('ROI_name', 'index', 'Datatype', 'ROI', 'ROI_array', 'curfit_dic', 'pheyear',
                             'coordinate_system', 'oritif_folder', 'ROI_tif', 'sparse_matrix', 'ori_dc_folder',
                             'huge_matrix', 'size_control_factor', 'dc_group_list', 'tiles', 'Nodata_value', 'Zoffset')

        # Read the metadata file
        metadata_file = file_filter(self.Phemetric_dc_filepath, ['metadata.json'])
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
                paraname_file = file_filter(self.Phemetric_dc_filepath, ['paraname.npy'])
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
                self.dc_filename = file_filter(self.Phemetric_dc_filepath, ['Phemetric_datacube.npy'])
                if len(self.dc_filename) == 0:
                    raise ValueError('There has no valid Phemetric datacube or the dc was missing!')
                elif len(self.dc_filename) > 1:
                    raise ValueError('There has more than one data file in the dc dir')
                else:
                    self.dc = np.load(self.dc_filename[0], allow_pickle=True)
            elif self.huge_matrix and not self.sparse_matrix:
                self.dc_filename = file_filter(self.Phemetric_dc_filepath, ['sequenced_datacube', '.npy'], and_or_factor='and')
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

    def _update_para_size_(self):
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
            self.coordinate_system = retrieve_srs(ds_temp)
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

    def compute_phemetrics(self, pheme_list: list, save2phemedc: bool = True, replace = False):

        # Process the pheme compute list
        pheme_list_temp = copy.copy(pheme_list)
        for pheme_temp in pheme_list:
            if pheme_temp not in self._support_pheme_list:
                raise ValueError(f'The {pheme_temp} is not supported')
            elif f'{self.pheyear}_{pheme_temp}' in self.paraname_list and replace is False:
                pheme_list_temp.remove(pheme_temp)
            elif f'{self.pheyear}_{pheme_temp}' in self.paraname_list and replace is True:
                self.remove_layer(f'{self.pheyear}_{pheme_temp}')
        pheme_list = pheme_list_temp

        # Identify the curfit function
        if self.curfit_dic['CFM'] == 'SPL':

            # Extract the pheme parameters
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

            # Separate the pheme list into the copy and computed series
            pheme_list_copy = [_ for _ in pheme_list if _ in ['trough_vi', 'GR', 'DR', 'DR2']]
            pheme_list_computed = [_ for _ in pheme_list if _ in ['SOS', 'EOS', 'peak_doy', 'EOM', 'MAVI', 'peak_vi', 'MAVI_std', 'TSVI']]

            # Copy the dc of pheme same as the para
            for pheme_temp in pheme_list_copy:
                para_index = [0, 3, 5, 6][['trough_vi', 'GR', 'DR', 'DR2'].index(pheme_temp)]
                arr_ = para_dic[para_index]
                arr_[np.isnan(arr_)] = self.Nodata_value
                self._add_layer(arr_, pheme_temp)

            # Compute the dc of pheme in different phenophases
            # Separate the dc for multiprocessing
            if len(pheme_list_computed) > 0:
                x_itr = int(np.floor(para_dic[0].shape[1] / os.cpu_count()))
                para_dic_separated, ul_pos_separated = [], []
                for _ in range(os.cpu_count()):
                    para_dic_temp = {}
                    for key_ in para_dic.keys():
                        if _ == os.cpu_count() - 1:
                            para_dic_temp[key_] = para_dic[key_][:, _ * x_itr:]
                        else:
                            para_dic_temp[key_] = para_dic[key_][:, _ * x_itr: (_ + 1) * x_itr]
                    para_dic_separated.append(para_dic_temp)
                    ul_pos_separated.append([0, _ * x_itr])

                with concurrent.futures.ProcessPoolExecutor(max_workers = int(RSDatacube.configuration['multiprocess_ratio'] * os.cpu_count())) as executor:
                    res = executor.map(compute_SPDL_pheme, para_dic_separated, repeat(pheme_list_computed), ul_pos_separated)
                res = list(res)

                # Reassign into the arr
                for pheme_ in pheme_list_computed:
                    pheme_arr = np.zeros_like(para_dic[list(para_dic.keys())[0]]) * np.nan
                    for _ in res:
                        pheme_arr[_[1][0]: _[1][0] + _[0][pheme_].shape[0], _[1][1]: _[1][1] + _[0][pheme_].shape[1]] = _[0][pheme_]
                    pheme_arr[np.isnan(pheme_arr)] = self.Nodata_value
                    self._add_layer(pheme_arr, pheme_)
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

        if isinstance(self.dc, NDSparseMatrix):
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
            create_folder(output_path)
        output_path = Path(output_path).path_name

        metadata_dic = {'ROI_name': self.ROI_name, 'index': self.index, 'Datatype': self.Datatype, 'ROI': self.ROI,
                        'ROI_array': self.ROI_array, 'ROI_tif': self.ROI_tif, 'sdc_factor': self.sdc_factor,
                        'coordinate_system': self.coordinate_system, 'sparse_matrix': self.sparse_matrix,
                        'huge_matrix': self.huge_matrix,
                        'size_control_factor': self.size_control_factor, 'oritif_folder': self.oritif_folder,
                        'dc_group_list': self.dc_group_list, 'tiles': self.tiles,
                        'pheyear': self.pheyear, 'curfit_dic': self.curfit_dic,
                        'Phemetric_factor': self.Phemetric_factor,
                        'Nodata_value': self.Nodata_value, 'Zoffset': self.Zoffset, 'ori_dc_folder': self.ori_dc_folder}

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
            create_folder(output_folder)
            output_folder = Path(output_folder).path_name

        ds_temp = gdal.Open(self.ROI_tif)
        for phe_ in phe_list:
            if not os.path.exists(output_folder + f'{str(phe_)}.TIF'):
                if self.sparse_matrix:
                    phe_arr = self.dc.SM_group[phe_].toarray()
                else:
                    phe_arr = self.dc[:, :, self.paraname_list.index(phe_)]
                phe_arr[phe_arr == self.Nodata_value] = np.nan
                write_raster(ds_temp, phe_arr, output_folder, f'{str(phe_)}.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)