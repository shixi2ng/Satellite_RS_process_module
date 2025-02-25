import scipy.sparse as sm
import pickle
import basic_function as bf
import numpy as np
import os
from tqdm.auto import tqdm


class NDSparseMatrix:

    ### The NDSparseMatrix is a data class specified for the huge and sparse N-dimensional matrix
    # The code strategy is different from the normal ND-array in the following aspects:
    # (1) The matrix is sliced by last dimension (Z-axis for the 3D array etc.)
    # (2) The matrix is not stored in one file but one file per layer in a folder
    # (3) Due to (2), the NDSparseMatrix is hard to index information in the z-dimension

    def __init__(self, *args, **kwargs):
        self.SM_group = {}
        self._matrix_type = None
        self._cols = -1
        self._rows = -1
        self._height = -1
        self.shape = [self._rows, self._cols, self._height]
        self.SM_namelist = None
        self.file_size = 0

        for kw_temp in kwargs.keys():
            if kw_temp not in ['SM_namelist']:
                raise KeyError(f'Please input the a right kwargs!')
            else:
                self.__dict__[kw_temp] = kwargs[kw_temp]

        if 'SM_namelist' in self.__dict__.keys():
            if self.SM_namelist is not None:
                if isinstance(self.SM_namelist, list):
                    if len(args) != len(self.SM_namelist):
                        raise ValueError(f'Please make sure the sm name list is consistent with the SM_group')
                else:
                    raise TypeError(f'Please input the SM_namelist under the list type!')
            else:
                if len(args) == 0:
                    self.SM_namelist = []
                else:
                    self.SM_namelist = [i for i in range(len(args))]
        else:
            if len(args) == 0:
                self.SM_namelist = []
            else:
                self.SM_namelist = [i for i in range(len(args))]
        i = 0
        for ele in args:
            if type(ele) not in (sm.spmatrix, sm.csr_matrix, sm.lil_matrix, sm.csc_matrix, sm.coo_matrix, sm.bsr_matrix, sm.dia_matrix, sm.dok_matrix):
                raise TypeError(f'Please input the {str(ele)} under sparse matrix type!')
            else:
                if self._matrix_type is None:
                    self._matrix_type = type(ele)
                elif self._matrix_type != type(ele):
                    raise TypeError(f'The {str(ele)} is not under the common type {str(self._matrix_type)}!')
                self.SM_group[self.SM_namelist[i]] = ele
            i += 1

        self._update_size_para()

    def __sizeof__(self):
        try:
            return len(pickle.dumps(self))
        except MemoryError:
            size = self.SM_namelist.__sizeof__()
            for temp in self.SM_group:
                size += len(pickle.dumps(temp))

    def __getitem__(self, keys):
        if len(keys) == 3:
            y_r, x_r, z_r = keys
        else:
            raise TypeError('ND matrix was under at least three dimension')

        arr_list = []
        if isinstance(z_r, (int, np.int8, np.int16, np.int32, np.int64)) or z_r.stop - z_r.start == 1 or z_r.stop - z_r.start == 0:
            if z_r.stop - z_r.start == 1 or z_r.stop - z_r.start == 0:
                z_r = z_r.start

            if isinstance(y_r, (int, np.int8, np.int16, np.int32, np.int64)) and isinstance(x_r, (int, np.int8, np.int16, np.int32, np.int64)):
                return self.SM_group[self.SM_namelist[z_r]][y_r, x_r].reshape(y_r.stop - y_r.start, x_r.stop - x_r.start, 1)
            else:
                return self.SM_group[self.SM_namelist[z_r]][y_r, x_r].toarray().reshape(y_r.stop - y_r.start, x_r.stop - x_r.start, 1)
        else:
            for _ in range(self.shape[2])[z_r]:
                if isinstance(x_r, (int, np.int8, np.int16, np.int32, np.int64)) and isinstance(y_r, (int, np.int8, np.int16, np.int32, np.int64)):
                    arr_list.append(np.array([[self.SM_group[self.SM_namelist[_]][y_r, x_r]]]))
                else:
                    arr_list.append(self.SM_group[self.SM_namelist[_]][y_r, x_r].toarray())
            return np.stack(arr_list, axis=2)

    def _update_size_para(self):
        for ele in self.SM_group.values():
            if self._cols == -1 or self._rows == -1:
                self._cols, self._rows = ele.shape[1], ele.shape[0]
            elif ele.shape[1] != self._cols or ele.shape[0] != self._rows:
                raise Exception(f'Consistency Error for the {str(ele)}')
        self._height = len(self.SM_namelist)
        self.shape = [self._rows, self._cols, self._height]

    def append(self, sm_matrix, name=None, pos=-1):
        if type(sm_matrix) not in (sm.spmatrix, sm.csr_matrix, sm.csc_matrix, sm.coo_matrix, sm.bsr_matrix, sm.dia_matrix, sm.dok_matrix, sm.lil_matrix):
            raise TypeError(f'The new sm_matrix is not a sm_matrix')
        elif type(sm_matrix) != self._matrix_type:
            if self._matrix_type is None:
                self._matrix_type = type(sm_matrix)
            else:
                raise TypeError(f'The new sm matrix is not under the same type within the 3d sm matrix')

        if name is None:
            try:
                name = int(self.SM_namelist[-1]) + 1
            except:
                name = 0

        if pos == -1:
            self.SM_namelist.append(name)
        elif pos not in range(len(self.SM_namelist)):
            print(f'The pos{str(pos)} is not in the range')
            self.SM_namelist.append(name)
        else:
            self.SM_namelist.insert(pos, name)

        self.SM_group[name] = sm_matrix
        self._update_size_para()

    def extend(self, sm_matrix_list, name=None):
        if type(sm_matrix_list) is not list:
            raise TypeError(f'Please input the sm_matrix_list as a list!')

        if name is not None and type(name) is not list:
            raise TypeError(f'Please input the name list as a list!')
        elif len(name) != len(sm_matrix_list):
            raise Exception(f'Consistency error')

        i = 0
        for sm_matrix in sm_matrix_list:
            if name is None:
                self.append(sm_matrix, name=name)
            else:
                self.append(sm_matrix, name=name[i])
            i += 1

    def save(self, output_path, overwritten_para=True):

        bf.create_folder(output_path)
        output_path = bf.Path(output_path).path_name
        i = 0

        with tqdm(total=len(self.SM_namelist), desc=f'Saving the N-D sparse matrix', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for sm_name in self.SM_namelist:
                if not os.path.exists(output_path + str(sm_name) + '.npz') or overwritten_para:
                    if isinstance(self.SM_group[sm_name], sm.lil_matrix):
                        sm_temp = sm.csc_matrix(self.SM_group[sm_name])
                        sm.save_npz(output_path + str(sm_name) + '.npz', sm_temp)
                    else:
                        sm.save_npz(output_path + str(sm_name) + '.npz', self.SM_group[sm_name])
                i += 1
                pbar.update()

        np.save(output_path + 'SMsequence.npz', np.array(self.SM_namelist))

    def load(self, input_path):

        input_path = bf.Path(input_path).path_name
        file_list = bf.file_filter(input_path, ['SMsequence.npz'])

        if len(file_list) == 0:
            raise ValueError('The header file is missing！')
        elif len(file_list) > 1:
            raise ValueError('There are more than one header file！')
        else:
            try:
                header_file = np.load(file_list[0], allow_pickle=True).astype(int)
            except ValueError:
                header_file = np.load(file_list[0], allow_pickle=True)

        self.SM_namelist = np.sort(header_file).tolist()
        self.SM_group = {}
        missing_sm = []

        for SM_name in self.SM_namelist:
            SM_arr_path = bf.file_filter(input_path, [f'{str(SM_name)}.npz'], and_or_factor='and')
            if len(SM_arr_path) == 0:
                missing_sm.append(SM_name)
                # raise ValueError(f'The file {str(SM_name)} is missing！')
            elif len(SM_arr_path) > 1:
                raise ValueError(f'There are more than one file sharing name {str(SM_name)}')
            else:
                try:
                    SM_arr_temp = sm.load_npz(SM_arr_path[0])
                except:
                    raise Exception(f'file {str(SM_name)} cannot be loaded')
                self.SM_group[SM_name] = SM_arr_temp

        if missing_sm:
            for _ in missing_sm:
                self.SM_namelist.remove(_)
            self.save(input_path, overwritten_para=True)

        # # Remove redundant npz
        # file_list = os.listdir(input_path)
        # for _ in file_list:
        #     if str(_.split('.npz')[0]) not in thalweg_temp.SM_namelist and _.split('.npz')[0] != 'SMsequence':
        #         try:
        #             os.remove(input_path + _)
        #         except:
        #             pass

        self._update_size_para()
        self._matrix_type = type(SM_arr_temp)
        return self

    def replace_layer(self, ori_layer_name, new_layer, new_layer_name=None):

        if type(new_layer) not in (sm.spmatrix, sm.csr_matrix, sm.csc_matrix, sm.coo_matrix, sm.bsr_matrix, sm.dia_matrix, sm.dok_matrix):
            raise TypeError(f'The new sm_matrix is not a sm_matrix')
        elif type(new_layer) != self._matrix_type:
            raise TypeError(f'The new sm_matirx is not under the same type within the 3d sm matrix')

        if new_layer_name is None:
            if ori_layer_name not in self.SM_namelist:
                raise ValueError(f'The {ori_layer_name} cannot be found')
            else:
                self.SM_group[ori_layer_name] = new_layer
        else:
            self.SM_group[ori_layer_name] = new_layer
            self.SM_namelist[self.SM_namelist.index(ori_layer_name)] = new_layer_name
        self._update_size_para()

    def add_layer(self, new_layer, new_layer_name, pos: int):

        if not isinstance(new_layer, (sm.spmatrix, sm.csr_matrix, sm.csc_matrix, sm.coo_matrix, sm.bsr_matrix, sm.dia_matrix, sm.dok_matrix)):
            raise TypeError(f'The new sm_matrix is not a sm_matrix')
        elif not isinstance(new_layer, self._matrix_type):
            raise TypeError(f'The new sm_matrix is not under the same type within the 3d sm matrix')

        try:
            self.SM_namelist.insert(pos, new_layer_name)
            self.SM_group[new_layer_name] = new_layer
        except:
            raise Exception('Error occurred when add layer!')

        self._update_size_para()

    def extend_layers(self, ndsm_temp):

        if not isinstance(ndsm_temp, NDSparseMatrix):
            raise TypeError('The extended layers should under ndsm type')
        else:
            for _ in range(ndsm_temp.shape[2]):
                self.add_layer(ndsm_temp.SM_group[ndsm_temp.SM_namelist[_]], ndsm_temp.SM_namelist[_], self.shape[2])
        self._update_size_para()

    def remove_layer(self, layer_name):

        if layer_name not in self.SM_namelist:
            raise ValueError(f'The {layer_name} cannot be found')
        else:
            self.SM_group.pop(layer_name)
            self.SM_namelist.remove(layer_name)
        self._update_size_para()

    def _understand_range(self, list_temp: list, range_temp: range):

        if len(list_temp) == 1:
            if list_temp[0] == 'all':
                return [min(range_temp), max(range_temp)]
            elif isinstance(list_temp[0], (int, np.int16, np.int32, np.int64)) and list_temp[0] in range_temp:
                return [list_temp[0], list_temp[0] + 1]
            else:
                raise ValueError('Please input a supported type!')

        elif len(list_temp) == 2:
            if isinstance(list_temp[0], (int, np.int16, np.int32, np.int64)) and isinstance(list_temp[1], (int, np.int16, np.int32, np.int64)):
                if list_temp[0] >= range_temp.start and list_temp[1] <=range_temp.stop and list_temp[0] <= list_temp[1]:
                    return [list_temp[0], list_temp[1]]
            else:
                raise ValueError('Please input a supported type!')

        elif len(list_temp) > 2:
            raise ValueError('Please input a supported type!')

    def extract_matrix(self, tuple_temp: tuple):

        if len(tuple_temp) != 3 or type(tuple_temp) != tuple:
            raise TypeError(f'Please input the index array in a 3D tuple')
        else:
            rows_range = self._understand_range(tuple_temp[0], range(self._rows + 1))
            cols_range = self._understand_range(tuple_temp[1], range(self._cols + 1))
            heights_range = self._understand_range(tuple_temp[2], range(self._height + 1))

        output_array = None
        height_temp = 0
        while height_temp < self._height:
            if height_temp in range(heights_range[0], heights_range[1]):
                if output_array is None:
                    if isinstance(self.SM_group[self.SM_namelist[height_temp]], sm.coo_matrix):
                        output_array = NDSparseMatrix(sm.coo_matrix(sm.csc_matrix(self.SM_group[self.SM_namelist[height_temp]])[rows_range[0]: rows_range[1], cols_range[0]: cols_range[1]]), SM_namelist=[self.SM_namelist[height_temp]])
                    else:
                        output_array = NDSparseMatrix(self.SM_group[self.SM_namelist[height_temp]][rows_range[0]: rows_range[1], cols_range[0]: cols_range[1]], SM_namelist=[self.SM_namelist[height_temp]])
                else:
                    if isinstance(self.SM_group[self.SM_namelist[height_temp]], sm.coo_matrix):
                        output_array.add_layer(sm.coo_matrix(sm.csc_matrix(self.SM_group[self.SM_namelist[height_temp]])[rows_range[0]: rows_range[1], cols_range[0]: cols_range[1]]), self.SM_namelist[height_temp], output_array.shape[2])
                    else:
                        output_array.add_layer(self.SM_group[self.SM_namelist[height_temp]][rows_range[0]: rows_range[1], cols_range[0]: cols_range[1]], self.SM_namelist[height_temp], output_array.shape[2])
            height_temp += 1

        output_array._cols, output_array._rows = -1, -1
        output_array._update_size_para()

        if output_array._cols != cols_range[1] - cols_range[0] or output_array._rows != rows_range[1] - rows_range[0] or output_array._height != heights_range[1] - heights_range[0]:
            raise Exception('Code error for the NDsparsematrix extraction')
        return output_array

    def _extract_matrix_y1x1zh(self, tuple_temp: tuple, nodata_export=False):

        # tt0, tt1, tt2 = 0, 0, 0
        # start_time = time.time()
        if len(tuple_temp) != 3 or type(tuple_temp) != tuple:
            raise TypeError(f'Please input the index array in a 3D tuple')
        elif len(tuple_temp[0]) != 1 or len(tuple_temp[1]) != 1:
            raise TypeError(f'This func is for y1x1zh datacube!')
        else:
            heights_range = self._understand_range(tuple_temp[2], range(self._height))
            rows_extract = tuple_temp[0][0]
            cols_extract = tuple_temp[1][0]
        # tt0 += time.time() - start_time

        date_temp, index_temp = [], []
        for height_temp in range(self._height):
            if height_temp in range(heights_range[0], heights_range[1]):
                date_tt = self.SM_namelist[height_temp]
                temp = self.SM_group[date_tt][rows_extract, cols_extract]
                if nodata_export:
                    date_temp.append(date_tt)
                    index_temp.append(temp)
                elif not nodata_export and temp != 0 and temp > 0:
                    date_temp.append(date_tt)
                    index_temp.append(temp)

        # tt1 += time.time() - start_time
        year_doy_all = np.array(bf.date2doy(date_temp))
        date_temp = np.mod(year_doy_all, 1000)
        index_temp = np.array(index_temp)

        # tt2 += time.time() - start_time
        # print(f'tt0:{str(tt0)}, tt1:{str(tt1)}, tt2:{str(tt2)}')
        return date_temp, index_temp, year_doy_all

    def _extract_matrix_y1x1zh_v2(self, tuple_temp: tuple, nodata_export= False):

        # tt0, tt1, tt2 = 0, 0, 0
        # start_time = time.time()
        if len(tuple_temp) != 3 or type(tuple_temp) != tuple:
            raise TypeError(f'Please input the index array in a 3D tuple')
        elif len(tuple_temp[0]) != 1 or len(tuple_temp[1]) != 1:
            raise TypeError(f'This func is for y1x1zh datacube!')
        else:
            heights_range = self._understand_range(tuple_temp[2], range(self._height))
            rows_extract = tuple_temp[0][0]
            cols_extract = tuple_temp[1][0]

        index_temp = []
        for height_temp in range(self._height):
            if height_temp in range(heights_range[0], heights_range[1]):
                date_tt = self.SM_namelist[height_temp]
                temp = self.SM_group[date_tt][rows_extract, cols_extract]
                if nodata_export:
                    index_temp.append(temp)
                elif not nodata_export and temp != 0:
                    index_temp.append(temp)

        # tt1 += time.time() - start_time
        index_temp = np.array(index_temp)

        return index_temp

    def drop_nanlayer(self):

        i = 0
        while i < len(self.SM_namelist):
            name = self.SM_namelist[i]
            if self.SM_group[name].data.shape[0] == 0:
                self.remove_layer(name)
                i -= 1
            i += 1
        self._update_size_para()

        return self

    def sum(self, axis: int, new_layer_name=None):

        if axis == 0 or axis == 1:
            for _ in self.SM_namelist:
                self.SM_group[_] = self.SM_group[_].sum(axis=axis)
            self._update_size_para()
        elif axis == 2:
            temp = None
            for _ in self.SM_namelist:
                if temp is None:
                    temp = self.SM_group[_].astype(np.float)
                else:
                    temp = temp + self.SM_group[_].astype(np.float)
            if new_layer_name is None:
                new_layer_name = 'sum'
            elif isinstance(new_layer_name, str):
                new_layer_name = new_layer_name
            self.SM_group = {new_layer_name: temp}
            self.SM_namelist = [new_layer_name]
            self._update_size_para()
        else:
            raise TypeError(f'The nd sparse matrix donot have a axis {str(axis)}')

        return self

