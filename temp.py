import basic_function as bf
import numpy as np
import scipy.sparse as sm


if __name__ == '__main__':
    i = 'OSAVI_20m'
    file = f'G:\A_veg\S2_all\Sentinel2_L2A_Output\Sentinel2_MYZR_FP_2020_datacube\{i}_sequenced_datacube\{i}_sequenced_datacube\\'
    files = bf.file_filter(file, ['.npz'], exclude_word_list=['npy'])
    for file_temp in files:
        array = sm.load_npz(file_temp)
        array = sm.coo_matrix(array.toarray().astype(np.uint16))
        array.col = array.col.astype(np.uint16)
        array.row = array.row.astype(np.uint16)
        sm.save_npz(file_temp, array)