import basic_function as bf
import gdal
import os


class MODIS_ds(object):

    def __init__(self, file_path, work_env=None):

        if work_env is None:
            try:
                self.work_env = bf.Path(os.path.dirname(os.path.dirname(file_path)) + '\\').path_name
            except:
                print('There has no base dir for the ori_folder and the ori_folder will be treated as the work env')
                self.work_env = bf.Path(file_path).path_name
        else:
            self.work_env = bf.Path(work_env).path_name

        # Init key variable
        self.ROI, self.ROI_name = None, None
        self.main_coordinate_system = None
        self.hdf_files = bf.file_filter(file_path, ['.hdf'], subfolder_detection=True)

        # Obtain the doy list
        self.doy_list = []
        for file in self.hdf_files:
            eles = file.split('\\')[-1].split('.')
            for ele in eles:
                if ele.startswith('A'):
                    self.doy_list.append(int(ele.split('A')[-1]))

        if len(self.doy_list) != len(self.hdf_files):
            raise Exception('Code Error when obtaining the doy list!')

        #

        # Define cache folder
        self.cache_folder, self.trash_folder = self.work_env + 'cache\\', self.work_env + 'trash\\'
        bf.create_folder(self.cache_folder)
        bf.create_folder(self.trash_folder)

        # Create output path
        self.output_path, self.shpfile_path, self.log_filepath = f'{self.work_env}MODIS_Output\\', f'{self.work_env}shpfile\\', f'{self.work_env}logfile\\'
        bf.create_folder(self.output_path)
        bf.create_folder(self.log_filepath)
        bf.create_folder(self.shpfile_path)

    def seq_hdf2tif(self, subname):
        for i in self.doy_list:
            self._hdf2tif(self.output_path, [q for q in self.hdf_files if f'.A{str(i)}.' in q], i, subname)

    def _hdf2tif(self, output_path: str, files: list, doy: int, subname_list: list, merge_factor=True, ROI: str = None, bounds: list=None, ras_size: list=None, crs: str=None):

        # Process the hdf file
        for file in files:
            if str(doy) not in file:
                raise TypeError(f'The {str(file)} does not belong to {str(doy)}!')

        # Process kwargs
        if ROI is not None and isinstance(ROI, str):
            if ROI.endswith('.shp'):
                self.ROI, self.ROI_name = ROI, ROI.split('\\')[-1].split('.')[0]
            else:
                self.ROI, self.ROI_name = None, None

        if not isinstance(merge_factor, bool):
            raise TypeError(f'The merge_factor should under bool type!')
        elif merge_factor is False and self.ROI is not None:
            raise TypeError(f'The merge factor should properly be set as True since the ROI has input!')

        if bounds is None:
            bounds = None
        elif not isinstance(bounds, list):
            raise TypeError(f'The bounds should under list type!')
        elif len(bounds) != 4:
            raise TypeError(f'The bounds should under list type (Xmin, Ymin, Xmax, Ymax)!')

        if ras_size is None:
            ras_size = None
        elif not isinstance(ras_size, list):
            raise TypeError(f'The bounds should under list type!')
        elif len(bounds) != 2:
            raise TypeError(f'The ras size should under list type (Ysize, Xsize)!')

        if crs is None:
            crs = None
        elif not isinstance(crs, str):
            raise TypeError(f'The csr should under str type!')

        # Create output folder
        ori_output_folder = f'{output_path}ori\\'
        roi_output_folder = f'{output_path}{str(self.ROI_name)}\\' if self.ROI_name is not None else None

        # Determine the subdataset
        ds_temp = gdal.Open(files[0])
        subset_ds = ds_temp.GetSubDatasets()
        subname_supported = [subset[0].split(':')[-1] for subset in subset_ds]
        subname_dic = {}
        for subname in subname_list:
            if subname not in subname_supported:
                raise TypeError(f'{subname} is not supported!')
            else:
                subname_dic[subname] = []

        # Separate all the hdffiles through the index
        for file in files:
            ds_temp = gdal.Open(files[0])
            subset_ds = ds_temp.GetSubDatasets()
            subname_temp = [subset[0].split(':')[-1] for subset in subset_ds]
            for subname in subname_list:
                if subname not in subname_temp:
                    raise TypeError(f'The file {file} is not consistency compared to other files in the ds!')
                else:
                    subname_dic[subname].append(subset_ds[subname_temp.index(subname)][0])

        # Create the tiffiles based on each index
        for subname in subname_list:
            for file in subname_dic[subname]:
                ds_temp = gdal.Open(file)
                vrt_temp = gdal.Warp(ds_temp)


if __name__ == '__main__':
    temp = MODIS_ds('G:\\A_veg\\MODIS_FPAR\\Ori\\')
    temp.seq_hdf2tif(['PAR'])








# def hdf2tif():
#     # USER SPECIFIED
#     dir = 'E:\\modis_temp\\hdf\\'
#
#     # Pre-defined para
#     xsize = 500
#     ysize = 500
#     output_dir = dir + 'output\\'
#     bf.create_folder(output_dir)
#
#     # Retrieve all the files in dir
#     # file_list = file_filter(dir, ['.hdf', '2022'], and_or_factor='and')
#     file_list = bf.file_filter(dir, ['.hdf'])
#
#     for file in file_list:
#         ds = gdal.Open(file)
#
#         subdatasets = ds.GetSubDatasets()
#         print('Number of subdatasets: {}'.format(len(subdatasets)))
#         for sd in subdatasets:
#             print('Name: {0}\nDescription:{1}\n'.format(*sd))
#
#         # USER SPECIFIED
#         LAI_ds = gdal.Open(subdatasets[1][0])
#         LAI_array = gdal.Open(subdatasets[1][0]).ReadAsArray()
#         temp_file = '/vsimem/' + file.split('\\')[-1].split('.hdf')[0] + '.tif'
#         dst_filename = output_dir + file.split('\\')[-1].split('.hdf')[0] + '.tif'
#         ulx, xres, xskew, uly, yskew, yres = LAI_ds.GetGeoTransform()
#         array2raster('/vsimem/' + file.split('\\')[-1].split('.hdf')[0] + '.tif', [ulx, uly], xsize, ysize, LAI_array)
#         gdal.Warp(dst_filename, '/vsimem/' + file.split('\\')[-1].split('.hdf')[0] + '.tif', dstSRS='EPSG:32649')
#         gdal.Unlink('/vsimem/' + file.split('\\')[-1].split('.hdf')[0] + '.tif')
