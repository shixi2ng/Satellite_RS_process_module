import ee
import os
import requests
import sympy
import datetime
import numpy as np
import time
import zipfile
import rivamap as rm
import geopandas as gp


#######################################################################################################################
# For how to activate the GEE python-api of your personal account, please follow the guide show in
# https://developers.google.com/earth-engine/guides/python_install
# Meantime, you can check the cookbook of GEE on https://developers.google.com/earth-engine
#######################################################################################################################


class GEE_ds(object):

    def __init__(self):
        self._band_output_list = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10']
        self._all_supported_index_list = ['RGB', 'QA', 'all_band', '4visual', 'NDVI', 'MNDWI', 'EVI', 'EVI2', 'OSAVI',
                                          'GNDVI', 'NDVI_RE', 'NDVI_RE2', 'AWEI', 'AWEInsh']
        self._band_tab = {'LE07_bandnum': ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'),
                          'LE07_bandname': ('BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'TIR', 'SWIR2'),
                          'LT05_bandnum': ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'SR_B8'),
                          'LT05_bandname': ('BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'TIR', 'SWIR2', 'PAN'),
                          'LC08_bandnum': ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'SR_B8', 'SR_B10'),
                          'LC08_bandname': ('AER', 'BLUE', 'GREEN', 'RED', 'NIR', 'SWIR', 'SWIR2', 'PAN', 'TIR'),
                          }
        self._support_satellite = ['LC08', 'LC09', 'LE05', 'LE07']
        self._built_in_index = built_in_index()

    def download_index_GEE(self, satellite, date_range, index, ROI, outputpath, export_QA_file=True):

        # Initialize the Earth Engine module
        ee.Initialize()

        ## Check if the satellite meet the requirement!
        if isinstance(satellite, str):
            if satellite not in self._support_satellite:
                raise ValueError('The input satellite is not supported!')
        else:
            raise TypeError('The input satellite should be a string type!')

        ## Check if the date range meet the requirement!
        if isinstance(date_range, (list, tuple)) and len(date_range):
            try:
                start_date, end_date = int(date_range[0]), int(date_range[1])
                end_date = datetime.date(year=int(end_date // 10000), month=int(np.mod(end_date, 10000) // 100), day=int(np.mod(end_date, 100))).strftime('%Y-%m-%d')
                start_date = datetime.date(year=int(start_date // 10000), month=int(np.mod(start_date, 10000) // 100), day=int(np.mod(start_date, 100))).strftime('%Y-%m-%d')
            except:
                raise TypeError('Both the start date and end date should under the YYYYMMDD format!')
        else:
            raise TypeError('The input date range should either be a list or tuple type!')

        ## Check if the index meet the requirement!
        if isinstance(index, str):
            if index not in self._all_supported_index_list:
                raise ValueError('The input index is not supported!')
            else:
                index_express = self._built_in_index.__dict__[index].split('=')[-1]

        else:
            raise TypeError('The input index should be a string type!')

        ## Check if the roi meet the requirement!
        if isinstance(ROI, str) and os.path.exists(ROI) and ROI.endswith('.shp'):
            shapefile = gp.read_file(ROI)
            geojson = shapefile.geometry[0].__geo_interface__
            roi = ee.Geometry(geojson)
            roi_name = ROI.split('\\')[-1].split('.shp')[0]
        elif isinstance(ROI, (list, tuple)) and len(ROI) == 4:
            try:
                roi = ee.Geometry.Rectangle([99.70322434775323, 33.80530886069177, 99.49654404990167, 33.73681471587109])
                roi_name = 'roi'
            except:
                raise ValueError('The input coordinate for ROI is invalid')
        else:
            raise TypeError('The input index should be a string type!')

        ## Create output path:
        if isinstance(outputpath, str):
            if not os.path.exists(outputpath):
                create_folder(outputpath)
            if not outputpath.endswith('\\'):
                outputpath = outputpath + '\\'
        else:
            raise TypeError('The output path is not under the string type')
        project_folder = f'{outputpath}{roi_name}_{index}_{start_date}_{end_date}\\'
        zip_folder = f'{outputpath}{roi_name}_{index}_{start_date}_{end_date}_Orizip\\'
        create_folder(project_folder)
        create_folder(zip_folder)

        # Load Landsat Collection 2 Level 2 Image Collection within the ROI and date range
        dataset = ee.ImageCollection(f'LANDSAT/{satellite}/C02/T1_L2').filterDate(start_date, end_date).filterBounds(roi).map(lambda image: image.clip(roi))

        # Function to calculate index
        def add_index(image):
            band_dic = {}
            for _ in self._built_in_index.index_dic[index][0]:
                band_dic[str(_)] = image.select(self._band_tab[f'{satellite}_bandnum'][self._band_tab[f'{satellite}_bandname'].index(str(_))])
            index_band = image.expression(index_express, band_dic).rename(index)
            return image.addBands(index_band)

        # Apply the index calculation to each image in the collection
        index_images = dataset.map(add_index)

        # Function to handle the export of each image as a zip
        def export_image(image, zip_folder, file_name):
            path = os.path.join(zip_folder, f"{file_name}.zip")
            url = image.select(index).getDownloadURL({
                'scale': 30,
                'region': roi,
                'crs': 'EPSG:4326',
                'fileFormat': 'GeoTIFF'
            })
            print(f"Downloading {file_name}...")
            st = time.time()
            response = requests.get(url, stream=True)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=4096):
                    f.write(chunk)

            if export_QA_file:
                url = image.select('QA_PIXEL').getDownloadURL({
                    'scale': 30,
                    'region': roi,
                    'crs': 'EPSG:4326',
                    'fileFormat': 'GeoTIFF'
                })
                print(f"Downloading {file_name}...")
                st = time.time()
                response = requests.get(url, stream=True)
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=4096):
                        f.write(chunk)
            print(f'Finish download {file_name} in {str(time.time() - st)[0:5]} s!')

        # Iterate through each image in the collection
        image_list = index_images.toList(index_images.size())

        for i in range(image_list.size().getInfo()):
            image = ee.Image(image_list.get(i))
            date = image.date().format('YYYYMMdd').getInfo()
            export_image(image, zip_folder, f"{index}_{date}")
            with zipfile.ZipFile(os.path.join(zip_folder, f"{index}_{date}.zip"), 'r') as zip_ref:
                zip_ref.extractall(project_folder)

        print("All exports Finished.")

    def remove_cloud_snow(self):
        ## 读取文件
        pass


class River_centreline(object):

    def __init__(self, MNDWI_tiffiles):
        pass

    def _extract_centreline_thr_(self):
        pass

    def _generate_centreline_thr_(self):
        pass


class built_in_index(object):

    def __init__(self, *args):
        self.NDVI = 'NDVI = (NIR - RED) / (NIR + RED)'
        self.OSAVI = 'OSAVI = 1.16 * (NIR - RED) / (NIR + RED + 0.16)'
        self.AWEI = 'AWEI = 4 * (GREEN - SWIR) - (0.25 * NIR + 2.75 * SWIR2)'
        self.AWEInsh = 'AWEInsh = BLUE + 2.5 * GREEN - 0.25 * SWIR2 - 1.5 * (NIR + SWIR1)'
        self.MNDWI = 'MNDWI = (GREEN - SWIR) / (SWIR + GREEN)'
        self.EVI = 'EVI = 2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)'
        self.EVI2 = 'EVI2 = 2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)'

        self._exprs2index(*args)
        self._built_in_index_dic()

    def _exprs2index(self, *args):
        for temp in args:
            if type(temp) is not str:
                raise ValueError(f'{temp} expression should be in a str type!')
            elif '=' in temp:
                self.__dict__[temp.split('=')[0]] = temp
            else:
                raise ValueError(f'{temp} expression should be in a str type!')

    def add_index(self, *args):
        self._exprs2index(*args)
        self._built_in_index_dic()

    def _built_in_index_dic(self):
        self.index_dic = {}
        for i in self.__dict__:
            if i != 'index_dic':
                var, func = convert_index_func(self.__dict__[i].split('=')[-1])
                self.index_dic[i] = [var, func]


def convert_index_func(expr: str):
    try:
        f = sympy.sympify(expr)
        dep_list = sorted(f.free_symbols, key=str)
        num_f = sympy.lambdify(dep_list, f)
        return dep_list, num_f
    except:
        raise ValueError(f'The {expr} is not valid!')


def create_folder(path_name, print_existence=False):
    if not os.path.exists(path_name):
        try:
            os.makedirs(path_name)
        except:
            print('Something went wrong during creating new folder')
            return
    else:
        if print_existence:
            print('Folder already exist  (' + path_name + ')')


if __name__ == '__main__':
    gee_api = GEE_ds()
    gee_api.download_index_GEE('LC08', (20060101, 20211231), 'MNDWI', [99.70322434775323, 33.80530886069177, 99.49654404990167, 33.73681471587109], 'G:\\A_HH_upper\\GEE\\')
    pass


