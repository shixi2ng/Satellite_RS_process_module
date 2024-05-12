import ee
import os
import requests
import rivamap as rm


#######################################################################################################################
# For how to activate the GEE python-api of your personal account, please follow the guide show in
# https://developers.google.com/earth-engine/guides/python_install
# Meantime, you can check the cookbook of GEE on https://developers.google.com/earth-engine
#######################################################################################################################


class GEE_Landsat_ds(object):

    def __int__(self):
        self._band_information = []
        self._index_function_ = []

    def download_index_GEE(self, satellite, date_range, index, ROI, outputpath):

        ## ROI代表

        ## 下一行需要把不同Landsat卫星的 MIR 和 Green 波段替换掉
        for _, bands in zip(['LC08', 'LE07', 'LE05'], [['SR_B3', 'SR_B6'], ['SR_B3', 'SR_B6'], ['SR_B3', 'SR_B6']]):

            # Initialize the Earth Engine module
            ee.Initialize()

            # Define the region of interest (ROI) as a GeoJSON geometry
            roi = ee.Geometry.Rectangle([99.70322434775323, 33.80530886069177, 99.49654404990167, 33.73681471587109])  # The coordinate of the upper left and lower right corner

            # Define the date range for image collection
            start_date = '2006-01-01'
            end_date = '2021-12-31'

            # Load Landsat 8 Collection 2 Level 2 Image Collection within the ROI and date range
            dataset = ee.ImageCollection(f'LANDSAT/{_}/C02/T1_L2').filterDate(start_date, end_date) \
                .filterBounds(roi).map(lambda image: image.clip(roi))

            # Function to calculate MNDWI
            def calculate_mndwi(image):
                mndwi = image.normalizedDifference(bands).rename('MNDWI')
                return image.addBands(mndwi)

            # Apply the MNDWI calculation to each image in the collection
            mndwi_images = dataset.map(calculate_mndwi)

            # Function to handle the export of each image as a TIFF
            def export_image(image, file_path, file_name):
                path = os.path.join(file_path, f"{file_name}.zip")
                url = image.select('MNDWI').getDownloadURL({
                    'scale': 30,
                    'region': roi,
                    'crs': 'EPSG:4326',
                    'fileFormat': 'GeoTIFF'
                })
                print(f"Downloading {file_name}...")
                print(url)
                response = requests.get(url, stream=True)
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=4096):
                        f.write(chunk)

            # Set the local directory to save images
            local_directory = 'G:\\A_HH_upper\\Jimai_MNDWI\\'

            # Iterate through each image in the collection
            image_list = mndwi_images.toList(mndwi_images.size())

            for i in range(image_list.size().getInfo()):
                image = ee.Image(image_list.get(i))
                date = image.date().format('YYYYMMdd').getInfo()
                export_image(image, local_directory, f"MNDWI_{_}_{date}")

            print("All exports initiated.")

class River_centreline(object):

    def __init__(self, MNDWI_tiffiles):
        pass

    def _extract_centreline_thr_(self):
        pass



