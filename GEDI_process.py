import h5py
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point,LineString, Polygon, shape
import geoviews as gv
from geoviews import opts, tile_sources as gvts
import Basic_function as bf
import sys
import requests as r
import os
import time
import concurrent.futures


def pointVisual(features, vdims):
    return (gvts.EsriImagery * gv.Points(features, vdims=vdims).options(tools=['hover'], height=500, width=900, size=5, 
                                                                        color='yellow', fontsize={'xticks': 10, 'yticks': 10, 
                                                                                                  'xlabel':16, 'ylabel': 16}))


def gedi_finder(product, bbox):
    # Define the base CMR granule search url, including LPDAAC provider name and max page size (2000 is the max allowed)
    cmr = "https://cmr.earthdata.nasa.gov/search/granules.json?pretty=true&provider=LPDAAC_ECS&page_size=2000&concept_id="

    # Set up dictionary where key is GEDI shortname + version and value is CMR Concept ID
    concept_ids = {'GEDI01_B.002': 'C1908344278-LPDAAC_ECS',
                   'GEDI02_A.002': 'C1908348134-LPDAAC_ECS',
                   'GEDI02_B.002': 'C1908350066-LPDAAC_ECS'}

    # CMR uses pagination for queries with more features returned than the page size
    page = 1
    bbox = bbox.replace(' ', '')  # Remove any white spaces
    try:
        # Send GET request to CMR granule search endpoint w/ product concept ID, bbox & page number, format return as json
        cmr_response = r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed']['entry']

        # If 2000 features are returned, move to the next page and submit another request, and append to the response
        while len(cmr_response) % 2000 == 0:
            page += 1
            cmr_response += r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox}&pageNum={page}").json()['feed'][
                'entry']

        # CMR returns more info than just the Data Pool links, below use list comprehension to return a list of DP links
        return [c['links'][0]['href'] for c in cmr_response]
    except:
        # If the request did not complete successfully, print out the response from CMR
        print(r.get(f"{cmr}{concept_ids[product]}&bounding_box={bbox.replace(' ', '')}&pageNum={page}").json())


class GEDI_l2_file(object):

    def __init__(self, ori_folder, work_env = None):
        bf.check_file_path(ori_folder)
        self.ori_folder = ori_folder
        if work_env is None:
            self.work_env = ''
            for i in self.ori_folder.split('\\'):
                if i != self.ori_folder.split('\\')[-2] and i != '':
                    self.work_env += i + '\\'
        else:
            self.work_env = work_env
        self.ori_file = bf.file_filter(self.ori_folder, ['.hdf'])
        self.all_DF = None
        self.shp_name = None

    def generate_metadata(self):
        print('--------------Start generate the metadata of GEDI datasets--------------')
        str_time = time.time()
        metadata_temp = []
        for filepath in self.ori_file:
            if 'GEDI02' in filepath:
                filename = filepath.split('\\')[-1]
                filename_list = filename.split('_')
                meta_temp = [filepath, 'GEDI_L2']
                meta_temp.append(filename_list[2][0:7])
                meta_temp.extend([filename_list[3], filename_list[6], filename_list[5], filename_list[9].split('.')[0]])
            else:
                print('There has some hdf-file not in L2!')
                sys.exit(-1)
            metadata_temp.append(meta_temp)
        meta_pd = pd.DataFrame(metadata_temp, columns=['Absolute dir', 'Level', 'Sensed DOY', 'Orbit Number', 'PPDS', 'Track Number', 'Product Version'])
        meta_pd.to_excel(self.work_env + 'Metadata.xlsx')
        print(f'--------------Finished in {round(time.time()-str_time)} seconds!--------------')

    def extract_shots_indi(self, filepath, file_itr):
        shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag = (
        [] for i in range(16))
        file_name = filepath.split('\\')[-1]
        try:
            beam_itr = 0
            beam_lat_lon_time = 0
            detection_time = 0
            append_time = 0
            GEDI_l2a_temp = h5py.File(filepath, 'r')
            GEDI_l2a_objs = []
            GEDI_l2a_temp.visit(GEDI_l2a_objs.append)
            gediSDS = [o for o in GEDI_l2a_objs if isinstance(GEDI_l2a_temp[o], h5py.Dataset)]
            beamNames = [g for g in GEDI_l2a_temp.keys() if g.startswith('BEAM')]
            for beam_temp in beamNames:
                # Open the SDS:
                # print(f'Start process {beam_temp} of {file_name} ({file_itr} of {self.file_num_all}))')
                start_time = time.time()
                beam_itr += 1
                time_sta = time.time()
                lat_array = np.array(GEDI_l2a_temp[f'{beam_temp}/lat_lowestmode'])
                lon_array = np.array(GEDI_l2a_temp[f'{beam_temp}/lon_lowestmode'])
                lat = [q for q in lat_array]
                lon = [q for q in lon_array]
                ll = np.array([lon, lat]).transpose().tolist()
                ll = [tuple(q) for q in ll]
                ll = LineString(ll)
                beam_lat_lon_time += time.time()-time_sta
                # print(f'beam lat lon generation consumes ca. {str(beam_lat_lon_time)} seconds.')
                if self.shpfile_gp.intersects(ll)[0]:
                    time_sta = time.time()
                    shot_number = np.array(GEDI_l2a_temp[f'{beam_temp}/shot_number'])
                    digital_elevation_model = np.array(GEDI_l2a_temp[f'{beam_temp}/digital_elevation_model'])
                    elev_lowestmode = np.array(GEDI_l2a_temp[f'{beam_temp}/elev_lowestmode'])
                    elev_highestreturn = np.array(GEDI_l2a_temp[f'{beam_temp}/elev_highestreturn'])
                    rh = np.array(GEDI_l2a_temp[f'{beam_temp}/rh'])
                    quality_flag_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/quality_flag'])
                    degrade_flag = np.array(GEDI_l2a_temp[f'{beam_temp}/degrade_flag'])
                    sensitivity_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/sensitivity'])
                    urban_proportion = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/urban_proportion'])
                    water_rate = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/landsat_water_persistence'])
                    leaf_off_flag_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/leaf_off_flag'])
                    for h in range(shot_number.shape[0]):
                        if self.lat_min < lat_array[h] < self.lat_max:
                            if self.shpfile_gp.contains(Point(lon_array[h], lat_array[h]))[0]:
                                time_sta_2 = time.time()
                                zLat.append(lat_array[h])
                                zLon.append(lon_array[h])
                                shotNum.append(shot_number[h])
                                dem.append(digital_elevation_model[h])
                                zElevation.append(elev_lowestmode[h])
                                zHigh.append(elev_highestreturn[h])
                                rh25.append(rh[h, 24])
                                rh98.append(rh[h, 97])
                                rh100.append(rh[h, 99])
                                quality.append(quality_flag_temp[h])
                                degrade.append(degrade_flag[h])
                                sensitivity.append(sensitivity_temp[h])
                                beamI.append(beam_temp)
                                urban_rate.append(urban_proportion[h])
                                Landsat_water_rate.append(water_rate[h])
                                leaf_off_flag.append(leaf_off_flag_temp[h])
                                append_time += time.time() - time_sta_2
                                # print(f'append consumes ca. {str(append_time)} seconds.')
                    detection_time += time.time() - time_sta
                    print(f'Finished in {str(time.time()-start_time)} seconds {beam_temp} of {file_name} ({file_itr} of {self.file_num_all})).')
        except:
            print(f'The {file_name} has some issues')
        if len(zLat) == len(leaf_off_flag):
            return shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag
        else:
            shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag = (
                [] for i in range(16))
            print('The output list consistency is invalid!')
            return shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh98, rh100, quality, degrade, sensitivity, beamI, urban_rate, Landsat_water_rate, leaf_off_flag


    def extract_shots_elevation_infor(self, shp_file=None, quality_flag=None):
        # Check the shp file
        if shp_file is None:
            print('Please mention that the GEDI dataset is extremely large! It takes times extracting all the shots information!')
            while True:
                continue_flag = str(input("Continue?(y/n):")).lower()
                if continue_flag == 'y':
                    self.shp_name = 'entire'
                    break
                elif continue_flag == 'n':
                    sys.exit(-1)
                else:
                    print('Invalid input!')
        else:
            if os.path.exists(shp_file) and shp_file.endswith('.shp'):
                try:
                    shpfile_gp = gp.read_file(shp_file)
                    self.shp_name = shp_file.split('\\')[-1].split('.')[0]
                except:
                    print('Something error during the shpfile input!')
                    sys.exit(-1)
                if shpfile_gp.crs != 'EPSG:4326':
                    shpfile_gp = shpfile_gp.to_crs(4326)
            else:
                print('Please input a valid shp file path!')
                sys.exit(-1)

        lat_min, lat_max, lon_min, lon_max = shpfile_gp.bounds['miny'][0], shpfile_gp.bounds['maxy'][0], shpfile_gp.bounds['minx'][0], shpfile_gp.bounds['maxx'][0]
        shotNum_all, dem_all, zElevation_all, zHigh_all, zLat_all, zLon_all, rh25_all, rh98_all, rh100_all, quality_all, degrade_all, sensitivity_all, beamI_all, urban_rate_all, Landsat_water_rate_all, leaf_off_flag_all = ([] for i in range(16))
        file_itr = 0
        file_num_all = len(self.ori_file)
        print('-------------------------------- Start extract GEDI data --------------------------------')
        for filepath in self.ori_file:
            beam_itr = 0
            beam_lat_lon_time = 0
            detection_time = 0
            append_time = 0
            file_itr += 1
            GEDI_l2a_temp = h5py.File(filepath, 'r')
            GEDI_l2a_objs = []
            GEDI_l2a_temp.visit(GEDI_l2a_objs.append)
            file_name = filepath.split('\\')[-1]
            gediSDS = [o for o in GEDI_l2a_objs if isinstance(GEDI_l2a_temp[o], h5py.Dataset)]
            beamNames = [g for g in GEDI_l2a_temp.keys() if g.startswith('BEAM')]
            shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh98, rh100 ,quality ,degrade, sensitivity ,beamI, urban_rate, Landsat_water_rate, leaf_off_flag = ([] for i in range(16))
            for beam_temp in beamNames:
                # Open the SDS:
                print(f'Start process {beam_temp} of {file_name} ({file_itr} of {file_num_all}))')
                start_time = time.time()
                beam_itr += 1
                time_sta = time.time()
                lat_array = np.array(GEDI_l2a_temp[f'{beam_temp}/lat_lowestmode'])
                lon_array = np.array(GEDI_l2a_temp[f'{beam_temp}/lon_lowestmode'])
                lat = [q for q in lat_array]
                lon = [q for q in lon_array]
                ll = np.array([lon, lat]).transpose().tolist()
                ll = [tuple(q) for q in ll]
                ll = LineString(ll)
                beam_lat_lon_time += time.time()-time_sta
                # print(f'beam lat lon generation consumes ca. {str(beam_lat_lon_time)} seconds.')
                if shpfile_gp.intersects(ll)[0]:
                    time_sta = time.time()
                    shot_number = np.array(GEDI_l2a_temp[f'{beam_temp}/shot_number'])
                    digital_elevation_model = np.array(GEDI_l2a_temp[f'{beam_temp}/digital_elevation_model'])
                    elev_lowestmode = np.array(GEDI_l2a_temp[f'{beam_temp}/elev_lowestmode'])
                    elev_highestreturn = np.array(GEDI_l2a_temp[f'{beam_temp}/elev_highestreturn'])
                    rh = np.array(GEDI_l2a_temp[f'{beam_temp}/rh'])
                    quality_flag_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/quality_flag'])
                    degrade_flag = np.array(GEDI_l2a_temp[f'{beam_temp}/degrade_flag'])
                    sensitivity_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/sensitivity'])
                    urban_proportion = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/urban_proportion'])
                    water_rate = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/landsat_water_persistence'])
                    leaf_off_flag_temp = np.array(GEDI_l2a_temp[f'{beam_temp}/land_cover_data/leaf_off_flag'])
                    for h in range(shot_number.shape[0]):
                        if lat_min < lat_array[h] < lat_max:
                            if shpfile_gp.contains(Point(lon_array[h], lat_array[h]))[0]:
                                time_sta_2 = time.time()
                                zLat.append(lat_array[h])
                                zLon.append(lon_array[h])
                                shotNum.append(shot_number[h])
                                dem.append(digital_elevation_model[h])
                                zElevation.append(elev_lowestmode[h])
                                zHigh.append(elev_highestreturn[h])
                                rh25.append(rh[h, 24])
                                rh98.append(rh[h, 97])
                                rh100.append(rh[h, 99])
                                quality.append(quality_flag_temp[h])
                                degrade.append(degrade_flag[h])
                                sensitivity.append(sensitivity_temp[h])
                                beamI.append(beam_temp)
                                urban_rate.append(urban_proportion[h])
                                Landsat_water_rate.append(water_rate[h])
                                leaf_off_flag.append(leaf_off_flag_temp[h])
                                append_time += time.time() - time_sta_2
                                # print(f'append consumes ca. {str(append_time)} seconds.')
                    detection_time += time.time() - time_sta
                    print(f'Finished in {str(time.time()-start_time)} seconds.')
                # print()
            shotNum_all.extend(shotNum)
            dem_all.extend(dem)
            zElevation_all.extend(zElevation)
            zHigh_all.extend(zHigh)
            zLat_all.extend(zLat)
            zLon_all.extend(zLon)
            rh25_all.extend(rh25)
            rh98_all.extend(rh98)
            rh100_all.extend(rh100)
            quality_all.extend(quality)
            degrade_all.extend(degrade)
            sensitivity_all.extend(sensitivity)
            beamI_all.extend(beamI)
            urban_rate_all.extend(urban_rate)
            Landsat_water_rate_all.extend(Landsat_water_rate)
            leaf_off_flag_all.extend(leaf_off_flag)
            print(f'Finished processed for file {file_name} in {str(detection_time + beam_lat_lon_time)} seconds, detection consumes ca. {str(detection_time)} seconds; Beam lat lon generation consumes ca. {str(beam_lat_lon_time)} seconds!')
        self.all_DF = pd.DataFrame({'Shot Number': shotNum_all, 'Beam': beamI_all, 'Latitude': zLat_all, 'Longitude': zLon_all, 'Tandem-X DEM': dem_all, 'Elevation (m)': zElevation_all, 'Canopy Elevation (m)': zHigh_all, 'Canopy Height (rh100)': rh100_all, 'RH 98': rh98_all, 'RH 25': rh25_all, 'Quality Flag': quality_all, 'Degrade Flag': degrade_all, 'Sensitivity': sensitivity_all, 'Urban rate': urban_rate_all, 'Landsat water rate': Landsat_water_rate_all, 'Leaf off flag': leaf_off_flag_all})
        bf.create_folder(self.work_env + 'Output\\')
        self.all_DF.to_excel(f'{self.work_env}Output\\{self.shp_name}_all.xlsx')

        # i = 0
        # while i < self.all_DF.shape[0]:
        #     Point_temp = Point(self.all_DF.iloc[i]['Latitude'], self.all_DF.iloc[i]['Longitude'])
        #     if not shpfile_gp.contains(Point_temp):
        #         self.all_DF = self.all_DF.drop([i])
        #     i += 1

        # Set any poor quality returns to NaN
        self.all_DF = self.all_DF.where(self.all_DF['Quality Flag'].ne(0))
        # self.all_DF = self.all_DF.where(self.all_DF['Degrade Flag'] < 1)
        # self.all_DF = self.all_DF.where(self.all_DF['Sensitivity'] > 0.95)
        self.all_DF = self.all_DF.dropna()
        bf.create_folder(self.work_env + 'Output\\')
        self.all_DF.to_excel(f'{self.work_env}Output\\{self.shp_name}_qf.xlsx')

        # Take the lat/lon dataframe and convert each lat/lon to a shapely point
        self.all_DF['geometry'] = self.all_DF.apply(lambda row: Point(row.Longitude, row.Latitude), axis=1)
        # Convert to geodataframe
        self.all_DF = gp.GeoDataFrame(self.all_DF)
        self.all_DF = self.all_DF.drop(columns=['Latitude', 'Longitude'])
        self.all_DF['Shot Number'] = self.all_DF['Shot Number'].astype(str)  # Convert shot number to string


    def mp_extract_shots_elevation_infor(self, shp_file=None, quality_flag=None):
        if shp_file is None:
            print('Please mention that the GEDI dataset is extremely large! It takes times extracting all the shots information!')
            while True:
                continue_flag = str(input("Continue?(y/n):")).lower()
                if continue_flag == 'y':
                    self.shp_name = 'entire'
                    break
                elif continue_flag == 'n':
                    sys.exit(-1)
                else:
                    print('Invalid input!')
        else:
            if os.path.exists(shp_file) and shp_file.endswith('.shp'):
                try:
                    shpfile_gp = gp.read_file(shp_file)
                    self.shp_name = shp_file.split('\\')[-1].split('.')[0]
                except:
                    print('Something error during the shpfile input!')
                    sys.exit(-1)
                if shpfile_gp.crs != 'EPSG:4326':
                    shpfile_gp = shpfile_gp.to_crs(4326)
            else:
                print('Please input a valid shp file path!')
                sys.exit(-1)

        self.shpfile_gp = shpfile_gp
        self.lat_min, self.lat_max, lon_min, lon_max = shpfile_gp.bounds['miny'][0], shpfile_gp.bounds['maxy'][0], shpfile_gp.bounds['minx'][0], shpfile_gp.bounds['maxx'][0]
        shotNum_all, dem_all, zElevation_all, zHigh_all, zLat_all, zLon_all, rh25_all, rh98_all, rh100_all, quality_all, degrade_all, sensitivity_all, beamI_all, urban_rate_all, Landsat_water_rate_all, leaf_off_flag_all = ([] for i in range(16))
        self.file_num_all = len(self.ori_file)
        self.file_itr = range(1, self.file_num_all + 1)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for shotNum, dem, zElevation, zHigh, zLat, zLon, rh25, rh98, rh100, quality, degrade, sensitivity ,beamI, urban_rate, Landsat_water_rate, leaf_off_flag in executor.map(self.extract_shots_indi, self.ori_file, self.file_itr):
                shotNum_all.extend(shotNum)
                dem_all.extend(dem)
                zElevation_all.extend(zElevation)
                zHigh_all.extend(zHigh)
                zLat_all.extend(zLat)
                zLon_all.extend(zLon)
                rh25_all.extend(rh25)
                rh98_all.extend(rh98)
                rh100_all.extend(rh100)
                quality_all.extend(quality)
                degrade_all.extend(degrade)
                sensitivity_all.extend(sensitivity)
                beamI_all.extend(beamI)
                urban_rate_all.extend(urban_rate)
                Landsat_water_rate_all.extend(Landsat_water_rate)
                leaf_off_flag_all.extend(leaf_off_flag)
        self.all_DF = pd.DataFrame(
            {'Shot Number': shotNum_all, 'Beam': beamI_all, 'Latitude': zLat_all, 'Longitude': zLon_all,
             'Tandem-X DEM': dem_all, 'Elevation (m)': zElevation_all, 'Canopy Elevation (m)': zHigh_all,
             'Canopy Height (rh100)': rh100_all, 'RH 98': rh98_all, 'RH 25': rh25_all, 'Quality Flag': quality_all,
             'Degrade Flag': degrade_all, 'Sensitivity': sensitivity_all, 'Urban rate': urban_rate_all,
             'Landsat water rate': Landsat_water_rate_all, 'Leaf off flag': leaf_off_flag_all})
        bf.create_folder(self.work_env + 'Output\\')
        self.all_DF.to_excel(f'{self.work_env}Output\\{self.shp_name}_all.xlsx')

        # i = 0
        # while i < self.all_DF.shape[0]:
        #     Point_temp = Point(self.all_DF.iloc[i]['Latitude'], self.all_DF.iloc[i]['Longitude'])
        #     if not shpfile_gp.contains(Point_temp):
        #         self.all_DF = self.all_DF.drop([i])
        #     i += 1

        # Set any poor quality returns to NaN
        self.all_DF = self.all_DF.where(self.all_DF['Quality Flag'].ne(0))
        # self.all_DF = self.all_DF.where(self.all_DF['Degrade Flag'] < 1)
        # self.all_DF = self.all_DF.where(self.all_DF['Sensitivity'] > 0.95)
        self.all_DF = self.all_DF.dropna()
        bf.create_folder(self.work_env + 'Output\\')
        self.all_DF.to_excel(f'{self.work_env}Output\\{self.shp_name}_qf.xlsx')

        # Take the lat/lon dataframe and convert each lat/lon to a shapely point
        self.all_DF['geometry'] = self.all_DF.apply(lambda row: Point(row.Longitude, row.Latitude), axis=1)
        # Convert to geodataframe
        self.all_DF = gp.GeoDataFrame(self.all_DF)
        self.all_DF = self.all_DF.drop(columns=['Latitude', 'Longitude'])
        self.all_DF['Shot Number'] = self.all_DF['Shot Number'].astype(str)  # Convert shot number to string

    def visualise_shots(self):
        vdims = []
        for f in self.all_DF:
            if f not in ['geometry']:
                vdims.append(f)

        visual = pointVisual(self.all_DF, vdims=vdims)
        # Plot the basemap and geoviews Points, defining the color as the Canopy Height for each shot
        (gvts.EsriImagery * gv.Points(self.all_DF, vdims=vdims).options(color='Canopy Height (rh100)', cmap='plasma', size=3,
                                                                  tools=['hover'],
                                                                  clim=(0, 102), colorbar=True, clabel='Meters',
                                                                  title='GEDI Canopy Height over Redwood National Park: June 19, 2019',
                                                                  fontsize={'xticks': 10, 'yticks': 10, 'xlabel': 16,
                                                                            'clabel': 12,
                                                                            'cticks': 10, 'title': 16,
                                                                            'ylabel': 16})).options(height=500,
                                                                                                    width=900)
        (gvts.EsriImagery * gv.Points(self.all_DF, vdims=vdims).options(color='Elevation (m)', cmap='terrain', size=3,
                                                                  tools=['hover'],
                                                                  clim=(min(self.all_DF['Elevation (m)']),
                                                                        max(self.all_DF['Elevation (m)'])),
                                                                  colorbar=True, clabel='Meters',
                                                                  title='GEDI Elevation over Redwood National Park: June 19, 2019',
                                                                  fontsize={'xticks': 10, 'yticks': 10, 'xlabel': 16,
                                                                            'clabel': 12,
                                                                            'cticks': 10, 'title': 16,
                                                                            'ylabel': 16})).options(height=500,
                                                                                                    width=900)

    def output_shpfile(self):
        outName = self.shp_name + '.shp'
        self.all_DF.to_file(f'{self.work_env}Output\\{outName}', driver='ESRI Shapefile')


if __name__ == '__main__':
    sample_YTR = GEDI_l2_file('E:\\A_Veg_phase2\\Entire_YTR\\Group2\\GEDI_ori_file\\')
    sample_YTR.generate_metadata()
    sample_YTR.mp_extract_shots_elevation_infor(shp_file='E:\\A_Veg_phase2\\Entire_YTR\\shpfile\\MID_YZR.shp')



