import sys
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import snappy
from sklearn.neural_network import MLPRegressor
import Landsat_main_v1
import Sentinel_main_V2
import numpy
import os
import gdal
import basic_function
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.figure as fg
import seaborn as sns
from pylab import mpl
import datetime


# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()
mask_path = 'E:\\A_Chl-a\\Study_area\\xjb.shp'
study_area = 'xjb'
all_band = 'E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\'
qi_band = 'E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\QI\\'
output_path = 'E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\xjb\\'
extract_path = 'E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\extract\\'
NDWI_PATH = 'E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\NDWI\\'
Basic_function.create_folder(output_path)
Basic_function.create_folder(extract_path)

# Basic_function.create_folder('E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\NDWI\\xjb\\')
# for i in Basic_function.file_filter('E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\NDWI\\', ['.tif']):
#     Basic_function.extract_by_mask(i, mask_path, 'E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\NDWI\\xjb\\', xRes=10, yRes=10, coordinate='EPSG:32648')
#
# for i in Basic_function.file_filter(extract_path, ['.tif']):
#     Basic_function.extract_by_mask(i, mask_path, output_path, xRes=10, yRes=10, coordinate='EPSG:32648')
#



s2_output = 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Sentinel_2_output\\'
xlsx_file = Basic_function.file_filter(s2_output, ['_pre'])
sa_list = []
sa_dic = {}
year_min = 30000000
date_max = 0
for i in xlsx_file:
    sa_temp = i[i.find('S2_output')+10: i.find('_pre')]
    sa_list.append(sa_temp)
    xlsx_temp = pd.read_excel(i)
    ds_temp = np.array(xlsx_temp[['Date','pre_chl_a']])

    sa_dic[sa_temp] = ds_temp
    year_min = min(int(year_min), int(np.min(ds_temp[:,0])))
    date_max = max(int(date_max), int(np.max(ds_temp[:,0])))

fig1, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
y_min = int(year_min//10000)
y_max = int(date_max//10000)
m_min = int(np.mod(year_min, 10000)//100)
m_max = int(np.mod(date_max, 10000)//100) + 1
year_min_ord = datetime.date(year=int(year_min//10000), month=int(np.mod(year_min, 10000)//100), day=1).toordinal()
date_max_ord = datetime.date(year=int(date_max//10000), month=int(np.mod(date_max, 10000)//100) + 1, day=1).toordinal() - 1
site_all = ['XN', 'XJB-1','XJB-2','XJB-3', 'DW', 'ZD']
site_dic = {}
for site in site_all:
    site_temp = []
    for sa in sa_list:
        if site == 'XJB-1':
            site_n_list = ['XJB01','XJB02','XJB03','XJB04','XJB05', 'XJB06','XJB07']
            for site_t in site_n_list:
                if site_t in sa:
                    ds_temp = sa_dic[sa]
                    site_temp.append(ds_temp[:, 1].reshape([-1]).tolist())
                    ds_temp = np.concatenate([ds_temp, np.array([datetime.date(year=int(d // 10000),month=int(np.mod(d, 10000) // 100),day=int(np.mod(d,100))).toordinal() - year_min_ord for d in ds_temp[:, 0]]).reshape([ds_temp.shape[0], 1])], axis=1)

        elif site == 'XJB-2':
            site_n_list = ['XJB08', 'XJB09', 'XJB10', 'XJB11', 'XJB12']
            for site_t in site_n_list:
                if site_t in sa:
                    ds_temp = sa_dic[sa]
                    site_temp.append(ds_temp[:, 1].reshape([-1]).tolist())
                    ds_temp = np.concatenate([ds_temp, np.array([datetime.date(year=int(d // 10000),month=int(np.mod(d, 10000) // 100),day=int(np.mod(d,100))).toordinal() - year_min_ord for d in ds_temp[:, 0]]).reshape([ds_temp.shape[0], 1])], axis=1)

        elif site == 'XJB-3':
            site_n_list = ['XJB13','XJB14','XJB15','XJB16','XJB17', 'XJB18', 'XJB19', 'XJB20', 'XJB21']
            for site_t in site_n_list:
                if site_t in sa:
                    ds_temp = sa_dic[sa]
                    site_temp.append(ds_temp[:, 1].reshape([-1]).tolist())
                    ds_temp = np.concatenate([ds_temp, np.array([datetime.date(year=int(d // 10000),month=int(np.mod(d, 10000) // 100),day=int(np.mod(d,100))).toordinal() - year_min_ord for d in ds_temp[:, 0]]).reshape([ds_temp.shape[0], 1])], axis=1)

        elif site in sa:
            ds_temp = sa_dic[sa]
            site_temp.append(ds_temp[:,1].reshape([-1]).tolist())
            ds_temp = np.concatenate([ds_temp, np.array([datetime.date(year=int(d//10000), month=int(np.mod(d, 10000)//100), day=int(np.mod(d,100))).toordinal()-year_min_ord for d in ds_temp[:,0]]).reshape([ds_temp.shape[0], 1])], axis=1)
            # ax.plot(ds_temp[:,2], ds_temp[:,1], lw=2)
    site_dic[site] = site_temp

plt.rc('font', size=14)
plt.rc('axes', linewidth=1.5)
plt.rc('font', family='Times New Roman')
fig5 = plt.figure(figsize=(15, 9), tight_layout=True)
gs = gridspec.GridSpec(3, 2)

# ['XN', 'XJB-01','XJB-02','XJB-03' 'DW', 'ZD']
ax_dic = {}
ax_dic['XJB-1'] = fig5.add_subplot(gs[0, 0])
ax_dic['XJB-2'] = fig5.add_subplot(gs[1, 0])
ax_dic['XJB-3'] = fig5.add_subplot(gs[2, 0])
ax_dic['XN'] = fig5.add_subplot(gs[0, 1])
ax_dic['ZD'] = fig5.add_subplot(gs[1, 1])
ax_dic['DW'] = fig5.add_subplot(gs[2, 1])
x_tick = []
x_tick_label = []
for i in range(y_min, y_max + 1):
    m_sta = int(1)
    m_end = int(12)
    if i == y_min:
        m_sta = m_min
    elif i == y_max:
        m_end = m_max
    for m in range(m_sta, m_end + 1, 6):
        x_tick.append(datetime.date(year=i, month=m, day=15).toordinal() - year_min_ord)
        x_tick_label.append(str(i) + '-' + str(m))

for temp in site_all:
    site_temp = np.array(site_dic[temp])
    site_max = np.nanmax(site_temp, axis=0)
    site_min = np.nanmin(site_temp, axis=0)
    site_temp = np.nanmean(site_temp, axis=0)
    date_temp = ds_temp[:,2].reshape([-1])
    date_temp = np.delete(date_temp, np.argwhere(np.isnan(site_temp)))
    site_max = np.delete(site_max, np.argwhere(np.isnan(site_temp)))
    site_min = np.delete(site_min, np.argwhere(np.isnan(site_temp)))
    site_temp = np.delete(site_temp, np.argwhere(np.isnan(site_temp)))
    if 'XJB' in temp:
        ax_dic[temp].plot(date_temp, site_max, lw=3, label=temp,zorder=1)
    else:
        ax_dic[temp].plot(date_temp, site_max, lw=3, c=(1,0,0), label=temp, zorder=1)
    ax_dic[temp].plot(np.linspace(0,date_max_ord - year_min_ord,100), np.linspace(30,30,100), lw=2, c=(0, 0, 0), ls='--', zorder=0)
    for y in range(y_min, y_max + 1):
        ax_dic[temp].plot(np.linspace(datetime.date(year=y, month=1,  day=1).toordinal()-year_min_ord, datetime.date(year=y, month=1,  day=1).toordinal()-year_min_ord, 100), np.linspace(0, 70, 100), lw=2, c=(0.5, 0.5, 0.5),
                      ls=':', zorder=0)
    # ax.fill_between(date_temp,site_min, site_max )
    ax_dic[temp].set_xlim([30, date_max_ord - year_min_ord])
    ax_dic[temp].set_ylim([0, 70])
    ax_dic[temp].set_xticks(x_tick[1:])
    ax_dic[temp].set_xticklabels(x_tick_label[1:])
    ax_dic[temp].set_yticks([0,15, 30,50,70])
    ax_dic[temp].set_yticklabels(['0','15','30','50','70'])
    ax_dic[temp].legend()

plt.savefig('E:\\A_Chl-a\\Fig\\Fig2\\sa_sim_chla.jpg')
plt.cla()
plt.clf()

mpl.rcParams['font.sans-serif'] = ['SimSun'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False

s2_fig = 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\S2_fig.xlsx'
sa_fig_df = pd.read_excel(s2_fig)
sa_fig_ds = np.array(sa_fig_df)[:, 0:9]
# plt.rc('font', family='Times New Roman')
plt.rc('font', size=16)
plt.rc('axes', linewidth=3)
plt.rc('axes', axisbelow=True)
fig1 = plt.figure(figsize=(12, 14), constrained_layout=True)
gs = gridspec.GridSpec(2, 1)
ax = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0])

for q in range(1,sa_fig_ds.shape[0]):
    if q == 40:
        ax.plot(sa_fig_ds[0, :], sa_fig_ds[q, :], lw=6, color=(256 / 256, 30 / 256, 30 / 256), zorder=4)
    else:
        ax.plot(sa_fig_ds[0,:], sa_fig_ds[q,:], lw=1.5, color=(120/256,120/256,120/256), zorder=1, **{'ls':'-'})

s2_fig2 = 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Original_insitu_data.xlsx'
sa_fig2_df = pd.read_excel(s2_fig2)
sa_fig_ds2 = np.array(sa_fig2_df)[:, 59:559]
for q in range(1,sa_fig_ds2.shape[0]):
    if q == 39:
        ax2.plot(np.linspace(400,900,500), sa_fig_ds2[q, :], lw=6, color=(256 / 256, 30 / 256, 30 / 256), zorder=4)
    else:
        ax2.plot(np.linspace(400,900,500), sa_fig_ds2[q,:], lw=1.5, color=(120/256,120/256,120/256), zorder=1, **{'ls':'-'})

ax2.set_xlabel('波长（nm）', fontname='SimSun', fontweight='bold')
ax2.set_ylabel('光谱反射率', fontname='SimSun',fontweight='bold')
ax.set_ylabel('光谱反射率', fontname='SimSun',fontweight='bold')
ax.set_ylim([0, 0.07])
ax.set_xlim([443,783])
ax2.set_ylim([0, 0.07])
ax2.set_xlim([443,783])
plt.savefig('E:\\A_Chl-a\\Fig\\s2_band.jpg')
plt.cla()
plt.clf()

data_file = 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Original_insitu_data.xlsx'
SRF_file = 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\S2-SRF.xlsx'

SRF_list = {}
data_list = pd.read_excel(data_file)
SRF_list['S2A'] = pd.read_excel(SRF_file, sheet_name='Spectral Responses (S2A)')
SRF_list['S2B'] = pd.read_excel(SRF_file, sheet_name='Spectral Responses (S2B)')
output_pd = data_list[['No', 'Site', 'Date',  'X', 'Y', 'Chl-a 0.1m', 'Chl-a 1m']]
output_pd['Lat'] = [float(lat_temp[0:2]) + float(lat_temp[3:5]) / 60 + float(lat_temp[6:11]) / 3600 for lat_temp in data_list['Lat']]
output_pd['Lon'] = [float(lon_temp[0:3]) + float(lon_temp[4:6]) / 60 + float(lon_temp[7:12]) / 3600 for lon_temp in data_list['Lon']]

#Create Output folder
file_path = 'E:\\A_Chl-a\\Sample_XJB\\Original_zipfile\\'
output_path = 'E:\\A_Chl-a\\Sample_XJB\\'
corrupted_filepath = output_path + 'Corrupted_S2_file\\'
l2a_output_path = output_path + 'Sentinel2_L2A_output\\'
QI_output_path = output_path + 'Sentinel2_L2A_output\\QI\\'
Sentinel_main_V2.create_folder(l2a_output_path)
Sentinel_main_V2.create_folder(QI_output_path)
Sentinel_main_V2.create_folder(corrupted_filepath)
# built-in parameters Configuration
overwritten_para_vis = False
overwritten_para_clipped = False
overwritten_para_cloud = True
overwritten_para_datacube = True
overwritten_para_sequenced_datacube = True

# Input Snappy data style
snappy.GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
HashMap = snappy.jpy.get_type('java.util.HashMap')
WriteOp = snappy.jpy.get_type('org.esa.snap.core.gpf.common.WriteOp')
BandDescriptor = snappy.jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
np.seterr(divide='ignore', invalid='ignore')

Sentinel2_metadata = Sentinel_main_V2.generate_S2_metadata(file_path, output_path)
# Generate VIs in GEOtiff format
i = 0
VI_list = ['QI', 'all_band', 'NDWI']
while i < Sentinel2_metadata.shape[0]:
    Sentinel_main_V2.generate_vi_file(VI_list, i, output_path, Sentinel2_metadata.shape[0], overwritten_para_vis, Sentinel2_metadata)
    try:
        cache_output_path = 'C:\\Users\\sx199\\.snap\\var\\cache\\s2tbx\\l2a-reader\\8.0.7\\'
        cache_path = [cache_output_path + temp for temp in os.listdir(cache_output_path)]
        Sentinel_main_V2.remove_all_file_and_folder(cache_path)
    except:
        print('process occupied')
    i += 1

for i in Basic_function.file_filter(all_band, ['.tif']):
    if not os.path.exists(extract_path + i[i.rindex('\\')+1:]):
        date = Basic_function.obtain_date_in_file_name(i)
        loc = i[i.find('48'): i.find('48') + 5]
        QI_file = Basic_function.file_filter(qi_band, [str(date), str(loc)], and_or_factor='and')
        NDWI_PATH = Basic_function.file_filter(qi_band, [str(date), str(loc)], and_or_factor='and')
        if len(QI_file) == 0:
            print('coherent pro in cloud removal')
            sys.exit(-1)
        else:
            ds_ori = gdal.Open(i)
            QI_raster = Basic_function.file2raster(QI_file[0])
            NDWI_raster = Basic_function.file2raster(NDWI_PATH[0])
            all_band_raster = Basic_function.file2raster(i)
            all_band_raster[QI_raster >= 7] = 0
            all_band_raster[QI_raster == 3] = 0
            all_band_raster[NDWI_raster <= -0.03] = 0
            Landsat_main_v1.write_raster(ds_ori, all_band_raster, extract_path, i[i.rindex('\\')+1:])

# ndwi_ds = gdal.Open(Basic_function.file_filter(extract_path, [i[i.find('S2') + 4: i.find('S2') + 9],
#                                                               [i[i.find('S2') + 9: i.find('S2') + 17], '.tif']])[0])
# ori_ds = gdal.Open(i)

site_all = np.unique(np.array(output_pd['Site'])).tolist()
band = ['B1_', 'B2_', 'B3', 'B4','B5','B6','B7']
date = np.unique(np.array([str(Basic_function.obtain_date_in_file_name(q)) for q in Basic_function.file_filter(output_path, ['.tif'])]))
date = date.tolist()

no = 0
list_all = []

for site in site_all:
    if not os.path.exists('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Sentinel_2_output\\S2_output_' + str(site) + '.xlsx'):
        site_list=[]
        x_centre = np.nanmean(np.array(output_pd[output_pd['Site'] == site]['X']))
        y_centre = np.nanmean(np.array(output_pd[output_pd['Site'] == site]['Y']))
        for date_temp in date:
            band_dic = {}
            for band_temp in band:
                all_file = Basic_function.file_filter(output_path, [str(band_temp), str(date_temp), '.tif'], and_or_factor='and')
                file_temp = np.nan
                for file_t in all_file:
                    raster_ds = gdal.Open(file_t)
                    file_temp = Basic_function.query_with_cor(raster_ds, x_centre, y_centre, half_width=5, nanvalue=0) / 10000
                    if not np.isnan(file_temp):
                        break
                band_dic[band_temp] = file_temp
            band_list = [band_dic[q] for q in band]
            try:
                d05 = band_dic['B6']*(1/band_dic['B4']-1/band_dic['B5'])
                nir_r = band_dic['B5'] / band_dic['B4']
                b_g = band_dic['B3'] / band_dic['B2_']
                ndci = (band_dic['B5'] - band_dic['B4'])/ (band_dic['B5'] + band_dic['B4'])
                flh = band_dic['B5'] - (band_dic['B4'] + band_dic['B6']) / 2
            except:
                d05, nir_r,b_g,ndci,flh = np.nan
            temp_list = [site, date_temp, x_centre, y_centre, d05, nir_r, b_g, ndci, flh]
            temp_list.extend(band_list)
            site_list.append(temp_list)
        pd_site = pd.DataFrame(site_list,columns=['Site', 'Date', 'X', 'Y', 'D05', 'NIR_RED', 'BLUE_GREEN', 'NDCI', 'FLH', 'B1', 'B2','B3', 'B4', 'B5', 'B6', 'B7'])
        pd_site.to_excel('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Sentinel_2_output\\S2_output_' + str(site) + '.xlsx')
    else:
        site_list = pd.read_excel('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Sentinel_2_output\\S2_output_' + str(site) + '.xlsx')
        site_t = site_list[['Site', 'Date', 'X', 'Y', 'D05', 'NIR_RED', 'BLUE_GREEN', 'NDCI', 'FLH', 'B1', 'B2','B3', 'B4', 'B5', 'B6', 'B7']]
        site_list = np.array(site_t).tolist()
    list_all.append(site_list)
pd_S2 = pd.DataFrame(list_all,columns=['Site', 'Date', 'X', 'Y', 'D05', 'NIR_RED', 'BLUE_GREEN', 'NDCI', 'FLH', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
pd_S2.to_excel('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Sentinel_2_output\\S2_output.xlsx')

# Check VI file consistency
Sentinel_main_V2.check_vi_file_consistency(l2a_output_path, VI_list)

specific_name_list = ['clipped', 'cloud_free', 'data_cube', 'sequenced_data_cube']
# Process files
VI_list = ['QI', 'NDVI', 'NDWI', 'EVI', 'EVI2', 'OSAVI', 'GNDVI', 'NDVI_RE', 'NDVI_2', 'NDVI_RE2']
Sentinel_main_V2.vi_process(l2a_output_path, VI_list, study_area, specific_name_list, overwritten_para_clipped, overwritten_para_cloud, overwritten_para_datacube, overwritten_para_sequenced_datacube)

# Inundated detection

# Spectral unmixing

# Curve fitting
mndwi_threshold = -0.15
fig_path = l2a_output_path + 'Fig\\'
pixel_limitation = Sentinel_main_V2.cor_to_pixel([[778602.523, 3322698.324], [782466.937, 3325489.535]], l2a_output_path + 'NDVI_' + study_area + '\\cloud_free\\')
Sentinel_main_V2.curve_fitting(l2a_output_path, VI_list, study_area, pixel_limitation, fig_path, mndwi_threshold)
# Generate Figure
# NDWI_DATA_CUBE = np.load(NDWI_data_cube_path + 'data_cube_inorder.npy')
# NDVI_DATA_CUBE = np.load(NDVI_data_cube_path + 'data_cube_inorder.npy')
# DOY_LIST = np.load(NDVI_data_cube_path + 'doy_list.npy')
# fig_path = output_path + 'Sentinel2_L2A_output\\Fig\\'
# create_folder(fig_path)
# create_NDWI_NDVI_CURVE(NDWI_DATA_CUBE, NDVI_DATA_CUBE, DOY_LIST, fig_path)