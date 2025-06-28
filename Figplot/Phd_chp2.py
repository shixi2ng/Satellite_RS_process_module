import copy
import os.path
import traceback
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from RSDatacube.RSdc import *
from skimage import io, feature
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
from River_GIS.River_GIS import *
from scipy.stats import pearsonr, kurtosis, variation, cramervonmises_2samp, wasserstein_distance
import matplotlib
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import interpolate
from RF.RFR_model import Ensemble_bagging_contribution
import rasterio
import geopandas as gpd
from rasterio.mask import mask

def table2_2():
    pre_tgd_file = 'G:\A_PhD_Main_paper\Chap.2\Table\Table.2.1\\pre_tgd.TIF'
    post_tgd_file = 'G:\A_PhD_Main_paper\Chap.2\Table\Table.2.1\\post_tgd.TIF'
    shp_file = [f'G:\A_PhD_Main_paper\Chap.2\Table\Table.2.1\\shp\\{sec}_all.shp' for sec in ['ch', 'hh', 'jj', 'yz']]
    shp_mcb_file = [f'G:\A_PhD_Main_paper\Chap.2\Table\Table.2.1\\shp\\{sec}_mcb.shp' for sec in ['ch', 'hh', 'jj', 'yz']]
    sections = ['ch', 'hh', 'jj', 'yz']

    # 统计函数
    def count_valid_pixels_by_mask(raster_path, vector_path):
        # 打开矢量文件
        vector_ds = ogr.Open(vector_path)
        layer = vector_ds.GetLayer()

        # 打开栅格文件
        raster_ds = gdal.Open(raster_path)
        band = raster_ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        # 使用掩膜裁剪
        tmp_path = "/vsimem/temp_clip.tif"
        gdal.Warp(tmp_path, raster_ds, cutlineDSName=vector_path,
                  cropToCutline=True, dstNodata=nodata, xRes=raster_ds.GetGeoTransform()[1],
                  yRes=abs(raster_ds.GetGeoTransform()[5]), outputType=gdal.GDT_Float32)

        # 读取裁剪后数据
        clipped_ds = gdal.Open(tmp_path)
        clipped_band = clipped_ds.GetRasterBand(1)
        clipped_array = clipped_band.ReadAsArray()

        # 清除临时文件
        gdal.Unlink(tmp_path)

        if clipped_array is None:
            return 0  # 无交集
        else:
            valid_count = np.count_nonzero(np.logical_and(clipped_array != nodata, clipped_array <0.95))
            return valid_count

    # 输出统计
    print("Section | Region | Pre_TGD | Post_TGD")
    print("----------------------------------------")

    for sec, shp_all, shp_mcb in zip(sections, shp_file, shp_mcb_file):
        pre_all = count_valid_pixels_by_mask(pre_tgd_file, shp_all) * 0.0009
        post_all = count_valid_pixels_by_mask(post_tgd_file, shp_all)* 0.0009
        pre_mcb = count_valid_pixels_by_mask(pre_tgd_file, shp_mcb)* 0.0009
        post_mcb = count_valid_pixels_by_mask(post_tgd_file, shp_mcb)* 0.0009

        print(f"{sec:<7} | all    | {pre_all:<8} | {post_all:<8}")
        print(f"{sec:<7} | mcb    | {pre_mcb:<8} | {post_mcb:<8}")

def fig2_4():

    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=1)

    wl1 = HydroStationDS()
    wl1.import_from_standard_files('G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\',
                                   'G:\\A_1Dflow_sed\\Hydrodynamic_model\\Original_water_level\\对应表.csv')

    sec_wl_diff, sec_ds_diff = [], []
    sec_dis = [0, 63.83, 153.87, 306.77, 384.16, 423.15, 653.115, 955]
    sec_name = ['宜昌', '枝城', '螺山', '汉口']
    for sec in sec_name:
        fig14_df = wl1.hydrostation_inform_df[sec]
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        ds_pri, ds_post = [], []
        year_dic = {}
        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            discharge = fig14_df['flow/m3/s'][year_temp].tolist()
            flow_temp = fig14_df['water_level/m'][year_temp].tolist() - wl1.waterlevel_offset_list[wl1.hydrostation_name_list.index(sec)]
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year > 2003:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                    ds_post.append(discharge[0:365])

                if 1998 <= year <= 2003:
                    wl_pri.append(flow_temp[0:365])
                    ds_pri.append(discharge[0:365])
                    sd_pri.append(sed_temp[0:365])

        diff_dis = np.array(np.nanmean(wl_post, axis=0)) - np.array(np.nanmean(wl_pri, axis=0))
        sec_wl_diff.append(diff_dis[122: 304].tolist())
        diff_dis = np.array(np.nanmean(ds_post, axis=0)) - np.array(np.nanmean(ds_pri, axis=0))
        sec_ds_diff.append(diff_dis[122: 304].tolist())

    plt.close()
    # plt.rcParams['font.family'] = ['Arial', 'SimHei']
    # plt.rc('font', size=18)
    # plt.rc('axes', axisbelow=True)
    # plt.rc('axes', linewidth=3)
    # fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    # # ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
    # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    # # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
    # # ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='--', c=(0, 0, 0))
    # ax_temp.set_xlim(-50, 975)
    # ax_temp.set_ylim(-4, 1)
    # # ax_temp.set_yticks(ytick)
    # bplot = ax_temp.boxplot(sec_wl_diff, widths=30, positions=sec_dis, notch=True, showfliers=False, whis=(5, 95),
    #                         patch_artist=True, medianprops={"color": "blue", "linewidth": 2.8},
    #                         boxprops={"linewidth": 1.8}, whiskerprops={"linewidth": 1.8},
    #                         capprops={"color": "C0", "linewidth": 1.8})
    #
    # ax_temp.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    # ax_temp.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'])
    # # ax_temp.set_xticklabels(['Yichang', 'Zhicheng', 'Jianli', 'Lianhuatang', 'Luoshan', 'Hankou', 'Jiujiang'], fontname='Arial', fontsize=24)
    # ax_temp.set_ylabel('Water level difference/m', fontname='Arial', fontsize=28, fontweight='bold')
    # colors = []
    #
    # for patch in bplot['boxes']:
    #     patch.set_facecolor((208 / 256, 156 / 256, 44 / 256))
    # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    # plt.savefig(
    #     f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\along_wl_nc.png',
    #     dpi=500)
    # plt.close()

    # plt.close()
    # plt.rcParams['font.family'] = ['Arial', 'SimHei']
    # plt.rc('font', size=18)
    # plt.rc('axes', axisbelow=True)
    # plt.rc('axes', linewidth=3)
    # fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    # # ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
    # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    # # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
    # # ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='--', c=(0, 0, 0))
    # ax_temp.set_xlim(-50, 975)
    # # ax_temp.set_yticks(ytick)
    # bplot = ax_temp.boxplot(sec_ds_diff, widths=20, positions=sec_dis, notch=True, showfliers=False,
    #                         patch_artist=True, medianprops={"color": "blue", "linewidth": 2.8},
    #                         boxprops={"linewidth": 1.8}, whiskerprops={"linewidth": 1.8},
    #                         capprops={"color": "C0", "linewidth": 1.8})
    # ax_temp.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    # ax_temp.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'])
    # # ax_temp.set_xticklabels(['Yichang', 'Zhicheng', 'Jianli', 'Lianhuatang', 'Luoshan', 'Hankou', 'Jiujiang'], fontname='Arial', fontsize=24)
    # ax_temp.set_ylabel('Discharge difference/m', fontname='Arial', fontsize=28, fontweight='bold')
    # colors = []
    #
    # for patch in bplot['boxes']:
    #     patch.set_facecolor((208 / 256, 156 / 256, 44 / 256))
    # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    # plt.savefig(
    #     f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\along_ds_nc.png',
    #     dpi=500)
    # plt.close()
    wl_pri2, wl_post2 = [], []
    wl_pri3, wl_post3 = [], []

    for sec, r1, l1, ytick, in zip(['宜昌', '枝城', '螺山', '汉口'], [(36, 54), (34, 50), (14, 34), (10, 30)],
                                   [48, 44, 29, 24],
                                   [[36, 39, 42, 45, 48, 51, 54], [34, 38, 42, 46, 50], [14, 18, 22, 26, 30, 34],
                                    [10, 15, 20, 25, 30]]):
        fig14_df = wl1.hydrostation_inform_df[sec]
        year_dic = {}
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        ds_pri, ds_post = [], []

        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            discharge = fig14_df['flow/m3/s'][year_temp].tolist()
            flow_temp = fig14_df['water_level/m'][year_temp].tolist() - wl1.waterlevel_offset_list[
                wl1.hydrostation_name_list.index(sec)]
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year >= 2003:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                    ds_post.append(discharge[0:365])
                    if sec == '宜昌':
                        wl_post2.extend(flow_temp[0: 365])
                    elif sec == '汉口':
                        wl_post3.extend(flow_temp[0: 365])
                elif year < 2003:
                    wl_pri.append(flow_temp[0:365])
                    if sec == '宜昌':
                        wl_pri2.extend(flow_temp[0: 365])
                    elif sec == '汉口':
                        wl_pri3.extend(flow_temp[0: 365])

                if 1950 <= year <= 2003:
                    ds_pri.append(discharge[0:365])
                    sd_pri.append(sed_temp[0:365])

        wl_post = np.array(wl_post)
        sd_post = np.array(sd_post)
        wl_pri = np.array(wl_pri)
        sd_pri = np.array(sd_pri)
        ds_pri = np.array(ds_pri)
        ds_post = np.array(ds_post)

        sd_pri[sd_pri == 0] = np.nan
        sd_post[sd_post == 0] = np.nan

        plt.rcParams['font.family'] = ['Arial', 'SimHei']
        plt.rc('font', size=22)
        plt.rc('axes', linewidth=1)
        fig_temp, ax_temp = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        ax_temp.fill_between(np.linspace(122, 304, 121), np.linspace(r1[1], r1[1], 121),
                             np.linspace(r1[0], r1[0], 121), alpha=1, color=(0.9, 0.9, 0.9))
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_post, axis=0).reshape([365]),
                             np.nanmin(wl_post, axis=0).reshape([365]), alpha=0.3, color=(0.8, 0.2, 0.1), zorder=3)
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_pri, axis=0).reshape([365]),
                             np.nanmin(wl_pri, axis=0).reshape([365]), alpha=0.3, color=(0.1, 0.2, 0.8), zorder=2)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1),
                     zorder=4)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0),
                     zorder=4)
        ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='-.', c=(0, 0, 0))
        ax_temp.set_xlim(1, 365)
        ax_temp.set_ylim(r1[0], r1[1])
        ax_temp.set_yticks(ytick)

        print(sec)
        print(f'pre-wl-flood-{str(np.nanmean(np.nanmean(wl_pri, axis=0)[122: 304]))}')
        print(f'post-wl-flood-{str(np.nanmean(np.nanmean(wl_post, axis=0)[122: 304]))}')
        print(f'pre-wl-flood-{str(np.nanmean(np.nanmean(wl_pri, axis=0)[np.r_[0:122, 305:365]]))}')
        print(f'post-wl-flood-{str(np.nanmean(np.nanmean(wl_post, axis=0)[np.r_[0:122, 305:365]]))}')

        a = [15, 45, 75, 106, 136, 167, 197, 228, 258, 289, 319, 350]
        c = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        ax_temp.set_xticks(a)
        ax_temp.set_xticklabels(c,  fontsize=24)
        ax_temp.set_ylabel('水位/m', fontsize=24, fontweight='bold')
        # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        plt.savefig(
            f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_wl_nc.png',
            dpi=500)
        plt.close()

        # fig_temp, ax_temp = plt.subplots(figsize=(11, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(sd_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2004)], np.nanmean(sd_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(255/256, 155/256, 37/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2004)], [np.nanmean(np.nanmean(sd_pri[:, 150: 300], axis=1)) for _ in range(1990, 2004)], linewidth=5, c=(255/256, 155/256, 37/256))
        # ax_temp.bar([_ for _ in range(2004, 2021)], np.nanmean(sd_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 92/256, 171/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2004, 2021)], [np.nanmean(np.nanmean(sd_post[:, 150: 300], axis=1)) for _ in range(2004, 2021)], linewidth=5, c=(0/256, 92/256, 171/256))
        # ax_temp.set_xlabel('Year', fontname='Arial', fontsize=24, fontweight='bold')
        # ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=24, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_annual_sd_nc.png', dpi=500)

        # fig_temp, ax_temp = plt.subplots(figsize=(15, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(ds_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2004)], np.nanmean(ds_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(256/256, 200/256, 87/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2004)], [np.nanmean(np.nanmean(ds_pri[:, 150: 300], axis=1)) for _ in range(1990, 2004)], linewidth=4, c=(255/256, 200/256, 87/256))
        # ax_temp.bar([_ for _ in range(2004, 2021)], np.nanmean(ds_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 72/256, 151/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2004, 2021)], [np.nanmean(np.nanmean(ds_post[:, 150: 300], axis=1)) for _ in range(2004, 2021)], linewidth=3, c=(0/256, 72/256, 151/256))
        # ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_annual_ds_nc.png', dpi=500)

        plt.rc('axes', axisbelow=True)
        plt.rc('axes', linewidth=3)
        fig_temp, ax_temp = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax_temp.grid( axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        # ax_temp.fill_between(np.linspace(175, 300, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_post, axis=0).reshape([365]), np.nanmin(sd_post, axis=0).reshape([365]), alpha=0.3, color=(0/256, 92/256, 171/256), zorder=3)
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_pri, axis=0).reshape([365]), np.nanmin(sd_pri, axis=0).reshape([365]), alpha=0.3, color=(255/256, 155/256, 37/256), zorder=2)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_pri, axis=0).reshape([365]), lw=5, c=(255/256, 155/256, 37/256), zorder=4)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_post, axis=0).reshape([365]), lw=5, c=(0/256, 92/256, 171/256), zorder=4)
        # ax_temp.plot(np.linspace(1,365,365), np.linspace(l1,l1,365), lw=2, ls='--', c=(0,0,0))
        ax_temp.set_xlim(1, 365)
        # ax_temp.set_ylim(r1[0], r1[1])
        # ax_temp.set_yticks(ytick)
        cc = np.nanmean(sd_post, axis=0)/np.nanmean(sd_pri, axis=0)
        print('sd' + str(1- np.nanmean(cc[120: 300])))
        print('sd' +str(1 - np.nanmean(cc[np.r_[0:122, 305:365]])))
        plt.yscale("log")
        a = [15, 45, 75, 106, 136, 167, 197, 228, 258, 289, 319, 350]
        c = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        ax_temp.set_xticks(a)
        ax_temp.set_xticklabels(c, fontname='Arial', fontsize=24)
        # ax_temp.set_xlabel('月份', fontname='Arial', fontsize=28, fontweight='bold')
        ax_temp.set_ylabel('悬移质含沙量/kg/m^3)', fontname='Arial', fontsize=28, fontweight='bold')
        # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        plt.savefig(f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_sd.png', dpi=500)

        if sec == '宜昌':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            # ax_temp.bar([_ for _ in range(1985, 2004)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2004, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            ax_temp.plot([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8),
                         linewidth=3, ls='-', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2),
                         linewidth=3, ls='-', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(50., 50., 100),
                                 edgecolor='none', facecolor=(0.4, 0.4, 0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(50, 50, 100), color=(0, 0, 0), ls='-.', lw=2,
                         label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(50, 50, 100), np.linspace(52, 52, 100),
                                 edgecolor='none', facecolor=(0.8, 0.8, 0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(52, 52, 100), color=(0, 0, 0), ls='--', lw=2,
                         label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(0, 0, 1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(1, 0, 0), linewidth=3)
            # ax_temp.plot(np.linspace([2003.5, 2003.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))
            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.legend(fontsize=20, ncol=2)
            ax_temp.set_yticks([45, 47, 49, 51, 53, 55])
            ax_temp.set_yticklabels(['45', '47', '49', '51', '53', '55'], fontname='Arial', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(45, 55)
            plt.savefig(
                f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_annual_wl_nc.png',
                dpi=500)
            plt.close()

        if sec == '汉口':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            # ax_temp.bar([_ for _ in range(1985, 2004)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2004, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.plot([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8),
                         linewidth=3,
                         ls='-', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2),
                         linewidth=3,
                         ls='-', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(24, 24, 100),
                                 edgecolor='none', facecolor=(0.4, 0.4, 0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(24, 24, 100), color=(0, 0, 0), ls='-.', lw=2,
                         label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(24, 24, 100), np.linspace(26, 26, 100),
                                 edgecolor='none', facecolor=(0.8, 0.8, 0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(26, 26, 100), color=(0, 0, 0), ls='--', lw=2,
                         label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2003)], np.nanmax(wl_pri, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(0, 0, 1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2003, 2021)], np.nanmax(wl_post, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(1, 0, 0), linewidth=3)
            # ax_temp.plot(np.linspace([2003.5, 2003.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))

            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_yticks([20, 22, 24, 26, 28, 30])
            ax_temp.set_yticklabels(['20', '22', '24', '26', '28', '30'], fontname='Arial', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(20, 30)
            plt.savefig(
                f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_annual_wl_nc.png',
                dpi=500)
            plt.close()

        # fig_temp, ax_temp = plt.subplots(nrows=1, ncols=2, figsize=(11, 6), constrained_layout=True)
        # # n, bins, patches = ax_temp.hist(wl_pri2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
        # x = np.linspace(min(wl_pri2), max(wl_pri2))
        # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
        # y = y.cumsum()
        # y /= y[-1]
        #
        # # # Complementary cumulative distributions.
        # # b = plt.ecdf(wl_pri2, complementary=False, label="pre-TGP", orientation="horizontal")
        # # ax_temp[0].ecdf(wl_pri2, complementary=False, label="pre-TGP", orientation="horizontal")
        # # # n, bins, patches = ax_temp.hist(wl_post2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
        # # x = np.linspace(min(wl_post2), max(wl_post2))
        # # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
        # # y = y.cumsum()
        # # y /= y[-1]
        # # # Complementary cumulative distributions.
        # # a = plt.ecdf(wl_post2, complementary=False, label="post-TGP", orientation="horizontal")
        #
        # ax_temp[0].set_yticks([37, 43, 49, 55])
        # ax_temp[0].set_yticklabels(['37', '43', '49', '55'], fontname='Arial', fontsize=22)
        # ax_temp[0].set_xticks([0, 0.25, 0.5, 0.75, 1])
        # ax_temp[0].set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Arial', fontsize=22)
        # # ax_temp.set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        # ax_temp[0].set_xlabel('Density', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp[0].set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp[0].set_xlim([-0.03, 1.03])
        # ax_temp[0].set_ylim([37, 55])
        # ax_temp[0].legend()
        #
        # x = np.linspace(min(wl_pri3), max(wl_pri3))
        # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
        # y = y.cumsum()
        # y /= y[-1]
        #
        # # Complementary cumulative distributions.
        # ax_temp[1].ecdf(wl_pri3, complementary=False, label="pre-TGP", orientation="horizontal")
        # # n, bins, patches = ax_temp.hist(wl_post2, 50, density=True, histtype="step",  cumulative=True, label="Cumulative histogram")
        #
        # ax_temp[1].set_yticks([12, 18, 24, 30])
        # ax_temp[1].set_yticklabels(['12', '18', '24', '30'], fontname='Arial', fontsize=22)
        # ax_temp[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
        # ax_temp[1].set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Arial', fontsize=22)
        # x = np.linspace(min(wl_post3), max(wl_post3))
        # y = ((1 / (np.sqrt(2 * np.pi) * 50)) * np.exp(-0.5 * (1 / 50 * (x - 50)) ** 2))
        # y = y.cumsum()
        # y /= y[-1]
        # # Complementary cumulative distributions.
        # ax_temp[1].ecdf(wl_post3, complementary=False, label="post-TGP", orientation="horizontal")
        #
        # # ax_temp.set_xticks(a)
        # # ax_temp.set_xticklabels(c, fontname='Arial', fontsize=24)
        # # ax_temp.set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        # ax_temp[1].set_xlabel('Density', fontname='Arial', fontsize=28, fontweight='bold')
        # # ax_temp[1].set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp[1].set_xlim([-0.03, 1.03])
        # ax_temp[1].set_ylim([12, 30])
        # ax_temp[1].legend()
        # plt.savefig(f'G:\\A_PhD_Main_paper\\Chap.2\\Figure\\Fig.2.4\\{sec}_wl_freq_nc.png', dpi=500)

        fig_temp, ax_temp = plt.subplots(nrows=1, ncols=1, figsize=(11, 6), constrained_layout=True)
        wl_dic = {'wl': [], 'status': []}
        s_ = 36
        for _ in wl_pri2:
            wl_dic['status'].append('Pre-TGP period')
            wl_dic['wl'].append(int(np.floor(_)))

        for _ in wl_post2:
            wl_dic['status'].append('Post-TGP period')
            wl_dic['wl'].append(int(np.floor(_)))

        sns.histplot(data=wl_dic, x="wl", hue="status",
                     palette=[(127 / 256, 163 / 256, 222 / 256), (247 / 256, 247 / 256, 247 / 256)], multiple="dodge",
                     shrink=1.45, stat='density', alpha=0.9)

        # # Manually add dashed lines for category 'C'
        i = 0
        for container in ax_temp.containers:
            for patch in container.patches:
                if np.mod(i, 2) == 0:  # This checks if the patch is for category 'C'
                    patch.set_hatch('/')  # Set dashed lines
                    patch.set_facecolor((247 / 256, 247 / 256, 247 / 256))
                elif np.mod(i, 2) == 1:
                    patch.set_hatch('')  # This checks if the patch is for category 'C' # Set dashed lines
                    patch.set_facecolor((127 / 256, 163 / 256, 222 / 256))
            i += 1

        ax_temp.set_xticks([38, 42, 46, 50, 54])
        ax_temp.set_xticklabels(['38', '42', '46', '50', '54'], fontname='Arial', fontsize=22)
        ax_temp.set_yticks([0, 0.05, 0.10, 0.15, 0.2, 0.25])
        ax_temp.set_yticklabels(['0%', '5%', '10%', '15%', '20%', '25%'], fontname='Arial', fontsize=22)
        ax_temp.set_ylabel('Density', fontname='Arial', fontsize=26, fontweight='bold')
        ax_temp.set_xlabel('Water level/m', fontname='Arial', fontsize=26, fontweight='bold')
        ax_temp.set_ylim([0, 0.25])
        ax_temp.set_xlim([35.5, 52])
        ax_temp.get_legend()

        plt.savefig(
            f'G:\\B_papers_patents\\RA_Dam operations enhance floodplain vegetation resistance and resilience but compress lateral heterogeneity\\A_fig_nc\\A_NC_Fig1\\{sec}_hist.png',
            dpi=500)
        plt.close()
        pass


fig2_4()