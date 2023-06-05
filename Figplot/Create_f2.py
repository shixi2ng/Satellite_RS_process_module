import os.path

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import gdal
import ogr
import copy
from matplotlib.colors import LogNorm
import sys
from matplotlib.colors import LinearSegmentedColormap
import basic_function as bf
import seaborn as sns


def poly2(x, a, b, c):
    return a * x ** 2 + b * x + c

def fig1_func():
    fig1, ax1 = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 15)
    ax1.set_ylim(0, 15)
    pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='UAV_Final')
    x_2020 = pd_temp[pd_temp['Date']//1000 == 2020]['Canopy_Hei']
    y_2020 = pd_temp[pd_temp['Date']//1000 == 2020]['MEAN']
    x_2021 = pd_temp[pd_temp['Date']//1000 == 2021]['Canopy_Hei']
    y_2021 = pd_temp[pd_temp['Date']//1000 == 2021]['MEAN']
    x_2022 = pd_temp[pd_temp['Date']//1000 == 2022]['Canopy_Hei']
    y_2022 = pd_temp[pd_temp['Date']//1000 == 2022]['MEAN']
    ax1.scatter(x_2020, y_2020, s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8,linewidth=1.2, marker='^', zorder=5)
    ax1.scatter(x_2021, y_2021, s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
    ax1.scatter(x_2022, y_2022, s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(230/256,76/256,0/256), linewidth=1.2, marker='o', zorder=5)
    ax1.plot(np.linspace(0, 15), np.linspace(0, 15), color=(0.2, 0.2, 0.2), ls = '-.', linewidth = 2)
    ax1.plot(np.linspace(0, 15), np.linspace(2, 17), color=(0.2, 0.2, 0.2), ls='--', linewidth=1)
    ax1.plot(np.linspace(0, 15), np.linspace(-2, 13), color=(0.2, 0.2, 0.2), ls='--', linewidth=1)
    ax1.set_xticks([0, 3, 6, 9])
    ax1.set_xticklabels(['0', '3', '6', '9', '12', '15'], fontname='Times New Roman', fontsize=12)
    ax1.set_yticks([0, 3, 6, 9, 12, 15])
    ax1.set_yticklabels(['0', '3', '6', '9', '12', '15'], fontname='Times New Roman', fontsize=12)

    # ax1.set_xlabel('Date', fontname='Times New Roman', fontsize=40, fontweight='bold')
    # ax1.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=40, fontweight='bold')

    ax1.fill_between(np.linspace(0, 15, 100), np.linspace(2, 17, 100), np.linspace(-2, 13, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig4\\fig4.png', dpi=300)


def fig2_func():
    fig2, ax1 = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    x = pd_temp['UAV']
    y = pd_temp['Mean ']
    ax1.scatter(x, y, s=20 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(168/256,0/256,0/256), alpha=1,linewidth=1.2, marker='^', zorder=5)
    ax1.plot(np.linspace(0, 15), np.linspace(0, 15), color=(0.2, 0.2, 0.2), ls = '-.', linewidth = 2)
    ax1.plot(np.linspace(0, 15), np.linspace(1, 16), color=(0.2, 0.2, 0.2), ls='--', linewidth=1)
    ax1.plot(np.linspace(0, 15), np.linspace(-1, 14), color=(0.2, 0.2, 0.2), ls='--', linewidth=1)
    ax1.set_xticks([0, 1, 2, 3, 4, 5])
    ax1.set_xticklabels(['0', '1', '2', '3', '4', '5'], fontname='Times New Roman', fontsize=12)
    ax1.set_yticks([0, 1, 2, 3, 4, 5])
    ax1.set_yticklabels(['0', '1', '2', '3', '4', '5'], fontname='Times New Roman', fontsize=12)

    # ax1.set_xlabel('Date', fontname='Times New Roman', fontsize=40, fontweight='bold')
    # ax1.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=40, fontweight='bold')

    ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig4\\fig5.png', dpi=300)


def fig3_func():
    plt.rc('font', family='Times New Roman')
    data_t = pd.read_excel('G:\\GEDI_MYR\\Hankou_table\\hankou.xlsx', sheet_name='UAV_Final')
    data_t = data_t.loc[0:24, ['Canopy_Hei', 'RH_98', 'RH90', 'RH75', 'RH25', 'MEAN']]
    ch_err = np.array(data_t['Canopy_Hei'] - data_t['MEAN'])
    rh98_err = np.array(data_t['RH_98'] - data_t['MEAN'])
    rh90_err = np.array(data_t['RH90'] - data_t['MEAN'])
    rh75_err = np.array(data_t['RH75'] - data_t['MEAN'])
    rh25_err = np.array(data_t['RH25'] - data_t['MEAN'])
    fig2, ax1 = plt.subplots(figsize=(6, 3), constrained_layout=True)
    d = [rh25_err, rh75_err, rh90_err,  rh98_err, ch_err]
    # ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    # ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(-10, 2)
    ax1.set_ylim(0.5, 5.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    box = ax1.boxplot(d, vert=False, labels=['RH25-UAV', 'RH75-UAV', 'RH90-UAV', 'RH98-UAV', 'RH100-UAV'],  notch=True, widths=0.5, patch_artist=True, whis=(5, 95), showfliers=False, zorder=4)
    ax1.scatter([np.mean(_) for _ in d], [1,2,3,4,5], marker='d', facecolor=(1,1,1), s=25, edgecolor=(0,0,0), lw=0.5, zorder=4)
    plt.setp(box['boxes'], color=(89/256,117/256,164/256), edgecolor=(60/256,60/256,60/256))
    ax1.plot(np.linspace(0,0,100), np.linspace(-100,100,100), c=(0,0,0), lw=1, zorder=3, ls ='--')
    # ax1.set_yticklabels('平均误差/m')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig5\\fig5.png', dpi=300)


def fig4_func():
    plt.rc('font', family='Times New Roman')
    data_t = pd.read_excel('G:\\GEDI_MYR\\Hankou_table\\hankou.xlsx', sheet_name='UAV_Final')
    data_t = data_t.loc[0:24, ['Canopy_Hei', 'RH_98', 'RH90', 'RH75', 'RH25', 'MEAN']]

    fig2, ax1 = plt.subplots(figsize=(6, 0.8), constrained_layout=True)
    ax1.xaxis.tick_top()
    d = [[0.097012,0.123635,0.0306,0.132558,0.002834,0.072849,0.122362,0.113934,-0.0023,-0.09]]
    # ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    # ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(-1, 0.2)
    ax1.set_ylim(0.5, 1.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    box = ax1.boxplot(d, labels = ['实测-UAV'], vert=False,  notch=False, widths=0.5, patch_artist=True, whis=(0, 100), showfliers=False, zorder=4)
    ax1.scatter([np.mean(_) for _ in d],[1], marker='d', facecolor=(1,1,1), s=25, edgecolor=(0,0,0), lw=0.5, zorder=4)
    plt.setp(box['boxes'], color=(167/256,0/256,0/256), edgecolor=(60/256,60/256,60/256))
    ax1.plot(np.linspace(0,0,100), np.linspace(-100,100,100), c=(0,0,0), lw=1, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig5\\fig51.png', dpi=300)


def fig5_func():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    # plt.style.use("ggplot")
    data_t = pd.read_excel('G:\\A_veg\\Paper\\Table\\Tab3.xlsx', sheet_name='Output')
    data_t = data_t.loc[:, ['Name', 'Percen', 'Type']].sort_values('Percen')[13:27]
    edge_c = [(1,1,1), (1,1,1), (1,1,1)]
    color_c = [(178/256,178/256,178/256), (112/256, 160/256, 205/256), (112/256, 160/256, 205/256)]
    c_all = []
    e_all = []
    for _ in data_t['Type']:
        c_all.append(color_c[_])
        e_all.append(edge_c[_])

    fig2, ax1 = plt.subplots(figsize=(3.5, 5), constrained_layout=True)
    ax1.xaxis.tick_top()
    ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 0.2)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax1.set_xticklabels(['0%', '5%', '10%', '15%', '20%'], font='Times New Roman')
    ax1.set_ylim(-0.5, 12.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    ax1.barh(data_t['Name'], width= data_t['Percen'], height=0.9, color=c_all, edgecolor=e_all)
    ax1.plot(np.linspace(0.04,0.040,100), np.linspace(-100,100,100), c=(0,0,0), lw=1.5, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig6\\fig5.png', dpi=300)

    data_t = pd.read_excel('G:\\A_veg\\Paper\\Table\\Tab3.xlsx', sheet_name='Output')
    data_t = data_t.loc[:, ['Name', 'Percen', 'Type']].sort_values('Percen')[0:13]
    edge_c = [(1,1,1), (1,1,1), (1,1,1)]
    color_c = [(178/256,178/256,178/256), (112/256, 160/256, 205/256), (112/256, 160/256, 205/256)]
    c_all = []
    e_all = []
    for _ in data_t['Type']:
        c_all.append(color_c[_])
        e_all.append(edge_c[_])

    fig2, ax1 = plt.subplots(figsize=(3.5, 5), constrained_layout=True)
    ax1.xaxis.tick_top()
    ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 0.2)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax1.set_xticklabels(['0%', '5%', '10%', '15%', '20%'], font='Times New Roman')
    ax1.set_ylim(-0.5, 12.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    ax1.barh(data_t['Name'], width= data_t['Percen'], height=0.9, color=c_all, edgecolor=e_all)
    ax1.plot(np.linspace(0.04,0.040,100), np.linspace(-100,100,100), c=(0,0,0), lw=1.5, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig6\\fig51.png', dpi=300)


def fig6_func():

    species = ("Index", "2020", "Gentoo")
    penguin_means = {
        'Bill Depth': (18.35, 18.43, 14.98),
        'Bill Length': (38.79, 48.83, 47.50),
        'Flipper Length': (189.95, 195.82, 217.19),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.

    plt.show()
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    # plt.style.use("ggplot")
    data_t = pd.read_excel('G:\\A_veg\\Paper\\Table\\Tab3.xlsx', sheet_name='Output')
    data_t = data_t.loc[:, ['Name', 'Percen', 'Type']].sort_values('Percen')[13:27]
    edge_c = [(1,1,1), (1,1,1), (1,1,1)]
    color_c = [(178/256,178/256,178/256), (112/256, 160/256, 205/256), (112/256, 160/256, 205/256)]
    c_all = []
    e_all = []
    for _ in data_t['Type']:
        c_all.append(color_c[_])
        e_all.append(edge_c[_])

    fig2, ax1 = plt.subplots(figsize=(3.5, 5), constrained_layout=True)
    ax1.xaxis.tick_top()
    ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 0.2)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax1.set_xticklabels(['0%', '5%', '10%', '15%', '20%'], font='Times New Roman')
    ax1.set_ylim(-0.5, 12.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    ax1.barh(data_t['Name'], width= data_t['Percen'], height=0.9, color=c_all, edgecolor=e_all)
    ax1.plot(np.linspace(0.04,0.040,100), np.linspace(-100,100,100), c=(0,0,0), lw=1.5, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig6\\fig5.png', dpi=300)

    data_t = pd.read_excel('G:\\A_veg\\Paper\\Table\\Tab3.xlsx', sheet_name='Output')
    data_t = data_t.loc[:, ['Name', 'Percen', 'Type']].sort_values('Percen')[0:13]
    edge_c = [(1,1,1), (1,1,1), (1,1,1)]
    color_c = [(95/256,158/256,110/256), (204/256,137/256,99/256), (89/256,117/256,164/256)]
    c_all = []
    e_all = []
    for _ in data_t['Type']:
        c_all.append(color_c[_])
        e_all.append(edge_c[_])

    fig2, ax1 = plt.subplots(figsize=(3.5, 5), constrained_layout=True)
    ax1.xaxis.tick_top()
    ax1.set_axisbelow(True)
    # ax1.yaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.xaxis.grid(True, color=(0.8,0.8,0.8),lw=0.5)
    ax1.set_xlim(0, 0.2)
    ax1.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax1.set_xticklabels(['0%', '5%', '10%', '15%', '20%'], font='Times New Roman')
    ax1.set_ylim(-0.5, 12.5)
    # pd_temp = pd.read_excel('G:\GEDI_MYR\Hankou_table\\hankou.xlsx', sheet_name='Insitu-Final')
    # x = pd_temp['UAV']
    # y = pd_temp['Mean ']
    ax1.barh(data_t['Name'], width= data_t['Percen'], height=0.9, color=c_all, edgecolor=e_all)
    ax1.plot(np.linspace(0.04,0.040,100), np.linspace(-100,100,100), c=(0,0,0), lw=1.5, zorder=3, ls ='--')

    # ax1.fill_between(np.linspace(0, 15, 100), np.linspace(1, 16, 100), np.linspace(-1, 14, 100), color=(0.8, 0.8, 0.8), alpha=0.5, zorder=1)
    # plt.show()
    plt.savefig('G:\A_veg\Paper\Figure\Fig6\\fig51.png', dpi=300)


def fig7_func():
    UAV_tif_folder = 'E:\\A_PhD_Main_stuff\\江滩LiDAR\\LiDAR_detect_res\\tiffiles\\'
    shpfile = 'E:\\A_PhD_Main_stuff\\江滩LiDAR\\LiDAR_detect_res\hk_shp\\hk.shp'
    plt.style.use('_mpl-gallery-nogrid')

    for date_t in ['peak_2022', 'peak_2021', 'peak_2020', 'peak_2019']:
        GEDI_tif_folder = f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\predicted_feature_tif\\'
        bf.create_folder(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\Compare_fig\\')
        bf.create_folder(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\Compare_tab\\')
        for _ in ['230205', '220706']:
            for chl in range(5, 10):
                for mod in range(3):
                    UAV_file = bf.file_filter(UAV_tif_folder, [str(_), f'chl{str(chl)}', '.tif'], and_or_factor='and', exclude_word_list=['aux', 'ovr'])
                    GEDI_file = bf.file_filter(GEDI_tif_folder, [f'heil{str(chl)}', f'mod{str(mod)}', '.tif'], and_or_factor='and', exclude_word_list=['aux', 'ovr'])
                    if len(UAV_file) == 1 and len(GEDI_file) == 1:
                        gdal.Warp(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_gedi.TIF', GEDI_file[0], cutlineDSName=shpfile, cropToCutline=True, xRes=10, yRes=10, outputType=gdal.GDT_Float32)
                        gdal.Warp(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_UAV.TIF', UAV_file[0], cutlineDSName=shpfile, cropToCutline=True, xRes=10, yRes=10, outputType=gdal.GDT_Float32)
                    else:
                        raise Exception(f'Date {_} chl {str(chl)} mod {str(mod)} is wrong!')

                    gedi_ds = gdal.Open(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_gedi.TIF')
                    uav_ds = gdal.Open(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_UAV.TIF')
                    gedi_nodata = gedi_ds.GetRasterBand(1).GetNoDataValue()
                    uav_nodata = uav_ds.GetRasterBand(1).GetNoDataValue()
                    gedi_arr = gedi_ds.GetRasterBand(1).ReadAsArray()
                    uav_arr = uav_ds.GetRasterBand(1).ReadAsArray()
                    if gedi_arr.shape == uav_arr.shape:
                        combine_arr = np.stack((gedi_arr.flatten(), uav_arr.flatten()), axis=0)

                        combine_arr = np.delete(combine_arr, np.argwhere(np.isnan(combine_arr))[:, 1], axis=1)
                        combine_arr = np.delete(combine_arr, np.argwhere(combine_arr == -3.402823e+38)[:, 1], axis=1)
                        combine_arr = np.delete(combine_arr, np.argwhere(combine_arr <= 0)[:, 1], axis=1)

                        df_temp = pd.DataFrame(combine_arr.transpose(), columns=['gedi_ch', 'uav_ch'])
                        df_temp.to_csv(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\Compare_tab\\date{_}_chl{str(chl)}_mod{str(mod)}.csv')
                        fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
                        # ax.scatter(combine_arr[1, :], combine_arr[0, :], s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
                        ax.hist2d(combine_arr[0, :], combine_arr[1, :], bins=100, range=[[0, chl], [0, chl]])
                        ax.plot(np.linspace(0, chl, chl), np.linspace(0, chl, chl), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
                        ax.set_xlabel('GEDI', fontname='Times New Roman', fontsize=40, fontweight='bold')
                        ax.set_ylabel('UAV', fontname='Times New Roman', fontsize=40, fontweight='bold')
                        RMSE_NDVI = np.sqrt(np.nanmean((combine_arr[1, :] - combine_arr[0, :]) ** 2))
                        MAE_NDVI = np.nanmean(np.absolute(combine_arr[1, :] - combine_arr[0, :]))
                        ax.text(2, chl-1, f'MAE:{str(MAE_NDVI)}', style='italic',)
                        ax.text(2, chl-2, f'RMSE:{str(RMSE_NDVI)}', style='italic', )
                        ax.set_ylim(0, chl)
                        ax.set_xlim(0, chl)
                        plt.savefig(f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\{str(date_t)}\\Compare_fig\\date{_}_chl{str(chl)}_mod{str(mod)}.png', dpi=300)
                        fig, ax = None, None
                        gdal.Unlink(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_UAV.TIF')
                        gdal.Unlink(f'/vsimem/date{_}_chl{str(chl)}_mod{str(mod)}_gedi.TIF')
                    else:
                        raise Exception(f'Date {_} chl {str(chl)} mod {str(mod)} facing crop error!')


def fig8_func():
    # Make figure and axes
    fig, ax = plt.subplots(2, 2)
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=12)
    # Shift the second slice using explode
    patches, texts, autotexts1 = ax[0, 0].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                  colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0.1, 0, 0, 0))
    patches, texts, autotexts2 = ax[0, 1].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0.1, 0, 0))
    patches, texts, autotexts3 = ax[1, 0].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0, 0.1, 0))
    patches, texts, autotexts4 =  ax[1, 1].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0, 0, 0.1))
    # ax.pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], )

    autotexts1[0].set_color('white')

    autotexts2[0].set_color('white')

    autotexts3[0].set_color('white')

    autotexts4[0].set_color('white')
    plt.savefig('G:\A_veg\Paper\Figure\Fig8\\fig83.png', dpi=300)


def fig9_func():
    # Make figure and axes
    fig, ax = plt.subplots(2, 2)
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=12)
    # Shift the second slice using explode
    patches, texts, autotexts1 = ax[0, 0].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                  colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0.1, 0, 0, 0))
    patches, texts, autotexts2 = ax[0, 1].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0.1, 0, 0))
    patches, texts, autotexts3 = ax[1, 0].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0, 0.1, 0))
    patches, texts, autotexts4 =  ax[1, 1].pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], autopct='%.0f%%',
                 colors=[(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)], shadow=False, explode=(0, 0, 0, 0.1))
    # ax.pie([7828, 2206, 4481, 5268], labels=['2019', '2020', '2021', '2022'], )

    autotexts1[0].set_color('white')

    autotexts2[0].set_color('white')

    autotexts3[0].set_color('white')

    autotexts4[0].set_color('white')
    plt.savefig('G:\A_veg\Paper\Figure\Fig8\\fig83.png', dpi=300)


def fig10_func():
    pd_temp = pd.read_csv('G:\A_veg\Paper\Figure\Fig8\\4year_ass2.csv')
    entire = np.array(pd_temp['Canopy Height (rh100)'])

    for i, c in zip([2019, 2020, 2021, 2022], [(51/256, 62/256, 81/256), (112/256, 160/256, 205/256), (196/256, 121/256, 0/256), (178/256, 178/256, 178/256)]):
        plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
        plt.rc('font', size=45)
        plt.rc('axes', linewidth=3)
        year = np.array(pd_temp[pd_temp['Year '] == i]['Canopy Height (rh100)'])
        fig, ax = plt.subplots(figsize=(10.5, 8.2), constrained_layout=True)
        plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
        plt.rc('font', size=12)
        ax.set_xlim([1, 8])
        # Shift the second slice using explode
        a1 = sns.kdeplot(data=entire, fill=True, color=(169/256, 2/256, 38/256), zorder=1, alpha=1, )
        a2 = sns.kdeplot(data=year, fill=True, color=c, alpha=1, zorder=3)
        ax.set_ylabel('概率密度')
        ax.set_xlabel('高度/m')
        plt.savefig(f'G:\A_veg\Paper\Figure\Fig8\\fig8_{str(i)}.png', dpi=300)


def fig11_func():
    pd1 = pd.read_csv('G:\\A_veg\\Paper\\Figure\\Fig9\\date220706_chl8_mod0.csv')
    pd2 = pd.read_csv('G:\\A_veg\\Paper\\Figure\\Fig9\\date220706_chl8_mod1.csv')
    pd1 = pd.merge(pd1, pd2, on=['Unnamed: 0'], how='left')
    chl = 8
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=14)
    fig, ax = plt.subplots(figsize=(5.8, 5), constrained_layout=True)
    # ax.scatter(combine_arr[1, :], combine_arr[0, :], s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
    qq = ax.hist2d(pd1['gedi_ch_x'], pd1['uav_ch_x'], bins=80, range=[[1, 6], [1, 6]], cmap=plt.cm.BuPu)
    fig.colorbar(qq[3], ax=ax)
    ax.plot(np.linspace(0, chl, chl), np.linspace(0, chl, chl), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
    ax.plot(np.linspace(0, chl, chl), np.linspace(0.89, chl+0.89, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax.plot(np.linspace(0, chl, chl), np.linspace(-0.89, chl-0.89, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax.fill_between(np.linspace(0, chl, chl), np.linspace(-0.89, chl - 0.89, chl), np.linspace(0.89, chl+0.89, chl), color=(0.3, 0.3, 0.3), alpha=0.1)
    ax.set_xlabel('外推植被高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax.set_ylabel('UAV航测高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')

    RMSE_NDVI = np.sqrt(np.nanmean((pd1['gedi_ch_x'] - pd1['uav_ch_x']) ** 2))
    MAE_NDVI = np.nanmean(np.absolute(pd1['gedi_ch_x'] - pd1['uav_ch_x']))
    ax.text(1.3, 6 - 0.5, f'MAE={str(MAE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax.text(1.3, 6 - 1, f'RMSE={str(RMSE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax.set_ylim(1, 6)
    ax.set_xlim(1, 6)
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig9\\fig91.png', dpi=300)

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=14)
    fig2, ax2 = plt.subplots(figsize=(5, 5), constrained_layout=True)
    # ax.scatter(combine_arr[1, :], combine_arr[0, :], s=12 ** 2, edgecolor=(0.1, 0.1, 0.1), facecolor=(47/256,85/256,151/256), alpha=0.8, linewidth=1.2, marker='^', zorder=5)
    qq = ax2.hist2d(pd1['gedi_ch_y'], pd1['uav_ch_y'], bins=80, range=[[1, 6], [1, 6]], cmap=plt.cm.BuPu)
    ax2.plot(np.linspace(0, chl, chl), np.linspace(0, chl, chl), c=(0, 0, 0), lw=1.5, zorder=3, ls='--')
    ax2.plot(np.linspace(0, chl, chl), np.linspace(0.93, chl+0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax2.plot(np.linspace(0, chl, chl), np.linspace(-0.93, chl-0.93, chl), c=(0.3, 0.3, 0.3), lw=0.5, zorder=3, ls='--')
    ax2.fill_between(np.linspace(0, chl, chl), np.linspace(-0.93, chl -0.93, chl), np.linspace(+0.93, chl +0.93, chl), color=(0.3, 0.3, 0.3), alpha=0.1)
    ax2.set_xlabel('外推植被高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax2.set_ylabel('UAV航测高度/m', fontname='Times New Roman', fontsize=20, fontweight='bold')
    RMSE_NDVI = np.sqrt(np.nanmean((pd1['gedi_ch_y'] - pd1['uav_ch_y']) ** 2))
    MAE_NDVI = np.nanmean(np.absolute(pd1['gedi_ch_y'] - pd1['uav_ch_y']))
    ax2.text(1.3, 6 - 0.5, f'MAE={str(MAE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax2.text(1.3, 6 - 1, f'RMSE={str(RMSE_NDVI)[0: 4]}m', c=(0, 0, 0), fontsize=18)
    ax2.set_ylim(1, 6)
    ax2.set_xlim(1, 6)
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig9\\fig92.png', dpi=300)


def fig12_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=3)

    file = 'G:\\A_veg\\Paper\\Figure\\Fig10\\ch_out_mod1_heil8.tif'
    ds = gdal.Open(file)
    arr = ds.GetRasterBand(1).ReadAsArray()
    mean = []
    min = []
    max = []
    for i in range(arr.shape[0]):
        q = arr[i, :].flatten()
        q = np.delete(q, np.argwhere(np.isnan(q)))
        mean.append(np.nanmean(q))
        if q != np.array([]):
            min.append(np.sort(q)[int(q.shape[0] * 0.05)])
            max.append(np.sort(q)[int(q.shape[0] * 0.95)])
        else:
            min.append(np.nan)
            max.append(np.nan)

    mean_re, min_re, max_re = [], [], []
    for _ in range(int(len(mean)/20)):
        qq = (_ + 1) * 20
        if qq > len(mean):
            mean_re.append(np.nanmean(mean[_ * 20: -1]))
            min_re.append(np.nanmean(min[_ * 20: -1]))
            max_re.append(np.nanmean(max[_ * 20: -1]))
        else:
            mean_re.append(np.nanmean(mean[_ * 20: (_ + 1) * 20]))
            min_re.append(np.nanmean(min[_ * 20: (_ + 1) * 20]))
            max_re.append(np.nanmean(max[_ * 20: (_ + 1) * 20]))

    min, max, mean = min_re, max_re, mean_re
    fig, ax = plt.subplots(figsize=(int(arr.shape[0] / 2000), 2.3), constrained_layout=True)
    ax.plot(range(len(mean)), mean, c=(0,0,0), lw=2)
    ax.fill_between(np.linspace(0, len(max), len(max)), min, max, color=(197/255, 197/255, 197/255), alpha=0.5, lw=2)
    ax.set_xlim([0, len(max)])
    ax.set_ylim([2.6, 5.8])
    ax.set_yticks([2.6, 3.4, 4.2, 5.0, 5.8])
    ax.set_yticklabels(['2', '3', '4', '5', '6'], fontname='Times New Roman', fontsize=12)

    # current_axes = plt.axes()
    # current_axes.get_xaxis().set_visible(False)
    # ax[1, 0] = plt.imshow(arr, cmap=copy.copy(plt.cm.plasma))
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig10\\fig10_y.png', dpi=600)


    mean = []
    min = []
    max = []
    for i in range(arr.shape[1]):
        q = arr[:, i].flatten()
        q = np.delete(q, np.argwhere(np.isnan(q)))
        mean.append(np.nanmean(q))
        if q != np.array([]):
            min.append(np.sort(q)[int(q.shape[0] * 0.05)])
            max.append(np.sort(q)[int(q.shape[0] * 0.95)])
        else:
            min.append(np.nan)
            max.append(np.nan)

    mean_re, min_re, max_re = [], [], []
    for _ in range(int(len(mean)/20)):
        qq = (_ + 1) * 20
        if qq > len(mean):
            mean_re.append(np.nanmean(mean[_ * 20: -1]))
            min_re.append(np.nanmean(min[_ * 20: -1]))
            max_re.append(np.nanmean(max[_ * 20: -1]))
        else:
            mean_re.append(np.nanmean(mean[_ * 20: (_ + 1) * 20]))
            min_re.append(np.nanmean(min[_ * 20: (_ + 1) * 20]))
            max_re.append(np.nanmean(max[_ * 20: (_ + 1) * 20]))

    min, max, mean = min_re, max_re, mean_re
    fig, ax = plt.subplots(figsize=(int(arr.shape[1] / 2000), 2.3), constrained_layout=True)
    ax.plot(range(len(mean)), mean, c=(0,0,0), lw=2)
    ax.fill_between(np.linspace(0, len(max), len(max)), min, max, color=(197/255, 197/255, 197/255), alpha=0.5, lw=2)
    ax.set_xlim([0, len(max)])
    ax.set_ylim([2.6, 5.8])
    ax.set_yticks([2.6, 3.4, 4.2, 5.0, 5.8])
    ax.set_yticklabels(['2', '3', '4', '5', '6'], fontname='Times New Roman', fontsize=12)

    plt.savefig(f'G:\A_veg\Paper\Figure\Fig10\\fig10_x.png', dpi=300)


def fig12_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=12)
    plt.rc('axes', linewidth=3)

    file = 'G:\\A_veg\\Paper\\Figure\\Fig10\\ch_out_mod1_heil8.tif'
    ds = gdal.Open(file)
    arr = ds.GetRasterBand(1).ReadAsArray()
    mean = []
    min = []
    max = []
    for i in range(arr.shape[0]):
        q = arr[i, :].flatten()
        q = np.delete(q, np.argwhere(np.isnan(q)))
        mean.append(np.nanmean(q))
        if q != np.array([]):
            min.append(np.sort(q)[int(q.shape[0] * 0.05)])
            max.append(np.sort(q)[int(q.shape[0] * 0.95)])
        else:
            min.append(np.nan)
            max.append(np.nan)

    mean_re, min_re, max_re = [], [], []
    for _ in range(int(len(mean)/20)):
        qq = (_ + 1) * 20
        if qq > len(mean):
            mean_re.append(np.nanmean(mean[_ * 20: -1]))
            min_re.append(np.nanmean(min[_ * 20: -1]))
            max_re.append(np.nanmean(max[_ * 20: -1]))
        else:
            mean_re.append(np.nanmean(mean[_ * 20: (_ + 1) * 20]))
            min_re.append(np.nanmean(min[_ * 20: (_ + 1) * 20]))
            max_re.append(np.nanmean(max[_ * 20: (_ + 1) * 20]))

    min, max, mean = min_re, max_re, mean_re
    fig, ax = plt.subplots(figsize=(int(arr.shape[0] / 2000), 2.3), constrained_layout=True)
    ax.plot(range(len(mean)), mean, c=(0,0,0), lw=2)
    ax.fill_between(np.linspace(0, len(max), len(max)), min, max, color=(197/255, 197/255, 197/255), alpha=0.5, lw=2)
    ax.set_xlim([0, len(max)])
    ax.set_ylim([2.6, 5.8])
    ax.set_yticks([2.6, 3.4, 4.2, 5.0, 5.8])
    ax.set_yticklabels(['2', '3', '4', '5', '6'], fontname='Times New Roman', fontsize=12)

    # current_axes = plt.axes()
    # current_axes.get_xaxis().set_visible(False)
    # ax[1, 0] = plt.imshow(arr, cmap=copy.copy(plt.cm.plasma))
    plt.savefig(f'G:\A_veg\Paper\Figure\Fig10\\fig10_y.png', dpi=600)


    mean = []
    min = []
    max = []
    for i in range(arr.shape[1]):
        q = arr[:, i].flatten()
        q = np.delete(q, np.argwhere(np.isnan(q)))
        mean.append(np.nanmean(q))
        if q != np.array([]):
            min.append(np.sort(q)[int(q.shape[0] * 0.05)])
            max.append(np.sort(q)[int(q.shape[0] * 0.95)])
        else:
            min.append(np.nan)
            max.append(np.nan)

    mean_re, min_re, max_re = [], [], []
    for _ in range(int(len(mean)/20)):
        qq = (_ + 1) * 20
        if qq > len(mean):
            mean_re.append(np.nanmean(mean[_ * 20: -1]))
            min_re.append(np.nanmean(min[_ * 20: -1]))
            max_re.append(np.nanmean(max[_ * 20: -1]))
        else:
            mean_re.append(np.nanmean(mean[_ * 20: (_ + 1) * 20]))
            min_re.append(np.nanmean(min[_ * 20: (_ + 1) * 20]))
            max_re.append(np.nanmean(max[_ * 20: (_ + 1) * 20]))

    min, max, mean = min_re, max_re, mean_re
    fig, ax = plt.subplots(figsize=(int(arr.shape[1] / 2000), 2.3), constrained_layout=True)
    ax.plot(range(len(mean)), mean, c=(0,0,0), lw=2)
    ax.fill_between(np.linspace(0, len(max), len(max)), min, max, color=(197/255, 197/255, 197/255), alpha=0.5, lw=2)
    ax.set_xlim([0, len(max)])
    ax.set_ylim([2.6, 5.8])
    ax.set_yticks([2.6, 3.4, 4.2, 5.0, 5.8])
    ax.set_yticklabels(['2', '3', '4', '5', '6'], fontname='Times New Roman', fontsize=12)

    plt.savefig(f'G:\A_veg\Paper\Figure\Fig10\\fig10_x.png', dpi=300)


def fig14_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=20)
    plt.rc('axes', linewidth=2)

    res_dic = {}
    for section in ['ch', 'jj', 'hh', 'yz']:
        shpfile = f'G:\\A_veg\\Paper\\Figure\\Fig11\\{section}.shp'
        for year in ['2019', '2020', '2021', '2022']:
            if not os.path.exists(f'G:\\A_veg\\Paper\\Figure\\Fig11\\tif\\{section}_{year}.tif'):
                tif_file = f'G:\\A_veg\\S2_all\\Feature_table4heightmap\\peak_{year}\\predicted_feature_tif\\ch_out_mod0_heil8.tif'
                gdal.Warp(f'G:\\A_veg\\Paper\\Figure\\Fig11\\tif\\{section}_{year}.tif', tif_file, cutlineDSName=shpfile, cropToCutline=True, xRes=10, yRes=10, outputType=gdal.GDT_Float32,dstNodata=np.nan)
            ds = gdal.Open(f'G:\\A_veg\\Paper\\Figure\\Fig11\\tif\\{section}_{year}.tif')
            res_dic[f'{section}_{year}'] = ds.GetRasterBand(1).ReadAsArray()

    a = [np.nanmean(res_dic[f'{section}_{2020}'] - res_dic[f'{section}_{str(int(2020) - 1)}']) for section in
         ['ch', 'jj', 'hh', 'yz']]

    # Comparable
    for section in ['ch', 'jj', 'hh', 'yz']:

        range = None
        for year in ['2019', '2020', '2021', '2022']:
            if range is None:
                range = copy.copy(res_dic[f'{section}_{year}'])
                range[~np.isnan(range)] = 1
            else:
                range[np.isnan(res_dic[f'{section}_{year}'])] = np.nan

        data_mean = None
        data_max = []
        data_min = []
        for year in ['2019', '2020', '2021', '2022']:
            _ = res_dic[f'{section}_{year}'] * range
            _ = _.flatten()
            _ = np.delete(_, np.argwhere(np.isnan(_)))
            _.sort()

            q = np.ones_like(_) * (int(year) - 2019)

            if data_mean is None:
                data_mean = np.stack([q, _], axis=0)
            else:
                data_mean = np.concatenate((data_mean, np.stack([q, _], axis=0)), axis=1)
            data_max.append(_[int(len(_) * 0.025)])
            data_min.append(_[int(len(_) * 0.975)])
            res_dic[f'{section}_{year}_v'] = _

        data_mean = pd.DataFrame(data_mean.transpose(), columns=['x', 'y'])
        data_all = [res_dic[f'{section}_{year}_v'] for year in ['2019', '2020', '2021', '2022']]
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

        paras1, extras = curve_fit(poly2, [0, 1, 2, 3], data_max, maxfev=50000)
        paras2, extras = curve_fit(poly2, [0, 1, 2, 3], data_min, maxfev=50000)
        paras, extras = curve_fit(poly2, np.array(data_mean['x']), np.array(data_mean['y']), maxfev=50000)

        # bplot1 = ax.boxplot(data_all, notch=False, positions = [2019, 2020, 2021, 2022],  widths=0.2, patch_artist=True, whis=(5, 95), showfliers=False, zorder=1, showmeans=False, )
        # colors = [(51/256, 62/256, 81/256), (112/256, 159/256, 204/256), (195/256, 121/256, 0/256), (177/256, 177/256, 177/256)]
        # for patch, color in zip(bplot1['boxes'], colors):
        #     patch.set_facecolor(color)
        #     patch._linewidth = 2

        sns.violinplot(data=data_mean, x="x", y="y", alpha=.55, zorder=2, legend=False, width=0.6, linewidth=0.9, split=True, palette=[(189/256, 215/256, 231/256), (107/256, 174/256, 214/256), (49/256, 130/256, 189/256), (8/256, 81/256, 156/256)])
        ax.fill_between(np.linspace(-0.5, 4.5, 100), poly2(np.linspace(-0.5, 4.5, 100), paras1[0], paras1[1], paras1[2]), poly2(np.linspace(0.5, 4.5, 100), paras2[0], paras2[1], paras2[2]), zorder=1, alpha = 0.1, fc=(0.5, 0.5, 0.5))
        ax.plot(np.linspace(-0.5, 4.5, 100), poly2(np.linspace(-0.5, 4.5, 100), paras[0], paras[1], paras[2]), zorder=4, c=(0.8, 0.1, 0.1), lw=4)
        ax.plot(np.linspace(-0.5, 4.5, 100), poly2(np.linspace(-0.5, 4.5, 100), paras1[0], paras1[1], paras1[2]), c=(0.1, 0.1, 0.1), lw=2, ls='--', zorder=1)
        ax.plot(np.linspace(-0.5, 4.5, 100), poly2(np.linspace(-0.5, 4.5, 100), paras2[0], paras2[1], paras2[2]), c=(0.1, 0.1, 0.1), lw=2, ls='--', zorder=1)

        ax.set_xlim([-0.5, 3.5])
        ax.set_ylim([2, 7])
        plt.savefig(f'G:\\A_veg\\Paper\\Figure\\Fig11\\fig\\{section}.png', dpi=300)
        fig, ax = None, None

    sns.set_theme(style="whitegrid", )
    plt.rc('axes', linewidth=2)
    plt.rc('axes', edgecolor=(0,0,0))
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

    # dic_temp = {'sn':[], 'ch': []}
    # for section in ['ch', 'jj', 'hh', 'yz']:
    #     _ =  res_dic[f'{section}_2022']
    #     _ = _.flatten()
    #     _ = np.delete(_, np.argwhere(np.isnan(_)))
    #     dic_temp['ch'].extend(_.tolist())
    #     dic_temp['sn'].extend([section for temp in range(_.shape[0])])
    # df_temp = pd.DataFrame(dic_temp)
    # seaborn.boxenplot(data=df_temp, x = 'sn', y='ch', scale="area", saturation = 0.6, width=0.6, order=['yz', 'jj', 'ch','hh'],  color="b")
    # ax.set_ylim([2, 7])
    # plt.savefig(f'G:\\A_veg\\Paper\\Figure\\Fig11\\fig\\sect4.png', dpi=300)

    # Diff
    a = [np.nanmean(res_dic[f'{section}_{2020}'] - np.nanmean(res_dic[f'{section}_{str(int(2020) - 1)}'])) for section in ['ch', 'jj', 'hh', 'yz']]

    for section in ['ch', 'jj', 'hh', 'yz']:
        for year in ['2020', '2021', '2022']:
            arr_temp = res_dic[f'{section}_{year}'] - res_dic[f'{section}_{str(int(year) - 1)}']
            arr_temp = arr_temp.flatten()
            arr_temp = np.delete(arr_temp, np.argwhere(np.isnan(arr_temp)))
            res_dic[f'{str(int(year) - 1)}_{year}_{section}'] = arr_temp

        # fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        # seaborn.violinplot([res_dic[f'2019_2020_{section}'], res_dic[f'2020_2021_{section}'], res_dic[f'2021_2022_{section}']])
        # plt.savefig(f'G:\\A_veg\\Paper\\Figure\\Fig11\\fig\\{section}.png', dpi=300)
        # fig, ax = None, None


fig11_func()



