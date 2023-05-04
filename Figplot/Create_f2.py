import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.gridspec as gridspec
import gdal
import ogr
import copy
import sys
from matplotlib.colors import LinearSegmentedColormap
import basic_function as bf
import seaborn as sns


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
    ax1.set_xticks([0, 3, 6, 9, 12, 15])
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

fig6_func()



