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



fig2_func()


