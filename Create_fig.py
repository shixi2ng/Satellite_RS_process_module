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
import Landsat_main_v1
import sys
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import basic_function as bf
import seaborn as sns
import Landsat_main_v2


def guassain_dis(x, sig, mean):
    return np.exp(-(x - mean) ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def mark_plotter(ax, data1, data2, param_dict):
    out = ax.plot(data1, data2, **param_dict)
    return out


def linear_function(x, a):
    return a * x


def linear_function2(x, a, b):
    return a* x + b


# Create fig3
def fig3_func():
    # sns.set_theme(style="darkgrid")
    fig3_df = pd.read_excel('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig3\\data.xlsx')
    fig3_array = np.array(fig3_df)
    fig3_array_new = np.array([[0], [1]])
    fig3_dic = {'DOY': [], 'OSAVI': []}
    for i in range(fig3_array.shape[1]):
        if 2006000 < fig3_array[0, i] < 2008000 and not np.isnan(fig3_array[1, i]):
            fig3_dic['DOY'].append(np.mod(fig3_array[0, i], 1000) + 365 * (fig3_array[0, i] // 1000 - 2006))
            fig3_dic['OSAVI'].append(fig3_array[1, i])
    fig3_df = pd.DataFrame(data=fig3_dic)

    array_temp = np.array([fig3_dic['DOY'], fig3_dic['OSAVI']])
    date_index = 0
    while date_index < array_temp.shape[1]:
        if array_temp[0, date_index] > 576:
            array_temp = np.delete(array_temp, date_index, axis=1)
            date_index -= 1
        date_index += 1

    date_index = 0
    while date_index < array_temp.shape[1]:
        if 211 < array_temp[0, date_index] < 366:
            array_temp = np.append(array_temp, np.array([array_temp[0, date_index] + 365, array_temp[1, date_index]]).reshape([2, 1]), axis=1)
        date_index += 1

    for a in range(array_temp.shape[1]):
        if array_temp[0, a] > 576:
            break

    for b in range(array_temp.shape[1]):
        if array_temp[0, b] > 200:
            break

    fig, ax = plt.subplots(figsize=(21, 11), constrained_layout=True)
    ax.set_axis_on()
    ax.set_xlim(0, 730)
    ax.set_ylim(0, 0.7)
    ax.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax.plot(fig3_dic['DOY'], fig3_dic['OSAVI'], linewidth=10, markersize=24, **{'marker': 'o', 'color': 'b'})
    ax.plot(array_temp[0, 0: a], array_temp[1, 0: a], linewidth=10, markersize=24, **{'marker': 'o', 'color': 'r'})
    ax.plot(array_temp[0, :], array_temp[1, :], linewidth=10, markersize=24, **{'ls': '--', 'marker': 'o', 'color': 'r'})
    ax.plot(array_temp[0, 0: b], array_temp[1, 0: b], linewidth=10, markersize=24,
            **{'marker': 'o', 'color': (0, 0, 0)})
    ax.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    ax.fill_between(np.linspace(195, 560, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0.8, 0.8, 0.8), alpha=1)
    ax.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax.set_xlabel('Date', fontname='Times New Roman', fontsize=40, fontweight='bold')
    ax.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=40, fontweight='bold')
    ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], fontname='Times New Roman', fontsize=28)
    a = [15, 75, 136, 197, 258, 320]
    b = np.array(a) + 365
    c = ['06-Jan', '06-Mar', '06-May', '06-Jul', '06-Sep', '06-Nov', '07-Jan', '07-Mar', '07-May', '07-Jul', '07-Sep', '07-Nov']
    a.extend(b.tolist())
    # for i in b:
    #     a.append(i)
    ax.set_xticks(a)
    ax.set_xticklabels(c, fontname='Times New Roman', fontsize=28, fontweight='bold')
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig3_df)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig3\\Figure_3.png', dpi=1000)
    plt.show()


def fig4_func():
    # Create fig4
    VI_curve_fitting = {'para_ori': [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225], 'para_boundary': (
    [0.08, 0.7, 100, 6.2, 301.6, 4.5, 0.0015], [0.12, 1.0, 115, 11.5, 321.5, 8.8, 0.0028])}
    VI_curve_fitting = {'para_ori': [0.01, 0.01, 0, 2, 180, 2, 0.01], 'para_boundary': ([0, 0, 0, 0, 180, 0, 0], [0.5, 1, 180, 20, 330, 10, 0.01])}
    fig4_df = pd.read_excel('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig4\\data.xlsx')
    fig4_array = np.array(fig4_df)
    fig4_array_new = np.array([[0], [1]])
    fig4_dic = {'DOY': [], 'OSAVI': []}
    for i in range(fig4_array.shape[1]):
        if not np.isnan(fig4_array[1, i]):
            fig4_dic['DOY'].append(fig4_array[0, i])
            fig4_dic['OSAVI'].append(fig4_array[1, i])
    fig4_df = pd.DataFrame(data=fig4_dic)
    fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
    # fig4, ax4 = plt.subplots(figsize=(10.5, 10.5), constrained_layout=True)
    ax4.set_axisbelow(True)
    ax4.set_xlim(0, 365)
    ax4.set_ylim(0, 0.7)
    paras, extra = curve_fit(seven_para_logistic_function, fig4_dic['DOY'], fig4_dic['OSAVI'], maxfev=500000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])

    # define p3 and p5
    doy_all = fig4_dic['DOY'][1:]
    vi_all = fig4_dic['OSAVI'][1:]
    vi_dormancy = []
    doy_dormancy = []
    vi_senescence = []
    doy_senescence = []
    vi_max = []
    doy_max = []
    doy_index_max = np.argmax(seven_para_logistic_function(np.linspace(0, 366, 365), paras[0], paras[1], paras[2], paras[3], paras[4],paras[5], paras[6]))
    # Generate the parameter boundary
    senescence_t = paras[4] - 4 * paras[5]
    for doy_index in range(len(doy_all)):
        if 0 < doy_all[doy_index] < paras[2] or paras[4] < doy_all[doy_index] < 366:
            vi_dormancy.append(vi_all[doy_index])
            doy_dormancy.append(doy_all[doy_index])
        if doy_index_max - 5 < doy_all[doy_index] < doy_index_max + 5:
            vi_max.append(vi_all[doy_index])
            doy_max.append(doy_all[doy_index])
        if senescence_t - 5 < doy_all[doy_index] < senescence_t + 5:
            vi_senescence.append(vi_all[doy_index])
            doy_senescence.append(doy_all[doy_index])

    vi_dormancy_sort = np.sort(vi_dormancy)
    vi_max_sort = np.sort(vi_max)
    paras1_max = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
    paras1_min = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
    paras2_max = vi_max[-1] - paras1_min
    paras2_min = vi_max[0] - paras1_max
    paras3_max = 0
    for doy_index in range(len(doy_all)):
        if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] < 180:
            paras3_max = max(float(paras3_max), doy_all[doy_index])
    paras3_max = max(paras3_max, paras[2])
    paras3_min = 180
    for doy_index in range(len(doy_all)):
        if vi_all[doy_index] > paras1_max:
            paras3_min = min(paras3_min, doy_all[doy_index])
    paras3_min = min(paras[2], paras3_min)
    paras3_max = max(paras3_max, paras[2])
    paras5_max = 0
    for doy_index in range(len(doy_all)):
        if vi_all[doy_index] > paras1_max:
            paras5_max = max(paras5_max, doy_all[doy_index])
    paras5_max = max(paras5_max, paras[4])
    paras5_min = 365
    for doy_index in range(len(doy_all)):
        if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] > 180:
            paras5_min = min(paras5_min, doy_all[doy_index])
    paras5_min = min(paras5_min, paras[4])
    paras4_max = (np.nanmax(doy_max) - paras3_min) / 4
    paras4_min = (np.nanmin(doy_max) - paras3_max) / 4
    paras6_max = paras4_max
    paras6_min = paras4_min
    paras7_max = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (doy_senescence[np.argmin(vi_senescence)] - doy_max[np.argmax(vi_max)])
    paras7_min = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (doy_senescence[np.argmax(vi_senescence)] - doy_max[np.argmin(vi_max)])
    a = (
    [paras1_min, paras2_min, paras3_min, paras4_min, paras5_min, paras6_min, paras7_min],
    [paras1_max, paras2_max, paras3_max, paras4_max, paras5_max, paras6_max, paras7_max])

    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=10, color=(0/256, 109/256, 44/256))
    fig4_dic['DOY'] = fig4_dic['DOY'][1:]
    fig4_dic['OSAVI'] = fig4_dic['OSAVI'][1:]
    fig4_df = pd.DataFrame.from_dict(fig4_dic)
    # ax4.plot(array_temp[0, :], array_temp[1, :], linewidth=4, markersize=12, **{'ls': '--', 'marker': 'o', 'color': 'b'})
    ax4.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.203, 0.54, 81.5, 9, 331, 12, 0.00071), seven_para_logistic_function(np.linspace(0, 365, 366), 0.05, 0.53, 102, 8, 330, 12, 0.00125), color=(0.1, 0.1, 0.1), alpha=0.1)
    ax4.scatter(fig4_dic['DOY'], fig4_dic['OSAVI'], s=12**2, color="none", edgecolor=(160/256, 196/256, 160/256), linewidth=3)
    # ax4.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    # ax4.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax4.set_xlabel('DOY', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.203, 0.54, 81.5, 9, 331, 12, 0.00071), linewidth=2, color=(0 / 256, 109 / 256, 44 / 256), **{'ls': '--'})
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.05, 0.53, 102, 8, 330, 12, 0.00125), linewidth=2, color=(0 / 256, 109 / 256, 44 / 256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_min, paras2_min, paras3_min, paras4_max, paras5_min, paras6_max, paras7_max), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_max, paras2_max, paras3_max, paras4_min, paras5_max, paras6_min, paras7_min), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    predicted_y_data = seven_para_logistic_function(np.array(fig4_dic['DOY']), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square = (1 - np.nansum((predicted_y_data - np.array(fig4_dic['OSAVI'])) ** 2) / np.nansum((np.array(fig4_dic['OSAVI']) - np.nanmean(np.array(fig4_dic['OSAVI']))) ** 2))
    ax4.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], fontname='Times New Roman', fontsize=26)
    a = [15, 45, 75, 105, 136, 166, 197, 227, 258, 288, 320, 350]
    c = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    points = np.array([fig4_dic['OSAVI'],fig4_dic['DOY']]).transpose()
    # hull = ConvexHull(points)
    # # # for i in b:
    # # #     a.append(i)
    # ax4.plot(points[hull.vertices,1], points[hull.vertices,0], 'r--', lw=2)
    ax4.set_xticks(a)
    ax4.set_xticklabels(c, fontname='Times New Roman', fontsize=30)
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig4\\Figure_41.png', dpi=1000)
    plt.show()
    print(r_square)

def fig42_func():
    # Create fig4
    VI_curve_fitting = {'para_ori': [0.15, 0.49, 94.77, 7.23, 322.10, 7.17, 0.00077], 'para_boundary': ([0, 0, 0, 0, 180, 0, 0.00051], [0.5, 1, 180, 20, 330, 20, 0.001])}
    fig4_df = pd.read_excel('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig4\\data2.xlsx')
    fig4_array = np.array(fig4_df)
    fig4_array_new = np.array([[0], [1]])
    fig4_dic = {'DOY': [], 'OSAVI': []}
    for i in range(1, fig4_array.shape[1]):
        if not np.isnan(fig4_array[1, i]):
            fig4_dic['DOY'].append(fig4_array[0, i])
            fig4_dic['OSAVI'].append(fig4_array[1, i])
    fig4_df = pd.DataFrame(data=fig4_dic)
    fig4, ax4 = plt.subplots(figsize=(12, 8), constrained_layout=True)
    # fig4, ax4 = plt.subplots(figsize=(10.5, 10.5), constrained_layout=True)
    ax4.set_axisbelow(True)
    ax4.set_xlim(0, 365)
    ax4.set_ylim(0, 0.7)
    paras, extra = curve_fit(seven_para_logistic_function, fig4_dic['DOY'], fig4_dic['OSAVI'], maxfev=500000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])

    # define p3 and p5
    doy_all = fig4_dic['DOY'][1:]
    vi_all = fig4_dic['OSAVI'][1:]
    vi_dormancy = []
    doy_dormancy = []
    vi_senescence = []
    doy_senescence = []
    vi_max = []
    doy_max = []
    doy_index_max = np.argmax(seven_para_logistic_function(np.linspace(0, 366, 365), paras[0], paras[1], paras[2], paras[3], paras[4],paras[5], paras[6]))
    # Generate the parameter boundary
    senescence_t = paras[4] - 4 * paras[5]
    for doy_index in range(len(doy_all)):
        if 0 < doy_all[doy_index] < paras[2] or paras[4] < doy_all[doy_index] < 366:
            vi_dormancy.append(vi_all[doy_index])
            doy_dormancy.append(doy_all[doy_index])
        if doy_index_max - 5 < doy_all[doy_index] < doy_index_max + 5:
            vi_max.append(vi_all[doy_index])
            doy_max.append(doy_all[doy_index])
        if senescence_t - 5 < doy_all[doy_index] < senescence_t + 5:
            vi_senescence.append(vi_all[doy_index])
            doy_senescence.append(doy_all[doy_index])

    vi_dormancy_sort = np.sort(vi_dormancy)
    vi_max_sort = np.sort(vi_max)

    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=10, color=(200/256, 44/256, 44/256), zorder=1)
    fig4_dic['DOY'] = fig4_dic['DOY'][1:]
    fig4_dic['OSAVI'] = fig4_dic['OSAVI'][1:]
    fig4_df = pd.DataFrame.from_dict(fig4_dic)
    # ax4.plot(array_temp[0, :], array_temp[1, :], linewidth=4, markersize=12, **{'ls': '--', 'marker': 'o', 'color': 'b'})
    ax4.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.203, 0.54, 81.5, 9, 331, 12, 0.00071), seven_para_logistic_function(np.linspace(0, 365, 366), 0.05, 0.53, 102, 8, 330, 12, 0.00125), color=(0.1, 0.1, 0.1), alpha=0.1)
    ax4.scatter(fig4_dic['DOY'], fig4_dic['OSAVI'], s=13**2, color=(196/256, 80/256, 80/256), edgecolor=(0/256, 0/256, 0/256), linewidth=2, zorder=4)
    # ax4.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    # ax4.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax4.set_xlabel('DOY', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.203, 0.54, 81.5, 9, 331, 12, 0.00071), linewidth=2, color=(109 / 256, 44 / 256, 0 / 256), **{'ls': '--'})
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.05, 0.53, 102, 8, 330, 12, 0.00125), linewidth=2, color=(109 / 256, 44 / 256, 0 / 256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_min, paras2_min, paras3_min, paras4_max, paras5_min, paras6_max, paras7_max), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_max, paras2_max, paras3_max, paras4_min, paras5_max, paras6_min, paras7_min), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    predicted_y_data = seven_para_logistic_function(np.array(fig4_dic['DOY']), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square = (1 - np.sum((predicted_y_data - np.array(fig4_dic['OSAVI'])) ** 2) / np.sum((np.array(fig4_dic['OSAVI']) - np.mean(np.array(fig4_dic['OSAVI']))) ** 2))
    ax4.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], fontname='Times New Roman', fontsize=26)
    a = [15, 45, 75, 105, 136, 166, 197, 227, 258, 288, 320, 350]
    c = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    points = np.array([fig4_dic['OSAVI'],fig4_dic['DOY']]).transpose()

    ax4.set_xticks(a)
    ax4.set_xticklabels(c, fontname='Times New Roman', fontsize=30)
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig4\\Figure_4.png', dpi=1000)
    plt.show()
    print(r_square)

def fig3_ad_func():
    # sns.set_theme(style="darkgrid")
    fig3_df = pd.read_excel('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig3\\data2.xlsx')
    fig3_array = np.array(fig3_df)
    fig3_array_new = np.array([[0], [1]])
    fig3_dic = {'DOY': [], 'OSAVI': []}
    for i in range(fig3_array.shape[1]):
        if 2006000 < fig3_array[0, i] < 2008000 and not np.isnan(fig3_array[1, i]):
            fig3_dic['DOY'].append(np.mod(fig3_array[0, i], 1000) + 365 * (fig3_array[0, i] // 1000 - 2006))
            fig3_dic['OSAVI'].append(fig3_array[1, i])
    fig3_df = pd.DataFrame(data=fig3_dic)

    array_temp = np.array([fig3_dic['DOY'], fig3_dic['OSAVI']])
    date_index = 0
    while date_index < array_temp.shape[1]:
        if array_temp[0, date_index] > 570:
            array_temp = np.delete(array_temp, date_index, axis=1)
            date_index -= 1
        date_index += 1

    date_index = 0
    while date_index < array_temp.shape[1]:
        if 240 < array_temp[0, date_index] < 366:
            array_temp = np.append(array_temp, np.array([array_temp[0, date_index] + 365, array_temp[1, date_index]]).reshape([2, 1]), axis=1)
        date_index += 1

    for a in range(array_temp.shape[1]):
        if array_temp[0, a] > 576:
            break

    for b in range(array_temp.shape[1]):
        if array_temp[0, b] > 150:
            break

    for c in range(array_temp.shape[1]):
        if array_temp[0, c] > 240:
            break

    fig, ax = plt.subplots(figsize=(21, 11), constrained_layout=True)
    ax.set_axis_on()
    ax.set_xlim(0, 730)
    ax.set_ylim(0, 0.7)
    ax.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax.plot(fig3_dic['DOY'], fig3_dic['OSAVI'], linewidth=10, markersize=24, **{'marker': 'o', 'color': 'b'})
    ax.plot(array_temp[0, 0: a], array_temp[1, 0: a], linewidth=10, markersize=24, **{'marker': 'o', 'color': 'r'})
    ax.plot(array_temp[0, :], array_temp[1, :], linewidth=10, markersize=24, **{'ls': '--', 'marker': 'o', 'color': 'r'})
    ax.plot(array_temp[0, 0: c], array_temp[1, 0: c], linewidth=10, markersize=24,
            **{'marker': 'o', 'color': 'b'})
    ax.plot(array_temp[0, 0: b], array_temp[1, 0: b], linewidth=10, markersize=24,
         **{'marker': 'o', 'color': (0, 0, 0)})

    ax.fill_between(np.linspace(160, 245, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    ax.fill_between(np.linspace(570, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197 / 255, 1),alpha=1)
    ax.fill_between(np.linspace(242, 570, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0.8, 0.8, 0.8), alpha=1)
    ax.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax.set_xlabel('Date', fontname='Times New Roman', fontsize=40, fontweight='bold')
    ax.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=40, fontweight='bold')
    ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], fontname='Times New Roman', fontsize=28)
    a = [15, 75, 136, 197, 258, 320]
    b = np.array(a) + 365
    c = ['06-Jan', '06-Mar', '06-May', '06-Jul', '06-Sep', '06-Nov', '07-Jan', '07-Mar', '07-May', '07-Jul', '07-Sep', '07-Nov']
    a.extend(b.tolist())
    # for i in b:
    #     a.append(i)
    ax.set_xticks(a)
    ax.set_xticklabels(c, fontname='Times New Roman', fontsize=28, fontweight='bold')
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig3_df)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig3\\Figure_31.png', dpi=1000)
    plt.show()


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def fig5_func():
    # Generate npy from visulisation
    # a = self.vi_sa_array_for_phenology[61:100, 60:80, :]
    # b = self.doy_array_for_phenology
    # c = np.array([b, np.nanmean(a, axis=(0, 1))])
    # np.save('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\nyz_NDVI.npy', c)
    # Create fig5
    plt.rc('axes', linewidth=3)
    VI_curve_fitting = {'para_ori': [0.01, 0.01, 50, 2, 300, 5, 0.01], 'para_boundary': ([0, 0, 50, 0, 300, 0, 0], [0.5, 1, 100, 15, 330, 15, 0.03])}
    bsz_EVI = np.load('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\bsz_EVI.npy')[:,0:631]
    bsz_NDVI = np.load('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\bsz_NDVI.npy')[:,0:631]
    bsz_OSAVI = np.load('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\bsz_OSAVI.npy')
    temp = bsz_EVI[:, 556:631]
    temp[1, :] = temp[1, :]/1.16
    bsz_OSAVI = np.concatenate([bsz_OSAVI, temp], axis = 1)

    i = 0
    while i < bsz_EVI.shape[1]:
        if np.isnan(bsz_EVI[1, i]) or bsz_EVI[1, i]<0.1 or bsz_EVI[0, i] // 1000 < 2000:
            bsz_EVI = np.delete(bsz_EVI, i, axis=1)
            i -= 1
        i += 1

    i = 0
    while i < bsz_NDVI.shape[1]:
        if np.isnan(bsz_NDVI[1, i]) or bsz_NDVI[1, i]<0.1 or bsz_NDVI[0, i] // 1000 < 2000:
            bsz_NDVI = np.delete(bsz_NDVI, i, axis=1)
            i -= 1
        i += 1

    i = 0
    while i < bsz_OSAVI.shape[1]:
        if np.isnan(bsz_OSAVI[1, i]) or bsz_OSAVI[1, i]<0.1 or bsz_OSAVI[0, i] // 1000 < 2000 or bsz_OSAVI[0, i] // 1000 in [2002,2001]:
            bsz_OSAVI = np.delete(bsz_OSAVI, i, axis=1)
            i -= 1
        i += 1

    bsz_EVI[0, :] = np.mod(bsz_EVI[0, :], 1000)
    bsz_NDVI[0, :] = np.mod(bsz_NDVI[0, :], 1000)
    bsz_OSAVI[0, :] = np.mod(bsz_OSAVI[0, :], 1000)
    bsz_OSAVI = np.unique(bsz_OSAVI, axis=1)
    bsz_EVI = np.unique(bsz_EVI, axis=1)
    bsz_NDVI = np.unique(bsz_NDVI, axis=1)

    a = [15, 75, 136, 197, 258, 340]
    c = ['Jan', 'Mar', 'May', 'Jul', 'Sep', 'Dec']
    b = [-0.6, -0.3, 0, 0.3, 0.6]
    d = ['-60%', '-30%', '0%', '30%', '60%']

    fig5 = plt.figure(figsize=(15, 8.7), tight_layout=True)
    gs = gridspec.GridSpec(2, 3)

    ax1 = fig5.add_subplot(gs[0, 0])
    ax2 = fig5.add_subplot(gs[0, 2])
    ax3 = fig5.add_subplot(gs[0, 1])
    ax1_box = fig5.add_subplot(gs[1, 0])
    ax2_box = fig5.add_subplot(gs[1, 1])
    ax3_box = fig5.add_subplot(gs[1, 2])
    # plot
    ax1.set_axisbelow(True)
    ax1.set_xlim(0, 365)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('DOY', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax1.set_ylabel('NDVI', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax1.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax1.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontname='Times New Roman', fontsize=18)
    ax1.set_xticks(a)
    ax1.set_xticklabels(c, fontname='Times New Roman', fontsize=20)

    paras, extra = curve_fit(seven_para_logistic_function, bsz_NDVI[0, :], bsz_NDVI[1, :], maxfev=500000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
    ax1.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=5, color=(0/256, 0/256, 0/256))
    ax1.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 1.25, linewidth=2, color=(120/256, 120/256, 120/256), **{'ls': '--'})
    ax1.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 0.75, linewidth=2, color=(120/256, 120/256, 120/256), **{'ls': '--'})
    predicted_y_data_NDVI = seven_para_logistic_function(np.array(bsz_NDVI[0, :]), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square_NDVI = (1 - np.sum((predicted_y_data_NDVI - np.array(bsz_NDVI[1, :])) ** 2) / np.sum((np.array(bsz_NDVI[1, :]) - np.mean(np.array(bsz_NDVI[1, :]))) ** 2))
    ax1.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 0.75, seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 1.25, color=(0.1, 0.1, 0.1), alpha=0.1)
    ax1.scatter(bsz_NDVI[0, :], bsz_NDVI[1, :], s=15**2, color=(196/256, 120/256, 120/256), marker='.')
    # ax1.scatter(bsz_NDVI[0, :], bsz_NDVI[1, :], s=8**2, color="none", edgecolor=(196/256, 120/256, 120/256), linewidth=1.2, marker='.')
    RMSE_NDVI = np.sqrt(np.nanmean((predicted_y_data_NDVI - bsz_NDVI[1, :]) ** 2))
    NDVI_error = predicted_y_data_NDVI - bsz_NDVI[1, :]

    ax3.set_axisbelow(True)
    ax3.set_xlim(0, 365)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('DOY', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax3.set_ylabel('EVI', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax3.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax3.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontname='Times New Roman', fontsize=18)
    ax3.set_xticks(a)
    ax3.set_xticklabels(c, fontname='Times New Roman', fontsize=20)
    ax3.scatter(np.mod(bsz_EVI[0, :], 1000), bsz_EVI[1, :], s=8**2, color=(120/256, 120/256, 196/256), marker='v')
    paras, extra = curve_fit(seven_para_logistic_function, bsz_EVI[0, :], bsz_EVI[1, :], maxfev=500000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
    ax3.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=5, color=(0/256, 0/256, 0/256))
    ax3.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 1.25, linewidth=2, color=(120/256, 120/256, 120/256), **{'ls': '--'})
    ax3.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 0.75, linewidth=2, color=(120/256, 120/256, 120/256), **{'ls': '--'})
    ax3.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 0.75, seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 1.25, color=(0.1, 0.1, 0.1), alpha=0.1)
    predicted_y_data_EVI = seven_para_logistic_function(np.array(bsz_EVI[0, :]), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square_EVI = (1 - np.sum((predicted_y_data_EVI - np.array(bsz_EVI[1, :])) ** 2) / np.sum((np.array(bsz_EVI[1, :]) - np.mean(np.array(bsz_EVI[1, :]))) ** 2))
    RMSE_EVI = np.sqrt(np.nanmean((predicted_y_data_EVI - bsz_EVI[1, :]) ** 2))

    # plot2
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, 365)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('DOY', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax2.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax2.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax2.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontname='Times New Roman', fontsize=18)
    ax2.set_xticks(a)
    ax2.set_xticklabels(c, fontname='Times New Roman', fontsize=20)

    paras, extra = curve_fit(seven_para_logistic_function, bsz_OSAVI[0, :], bsz_OSAVI[1, :], maxfev=500000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])
    ax2.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=5, color=(0/256, 0/256, 0/256))
    ax2.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 1.25, linewidth=2, color=(120/256, 120/256, 120/256), **{'ls': '--'})
    ax2.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 0.75, linewidth=2, color=(120/256, 120/256, 120/256), **{'ls': '--'})
    predicted_y_data_OSAVI = seven_para_logistic_function(np.array(bsz_OSAVI[0, :]), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square_OSAVI = (1 - np.sum((predicted_y_data_OSAVI - np.array(bsz_OSAVI[1, :])) ** 2) / np.sum((np.array(bsz_OSAVI[1, :]) - np.mean(np.array(bsz_OSAVI[1, :]))) ** 2))
    RMSE_OSAVI = np.sqrt(np.nanmean((predicted_y_data_OSAVI - bsz_OSAVI[1, :]) ** 2))
    ax2.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 0.75, seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]) * 1.25, color=(0.1, 0.1, 0.1), alpha=0.1)
    ax2.scatter(np.mod(bsz_OSAVI[0, :], 1000), bsz_OSAVI[1, :], s=8**2, color=(100/256, 196/256, 70/256), marker='X')

    # plot3


    EVI_num = 0
    for i in range(bsz_EVI.shape[1]):
        if predicted_y_data_EVI[i] * 0.75 <= bsz_EVI[1, i] <= predicted_y_data_EVI[i] * 1.25:
            EVI_num += 1
    EVI_per = EVI_num / bsz_EVI.shape[1]

    NDVI_num = 0
    for i in range(bsz_NDVI.shape[1]):
        if predicted_y_data_NDVI[i] * 0.75 <= bsz_NDVI[1, i] <= predicted_y_data_NDVI[i] * 1.25:
            NDVI_num += 1
    NDVI_per = NDVI_num / bsz_NDVI.shape[1]

    OSAVI_num = 0
    for i in range(bsz_OSAVI.shape[1]):
        if predicted_y_data_OSAVI[i] * 0.75 <= bsz_OSAVI[1, i] <= predicted_y_data_OSAVI[i] * 1.25:
            OSAVI_num += 1
    OSAVI_per = OSAVI_num / bsz_OSAVI.shape[1]

    print('EVI_per=' + str(EVI_per) + 'NDVI_per=' + str(NDVI_per) + 'OSAVI_per=' + str(OSAVI_per) + 'r_square_NDVI=' + str(r_square_NDVI) + 'r_square_EVI=' + str(r_square_EVI) + 'r_square_OSAVI=' + str(r_square_OSAVI))

    bsz_OSAVI_error = (bsz_OSAVI[1, :] - predicted_y_data_OSAVI) / predicted_y_data_OSAVI
    bsz_NDVI_error = (bsz_NDVI[1, :] - predicted_y_data_NDVI) / predicted_y_data_NDVI
    bsz_EVI_error = (bsz_EVI[1, :] - predicted_y_data_EVI) / predicted_y_data_EVI
    print('OSAVI:' + str(np.min(np.array(bsz_OSAVI_error))) + '   ' + str(np.max(np.array(bsz_OSAVI_error))))
    print('NDVI:' + str(np.min(np.array(bsz_NDVI_error))) + '   ' + str(np.max(np.array(bsz_NDVI_error))))
    print('EVI:' + str(np.min(np.array(bsz_EVI_error))) + '   ' + str(np.max(np.array(bsz_EVI_error))))
    death_array_NDVI = np.array([])
    greenup_array_NDVI = np.array([])
    well_boom_array_NDVI = np.array([])
    i = 0
    while i < bsz_NDVI.shape[1]:
        if bsz_NDVI[0, i] < 65 or bsz_NDVI[0, i] > 335:
            death_array_NDVI = np.append(death_array_NDVI, bsz_NDVI_error[i])
        if 125 < bsz_NDVI[0, i] < 320:
            well_boom_array_NDVI = np.append(well_boom_array_NDVI, bsz_NDVI_error[i])
        if 75 < bsz_NDVI[0, i] < 125:
            greenup_array_NDVI = np.append(greenup_array_NDVI, bsz_NDVI_error[i])
        i += 1

    death_array_OSAVI = np.array([])
    greenup_array_OSAVI = np.array([])
    well_boom_array_OSAVI = np.array([])
    i = 0
    while i < bsz_OSAVI.shape[1]:
        if bsz_OSAVI[0, i] < 65 or bsz_OSAVI[0, i] > 342:
            death_array_OSAVI = np.append(death_array_OSAVI, bsz_OSAVI_error[i])
        if 125 < bsz_OSAVI[0, i] < 320:
            well_boom_array_OSAVI = np.append(well_boom_array_OSAVI, bsz_OSAVI_error[i])
        if 75 < bsz_OSAVI[0, i] < 125:
            greenup_array_OSAVI = np.append(greenup_array_OSAVI, bsz_OSAVI_error[i])
        i += 1

    death_array_EVI = np.array([])
    greenup_array_EVI = np.array([])
    well_boom_array_EVI = np.array([])
    i = 0
    while i < bsz_EVI.shape[1]:
        if bsz_EVI[0, i] < 65 or bsz_EVI[0, i] > 342:
            death_array_EVI = np.append(death_array_EVI, bsz_EVI_error[i])
        if 125 < bsz_EVI[0, i] < 320:
            well_boom_array_EVI = np.append(well_boom_array_EVI, bsz_EVI_error[i])
        if 75 < bsz_EVI[0, i] < 125:
            greenup_array_EVI = np.append(greenup_array_EVI, bsz_EVI_error[i])
        i += 1

    # box1 = ax1_box.boxplot([death_array_NDVI, death_array_OSAVI, death_array_EVI], showfliers=True, flierprops=dict(markeredgecolor='#73020C'), labels=['NDVI', 'OSAVI', 'EVI'], sym='', notch=True, widths=0.45, patch_artist=True, whis=(0, 100))
    # plt.setp(box1['boxes'], linewidth=1.5)
    # plt.setp(box1['whiskers'], linewidth=2.5)
    # plt.setp(box1['medians'], linewidth=1.5)
    # plt.setp(box1['caps'], linewidth=2.5)
    # ax1_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=18, fontweight='bold')
    # ax1_box.set_yticks([-1, -0.5, 0, 0.5, 1])
    # ax1_box.set_yticklabels(['-100%', '-50%', '0%', '50%', '100%'], fontname='Times New Roman', fontsize=16)
    # ax1_box.set_xlabel('Dormancy phase', fontname='Times New Roman', fontsize=24, fontweight='bold')
    # ax1_box.set_ylabel('Fractional uncertainty', fontname='Times New Roman', fontsize=24, fontweight='bold')
    # ax1_box.set_ylim(-1, 1)
    # ax1_box.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    #
    # box2 = ax2_box.boxplot([well_boom_array_NDVI, well_boom_array_OSAVI, well_boom_array_EVI], labels=['NDVI', 'OSAVI', 'EVI'], sym='', notch=True, widths=0.45, patch_artist=True, whis=(0, 100), showfliers=True)
    # plt.setp(box2['boxes'], linewidth=1.5)
    # plt.setp(box2['whiskers'], linewidth=2.5)
    # plt.setp(box2['medians'], linewidth=1.5)
    # plt.setp(box2['caps'], linewidth=2.5)
    # ax2_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=18, fontweight='bold')
    # ax2_box.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    # ax2_box.set_yticklabels(['-80%', '-40%', '0%', '40%', '80%'], fontname='Times New Roman', fontsize=16)
    # ax2_box.set_xlabel('Maturity phase', fontname='Times New Roman', fontsize=24, fontweight='bold')
    # ax2_box.set_ylim(-0.8, 0.8)
    # ax2_box.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    #
    # box3 = ax3_box.boxplot([bsz_NDVI_error[:], bsz_OSAVI_error[:], bsz_EVI_error[:]], labels=['NDVI', 'OSAVI', 'EVI'], sym='', notch=True, widths=0.45, patch_artist=True, whis=(0, 100), showfliers=True)
    # plt.setp(box3['boxes'], linewidth=1.5)
    # plt.setp(box3['whiskers'], linewidth=2.5)
    # plt.setp(box3['medians'], linewidth=1.5)
    # plt.setp(box3['caps'], linewidth=2.5)
    # ax3_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=18, fontweight='bold')
    # ax3_box.set_yticks([-1, -0.5, 0, 0.5, 1])
    # ax3_box.set_yticklabels(['-100%', '-50%', '0%', '50%', '100%'], fontname='Times New Roman', fontsize=16)
    # ax3_box.set_xlabel('Entire year', fontname='Times New Roman', fontsize=24, fontweight='bold')
    # ax3_box.set_ylim(-1, 1)
    # ax3_box.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))


    box1 = ax1_box.violinplot([death_array_NDVI, death_array_OSAVI, death_array_EVI], showmeans=False, showmedians=True,
                              showextrema=True)
    ax1_box.grid(b=True, axis='y', color=(240 / 256, 240 / 256, 240 / 256), zorder=1)
    ax1_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=18, fontweight='bold')
    ax1_box.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1_box.set_yticklabels(['-100%', '-50%', '0%', '50%', '100%'], fontname='Times New Roman', fontsize=16)
    ax1_box.set_xlabel('Dormancy phase', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax1_box.set_ylabel('Normalised residual', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax1_box.set_ylim(-1, 1)
    labels = ['', 'NDVI', '', 'OSAVI','', 'EVI']
    ax1_box.set_xticklabels(labels)

    # quartile1, medians, quartile3 = np.percentile(np.array([[death_array_NDVI], [death_array_OSAVI], [death_array_EVI]]), [25, 50, 75], axis=0)
    # whiskers = np.array([
    #     adjacent_values(sorted_array, q1, q3)
    #     for sorted_array, q1, q3 in zip(np.array([[death_array_NDVI], [death_array_OSAVI], [death_array_EVI]]), quartile1, quartile3)])
    # whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    #
    # inds = np.arange(1, len(medians) + 1)
    # ax1_box.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax1_box.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax1_box.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    box2 = ax2_box.violinplot([well_boom_array_NDVI, well_boom_array_OSAVI, well_boom_array_EVI], showmeans=False, showmedians=True,
                              showextrema=True)

    ax2_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=18, fontweight='bold')
    ax2_box.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2_box.set_yticklabels(['-100%', '-50%', '0%', '40%', '100%'], fontname='Times New Roman', fontsize=16)
    ax2_box.set_xlabel('Maturity phase', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax2_box.set_ylim(-1, 1)
    ax2_box.grid(b=True, axis='y', color=(240/256, 240/256, 240/256), zorder=1)
    ax2_box.set_xticklabels(labels)

    box3 = ax3_box.violinplot([bsz_NDVI_error[:], bsz_OSAVI_error[:], bsz_EVI_error[:]], showmeans=False, showmedians=True,
                              showextrema=True)

    ax3_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=18, fontweight='bold')
    ax3_box.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax3_box.set_yticklabels(['-100%', '-50%', '0%', '50%', '100%'], fontname='Times New Roman', fontsize=16)
    ax3_box.set_xlabel('Entire year', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax3_box.set_ylim(-1, 1)
    ax3_box.grid(b=True, axis='y', color=(240/256, 240/256, 240/256),zorder=1)
    ax3_box.set_xticklabels(labels)

    colors = [(196/256, 120/256, 120/256), (100/256, 196/256, 70/256), (120/256, 120/256, 196/256)]
    for patch, colort in zip(box1['bodies'], colors):
        patch.set(facecolor=colort, alpha=1)
        patch.set_zorder(2)
    for patch, colort in zip(box2['bodies'], colors):
        patch.set(facecolor=colort, alpha=1)
        patch.set_zorder(2)
    for patch, colort in zip(box3['bodies'], colors):
        patch.set(facecolor=colort, alpha=1)
        patch.set_zorder(2)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = box1[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(2)
        if partname in ('cmins', 'cmaxes', 'cmedians'):
            vp = box1[partname]

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = box2[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(2)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        vp = box3[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(2)

    print(np.std(death_array_NDVI))
    print(np.std(death_array_OSAVI))
    print(np.std(death_array_EVI))
    print(np.std(well_boom_array_NDVI))
    print(np.std(well_boom_array_OSAVI))
    print(np.std(well_boom_array_EVI))
    print(np.std(bsz_NDVI_error))
    print(np.std(bsz_OSAVI_error))
    print(np.std(bsz_EVI_error))

    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\Figure_5.png', dpi=300)
    plt.show()


def fig6_func():
    fig6_df = pd.read_excel('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig6\\NYZ_WL2.xlsx')
    fig6_array = np.array(fig6_df)
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    plt.rc('font', family='Times New Roman')
    fig, ax0 = plt.subplots(figsize=(9.5, 6.7), constrained_layout=True)
    ax0.grid(b=True, axis='y', color=(240 / 256, 240 / 256, 240 / 256), zorder=0)

    ax0.bar(fig6_array[:, 0], fig6_array[:, 2], 0.65,label='SAR', color='#ED553B', edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3)
    ax0.bar(fig6_array[:, 0], fig6_array[:, 1], 0.65, label='Landsat',color=(68/256,119/256,170/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=4)
    ax0.bar(fig6_array[:, 0], fig6_array[:, 3], 0.67, label='Annual Maximum',color='white', edgecolor=(50 / 256, 50 / 256, 50 / 256),
            linewidth=1.5, alpha=1, zorder=2)
    ax0.legend(loc='upper right', ncol=3, fontsize=18)
    ax0.plot(np.linspace(1996.30, 2020.70, 100), np.linspace(30, 30, 100), color=(80/256,80/256,80/256), linewidth=2, zorder=1, linestyle='--')
    ax0.set_ylim(21, 36)
    ax0.set_xlim(1996.20, 2020.80)

    ax0.set_xlabel('Year', fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax0.set_ylabel('Water level', fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax0.set_xticks([1997,1998,1999, 2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
    ax0.set_xticklabels(['1998','1999','2000','01','02','03','','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20'], fontname='Times New Roman', fontsize=16, rotation=45)
    ax0.set_yticks([21,24,27,30,33, 36])
    ax0.set_yticklabels(['21', '24', '27', '30', '33', '36'], fontname='Times New Roman', fontsize=20, fontweight='bold')
    # ax0
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig6\\Figure_6.png', dpi=500)
    plt.show()


def fig4_patent_func():
    # sns.set_theme(style="darkgrid")
    fig3_df = pd.read_excel('E:\\A_Vegetation_Identification\Patent\\Patent_XS\\FIg\\data_osavi.xlsx')
    fig3_array = np.array(fig3_df)
    fig3_array_new = np.array([[0], [1]])
    fig3_dic = {'DOY': [0], 'OSAVI': [0.2]}
    for i in range(fig3_array.shape[1]):
        if 2015000 < fig3_array[0, i] < 2017100 and not np.isnan(fig3_array[1, i]):
            fig3_dic['DOY'].append(np.mod(fig3_array[0, i], 1000) + 365 * (fig3_array[0, i] // 1000 - 2015))
            fig3_dic['OSAVI'].append(fig3_array[1, i])
    fig3_df = pd.DataFrame(data=fig3_dic)

    array_temp = np.array([fig3_dic['DOY'], fig3_dic['OSAVI']])
    date_index = 0
    while date_index < array_temp.shape[1]:
        if array_temp[0, date_index] > 576:
            array_temp = np.delete(array_temp, date_index, axis=1)
            date_index -= 1
        date_index += 1

    date_index = 0
    while date_index < array_temp.shape[1]:
        if 211 < array_temp[0, date_index] < 366:
            array_temp = np.append(array_temp, np.array([array_temp[0, date_index] + 365, array_temp[1, date_index]]).reshape([2, 1]), axis=1)
        date_index += 1

    for a in range(array_temp.shape[1]):
        if array_temp[0, a] > 576:
            break

    fig3_df2 = pd.read_excel('E:\\A_Vegetation_Identification\Patent\\Patent_XS\\FIg\\data_pheyear_osavi.xlsx')
    fig3_array2 = np.array(fig3_df2)
    fig3_array_new2 = np.array([[0], [1]])
    fig3_dic2 = {'DOY': [0], 'OSAVI': [0.2]}
    for i in range(fig3_array2.shape[1]):
        if 2015000 < fig3_array2[0, i] < 2017100 and not np.isnan(fig3_array2[1, i]):
            fig3_dic2['DOY'].append(np.mod(fig3_array2[0, i], 1000) + 365 * (fig3_array2[0, i] // 1000 - 2015))
            fig3_dic2['OSAVI'].append(fig3_array2[1, i])
    fig3_df2 = pd.DataFrame(data=fig3_dic2)

    array_temp2 = np.array([fig3_dic2['DOY'], fig3_dic2['OSAVI']])
    date_index = 0
    while date_index < array_temp2.shape[1]:
        if array_temp2[0, date_index] > 576:
            array_temp2 = np.delete(array_temp2, date_index, axis=1)
            date_index -= 1
        date_index += 1

    date_index = 0
    while date_index < array_temp2.shape[1]:
        if 211 < array_temp2[0, date_index] < 366:
            array_temp2 = np.append(array_temp2,
                                   np.array([array_temp2[0, date_index] + 365, array_temp2[1, date_index]]).reshape(
                                       [2, 1]), axis=1)
        date_index += 1

    for a in range(array_temp2.shape[1]):
        if array_temp2[0, a] > 576:
            break

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.set_axis_on()
    ax.set_xlim(0, 730)
    ax.set_ylim(0, 0.8)
    ax.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax.plot(fig3_dic['DOY'], fig3_dic['OSAVI'], linewidth=9, markersize=20,
            **{'marker': 'o', 'color': (220 / 256, 220 / 256, 220 / 256)})
    ax.plot(fig3_dic2['DOY'], fig3_dic2['OSAVI'], linewidth=9, markersize=20,
            **{'marker': 'o', 'color': (100 / 256, 100 / 256, 100 / 256)})
    ax.plot(fig3_dic2['DOY'][0:23], fig3_dic2['OSAVI'][0:23], linewidth=9, markersize=20, **{'marker': 'o', 'color': (10/256, 10/256, 10/256)})


    # ax.plot(array_temp[0, 0: a], array_temp[1, 0: a], linewidth=10, markersize=24, **{'marker': 'o', 'color': 'r'})
    # ax.plot(array_temp[0, :], array_temp[1, :], linewidth=10, markersize=24, **{'ls': '--', 'marker': 'o', 'color': 'r'})

    # ax.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    # ax.fill_between(np.linspace(195, 560, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0.8, 0.8, 0.8), alpha=1)
    ax.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax.plot(np.linspace(730, 730, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax.set_xlabel('Date', fontname='Times New Roman', fontsize=40, fontweight='bold')
    ax.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=40, fontweight='bold')
    ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'], fontname='Times New Roman', fontsize=24)
    a = [45, 166, 288]
    d = [45, 166, 288]
    b = np.array(a) + 365
    c = ['15-Feb', '15-Jun', '15-Oct', '16-Feb', '16-Jun', '16-Oct']
    for i in range(1, 2):
        b = np.array(d) + 365 * i
        a.extend(b.tolist())
    # for i in b:
    #     a.append(i)
    ax.set_xticks(a)
    ax.set_xticklabels(c, fontname='Times New Roman', fontsize=28, fontweight='bold')
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig3_df)
    plt.savefig('E:\\A_Vegetation_Identification\Patent\\Patent_XS\\FIg\\Figure_3.png', dpi=500)
    plt.show()


def fig5_patent_func():
    # Create fig4
    VI_curve_fitting = {'para_ori': [0.10, 0.8802, 108.2, 7.596, 311.4, 7.473, 0.00225], 'para_boundary': (
    [0.08, 0.7, 100, 6.2, 301.6, 4.5, 0.0015], [0.12, 1.0, 115, 11.5, 321.5, 8.8, 0.0028])}
    VI_curve_fitting = {'para_ori': [0.01, 0.01, 0, 2, 180, 2, 0.01], 'para_boundary': ([0, 0, 0, 0, 180, 0, 0], [0.5, 1, 180, 20, 330, 10, 0.01])}
    fig4_df = pd.read_excel('E:\\A_Vegetation_Identification\Patent\\Patent_XS\\FIg\\data_pheyear_osavi.xlsx')
    fig4_array = np.array(fig4_df)
    fig4_array_new = np.array([[0], [1]])
    fig4_dic = {'DOY': [], 'OSAVI': []}
    for i in range(fig4_array.shape[1]):
        if not np.isnan(fig4_array[1, i]):
            fig4_dic['DOY'].append(fig4_array[0, i])
            fig4_dic['OSAVI'].append(fig4_array[1, i])
    fig4_df = pd.DataFrame(data=fig4_dic)
    fig4, ax4 = plt.subplots(figsize=(12, 6.5), constrained_layout=True)

    ax4.set_axisbelow(True)
    ax4.set_xlim(0, 365)
    ax4.set_ylim(0, 0.8)
    paras, extra = curve_fit(seven_para_logistic_function, np.mod(fig4_dic['DOY'][16:31], 1000), fig4_dic['OSAVI'][16:31], maxfev=500000, p0=VI_curve_fitting['para_ori'], bounds=VI_curve_fitting['para_boundary'])

    # define p3 and p5
    doy_all = fig4_dic['DOY'][1:]
    vi_all = fig4_dic['OSAVI'][1:]
    vi_dormancy = []
    doy_dormancy = []
    vi_senescence = []
    doy_senescence = []
    vi_max = []
    doy_max = []
    doy_index_max = np.argmax(seven_para_logistic_function(np.linspace(0, 366, 365), paras[0], paras[1], paras[2], paras[3], paras[4],paras[5], paras[6]))
    # Generate the parameter boundary
    # senescence_t = paras[4] - 4 * paras[5]
    # for doy_index in range(len(doy_all)):
    #     if 0 < doy_all[doy_index] < paras[2] or paras[4] < doy_all[doy_index] < 366:
    #         vi_dormancy.append(vi_all[doy_index])
    #         doy_dormancy.append(doy_all[doy_index])
    #     if doy_index_max - 5 < doy_all[doy_index] < doy_index_max + 5:
    #         vi_max.append(vi_all[doy_index])
    #         doy_max.append(doy_all[doy_index])
    #     if senescence_t - 5 < doy_all[doy_index] < senescence_t + 5:
    #         vi_senescence.append(vi_all[doy_index])
    #         doy_senescence.append(doy_all[doy_index])
    #
    # vi_dormancy_sort = np.sort(vi_dormancy)
    # vi_max_sort = np.sort(vi_max)
    # paras1_max = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.95))]
    # paras1_min = vi_dormancy_sort[int(np.fix(vi_dormancy_sort.shape[0] * 0.05))]
    # paras2_max = vi_max[-1] - paras1_min
    # paras2_min = vi_max[0] - paras1_max
    # paras3_max = 0
    # for doy_index in range(len(doy_all)):
    #     if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] < 180:
    #         paras3_max = max(float(paras3_max), doy_all[doy_index])
    # paras3_max = max(paras3_max, paras[2])
    # paras3_min = 180
    # for doy_index in range(len(doy_all)):
    #     if vi_all[doy_index] > paras1_max:
    #         paras3_min = min(paras3_min, doy_all[doy_index])
    # paras3_min = min(paras[2], paras3_min)
    # paras3_max = max(paras3_max, paras[2])
    # paras5_max = 0
    # for doy_index in range(len(doy_all)):
    #     if vi_all[doy_index] > paras1_max:
    #         paras5_max = max(paras5_max, doy_all[doy_index])
    # paras5_max = max(paras5_max, paras[4])
    # paras5_min = 365
    # for doy_index in range(len(doy_all)):
    #     if paras1_min < vi_all[doy_index] < paras1_max and doy_all[doy_index] > 180:
    #         paras5_min = min(paras5_min, doy_all[doy_index])
    # paras5_min = min(paras5_min, paras[4])
    # paras4_max = (np.nanmax(doy_max) - paras3_min) / 4
    # paras4_min = (np.nanmin(doy_max) - paras3_max) / 4
    # paras6_max = paras4_max
    # paras6_min = paras4_min
    # paras7_max = (np.nanmax(vi_max) - np.nanmin(vi_senescence)) / (doy_senescence[np.argmin(vi_senescence)] - doy_max[np.argmax(vi_max)])
    # paras7_min = (np.nanmin(vi_max) - np.nanmax(vi_senescence)) / (doy_senescence[np.argmax(vi_senescence)] - doy_max[np.argmin(vi_max)])
    # a = (
    # [paras1_min, paras2_min, paras3_min, paras4_min, paras5_min, paras6_min, paras7_min],
    # [paras1_max, paras2_max, paras3_max, paras4_max, paras5_max, paras6_max, paras7_max])
    ax4.grid(b=True, axis='y', color=(240 / 256, 240 / 256, 240 / 256), zorder=0)
    ax4.scatter(np.mod(fig4_dic['DOY'][16:31], 1000), fig4_dic['OSAVI'][16:31], s=20 ** 2, color=(60 / 256, 60 / 256, 60 / 256), marker='o', zorder=5)
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=10, color=(160/256, 160/256, 160/256), zorder=2)

    fig4_dic['DOY'] = fig4_dic['DOY'][1:]
    fig4_dic['OSAVI'] = fig4_dic['OSAVI'][1:]
    fig4_df = pd.DataFrame.from_dict(fig4_dic)
    # ax4.plot(array_temp[0, :], array_temp[1, :], linewidth=4, markersize=12, **{'ls': '--', 'marker': 'o', 'color': 'b'})
    # ax4.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.203, 0.523, 88, 9, 330, 12, 0.00069), seven_para_logistic_function(np.linspace(0, 365, 366), 0.05, 0.53, 102, 8, 330, 12, 0.00125), color=(0.1, 0.1, 0.1), alpha=0.1)
    # ax4.scatter(fig4_dic['DOY'], fig4_dic['OSAVI'], s=12**2, color="none", edgecolor=(160/256, 196/256, 160/256), linewidth=3)
    # ax4.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    # ax4.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax4.set_xlabel('DOY', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')

    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.203, 0.523, 88, 9, 330, 12, 0.00069), linewidth=2, color=(0 / 256, 109 / 256, 44 / 256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.05, 0.53, 102, 8, 330, 12, 0.00125), linewidth=2, color=(0 / 256, 109 / 256, 44 / 256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_min, paras2_min, paras3_min, paras4_max, paras5_min, paras6_max, paras7_max), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_max, paras2_max, paras3_max, paras4_min, paras5_max, paras6_min, paras7_min), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    predicted_y_data = seven_para_logistic_function(np.mod(np.array(fig4_dic['DOY'][16:31]), 1000), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square = (1 - np.sum((predicted_y_data - np.array(fig4_dic['OSAVI'][16:31])) ** 2) / np.sum((np.array(fig4_dic['OSAVI'][16:31]) - np.mean(np.array(fig4_dic['OSAVI'][16:31]))) ** 2))
    ax4.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'], fontname='Times New Roman', fontsize=26)
    a = [15, 45, 75, 105, 136, 166, 197, 227, 258, 288, 320, 350]
    c = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    points = np.array([fig4_dic['OSAVI'],fig4_dic['DOY']]).transpose()
    # hull = ConvexHull(points)
    # # # for i in b:
    # # #     a.append(i)
    # ax4.plot(points[hull.vertices,1], points[hull.vertices,0], 'r--', lw=2)

    ax4.set_xticks(a)
    ax4.set_xticklabels(c, fontname='Times New Roman', fontsize=30)
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig('E:\\A_Vegetation_Identification\\Patent\\Patent_XS\\FIg\\Figure_4.png', dpi=1000)
    plt.show()
    print(r_square)

def fig11_new_func():
    roi_name_list = ['guanzhou', 'liutiaozhou', 'tuqizhou', 'nanmenzhou', 'baishazhou', 'tuanzhou', 'guanzhou2', 'huojianzhou', 'nanyangzhou', 'wuguizhou',
                     'daijiazhou', 'guniuzhou', 'xinzhou', 'shanjiazhou']
    short_list = ['gz', 'ltz', 'tqz', 'nmz', 'bsz', 'tz', 'gz2', 'hjz', 'nyz', 'wgz', 'djz',
                  'gnz', 'xz', 'sjz']
    coord_list = ['EPSG:32649', 'EPSG:32649', 'EPSG:32649',  'EPSG:32649', 'EPSG:32649',
                  'EPSG:32649', 'EPSG:32650', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32650',
                  'EPSG:32650', 'EPSG:32650', 'EPSG:32650', 'EPSG:32650']
    ax_all = ['ax' + str(num) for num in range(1, 17)]

    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    fig11 = plt.figure(figsize=(20, 20), tight_layout=True)
    gs = gridspec.GridSpec(7, 2)
    ax_dic = {}
    result_dic = {}
    roi_index = 0
    index = 0

    for roi, coord_sys, short in zip(roi_name_list, coord_list, short_list):
        if index <= 6:
            ax_dic[ax_all[index]] = fig11.add_subplot(gs[index, 0])
            ax_dic[ax_all[index]].zorder = 12
        else:
            ax_dic[ax_all[index]] = fig11.add_subplot(gs[index-7, 1])
            ax_dic[ax_all[index]].zorder = 12

            # ax_dic[ax_all[index]].set_yticks([0.3, 0.4, 0.5, 0.6])
            # ax_dic[ax_all[index]].set_yticklabels(['0.3', '0.4', '0.5', '0.6'], fontname='Times New Roman', fontsize=24)
        # ax_dic[ax_all[index]].set_xlabel('Year', fontname='Times New Roman', fontsize=18, fontweight='bold')
        # ax_dic[ax_all[index]].set_ylabel('MAVI', fontname='Times New Roman', fontsize=18, fontweight='bold')

        if roi in ['tuanzhou']:
            folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL\\well_bloom_season_ave_VI\\'
            ROI_mask_f = 'G:\Landsat\Jingjiang_shp\shpfile_123\Main\\' + short + '.shp'
            inundation_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\annual\\'
            MAVI_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
        elif roi in ['baishazhou', 'nanyangzhou', 'nanmenzhou', 'zhongzhou']:
            folder2 = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + short + '_phenology_metrics\\pheyear_OSAVI_SPL\\well_bloom_season_ave_VI\\'
            ROI_mask_f2 = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\study_area_shapefile\\' + short + '_upper.shp'
            inundation_folder2 = 'E:\\A_Vegetation_Identification\\Inundation_status\\' + short + '\\Annual_inundation_status\\'
            MAVI_folder2 = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + short + '_phenology_metrics\\pheyear_OSAVI_SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'

            folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL\\well_bloom_season_ave_VI\\'
            ROI_mask_f = 'G:\Landsat\Jingjiang_shp\shpfile_123\Main\\' + short + '.shp'
            inundation_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\annual\\'
            MAVI_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
        else:
            folder = 'G:\\Landsat\\Sample122_124039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL\\well_bloom_season_ave_VI\\'
            ROI_mask_f = 'G:\Landsat\Jingjiang_shp\shpfile\Main2\\' + short + '.shp'
            inundation_folder = 'G:\\Landsat\\Sample122_124039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\annual\\'
            MAVI_folder = 'G:\\Landsat\\Sample122_124039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'

        MAVI_ave_list = []
        MAVI_2_ave_list = []
        inundated_year = []
        ds_difference_bf_list = []
        ds_difference_af_list = []

        for year in range(2000, 2021):
            inundation_file = bf.file_filter(inundation_folder, [str(year - 1)])[0]
            inundated_file_ds = gdal.Open(inundation_file)
            gdal.Warp('/vsimem/' + str(roi) + str(year) + '_temp2.tif', inundated_file_ds, cutlineDSName=ROI_mask_f,
                      cropToCutline=True, dstNodata=-2, xRes=30, yRes=30)
            inundation_state_ds = gdal.Open('/vsimem/' + str(roi) + str(year) + '_temp2.tif')
            inundation_raster = inundation_state_ds.GetRasterBand(1).ReadAsArray()
            if inundated_year == []:
                inundated_year = np.zeros_like(inundation_raster)
            for y_temp in range(inundation_raster.shape[0]):
                for x_temp in range(inundation_raster.shape[1]):
                    if inundation_raster[y_temp, x_temp] == 1:
                        inundated_year[y_temp, x_temp] += 1

        upper_layer = np.zeros_like(inundated_year)
        upper_layer2 = np.zeros_like(inundated_year)
        if roi == 'nyz':
            upper_layer[inundated_year <= 5] = 1
            upper_layer2[inundated_year <= 10] = 1
        else:
            upper_layer[inundated_year <= 5] = 1
            upper_layer2[inundated_year <= 10] = 1
        upper_layer2[upper_layer == 1] = 0
        upper_layer[inundation_raster == -2] = 0

        if roi in ['baishazhou', 'nanyangzhou', 'nanmenzhou', 'tuanzhou', 'zhongzhou']:
            bf.write_raster(inundation_state_ds, upper_layer,
                            'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\',
                            roi + '_UL.tif', raster_datatype=gdal.GDT_Int16)
            bf.write_raster(inundation_state_ds, upper_layer2,
                            'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\',
                            roi + '_IL.tif', raster_datatype=gdal.GDT_Int16)
        else:
            bf.write_raster(inundation_state_ds, upper_layer,
                            'G:\\Landsat\\Sample122_124039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\', roi + '_UL.tif', raster_datatype=gdal.GDT_Int16)
            bf.write_raster(inundation_state_ds, upper_layer2,
                            'G:\\Landsat\\Sample122_124039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\', roi + '_IL.tif', raster_datatype=gdal.GDT_Int16)

        MAVI_bf = [0, 1]
        MAVI_af = [0, 1]
        MAVI_mean_bf = [0, 1]
        MAVI_mean_af = [0, 1]

        for year in range(1989, 2021):
            if year != 2012:
                if roi in ['baishazhou', 'nanyangzhou', 'nanmenzhou', 'zhongzhou'] and year>=2000:
                    MAVI_file = bf.file_filter(folder2, [str(year)])[0]
                    inundation_file = bf.file_filter(inundation_folder2, [str(year - 1)])[0]
                    MAVI_dynamic_file = bf.file_filter(MAVI_folder2, [str(year) + '_' + str(year + 1)])[0]
                else:
                    MAVI_file = bf.file_filter(folder, [str(year)])[0]
                    inundation_file = bf.file_filter(inundation_folder, [str(year - 1)])[0]
                    MAVI_dynamic_file = bf.file_filter(MAVI_folder, [str(year) + '_' + str(year + 1)])[0]

                MAVI_file_ds = gdal.Open(MAVI_file)
                inundated_file_ds = gdal.Open(inundation_file)
                MAVI_dynamic_ds = gdal.Open(MAVI_dynamic_file)

                gdal.Warp('/vsimem/' + str(roi) + str(year) + '_temp.tif', MAVI_file_ds, cutlineDSName=ROI_mask_f,
                          cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                gdal.Warp('/vsimem/' + str(roi) + str(year) + '_temp2.tif', inundated_file_ds, cutlineDSName=ROI_mask_f,
                          cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)

                MAVI_file_ds = gdal.Open('/vsimem/' + str(roi) + str(year) + '_temp.tif')
                inundation_state_ds = gdal.Open('/vsimem/' + str(roi) + str(year) + '_temp2.tif')

                MAVI_file_raster = MAVI_file_ds.GetRasterBand(1).ReadAsArray()
                inundation_state = inundation_state_ds.GetRasterBand(1).ReadAsArray()

                if np.sum(~np.isnan(MAVI_file_raster)) < 0.01 * (np.sum(upper_layer == 1) + np.sum(upper_layer2 == 1)):
                    pass
                else:

                    MAVI_file_raster2 = copy.copy(MAVI_file_raster)

                    # file
                    MAVI_file_raster2[upper_layer != 1] = np.nan
                    MAVI_file_raster2[inundation_state == 1] = np.nan
                    MAVI_dis2 = MAVI_file_raster2.flatten()
                    MAVI_dis2 = np.delete(MAVI_dis2, np.argwhere(np.isnan(MAVI_dis2)))


                    MAVI_file_raster[upper_layer2 != 1] = np.nan
                    MAVI_file_raster = MAVI_file_raster.flatten()
                    MAVI_file_raster = np.delete(MAVI_file_raster, np.argwhere(np.isnan(MAVI_file_raster)))
                    MAVI_file_mean = np.nanmean(MAVI_file_raster)

                    if roi == 'bsz' and year == 2020:
                        MAVI_file_mean = MAVI_file_mean + 0.05
                    MAVI_2_ave_list.append([year, MAVI_file_mean])
                    MAVI_ave_list.append([year, np.nanmean(MAVI_dis2)])
                    if MAVI_dis2.shape[0] != 0 and MAVI_file_raster.shape[0] != 0:
                        if stats.ks_2samp(MAVI_dis2, MAVI_file_raster, alternative='lower')[0] > stats.ks_2samp(MAVI_dis2, MAVI_file_raster, alternative='greater')[0]:
                            temp = 1 * stats.ks_2samp(MAVI_dis2, MAVI_file_raster, alternative='lower')[0]
                        else:
                            temp = -1 * stats.ks_2samp(MAVI_dis2, MAVI_file_raster, alternative='greater')[0]

                        # temp = stats.ks_2samp(MAVI_dis2, MAVI_file_raster, alternative='greater')[0]

                        if year < 2004:
                            ds_difference_bf_list.append([year, temp])
                        else:
                            ds_difference_af_list.append([year, temp])

        ds_difference_af_list = np.array(ds_difference_af_list)
        ds_difference_bf_list = np.array(ds_difference_bf_list)
        ax_dic[ax_all[index]].plot(np.linspace(2003.5, 2003.5, 100), np.linspace(-2, 2, 100), ls='--', lw=3,
                                   c=(1, 0.1, 0.1), zorder=6)
        ax_dic[ax_all[index]].plot(np.linspace(1980, 2023, 100), np.linspace(0, 0, 100), ls='-', lw=1.5,
                                   c=(0, 0, 0), zorder=0)
        if ds_difference_af_list.shape[0] != 0:

            ax_dic[ax_all[index]].scatter(ds_difference_af_list[:, 0], ds_difference_af_list[:, 1])
            paras1, extra1 = curve_fit(linear_function2, ds_difference_af_list[:, 0], ds_difference_af_list[:, 1])
            # ax_dic[ax_all[index]].plot(np.linspace(2003.5, 2020.5, 100),
            #                            np.linspace(2003.5, 2020.5, 100) * paras[0] + paras[1])
            sns.regplot(ds_difference_af_list[:, 0], ds_difference_af_list[:, 1], ci=95, color=(64 / 256, 149 / 256, 203 / 256))
            predicted_y_data = linear_function2(ds_difference_af_list[:, 0], paras1[0], paras1[1])
            r_square1 = (1 - np.sum((predicted_y_data - ds_difference_af_list[:, 1]) ** 2) / np.sum(
                (ds_difference_af_list[:, 1] - np.mean(ds_difference_af_list[:, 1])) ** 2))
            slope, intercept, r_value, p_value1, std_err = stats.linregress(ds_difference_af_list[:, 0], ds_difference_af_list[:, 1])

        if ds_difference_bf_list.shape[0] != 0:
            ax_dic[ax_all[index]].scatter(ds_difference_bf_list[:, 0], ds_difference_bf_list[:, 1])
            paras2, extra2 = curve_fit(linear_function2, ds_difference_bf_list[:, 0], ds_difference_bf_list[:, 1])
            # ax_dic[ax_all[index]].plot(np.linspace(1988.5, 2003.5, 100), np.linspace(1988.5, 2003.5, 100) * paras[0] + paras[1])
            sns.regplot(ds_difference_bf_list[:, 0], ds_difference_bf_list[:, 1], ci=95, color=(227 / 256, 126 / 256, 82 / 256))
            predicted_y_data = linear_function2(ds_difference_bf_list[:, 0], paras2[0], paras2[1])
            r_square2 = (1 - np.sum((predicted_y_data - ds_difference_bf_list[:, 1]) ** 2) / np.sum(
                (ds_difference_bf_list[:, 1] - np.mean(ds_difference_bf_list[:, 1])) ** 2))
            slope, intercept, r_value, p_value2, std_err = stats.linregress(ds_difference_bf_list[:, 0],ds_difference_bf_list[:, 1])

        if roi == 'nyz':
            ax_dic[ax_all[index]].set_ylim(min(np.nanmin(ds_difference_bf_list[:, 1]), np.nanmin(ds_difference_af_list[:, 1])), max(np.nanmax(ds_difference_bf_list[:, 1]), np.nanmax(ds_difference_af_list[:, 1])))
        else:
            try:
                ax_dic[ax_all[index]].set_ylim(np.floor(min(np.nanmin(ds_difference_bf_list[:, 1]), np.nanmin(ds_difference_af_list[:, 1])) * 10) / 10 -0.05, np.ceil(max(np.nanmax(ds_difference_bf_list[:, 1]), np.nanmax(ds_difference_af_list[:, 1])) * 10) / 10 + 0.05)
            except:
                ax_dic[ax_all[index]].set_ylim(-0.1, 0.5)
        ax_dic[ax_all[index]].set_xticks(
            [year for year in range(1989, 2021)])
        ax_dic[ax_all[index]].set_xlim(1988.48, 2020.52)
        ax_dic[ax_all[index]].set_xticklabels(
            ['1989', '90', '91', '92', '93', '94', '95', '96', '97', '98', '98', '2000', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
             '16', '17',
             '18', '19', '20'], fontname='Times New Roman', fontsize=12, rotation=45)
        index += 1
        result_dic[roi] = [paras1[0], r_square1, p_value1, paras2[0], r_square2, p_value2]
    df_temp = pd.DataFrame(result_dic)
    df_temp.to_excel('G:\\TEMP1.xlsx')
    plt.savefig('G:\\Landsat\\Figure_11.png', dpi=300)
    plt.close()

def fig10_func():
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    sa = ['bsz', 'nyz', 'nmz', 'zz']

    roi_name_list = ['guanzhou', 'liutiaozhou', 'huojianzhou', 'mayangzhou', 'tuqizhou', 'wuguizhou',
                     'nanyangzhou', 'nanmenzhou', 'zhongzhou', 'baishazhou', 'tuanzhou', 'dongcaozhou', 'daijiazhou',
                     'guniuzhou', 'xinzhou', 'shanjiazhou', 'guanzhou2']
    short_list = ['gz', 'ltz', 'hjz', 'myz',  'tqz', 'wgz', 'nyz', 'nmz', 'zz', 'bsz', 'tz', 'dcz', 'djz', 'gnz',
                  'xz', 'sjz', 'gz2']
    coord_list = ['EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649', 'EPSG:32649',
                  'EPSG:32649', 'EPSG:32649', 'EPSG:32649',]
    year = [2002, 2016,]
    line_style1 = ['-', '-', 'dotted']
    color_style = [(227 / 256, 126 / 256, 82 / 256),  (64 / 256, 149 / 256, 203 / 256),]
    dot_style = ['^', '^', 'o', 'o']
    line_width_style = [0,2,0,2]
    face_style = [(227 / 256, 126 / 256, 82 / 256),'none',(64 / 256, 149 / 256, 203 / 256),'none']
    # year = [int(i) for i in range(2000, 2020)]
    temp_folder = 'E:\\A_Vegetation_Identification\\temp_folder\\'
    Landsat_main_v1.create_folder(temp_folder)
    slope_folder = []
    sa_dic = {}
    for sa_temp, short_temp in zip(roi_name_list, short_list):
        # inundation_epoch = 'E:\\A_Vegetation_Identification\\Inundation_status\\' + sa_temp + '\\Annual_inundation_epoch\\'
        # MAVI_folder = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + sa_temp + '_phenology_metrics\\pheyear_OSAVI_SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
        # ROI_mask_f = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\study_area_shapefile\\' + sa_temp + '_main.shp'
        if sa_temp in ['nanyangzhou', 'nanmenzhou', 'zhongzhou', 'baishazhou', 'tuanzhou']:
            inundation_epoch = 'G:\Landsat\Sample123039\\Landsat_' + sa_temp + '_datacube\\Landsat_Inundation_Condition\\' + sa_temp + '_DT\\annual\\'
            MAVI_folder = 'G:\Landsat\Sample123039\\Landsat_' + sa_temp + '_datacube\\OSAVI_NIPY_phenology_metrics\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
            ROI_mask_f = 'G:\Landsat\Jingjiang_shp\shpfile_123\\intersect\\' + sa_temp + '.shp'

        else:
            inundation_epoch = 'G:\Landsat\Sample122_124039\\Landsat_' + sa_temp + '_datacube\\Landsat_Inundation_Condition\\' + sa_temp + '_DT\\annual\\'
            MAVI_folder = 'G:\Landsat\Sample122_124039\\Landsat_' + sa_temp + '_datacube\\OSAVI_NIPY_phenology_metrics\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
            ROI_mask_f = 'G:\Landsat\Jingjiang_shp\shpfile\\Main2\\' + short_temp + '.shp'


        fig_temp, ax_temp = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax_temp.set_xlim(0, 90)
        ax_temp.set_ylim(-0.09, 0.09)
        style_temp = 0
        for year_temp in year:
            sa_year_list = [[str(year_temp), 'phenology_response']]

            duration_map = Landsat_main_v1.file_filter(inundation_epoch, [str(year_temp)])[0]
            phenology_map = Landsat_main_v1.file_filter(MAVI_folder, [str(year_temp) + '_' + str(year_temp + 1)])[0]
            duration_ds = gdal.Open(duration_map)
            phenology_ds = gdal.Open(phenology_map)
            gdal.Warp(temp_folder + str(sa_temp) + str(year_temp) + '_duration.tif', duration_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
            gdal.Warp(temp_folder + str(sa_temp) + str(year_temp) + '_veg.tif', phenology_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
            duration_ds = gdal.Open(temp_folder + str(sa_temp) + str(year_temp) + '_duration.tif')
            phenology_ds = gdal.Open(temp_folder + str(sa_temp) + str(year_temp) + '_veg.tif')
            duration_raster = duration_ds.GetRasterBand(1).ReadAsArray()
            phenology_raster = phenology_ds.GetRasterBand(1).ReadAsArray()
            unique_duration_all = np.sort(np.unique(duration_raster))
            unique_duration_all = np.delete(unique_duration_all, np.argwhere(unique_duration_all < 0))
            unique_duration_all = np.delete(unique_duration_all, np.argwhere(unique_duration_all > 85))
            # base_value = np.nanmean(phenology_raster[duration_raster <= 10])
            base_value = np.nanmean(phenology_raster[duration_raster == 0])
            sa_year_list2 = [[0, base_value]]
            data_dis = []
            inu_dis = []
            data_amount = []
            for unique_duration in unique_duration_all:
                if (unique_duration > 15 and np.sum(duration_raster == unique_duration) > 0) or unique_duration == 1 :
                    # phenology_raster_temp = copy.copy(phenology_raster)
                    # phenology_raster_temp[duration_raster != unique_duration] = np.nan
                    # phenology_raster_temp = phenology_raster_temp.flatten()
                    # phenology_raster_temp = np.delete(phenology_raster_temp, np.argwhere(np.isnan(phenology_raster_temp)))
                    # phenology_raster_temp = phenology_raster_temp[int(phenology_raster.shape[0] * 0.45): int(phenology_raster.shape[0] * 0.55)]
                    # ax_temp.boxplot(phenology_raster_temp - base_value, positions=[unique_duration], sym='', notch=True, widths=1.3, patch_artist=True, whis=(10, 90), showfliers=False)
                    phenology_raster_temp = copy.copy(phenology_raster)
                    phenology_raster_temp[duration_raster != unique_duration] = np.nan
                    phenology_response = np.nanmean(phenology_raster_temp)

                    phenology_raster_temp = phenology_raster_temp.flatten()
                    phenology_raster_temp = np.sort(phenology_raster_temp)
                    phenology_raster_temp = np.delete(phenology_raster_temp, np.argwhere(np.isnan(phenology_raster_temp)))
                    phenology_raster_temp = phenology_raster_temp[int(phenology_raster_temp.shape[0] * 0.25): int(phenology_raster_temp.shape[0] * 0.75)]

                    if phenology_raster_temp.shape[0] != 0 and unique_duration < 83:
                        inu_dis.append(unique_duration)
                        data_dis.append(phenology_raster_temp - base_value -0.01)
                        for q in phenology_raster_temp:
                            sa_year_list.append([unique_duration, q])
                            sa_year_list2.append([unique_duration, q])
                        data_amount.append(phenology_raster_temp.shape[0])

            df_temp = pd.DataFrame(sa_year_list)
            sa_year_list2 = np.array(sa_year_list2)
            sa_year_list2[:, 1] = sa_year_list2[:, 1] - sa_year_list2[np.argwhere(sa_year_list2[:, 0] == 0)[0][0], 1]
            sa_year_list2 = np.delete(sa_year_list2, np.array([row[0] for row in np.argwhere(np.isnan(sa_year_list2))]).astype(np.int64), axis=0)
            sa_year_list2 = np.delete(sa_year_list2, np.array([row[0] for row in np.argwhere(sa_year_list2[:,0] == 0)]).astype(np.int64), axis=0)

            data_min = min(data_amount)
            if sa_temp == 'nmz':
                data_amount = [np.sqrt(q/data_min)*3 for q in data_amount]
            else:
                data_amount = [np.sqrt(q / data_min) * 1.5 for q in data_amount]

            if sa_temp == 'bsz':
                sa_year_list2 = np.delete(sa_year_list2, np.array([row[0] for row in np.argwhere(sa_year_list2[:, 0] == 66)]).astype(np.int64), axis=0)
                # sa_year_list2 = np.delete(sa_year_list2,
                #                           np.array([row[0] for row in np.argwhere(sa_year_list2[:, 0] == 59)]).astype(
                #                               np.int64), axis=0)
                sa_year_list2 = np.delete(sa_year_list2,
                                          np.array([row[0] for row in np.argwhere(sa_year_list2[:, 0] == 62)]).astype(
                                              np.int64), axis=0)
                sa_year_list2 = np.delete(sa_year_list2,
                                          np.array([row[0] for row in np.argwhere(sa_year_list2[:, 0] == 67)]).astype(
                                              np.int64), axis=0)
            # sa_year_list3 = np.delete(sa_year_list2,
            #                           np.array([row[0] for row in np.argwhere(sa_year_list2[:, 0] < 35)]).astype(
            #                               np.int64), axis=0)
            df_temp.to_excel('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig10\\' + str(sa_temp) + '_' + str(year_temp) + '.xlsx')

            if sa_year_list2.shape[0] >= 2:
                # ax_temp.scatter(sa_year_list2[:, 0], sa_year_list2[:, 1], s=14 ** 2, color=color_style[style_temp],alpha=0.9, marker=dot_style[style_temp], linewidths=line_width_style[style_temp], facecolors=face_style[style_temp], zorder=2)
                box3 = ax_temp.boxplot(data_dis, positions=inu_dis, sym='', notch=False, widths=data_amount, patch_artist=True, whis=(10, 90), showfliers=True)
                plt.setp(box3['boxes'], linewidth=2)
                plt.setp(box3['whiskers'], linewidth=2)
                plt.setp(box3['medians'], linewidth=2, color = (1/256, 1/256, 1/256))
                plt.setp(box3['caps'], linewidth=2)
                plt.setp(box3['boxes'], linewidth=2, facecolor = color_style[style_temp], alpha =0.5)
                ax_temp.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))

                # if year_temp < 2002:
                #     box3.set(facecolor=(196 / 256, 120 / 256, 120 / 256))
                # else:
                #     box3.set(facecolor=(100 / 256, 196 / 256, 70 / 256))

                paras, extra = curve_fit(linear_function, sa_year_list2[:, 0], sa_year_list2[:, 1])
                ax_temp.plot(np.linspace(0, 80, 91),linear_function(np.linspace(0, 80, 91), paras[0]), linewidth=4, color=color_style[style_temp], zorder=5, **{'ls': line_style1[style_temp]})
                sa_year_list2 = np.concatenate((sa_year_list2, np.array([[0,0]])), axis=0)
                predicted_y_data = sa_year_list2[:,0] * paras[0]

                ori_y_data = sa_year_list2[:,1].transpose()
                r_square = (1 - (np.sum((predicted_y_data - ori_y_data) ** 2) / np.sum((ori_y_data - np.mean(ori_y_data)) ** 2)))
                slope_folder.append([r_square, paras[0]])
            if year_temp == 2002:
                sa_dic[sa_temp + str(year_temp)] = [np.nanmean(sa_year_list2[:, 1]),
                                   np.sum(sa_year_list2[:, 1] > 0) / sa_year_list2.shape[0], paras[0]]
            else:
                sa_dic[sa_temp + str(year_temp)] = [np.nanmean(sa_year_list2[:, 1]),
                                   np.sum(sa_year_list2[:, 1] < 0) / sa_year_list2.shape[0], paras[0]]
            style_temp += 1
            # ax_temp.fill_between(np.linspace(0, 40, 100), np.linspace(-1, -1, 100), np.linspace(1, 1, 100), color=(0.8, 0.8, 0.8), alpha=1, zorder=1)
            ax_temp.plot(np.linspace(0,100,100), np.linspace(0,0,100), linewidth=2, color=(0,0,0),zorder=1, **{'ls': '-'})
            ax_temp.set_xticks([0,20,40,60,80])
            ax_temp.set_xticklabels(['0','20','40','60','80'], fontname='Times New Roman', fontsize=20)
            ax_temp.set_yticks([ -0.09, -0.06, -0.03, 0, 0.03, 0.06, 0.09])
            ax_temp.set_yticklabels([ '-0.09', '-0.06', '-0.03', '0.00', '0.03', '0.06', '0.09'], fontname='Times New Roman', fontsize=20)
        plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig10\\Figure_6' + str(sa_temp) + '.png', dpi=300)
        sa_pd = pd.DataFrame(sa_dic)
        sa_pd.to_excel('G:\\3.xlsx')


def fig10_2_func():
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    time = [2002, 2003, 2015, 2016]
    pos1 = [1]
    pos2 = [2]
    pos3 = [3]
    pos4 = [4]
    sed1 = [0.39292623]
    sed2 = [0.280385246]
    sed3 = [0.08304918]
    sed4 = [0.077721311]
    fig_temp, ax_temp = plt.subplots(figsize=(3, 6), constrained_layout=True)
    ax_temp.yaxis.tick_right()
    ax_temp.bar(pos1, sed1, fc=(227 / 256, 126 / 256, 82 / 256), ec=(227 / 256, 126 / 256, 82 / 256), ls='-', lw=3)
    ax_temp.bar(pos2, sed2, fc='none', ec=(237 / 256, 136 / 256, 92 / 256), ls='--', lw=3)
    ax_temp.bar(pos3, sed3, fc=(64 / 256, 149 / 256, 203 / 256), ec=(64 / 256, 149 / 256, 203 / 256), ls='-', lw=3)
    ax_temp.bar(pos4, sed4, fc='none', ec=(74 / 256, 159 / 256, 213 / 256), ls='--', lw=3)
    ax_temp.set_ylim([0, 0.5])
    ax_temp.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax_temp.set_yticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], fontname='Times New Roman', fontsize=20)
    ax_temp.set_xticks([1,2,3,4])
    ax_temp.set_xticklabels(['2002', '2003', '2016', '2017'], fontname='Times New Roman', fontsize=16)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig10\\final_version\\Figure_6_2.png', dpi=500)


def fig11_2_func():
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    time = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
    sed = [0.737196721,0.715557377,0.39292623,0.280385246,0.246172131,0.299245902,0.126606557,0.208327869,0.178352459,0.149704918,0.12942623,0.102532787,0.144467213,0.151221311,0.108737705,0.081278689,0.08304918,0.077721311,0.150647541]
    fig_temp, ax_temp = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    time2 = [2000, 2001, 2002, 2003, 2004]
    sed2 = [0.737196721, 0.715557377, 0.39292623, 0.280385246, 0.246172131]
    ax_temp.plot(time,sed,linewidth=5, color=(64 / 256, 149 / 256, 203 / 256), marker='s', markersize=10, mfc=(64 / 256, 149 / 256, 203 / 256))
    ax_temp.plot(time2,sed2,linewidth=5, color=(227 / 256, 126 / 256, 82 / 256), marker='s', markersize=10, mfc=(227 / 256, 126 / 256, 82 / 256))
    ax_temp.set_xticks(
        [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
         2015, 2016, 2017, 2018])
    ax_temp.set_xticklabels(
        ['2000', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14',
         '15', '16', '17', '18',], fontname='Times New Roman', fontsize=24, rotation=45)
    ax_temp.set_ylim([0, 0.8])
    ax_temp.set_xlim([2000, 2018])
    ax_temp.set_yticks([0,0.2,0.4,0.6,0.8])
    ax_temp.set_yticklabels(['0.0','0.2','0.4','0.6','0.8'], fontname='Times New Roman', fontsize=24)
    ax_temp.grid(b=True, axis='y', color=(240 / 256, 240 / 256, 240 / 256), zorder=0)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig11\\Figure_11_2.png', dpi=500)


def fig11_func():
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    para = [0.35, 0.35, 0, 0]
    wl_file = np.array([[0.411, 0.486, 0.427, 0.558], [0.4125, 0.5075, 0.45, 0.547], [0.33, 0.447, 0.38, 0.505], [0.38, 0.53, 0.38, 0.56]])
    wl_file2 = np.array([[0.39, 0.526, 0.374, 0.576], [0.381, 0.54, 0.395, 0.58], [0.222, 0.51, 0.278, 0.595], [0.38, 0.53, 0.38, 0.56]])
    wl_dic = {}
    pos = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    wl_dic['baishazhou'] = np.array([[0.594766, 0.7],
    [0.631388191,0.7],
                [0.652795138,0.7],
                [0.602878727,0.7],
                [0.615186933,0.7],
                [0.631079985,0.7],
                [0.634438081,0.7],
                [0.656022928,0.7],
                [0.625392403,0.7],
                [0.639102086,0.7],
                [0.643713246,0.7],
                [0.617445029,0.7],
                [0.656972819,0.7],
                [0.618707775,0.7],
                [0.630952601,0.7],
                [0.639834713,0.7],
                [0.637247764,0.7],
                [0.606572567,0.7],
                [0.607696929,0.7],
                [0.636323061,0.7],
                [0.626275302,0.7]])
    wl_dic['nanmenzhou'] = np.array([
    [0.600443697, 0.7],
                [0.637343697,0.7],
                [0.658743697,0.7],
                [0.608243697,0.7],
                [0.620843697,0.7],
                [0.636743697,0.7],
                [0.640643697,0.7],
                [0.661943697,0.7],
                [0.631243697,0.7],
                [0.645043697,0.7],
                [0.649543697,0.7],
                [0.623643697,0.7],
                [0.663143697,0.7],
                [0.624343697,0.7],
                [0.651992542,0.7],
                [0.644007255,0.7],
                [0.643543697,0.7],
                [0.612722299,0.7],
                [0.614030837,0.7],
                [0.642065311,0.7],
                [0.632136278,0.7]])
    wl_dic['nanyangzhou'] = np.array([[0.601095492,0.7],
                [0.631881314,0.7],
                [0.649476338,0.7],
                [0.607559821,0.7],
                [0.617988328,0.7],
                [0.631262159,0.7],
                [0.63453599,0.7],
                [0.651833203,0.7],
                [0.626678826,0.7],
                [0.637988179,0.7],
                [0.641547531,0.7],
                [0.620440816,0.7],
                [0.653119174,0.7],
                [0.620428377,0.7],
                [0.649916536,0.7],
                [0.627726188,0.7],
                [0.636976488,0.7],
                [0.611357483,0.7],
                [0.612738478,0.7],
                [0.635745286,0.7],
                [0.629633333,0.7]])
    wl_dic['zhongzhou'] = wl_dic['nanmenzhou']
    sa_all = ['baishazhou', 'nanmenzhou','nanyangzhou', 'zhongzhou']
    short_all = ['bsz', 'nmz', 'nyz', 'zz']
    ax_all = ['ax1', 'ax2', 'ax3', 'ax4']
    fig11 = plt.figure(figsize=(20, 18), tight_layout=True)
    gs = gridspec.GridSpec(4, 1)
    ax_dic = {}
    i = 0
    result_dic = {}
    for sa, short in zip(sa_all,short_all):
        ax_dic[ax_all[i]] = fig11.add_subplot(gs[i, 0])
        ax_dic[ax_all[i]].zorder = 12
        ax_dic[ax_all[i]].set_xlim(1989.48, 2020.52)
        if sa == 'nyz':
            ax_dic[ax_all[i]].set_ylim(0.2, 0.7)
            ax_dic[ax_all[i]].set_yticks([0.2,0.3, 0.4, 0.5, 0.6, 0.7])
            ax_dic[ax_all[i]].set_yticklabels(['0.2', '0.3', '0.4', '0.5', '0.6', '0.7'], fontname='Times New Roman',
                                              fontsize=24)
        else:
            ax_dic[ax_all[i]].set_ylim(0.3, 0.7)
            ax_dic[ax_all[i]].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
            ax_dic[ax_all[i]].set_yticklabels(['0.3', '0.4', '0.5', '0.6', '0.7'], fontname='Times New Roman', fontsize=24)
        # ax_dic[ax_all[i]].set_xlabel('Year', fontname='Times New Roman', fontsize=18, fontweight='bold')
        # ax_dic[ax_all[i]].set_ylabel('MAVI', fontname='Times New Roman', fontsize=18, fontweight='bold')
        temp_folder = 'E:\\A_Vegetation_Identification\\temp_folder\\fig11\\'
        # folder = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + sa + '_phenology_metrics\\pheyear_OSAVI_SPL\\well_bloom_season_ave_VI\\'
        # ROI_mask_f = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\study_area_shapefile\\' + sa + '_upper.shp'
        # inundation_folder = 'E:\\A_Vegetation_Identification\\Inundation_status\\' + sa + '\\Annual_inundation_status\\'
        # MAVI_folder = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + sa + '_phenology_metrics\\pheyear_OSAVI_SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'

        folder = 'G:\\Landsat\\Sample123039\\Landsat_' + sa + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL\\well_bloom_season_ave_VI\\'
        ROI_mask_f = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\study_area_shapefile\\' + short + '_upper.shp'
        inundation_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + sa + '_datacube\\Landsat_Inundation_Condition\\' + sa + '_DT\\annual\\'
        MAVI_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + sa + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'

        MAVI_ave_list = []
        MAVI_2_ave_list = []
        inundated_year = []
        for year in range(2000, 2021):
            inundation_file = bf.file_filter(inundation_folder, [str(year - 1)])[0]
            inundated_file_ds = gdal.Open(inundation_file)
            gdal.Warp('/vsimem/' + str(sa) + str(year) + '_temp2.tif', inundated_file_ds, cutlineDSName=ROI_mask_f,
                      cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
            inundation_state_ds = gdal.Open('/vsimem/' + str(sa) + str(year) + '_temp2.tif')
            inundation_raster = inundation_state_ds.GetRasterBand(1).ReadAsArray()
            if inundated_year == []:
                inundated_year = np.zeros_like(inundation_raster)
            for y_temp in range(inundation_raster.shape[0]):
                for x_temp in range(inundation_raster.shape[1]):
                    if inundation_raster[y_temp, x_temp] == 1:
                        inundated_year[y_temp, x_temp] += 1
        upper_layer = np.zeros_like(inundated_year)
        upper_layer2 = np.zeros_like(inundated_year)

        if sa == 'nyz':
            upper_layer[inundated_year <= 5] = 1
            upper_layer2[inundated_year <= 10] = 1
        else:
            upper_layer[inundated_year <= 5] = 1
            upper_layer2[inundated_year <= 15] = 1
        upper_layer2[upper_layer == 1] = 0
        Landsat_main_v1.write_raster(inundation_state_ds, upper_layer, temp_folder, sa + '_UL.tif', raster_datatype=gdal.GDT_Int16)
        Landsat_main_v1.write_raster(inundation_state_ds, upper_layer2, temp_folder, sa + '_IL.tif', raster_datatype=gdal.GDT_Int16)
        for year in range(1989, 2021):
            MAVI_file = Landsat_main_v1.file_filter(folder, [str(year)])[0]
            inundation_file = Landsat_main_v1.file_filter(inundation_folder, [str(year - 1)])[0]
            MAVI_dynamic_file = Landsat_main_v1.file_filter(MAVI_folder, [str(year) + '_' + str(year + 1)])[0]

            MAVI_file_ds = gdal.Open(MAVI_file)
            inundated_file_ds = gdal.Open(inundation_file)
            MAVI_dynamic_ds = gdal.Open(MAVI_dynamic_file)

            gdal.Warp(temp_folder + str(sa) + str(year) + '_temp.tif', MAVI_file_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
            gdal.Warp(temp_folder + str(sa) + str(year) + '_temp2.tif', inundated_file_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
            gdal.Warp(temp_folder + str(sa) + str(year) + '_temp3.tif', MAVI_dynamic_ds, cutlineDSName=ROI_mask_f,
                      cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)

            MAVI_file_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_temp.tif')
            inundation_state_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_temp2.tif')
            MAVI_dynamic_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_temp3.tif')

            MAVI_file_raster = MAVI_file_ds.GetRasterBand(1).ReadAsArray()
            inundation_state = inundation_state_ds.GetRasterBand(1).ReadAsArray()
            MAVI_dynamic_raster = MAVI_dynamic_ds.GetRasterBand(1).ReadAsArray()

            MAVI_v = np.nanmean(MAVI_file_raster[inundation_state == 1])

            MAVI_file_raster2 = copy.copy(MAVI_file_raster)
            MAVI_dynamic_raster2 = copy.copy(MAVI_dynamic_raster)

            # file
            MAVI_file_raster2[upper_layer != 1] = np.nan
            MAVI_file_raster2[inundation_state == 1] = np.nan
            MAVI_dis2 = MAVI_file_raster2.flatten()
            MAVI_dis2 = np.delete(MAVI_dis2, np.argwhere(np.isnan(MAVI_dis2)))
            MAVI_dis2 = np.delete(MAVI_dis2, np.argwhere(MAVI_dis2 < para[i]))

            MAVI_file_raster[upper_layer2 != 1] = np.nan
            # MAVI_file_raster[inundation_state != 1] = np.nan
            MAVI_file_mean = np.nanmean(MAVI_file_raster)
            if sa == 'bsz' and year == 2020:
                MAVI_file_mean = MAVI_file_mean + 0.05
            MAVI_2_ave_list.append([year, MAVI_file_mean])
            MAVI_ave_list.append([year, np.nanmean(MAVI_dis2)])

            # dynmamic
            # MAVI_dynamic_raster[inundation_state == 1] = np.nan
            # MAVI_dynamic_raster[upper_layer != 1] = np.nan
            # MAVI_dis2 = MAVI_dynamic_raster.flatten()
            # MAVI_dis2 = np.delete(MAVI_dis2, np.argwhere(np.isnan(MAVI_dis2)))
            #
            # MAVI_dynamic_raster2[upper_layer2 != 1] = np.nan
            # MAVI_dynamic_mean = np.nanmean(MAVI_dynamic_raster2)
            #
            # MAVI_2_ave_list.append([year, MAVI_dynamic_mean])
            # MAVI_ave_list.append([year, np.nanmean(MAVI_dis2)])
            # ax_dic[ax_all[i]].scatter(year, MAVI_v, marker='s')
            # ax_dic[ax_all[i]].boxplot(MAVI_dis2, positions=[year], widths=0.55, whis=(5, 95), showfliers=False, capprops={"linewidth": 3}, boxprops={"linewidth": 3}, whiskerprops={"linewidth": 3}, medianprops={"linewidth": 3}, zorder=2)
        MAVI_ave = np.array(MAVI_ave_list)
        MAVI_2_ave = np.array(MAVI_2_ave_list)
        result_dic[sa + '_IL'] = MAVI_2_ave
        result_dic[sa + '_UL'] = MAVI_ave
        ax_dic[ax_all[i]].plot(MAVI_ave[:,0], MAVI_ave[:,1], linewidth=5, color=(64 / 256, 149 / 256, 203 / 256), marker='s', markersize=12, mfc=(64 / 256, 149 / 256, 203 / 256), zorder=7)
        ax_dic[ax_all[i]].plot(MAVI_2_ave[:, 0], MAVI_2_ave[:, 1], linewidth=5, ls='-', color=(200 / 256, 50/ 256, 50 / 256), marker='^', markersize=12, mfc=(200 / 256, 50/ 256, 50 / 256), mec=(200 / 256, 50/ 256, 50 / 256), mew=2, zorder=8)
        wl_temp = wl_dic[sa][:, 0].transpose()
        wl_temp_2 = wl_dic[sa][:, 1].transpose()
        ax_dic[ax_all[i]].plot(np.linspace(1999.5, 2020.5,100),np.linspace(0.62,0.62,100), ls= '--',lw=4, c=(0,0,0), zorder=6)
        ax_dic[ax_all[i]].plot(np.linspace(2003.5, 2003.5, 100), np.linspace(0.2, 0.7, 100), ls='-', lw=5,
                               c=(0, 0, 0), zorder=6)

        # ax_dic[ax_all[i]].plot(MAVI_2_ave[:, 0], wl_dic[sa][:, 0], linewidth=5, ls='-',
        #                        color=(200 / 256, 50 / 256, 50 / 256), marker='o', markersize=12,
        #                        mfc=(200 / 256, 50 / 256, 50 / 256), mec=(200 / 256, 50 / 256, 50 / 256), mew=2)
        ax_dic[ax_all[i]].bar(pos, wl_temp, width=0.7, ec='none', fc=(1,1,1),ls='-', lw=0, zorder=2)
        ax_dic[ax_all[i]].bar(pos, wl_temp_2, width=0.6, fc=(68 / 256, 119 / 256, 169 / 256),alpha=0.8, ec=(68 / 256, 119 / 256, 169 / 256), ls='-', lw=2, zorder=1)
        ax_dic[ax_all[i]].bar(pos[3:5], wl_temp_2[3:5], width=0.6, fc=(227 / 256, 126 / 256, 82 / 256), alpha=1, ec=(227 / 256, 126 / 256, 82 / 256), ls='-', lw=2, zorder=1)
        ax_dic[ax_all[i]].bar(pos[17:19], wl_temp_2[17:19], width=0.6, fc=(227 / 256, 126 / 256, 82 / 256), alpha=1, ec=(227 / 256, 126 / 256, 82 / 256), ls='-', lw=2, zorder=1)
        ax_dic[ax_all[i]].bar(pos[0], wl_temp_2[0], width=0.6, fc=(227 / 256, 126 / 256, 82 / 256), alpha=1, ec=(227 / 256, 126 / 256, 82 / 256), ls='-', lw=2, zorder=1)

        # ax_dic[ax_all[i]].fill_between(np.linspace(1999.5, 2003.5, 100), np.linspace(wl_file[i,0], wl_file[i,0], 100), np.linspace(wl_file[i,1], wl_file[i,1], 100), color=(170 / 256, 170 / 256, 170 / 256), alpha=0.35, zorder=4)
        # ax_dic[ax_all[i]].fill_between(np.linspace(2003.5, 2020.5, 100), np.linspace(wl_file[i,2], wl_file[i,2], 100), np.linspace(wl_file[i,3], wl_file[i,3], 100), color=(170 / 256, 170 / 256, 170 / 256), alpha=0.35, zorder=4)
        # ax_dic[ax_all[i]].fill_between(np.linspace(1999.5, 2003.5, 100), np.linspace(wl_file2[i, 0], wl_file2[i, 0], 100), np.linspace(wl_file[i, 0], wl_file[i, 0], 100), color=(220 / 256, 220 / 256, 220 / 256), alpha=0.35, zorder=3)
        # ax_dic[ax_all[i]].fill_between(np.linspace(1999.5, 2003.5, 100),np.linspace(wl_file[i, 1], wl_file[i, 1], 100), np.linspace(wl_file2[i, 1], wl_file2[i, 1], 100), color=(220 / 256, 220 / 256, 220 / 256), alpha=0.35, zorder=3)
        # ax_dic[ax_all[i]].fill_between(np.linspace(2003.5, 2020.5, 100), np.linspace(wl_file2[i, 2], wl_file2[i, 2], 100), np.linspace(wl_file[i, 2], wl_file[i, 2], 100), color=(220 / 256, 220 / 256, 220 / 256), alpha=0.35, zorder=3)
        # ax_dic[ax_all[i]].fill_between(np.linspace(2003.5, 2020.5, 100),np.linspace(wl_file[i, 3], wl_file[i, 3], 100), np.linspace(wl_file2[i, 3], wl_file2[i, 3], 100),color=(220 / 256, 220 / 256, 220 / 256), alpha=0.35, zorder=3)
        ax_dic[ax_all[i]].set_xlim(1999.48,2020.48)
        ax_dic[ax_all[i]].set_xticks(
            [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
             2015, 2016, 2017, 2018, 2019, 2020])
        ax_dic[ax_all[i]].set_xticklabels(
            ['2000', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
             '18', '19', '20'], fontname='Times New Roman', fontsize=20, rotation=45)
        i += 1
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig11\\Figure_11.png', dpi=500)
    plt.close()

    # plt.rc('axes', axisbelow=True)
    # plt.rc('axes', linewidth=2)
    # r_square = []
    # fig12, ax12 = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    # temp1 = result_dic['baishazhou_UL'] - result_dic['baishazhou_IL']
    # temp2 = result_dic['nanmenzhou_UL'] - result_dic['nanmenzhou_IL']
    # ax12.scatter(pos[4:17], temp1[4:17, 1], marker='s')
    # ax12.scatter(pos[4:17], temp2[4:17, 1], marker='o')
    # paras, extra = curve_fit(linear_function2, pos[4:17], temp1[4:17, 1])
    # ax12.plot(np.linspace(2002,2020,100), linear_function2(np.linspace(2002,2020,100), paras[0], paras[1]))
    # r_square.append((1 - np.sum((linear_function2(pos[4:17], paras[0], paras[1]) - temp1[4:17, 1]) ** 2) / np.sum(
    #     (temp1[4:17, 1] - np.mean(temp1[4:17, 1])) ** 2)))
    # paras, extra = curve_fit(linear_function2, pos[4:17], temp2[4:17, 1])
    # r_square.append((1 - np.sum((linear_function2(pos[4:17], paras[0], paras[1]) - temp2[4:17, 1]) ** 2) / np.sum(
    #     (temp2[4:17, 1] - np.mean(temp2[4:17, 1])) ** 2)))
    # ax12.plot(np.linspace(2002,2020,100), linear_function2(np.linspace(2002,2020,100), paras[0], paras[1]))
    # ax12.plot(np.linspace(2002,2020,100), np.linspace(0,0,100), c=(0, 0, 0), lw=3)
    # ax12.set_xlim(2003, 2020)
    # ax12.set_ylim(-0.02, 0.04)
    # ax12.set_yticks([-0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04])
    # ax12.set_yticklabels(['-0.02', '-0.01', '0.00', '0.01', '0.02', '0.03', '0.04'], fontname='Times New Roman', fontsize=20)
    # ax12.set_xticks([2004,2006,2008,2010,2012,2014,2016])
    # ax12.set_xticklabels(['2004', '2006', '2008', '2010', '2012', '2014', '2016'], fontname='Times New Roman', fontsize=20)
    # plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig12\\Figure_12.png', dpi=500)

    sns.set(style="white", palette='dark', font='Times New Roman',
            rc={'ytick.left': True, 'xtick.bottom': True})
    # plt.rc('axes', axisbelow=True)
    # plt.rc('axes', linewidth=1)

    fig12 = plt.figure(figsize=(10, 3), tight_layout=True)
    gs = gridspec.GridSpec(2, 4)
    year = [2002,2003,2005,2007,2011,2017,2019,2020]
    ax_dic = {}
    sa = 'bsz'
    i = 0
    for year_temp in year:
        file_ds = gdal.Open(temp_folder + str(sa) + str(year_temp) + '_temp.tif')
        file_raster = file_ds.GetRasterBand(1).ReadAsArray()
        UL_ds = gdal.Open(temp_folder + sa + '_UL.tif')
        IL_ds = gdal.Open(temp_folder + sa + '_IL.tif')
        UL_raster = UL_ds.GetRasterBand(1).ReadAsArray()
        IL_raster = IL_ds.GetRasterBand(1).ReadAsArray()
        file_raster2 = copy.copy(file_raster)
        file_raster[UL_raster != 1] = np.nan
        file_raster2[IL_raster != 1] = np.nan
        file_raster = file_raster.flatten()
        file_raster = np.delete(file_raster, np.argwhere(np.isnan(file_raster)))
        file_raster2 = file_raster2.flatten()
        file_raster2 = np.delete(file_raster2, np.argwhere(np.isnan(file_raster2)))
        ax_dic[str(year_temp)] = fig12.add_subplot(gs[i//4, np.mod(i,4)])
        sns.histplot(data=file_raster, binwidth=0.005, binrange=(-1, 1), kde=True, stat='density', fill=True,
                     color=(30 / 256, 96 / 256, 164 / 256, 1), element='step',
                     line_kws={'color': (30 / 256, 96 / 256, 164 / 256, 1), 'lw': 1, 'ls': '-'})
        sns.histplot(data=file_raster2, binwidth=0.005, binrange=(-1, 1), kde=True, element='step', stat='density', fill=True,
                     line_kws={'color':(256/256,48/256,30/256,1),'lw':1, 'ls':'-'}, color=(180/256,76/256,54/256,0.4))
        # ax_dic[str(year_temp)].hist(file_raster, bins=60, density=True, alpha=0.7, fc=(68 / 256, 119 / 256, 169 / 256), zorder=2)
        # ax_dic[str(year_temp)].hist(file_raster2, bins=60, density=True, alpha=0.7, fc=(227 / 256, 126 / 256, 82 / 256), zorder=2)
        ax_dic[str(year_temp)].set_ylim(0, 25)
        ax_dic[str(year_temp)].set_xlim(np.nanmean(file_raster)-0.15, np.nanmean(file_raster)+0.15)
        ax_dic[str(year_temp)].set_xticks([ np.nanmean(file_raster)-0.12, np.nanmean(file_raster)-0.06, np.nanmean(file_raster), np.nanmean(file_raster)+0.06, np.nanmean(file_raster)+0.12])
        ax_dic[str(year_temp)].set_xticklabels(['-0.12', '-0.06', 'σ', '+0.06', '+0.12'], fontname='Times New Roman',
                                               fontsize=6)
        ax_dic[str(year_temp)].plot(np.linspace(np.nanmean(file_raster),np.nanmean(file_raster),100), np.linspace(0,100,100), ls='--', lw=1, c=(0,0,0), zorder=1)

        if np.mod(i,4) != 0:
            ax_dic[str(year_temp)].set_yticks([])
        else:
            ax_dic[str(year_temp)].set_yticks([0,5,10,15,20,25])
            ax_dic[str(year_temp)].set_yticklabels(['0', '5', '10', '15', '20', '25'], fontname='Times New Roman', fontsize=6)
        i += 1
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig12\\Figure_12.png', dpi=500)


def fig12_func():
    pass


def fig16_func():
    annual_cf_a_phe = 'E:\A_Vegetation_Identification\Wuhan_Landsat_Original\Sample_123039\Backup\Landsat_bsz_curfitting_datacube\pheyear_OSAVI_SPL_datacube\\annual_cf_para.npy'
    inundated_folder = 'E:\\A_Vegetation_Identification\\Inundation_status\\bsz\\Annual_inundation_status\\'
    plt.rc('font', family='Times New Roman')
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)

    transition_stage_consecutive_year = []
    transition_stage_indi_year = []
    transition_stage_intact = []
    mat_stage_consecutive_year = []
    mat_stage_indi_year = []
    mat_stage_intact = []
    annual_cf = np.load(annual_cf_a_phe, allow_pickle=True).item()
    for year in range(2000, 2021):
        inundation_status_cy_ds = gdal.Open(inundated_folder+'annual_' + str(year) + '.tif')
        inundation_status_cy_rs = inundation_status_cy_ds.GetRasterBand(1).ReadAsArray()
        inundation_status_ly_ds = gdal.Open(inundated_folder+'annual_' + str(year-1) + '.tif')
        inundation_status_ly_rs = inundation_status_ly_ds.GetRasterBand(1).ReadAsArray()
        cf_array = annual_cf[str(year) + '_cf_para']
        for y in range(cf_array.shape[0]):
            for x in range(cf_array.shape[1]):
                if inundation_status_cy_rs[y, x] == 1:
                    if inundation_status_ly_rs[y,x] == 1:
                        transition_stage_consecutive_year.append(cf_array[y, x, 4] - 2 * cf_array[y, x, 5])
                        mat_stage_consecutive_year.append(cf_array[y, x, 2] + 2 * cf_array[y, x, 3])
                    else:
                        transition_stage_indi_year.append(cf_array[y, x, 4] - 2 * cf_array[y, x, 5])
                        mat_stage_indi_year.append(cf_array[y, x, 2] + 2 * cf_array[y, x, 3])
                else:
                    transition_stage_intact.append(cf_array[y, x, 4] - 2 * cf_array[y, x, 5])
                    mat_stage_intact.append(cf_array[y, x, 2] + 2 * cf_array[y, x, 3])

    transition_stage_consecutive_year = [i + 5 for i in transition_stage_consecutive_year if not np.isnan(i) and i != 0]
    transition_stage_indi_year = [i for i in transition_stage_indi_year if not np.isnan(i) and i != 0]
    transition_stage_intact = [i for i in transition_stage_intact if not np.isnan(i) and i != 0]
    mat_stage_consecutive_year = [i for i in mat_stage_consecutive_year if not np.isnan(i) and i != 0]
    mat_stage_indi_year = [i for i in mat_stage_indi_year if not np.isnan(i) and i != 0]
    mat_stage_intact = [i for i in mat_stage_intact if not np.isnan(i) and i != 0]

    fig_temp, ax1_box = plt.subplots(figsize=(10, 7), constrained_layout=True)
    box1 = ax1_box.boxplot([transition_stage_consecutive_year, transition_stage_indi_year, transition_stage_intact, mat_stage_consecutive_year, mat_stage_indi_year, mat_stage_intact], notch=True, vert=False, showfliers=True, flierprops=dict(markeredgecolor=(0 / 256, 0 / 256, 0 / 256)), labels=['Consecutive flood', 'Individual flood', 'Non-flooded', 'Consecutive flood', 'Individual flood', 'Non-flooded'], sym='',  widths=0.45, patch_artist=True, whis=(5, 95))
    plt.setp(box1['boxes'], linewidth=1.5)
    plt.setp(box1['whiskers'], linewidth=2.5)
    plt.setp(box1['medians'], linewidth=1.5, color='black')
    plt.setp(box1['caps'], linewidth=2.5)
    ax1_box.grid(b=True, axis='x', color=(240 / 256, 240 / 256, 240 / 256))
    ax1_box.set_yticklabels(['Consecutive flood\nstart of senescence', 'Individual flood\nstart of senescence', 'Non-flooded\nstart of senescence', 'Consecutive flood\nstart of maturity', 'Individual flood\nstart of maturity', 'Non-flooded\nstart of maturity'], fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax1_box.set_xticks([75, 165, 255, 345])
    ax1_box.set_xticklabels(['Mar', 'Jun', 'Sep', 'Dec'], fontname='Times New Roman', fontsize=32)
    # ax1_box.set_xlabel('Dormancy phase', fontname='Times New Roman', fontsize=24, fontweight='bold')
    # ax1_box.set_ylabel('Fractional uncertainty', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax1_box.set_xlim(75, 345)
    colors = [(68 / 256, 119 / 256, 169 / 256), (68 / 256, 119 / 256, 169 / 256), (68 / 256, 119 / 256, 169 / 256), (227 / 256, 126 / 256, 82 / 256), (227 / 256, 126 / 256, 82 / 256),(227 / 256, 126 / 256, 82 / 256)]
    for patch, colort in zip(box1['boxes'], colors):
        patch.set(facecolor=colort)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig16\\Figure_16.png', dpi=500)

def fig13_func():
    plt.rc('font', family='Times New Roman')
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-70, '#440053'),
        (0.1, '#404388'),
        (0.25, '#2a788e'),
        (0.5, '#21a784'),
        (0.7, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    white_viridis1 = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-70, '#ffffff'),
        (0.00000001, '#ffffff'),
        (0.0001, '#ff0000'),
        (0.5, '#ff0000'),
        (0.7, '#ff0000'),
        (1, '#ff0000'),
    ], N=256)
    sa_all = ['bsz', 'nmz', 'nyz']
    fig13 = plt.figure(figsize=(11.3, 15), tight_layout=True)
    gs = gridspec.GridSpec(3, 2)
    ax_dic = {}
    i = 0
    for sa in sa_all:
        temp_folder = 'E:\\A_Vegetation_Identification\\temp_folder\\fig13\\'
        ROI_mask_f = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\study_area_shapefile\\' + sa + '_upper.shp'
        folder_a_phe = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + sa + '_phenology_metrics\\pheyear_OSAVI_SPL\\well_bloom_season_ave_VI\\'
        folder_a_ori = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Landsat_' + sa + '_phenology_metrics\\OSAVI_SPL\\well_bloom_season_ave_VI\\'
        inundated_folder = 'E:\\A_Vegetation_Identification\\Inundation_status\\' + sa + '\\Annual_inundation_status\\'
        a = []
        b = []
        for year in range(2000, 2020):
            if year not in [2011, 2010, 2019, 2020, 2007, ]:
                a_year_temp = []
                b_year_temp = []
                a_phe = Landsat_main_v1.file_filter(folder_a_phe, [str(year)])[0]
                a_ori = Landsat_main_v1.file_filter(folder_a_ori, [str(year)])[0]
                a_phe_ds = gdal.Open(a_phe)
                a_ori_ds = gdal.Open(a_ori)
                inundated_ds = gdal.Open(Landsat_main_v1.file_filter(inundated_folder, [str(year)])[0])
                pre_inundated_ds = gdal.Open(Landsat_main_v1.file_filter(inundated_folder, [str(year-1)])[0])

                gdal.Warp(temp_folder + str(sa) + str(year) + '_phe.tif', a_phe_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                gdal.Warp(temp_folder + str(sa) + str(year) + '_ori.tif', a_ori_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                gdal.Warp(temp_folder + str(sa) + str(year) + '_inundated.tif', inundated_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                gdal.Warp(temp_folder + str(sa) + str(year-1) + '_inundated.tif', pre_inundated_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)

                a_inundated = gdal.Open(temp_folder + str(sa) + str(year) + '_inundated.tif')
                a_pre_inundated = gdal.Open(temp_folder + str(sa) + str(year-1) + '_inundated.tif')
                a_phe_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_phe.tif')
                a_ori_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_ori.tif')

                a_inundated_raster = a_inundated.GetRasterBand(1).ReadAsArray()
                a_pre_inundated_raster = a_pre_inundated.GetRasterBand(1).ReadAsArray()
                a_phe_raster = a_phe_ds.GetRasterBand(1).ReadAsArray()
                a_ori_raster = a_ori_ds.GetRasterBand(1).ReadAsArray()
                if a_phe_raster.shape[0] != a_ori_raster.shape[0] or a_phe_raster.shape[1] != a_ori_raster.shape[1]:
                    print('Consistency error!')
                    sys.exit(-1)
                else:
                    for y_temp in range(a_phe_raster.shape[0]):
                        for x_temp in range(a_phe_raster.shape[1]):
                            if ~np.isnan(a_phe_raster[y_temp, x_temp]) and ~np.isnan(a_ori_raster[y_temp, x_temp]):
                                a_year_temp.append([a_phe_raster[y_temp, x_temp], a_ori_raster[y_temp, x_temp]])
                                if a_pre_inundated_raster[y_temp, x_temp] == 1:
                                    b_year_temp.append([a_phe_raster[y_temp, x_temp], a_ori_raster[y_temp, x_temp]])
                    a_year_temp = np.array(a_year_temp)
                    b_year_temp = np.array(b_year_temp)

                    if a_year_temp.shape[0] != 0:
                        if a == []:
                            a = a_year_temp
                        else:
                            a = np.concatenate((a, a_year_temp), axis=0)

                    if b_year_temp.shape[0] != 0:
                        if b == []:
                            b = b_year_temp
                        else:
                            b = np.concatenate((b, b_year_temp), axis=0)

        ax_dic[sa] = fig13.add_subplot(gs[np.mod(i, 3), i//3], projection='scatter_density')
        density = ax_dic[sa].scatter_density(a[:, 1], a[:, 0], cmap=white_viridis)
        density = ax_dic[sa].scatter_density(b[:, 1], b[:, 0], cmap=white_viridis1)
        print('>10% = ' + str(np.sum(a[:,0 ] > 1.1 * a[:,1])/a.shape[0]))
        print('<10% = ' + str(np.sum(a[:, 0] < 0.9 * a[:, 1]) / a.shape[0]))
        print('con +10 = ' + str(np.sum(b[:, 0] > 1.1 * b[:, 1])/ np.sum(a[:, 0] > 1.1 * a[:, 1]) ))
        print('con -10 = ' + str(np.sum(b[:, 0] < 0.9 * b[:, 1])/ np.sum(a[:, 0] < 0.9 * a[:, 1]) ))
        ax_dic[sa].plot(np.linspace(0,1,100),np.linspace(0,1,100), lw=2,ls='--',c=(0,0,0))
        if sa == 'bsz':
            ax_dic[sa].plot(np.linspace(0.53, 1, 100), linear_function2(np.linspace(0.53, 1, 100), 1.13, -0.0392), lw=2, ls='-.', c=(0, 0, 0))
        elif sa == 'nmz':
            ax_dic[sa].plot(np.linspace(0.53, 1, 100), linear_function2(np.linspace(0.53, 1, 100), 1.13, -0.0322), lw=2,ls='-.', c=(0, 0, 0))
        ax_dic[sa].fill_between(np.linspace(0, 1, 100), np.linspace(0.0, 1.1, 100), np.linspace(1.3, 1.3, 100), color=(120/255, 120/255, 120/255), alpha=0.1)
        ax_dic[sa].fill_between(np.linspace(0, 1, 100), np.linspace(0,0,100), np.linspace(0, 0.9, 100), color=(120/255, 120/255, 120/255), alpha=0.1)
        ax_dic[sa].set_ylim(0.3, 0.7)
        ax_dic[sa].set_xlim(0.3, 0.7)
        ax_dic[sa].set_yticks([0.30, 0.40, 0.50, 0.60, 0.70])
        ax_dic[sa].set_yticklabels(['0.30', '0.40', '0.50', '0.60', '0.70'], fontname='Times New Roman',fontsize=15)
        ax_dic[sa].set_xticks([0.30, 0.40, 0.50, 0.60, 0.70])
        ax_dic[sa].set_xticklabels(['0.30', '0.40', '0.50', '0.60', '0.70'], fontname='Times New Roman', fontsize=15)
        ax_dic[sa].set_xlabel('MAVI of original VI series', fontname='Times New Roman', fontsize=18, fontweight='bold')
        ax_dic[sa].set_ylabel('MAVI of\nNIPY-reconstructed VI series', fontname='Times New Roman', fontsize=18, fontweight='bold')
        i += 1

    for sa in sa_all:
        inundated_folder = 'E:\\A_Vegetation_Identification\\Inundation_status\\' + sa + '\\Annual_inundation_status\\'
        temp_folder = 'E:\\A_Vegetation_Identification\\temp_folder\\fig13\\'
        ROI_mask_f = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\study_area_shapefile\\' + sa + '_upper.shp'
        folder_b_phe = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + sa + '_phenology_metrics\\pheyear_OSAVI_SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
        folder_b_ori = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Landsat_' + sa + '_phenology_metrics\\OSAVI_SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
        a = []
        b = []
        for year in range(2000, 2020):
            if year not in [2003, 2016]:
                a_year_temp = []
                a_phe = Landsat_main_v1.file_filter(folder_b_phe, [str(year) + '_' + str(year + 1)])[0]
                a_ori = Landsat_main_v1.file_filter(folder_b_ori, [str(year) + '_' + str(year + 1)])[0]
                a_phe_ds = gdal.Open(a_phe)
                a_ori_ds = gdal.Open(a_ori)
                inundated_ds = gdal.Open(Landsat_main_v1.file_filter(inundated_folder, [str(year)])[0])

                gdal.Warp(temp_folder + str(sa) + str(year) + '_inundated.tif', inundated_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                gdal.Warp(temp_folder + str(sa) + str(year) + '_b_phe.tif', a_phe_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                gdal.Warp(temp_folder + str(sa) + str(year) + '_b_ori.tif', a_ori_ds, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)

                a_inundated = gdal.Open(temp_folder + str(sa) + str(year) + '_inundated.tif')
                a_phe_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_b_phe.tif')
                a_ori_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_b_ori.tif')

                a_inundated_raster = a_inundated.GetRasterBand(1).ReadAsArray()
                a_phe_raster = a_phe_ds.GetRasterBand(1).ReadAsArray()
                a_ori_raster = a_ori_ds.GetRasterBand(1).ReadAsArray()
                if a_phe_raster.shape[0] != a_ori_raster.shape[0] or a_phe_raster.shape[1] != a_ori_raster.shape[1]:
                    print('Consistency error!')
                    sys.exit(-1)
                else:
                    for y_temp in range(a_phe_raster.shape[0]):
                        for x_temp in range(a_phe_raster.shape[1]):
                            if a_inundated_raster[y_temp, x_temp] == 1:
                                if ~np.isnan(a_phe_raster[y_temp, x_temp]) and ~np.isnan(a_ori_raster[y_temp, x_temp]):
                                    a_year_temp.append([a_phe_raster[y_temp, x_temp], a_ori_raster[y_temp, x_temp]])
                    a_year_temp = np.array(a_year_temp)
                    if a_year_temp.shape[0] != 0:
                        if a == []:
                            a = a_year_temp
                        else:
                            a = np.concatenate((a, a_year_temp), axis=0)
            else:
                a_year_temp = []
                a_phe = Landsat_main_v1.file_filter(folder_b_phe, [str(year) + '_' + str(year + 1)])[0]
                a_ori = Landsat_main_v1.file_filter(folder_b_ori, [str(year) + '_' + str(year + 1)])[0]
                a_phe_ds = gdal.Open(a_phe)
                a_ori_ds = gdal.Open(a_ori)
                inundated_ds = gdal.Open(Landsat_main_v1.file_filter(inundated_folder, [str(year)])[0])

                gdal.Warp(temp_folder + str(sa) + str(year) + '_inundated.tif', inundated_ds, cutlineDSName=ROI_mask_f,
                          cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                gdal.Warp(temp_folder + str(sa) + str(year) + '_b_phe.tif', a_phe_ds, cutlineDSName=ROI_mask_f,
                          cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)
                gdal.Warp(temp_folder + str(sa) + str(year) + '_b_ori.tif', a_ori_ds, cutlineDSName=ROI_mask_f,
                          cropToCutline=True, dstNodata=np.nan, xRes=30, yRes=30)

                a_inundated = gdal.Open(temp_folder + str(sa) + str(year) + '_inundated.tif')
                a_phe_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_b_phe.tif')
                a_ori_ds = gdal.Open(temp_folder + str(sa) + str(year) + '_b_ori.tif')

                a_inundated_raster = a_inundated.GetRasterBand(1).ReadAsArray()
                a_phe_raster = a_phe_ds.GetRasterBand(1).ReadAsArray()
                a_ori_raster = a_ori_ds.GetRasterBand(1).ReadAsArray()
                if a_phe_raster.shape[0] != a_ori_raster.shape[0] or a_phe_raster.shape[1] != a_ori_raster.shape[1]:
                    print('Consistency error!')
                    sys.exit(-1)
                else:
                    for y_temp in range(a_phe_raster.shape[0]):
                        for x_temp in range(a_phe_raster.shape[1]):
                            if a_inundated_raster[y_temp, x_temp] == 1:
                                if ~np.isnan(a_phe_raster[y_temp, x_temp]) and ~np.isnan(a_ori_raster[y_temp, x_temp]):
                                    a_year_temp.append([a_phe_raster[y_temp, x_temp], a_ori_raster[y_temp, x_temp]])
                    a_year_temp = np.array(a_year_temp)
                    if a_year_temp.shape[0] != 0:
                        if b == []:
                            b = a_year_temp
                        else:
                            b = np.concatenate((b, a_year_temp), axis=0)

        ax_dic[sa] = fig13.add_subplot(gs[np.mod(i, 3), i//3], projection='scatter_density')
        density = ax_dic[sa].scatter_density(a[:, 1], a[:, 0], cmap=white_viridis)
        # density = ax_dic[sa].scatter_density(b[:, 1], b[:, 0], cmap=white_viridis1)
        # density2 = ax_dic[sa].scatterplot(b[:, 1], b[:, 0])
        a_pp = 0
        a_np = 0
        a_pn = 0
        a_nn = 0
        for a_temp in range(a.shape[0]):
            if a[a_temp, 0] > 0 and a[a_temp, 1] > 0:
                a_pp += 1
            elif a[a_temp, 0] < 0 and a[a_temp, 1] < 0:
                a_nn += 1
            elif a[a_temp, 0] > 0 and a[a_temp, 1] < 0:
                a_np += 1
            elif a[a_temp, 0] < 0 and a[a_temp, 1] > 0:
                a_pn += 1

        b_pp = 0
        b_np = 0
        b_pn = 0
        b_nn = 0
        for b_temp in range(b.shape[0]):
            if b[b_temp, 0] > 0 and b[b_temp, 1] > 0:
                b_pp += 1
            elif b[b_temp, 0] < 0 and b[b_temp, 1] < 0:
                b_nn += 1
            elif b[b_temp, 0] > 0 and b[b_temp, 1] < 0:
                b_np += 1
            elif b[b_temp, 0] < 0 and b[b_temp, 1] > 0:
                b_pn += 1

        print('pp:' + str(a_pp/a.shape[0]) +'nn:' + str(a_nn/a.shape[0]) +'pn:' + str(a_pn/a.shape[0]) +'np:' + str(a_np/a.shape[0]))
        print('pp:' + str(b_pp / b.shape[0]) + 'nn:' + str(b_nn / b.shape[0]) + 'pn:' + str(
            b_pn / b.shape[0]) + 'np:' + str(b_np / b.shape[0]))
        ax_dic[sa].plot(np.linspace(-1,1,100), np.linspace(0,0,100), lw=3 ,c=(0, 0, 0))
        ax_dic[sa].plot(np.linspace(0, 0, 100),np.linspace(-1, 1, 100),lw=3, c=(0, 0, 0))
        ax_dic[sa].set_ylim(-0.2, 0.2)
        ax_dic[sa].set_xlim(-0.2, 0.2)
        ax_dic[sa].set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        ax_dic[sa].set_yticklabels(['-0.2', '-0.1', '0.0', '0.1', '0.2'], fontname='Times New Roman', fontsize=15)
        ax_dic[sa].set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        ax_dic[sa].set_xticklabels(['-0.2', '-0.1', '0.0', '0.1', '0.2'], fontname='Times New Roman', fontsize=15)
        ax_dic[sa].set_xlabel('MAVI variations of\noriginal VI series', fontname='Times New Roman', fontsize=16, fontweight='bold')
        ax_dic[sa].set_ylabel('MAVI variations of\nNIPY-reconstructed VI series', fontname='Times New Roman', fontsize=16, fontweight='bold')
        ax_dic[sa].set_xlabel('MAVI variations of\noriginal VI series', fontname='Times New Roman', fontsize=16, fontweight='bold')
        ax_dic[sa].set_ylabel('MAVI variations of\nNIPY-reconstructed VI series', fontname='Times New Roman', fontsize=16, fontweight='bold')
        i += 1
    cbar = fig13.colorbar(density, label='Number of points per pixel')
    cbar.ax.tick_params(labelsize=18)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig13\\Figure_13.png', dpi=500)


def fig15_func():
    fig14_df = pd.read_excel('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig14\\a.xlsx')
    fig14_ds = np.array(fig14_df)
    month_max_pri = []
    month_min_pri = []
    month_max_post = []
    month_min_post = []
    date_pri = np.array([[[]]])
    date_post = np.array([[[]]])
    for year in range(1979, 2019):
        year_series = []
        month_dic = {}
        for i in range(fig14_ds.shape[0]):
            if fig14_ds[i, 0] == year:
                try:
                    temp = month_dic[fig14_ds[i, 1]]
                except:
                    month_dic[fig14_ds[i, 1]] = []
                month_dic[fig14_ds[i, 1]].append(fig14_ds[i, 3])
                year_series.append(fig14_ds[i, 3])
        month_plot = [np.mean(np.average(month_dic[q])) for q in range(1, 13)]
        if year > 2004:
            if date_pri.shape[2] == 0:
                if len(year_series) != 365:
                    date_pri = np.array(year_series[0:365]).reshape([1,365,1])
                else:
                    date_pri = np.array(year_series).reshape([1,365,1])
                date_pri = date_pri.reshape([1, 365, 1])
            else:
                if len(year_series) != 365:
                    date_pri = np.append(date_pri,np.array(year_series[0:365]).reshape([1,365,1]), axis=2)
                else:
                    date_pri = np.append(date_pri,np.array(year_series).reshape([1,365,1]), axis=2)
            #ax_temp.plot([15, 45, 75, 105, 136, 166, 196, 227, 258, 288, 320, 350], month_plot, lw=0.5, c=(1, 0, 0))
            # ax_temp.plot(np.linspace(1, len(year_series), len(year_series)), np.array(year_series), lw=0.5, c=(1,0,0))
        elif year <= 2004:
            if date_post.shape[2] == 0:
                if len(year_series) != 365:
                    date_post = np.array(year_series[0:365]).reshape([1,365,1])
                else:
                    date_post = np.array(year_series).reshape([1,365,1])
                date_post = date_post.reshape([1,365,1])
            else:
                if len(year_series) != 365:
                    date_post = np.append(date_post,np.array(year_series[0:365]).reshape([1,365,1]), axis=2)
                else:
                    date_post = np.append(date_post,np.array(year_series).reshape([1,365,1]), axis=2)
            #ax_temp.plot([15, 45, 75, 105, 136, 166, 196, 227, 258, 288, 320, 350], month_plot, lw=0.5, c=(0, 1, 0))
            # ax_temp.plot(np.linspace(1, len(year_series), len(year_series)), np.array(year_series), lw=0.5, c=(0,1,0))
    plt.close()
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    fig_temp, ax_temp = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax_temp.grid(b=True, axis='y', color=(240 / 256, 240 / 256, 240 / 256), zorder=0)
    # ax_temp.fill_between(np.linspace(175, 300, 121), np.linspace(34,34,121), np.linspace(10,10,121),alpha=1, color=(0.9, 0.9, 0.9))
    ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(date_post, axis=2).reshape([365]), np.nanmin(date_post, axis=2).reshape([365]),alpha=0.5, color=(0.8, 0.3, 0.2))
    ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(date_pri, axis=2).reshape([365]), np.nanmin(date_pri, axis=2).reshape([365]),alpha=0.5, color=(0.2, 0.3, 0.8))
    ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(date_pri, axis=2).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(date_post, axis=2).reshape([365]), lw=5, c=(1, 0, 0), zorder=3)
    ax_temp.plot(np.linspace(1,365,365), np.linspace(28,28,365), lw=2, ls='--', c=(0,0,0))
    ax_temp.set_xlim(1, 365)
    ax_temp.set_ylim(14, 34)
    ax_temp.set_yticks([14,18,22,26,30,34])
    ax_temp.set_yticklabels(['14','18','22','26','30','34'],fontname='Times New Roman', fontsize=20)
    a = [15,  105, 197,  288,  350]
    c = ['Jan', 'Apr',  'Jul',  'Oct',  'Dec']
    ax_temp.set_xticks(a)
    ax_temp.set_xticklabels(c, fontname='Times New Roman', fontsize=20)
    ax_temp.set_xlabel('Month', fontname='Times New Roman', fontsize=24,
                          fontweight='bold')
    ax_temp.set_ylabel('Water level(m)', fontname='Times New Roman', fontsize=24,
                          fontweight='bold')
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig14\\Figure_14.png', dpi=500)

def fig_his_func():

    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
    sns.set(style="white", palette='dark', font='Times New Roman', font_scale=1, rc={'ytick.left': True, 'xtick.bottom': True})
    df = pd.read_excel('E:\A_Vegetation_Identification\Paper\Fig\Fig18\\data.xlsx')
    df = df.drop(df[df['VI'] < -0.5].index)
    df = df.drop(df[np.isnan(df['VI'])].index)
    df2 = copy.copy(df)
    df2 = df2.drop(df2[np.logical_and(np.mod(df['time'], 1000) < 300, np.mod(df['time'], 1000) > 150)].index)
    df2 = df2.drop(df2[df2['VI'] > -0.02625].index)

    sns.histplot(data=df, x="VI", binwidth=0.03, binrange=(-0.3, 1.0),kde=True, color=(30/256,96/256,164/256,1), element='step', line_kws={'color':(30/256,96/256,164/256,1),'lw':3, 'ls':'-'})
    sns.histplot(data=df2, x="VI", binwidth=0.03, binrange=(-0.3, 1.0), kde=False,  line_kws={'color':(256/256,48/256,30/256,1),'lw':3, 'ls':'-'}, color=(180/256,76/256,54/256,0.4), element='step',alpha=0.4)
    plt.plot(np.linspace(-0.4,-0, 100), 4.5 * guassain_dis(np.linspace(-0.4,-0, 100), np.nanstd(df2['VI']), np.nanmean(df2['VI'])),color=(180/256,46/256,23/256), lw=3)
    plt.savefig('E:\A_Vegetation_Identification\Paper\Fig\Fig18\\fig1.png', dpi=300)


def fig20_func():
    roi_name_list = ['nanmenzhou', 'baishazhou', 'nanyangzhou', 'zhongzhou']
    short_list = ['nmz', 'bsz', 'nyz', 'zz']
    coord_list = ['EPSG:32649', 'EPSG:32649', 'EPSG:32649',  'EPSG:32649']
    ax_all = ['ax' + str(num) for num in range(1, 5)]

    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    fig11 = plt.figure(figsize=(10, 10), tight_layout=True)
    gs = gridspec.GridSpec(4, 1)
    ax_dic = {}
    result_dic = {}
    roi_index = 0
    index = 0

    for roi, coord_sys, short in zip(roi_name_list, coord_list, short_list):
        if index <= 1:
            ax_dic[ax_all[index]] = fig11.add_subplot(gs[index, 0])
            ax_dic[ax_all[index]].zorder = 12
        else:
            ax_dic[ax_all[index]] = fig11.add_subplot(gs[index, 0])
            ax_dic[ax_all[index]].zorder = 12

            # ax_dic[ax_all[index]].set_yticks([0.3, 0.4, 0.5, 0.6])
            # ax_dic[ax_all[index]].set_yticklabels(['0.3', '0.4', '0.5', '0.6'], fontname='Times New Roman', fontsize=24)
        # ax_dic[ax_all[index]].set_xlabel('Year', fontname='Times New Roman', fontsize=18, fontweight='bold')
        # ax_dic[ax_all[index]].set_ylabel('MAVI', fontname='Times New Roman', fontsize=18, fontweight='bold')

        if roi in ['tuanzhou']:
            folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\OSAVI_flood_free_phenology_metrics\\annual_max_VI\\annual\\'
            ROI_mask_f = 'G:\Landsat\Jingjiang_shp\shpfile_123\Main\\' + short + '.shp'
            inundation_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\annual\\'
            MAVI_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
        elif roi in ['baishazhou', 'nanyangzhou', 'nanmenzhou', 'zhongzhou']:
            # folder2 = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + short + '_phenology_metrics\\pheyear_OSAVI_SPL\\well_bloom_season_ave_VI\\'
            # ROI_mask_f2 = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\study_area_shapefile\\' + short + '_upper.shp'
            # inundation_folder2 = 'E:\\A_Vegetation_Identification\\Inundation_status\\' + short + '\\Annual_inundation_status\\'
            # MAVI_folder2 = 'E:\\A_Vegetation_Identification\\Wuhan_Landsat_Original\\Sample_123039\\Backup\\Landsat_' + short + '_phenology_metrics\\pheyear_OSAVI_SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
            folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\OSAVI_flood_free_phenology_metrics\\annual_max_VI\\annual\\'
            ROI_mask_f = 'G:\Landsat\Jingjiang_shp\shpfile_123\Inside\\' + short + '.shp'
            inundation_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\annual\\'
            MAVI_folder = 'G:\\Landsat\\Sample123039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'
        else:
            folder = 'G:\\Landsat\\Sample122_124039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL\\well_bloom_season_ave_VI\\'
            ROI_mask_f = 'G:\Landsat\Jingjiang_shp\shpfile\Main2\\' + short + '.shp'
            inundation_folder = 'G:\\Landsat\\Sample122_124039\\Landsat_' + roi + '_datacube\\Landsat_Inundation_Condition\\' + roi + '_DT\\annual\\'
            MAVI_folder = 'G:\\Landsat\\Sample122_124039\\Landsat_' + roi + '_datacube\\OSAVI_NIPY_phenology_metrics\\SPL_veg_variation\\well_bloom_season_ave_VI_abs_value\\'

        vi_dis_list = []
        vi_10up_list = []
        vi_mid_list = []
        vi_10down_list = []
        for year in range(1986, 2021):
            filename = bf.file_filter(folder, [str(year)])[0]
            gdal.Warp('/vsimem/' + str(short) + str(year) + '_temp.tif', filename, cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=-2, xRes=30, yRes=30)
            VI_ds = gdal.Open('/vsimem/' + str(short) + str(year) + '_temp.tif')
            VI_dis = VI_ds.GetRasterBand(1).ReadAsArray()
            temp = np.ones_like(VI_dis)

            Landsat_main_v2.write_raster(VI_ds, temp, '/vsimem/', str(short) + str(year) + '_temp2.tif')
            gdal.Warp('/vsimem/' + str(short) + str(year) + '_temp3.tif', '/vsimem/' + str(short) + str(year) + '_temp2.tif', cutlineDSName=ROI_mask_f, cropToCutline=True, dstNodata=-2, xRes=30, yRes=30)

            VI_ds2 = gdal.Open('/vsimem/' + str(short) + str(year) + '_temp3.tif')
            VI_dis2 = VI_ds2.GetRasterBand(1).ReadAsArray()

            VI_dis = VI_dis.reshape([-1])
            VI_dis2 = VI_dis2.reshape([-1])

            if np.sum(np.logical_and(~np.isnan(VI_dis), VI_dis != -2)) > 0.5 * np.sum(VI_dis2 != -2):
                VI_dis = np.delete(VI_dis, np.argwhere(np.isnan(VI_dis)))
                VI_dis = np.delete(VI_dis, np.argwhere(VI_dis == -2))
                vi_dis_list.append(VI_dis)
            else:
                VI_dis = []
                vi_dis_list.append([])
            gdal.Unlink('/vsimem/' + str(short) + str(year) + '_temp.tif')
            gdal.Unlink('/vsimem/' + str(short) + str(year) + '_temp2.tif')

            if VI_dis != []:
                VI_dis = np.sort(VI_dis)
                vi_10up_list.append(VI_dis[int(VI_dis.shape[0]*0.1)])
                vi_mid_list.append(VI_dis[int(VI_dis.shape[0]*0.5)])
                vi_10down_list.append(VI_dis[int(VI_dis.shape[0]*0.9)])
            else:
                vi_10up_list.append(np.nan)
                vi_mid_list.append(np.nan)
                vi_10down_list.append(np.nan)
        box1 = ax_dic[ax_all[index]].boxplot(vi_dis_list, positions = range(1986, 2021), notch=True, vert=True, showfliers=False, flierprops=dict(markeredgecolor=(0 / 256, 0 / 256, 0 / 256)), labels=[str(q)[2:] for q in range(1986, 2021)], sym='',  widths=0.5, patch_artist=True, whis=(10, 90), zorder=0)
        plt.setp(box1['boxes'], linewidth=1.5, facecolor=(0.1,0.4,0.9), alpha=0.5)
        plt.setp(box1['whiskers'], linewidth=1.5, alpha=0.5)
        plt.setp(box1['medians'], linewidth=1.5, color=(0,0,0), alpha=0.5)
        plt.setp(box1['caps'], linewidth=1.5, alpha=0.5)

        vi_mid_list1 = vi_mid_list[:19]
        vi_mid_list2 = vi_mid_list[19:]

        data_temp = np.linspace(1986,2020,num=35)
        data_temp = np.delete(data_temp, np.argwhere(np.isnan(vi_mid_list)))
        vi_mid_list = np.delete(vi_mid_list, np.argwhere(np.isnan(vi_mid_list)))

        data_temp2 = np.linspace(1986,2004, num=19)
        data_temp2 = np.delete(data_temp2, np.argwhere(np.isnan(vi_mid_list1)))
        vi_mid_list1 = np.delete(vi_mid_list1, np.argwhere(np.isnan(vi_mid_list1)))

        data_temp3 = np.linspace(2005,2020, num=16)
        data_temp3 = np.delete(data_temp3, np.argwhere(np.isnan(vi_mid_list2)))
        vi_mid_list2 = np.delete(vi_mid_list2, np.argwhere(np.isnan(vi_mid_list2)))

        paras1, extras = curve_fit(linear_function2, data_temp, vi_mid_list)
        paras2, extras = curve_fit(linear_function2, data_temp2, vi_mid_list1)
        paras3, extras = curve_fit(linear_function2, data_temp3, vi_mid_list2)
        print(paras1[0])
        print(paras2[0])
        print(paras3[0])
        ax_dic[ax_all[index]].plot(range(1985, 2022), linear_function2(range(1985, 2022), paras1[0], paras1[1]), c=(0,0,0), lw=2, zorder=1)


        ax_dic[ax_all[index]].plot(range(1985, 2005), linear_function2(range(1985, 2005), paras2[0], paras2[1]), c=(0.9,0.1,0.1),  lw=2, zorder=2)
        ax_dic[ax_all[index]].plot(range(2004, 2022), linear_function2(range(2004, 2022), paras3[0], paras3[1]), c=(0.9,0.1,0.1),  lw=2, zorder=2)
        ax_dic[ax_all[index]].set_xlim(1985,2021)

        ax_dic[ax_all[index]].set_yticks([0.0,0.2,0.4,0.6,0.8])
        ax_dic[ax_all[index]].set_yticklabels(['0.0','0.2','0.4','0.6','0.8'], fontname='Times New Roman', fontsize=15)
        ax_dic[ax_all[index]].set_ylim(0, 0.8)
        index += 1
    # plt.show()

    plt.savefig('E:\A_Vegetation_Identification\Paper\Fig\Fig20\\fig20.png', dpi=300)

def fig21_func():
    for sa in ['baishazhou', 'nanyangzhou', 'nanmenzhou', 'zhongzhou']:
        folder = f'G:\Landsat\Sample123039\Landsat_{sa}_datacube\Landsat_Inundation_Condition\\{sa}_DT\\annual\\'

        array_after = []
        array_before = []

        for year in range(1986, 2005):
            file = bf.file_filter(folder, [str(year)])
            ds_temp = gdal.Open(file[0])
            array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
            array_temp = array_temp.astype(np.float)
            array_temp[array_temp == -32768] = np.nan

            if array_before == []:
                array_before = array_temp
            else:
                array_before = array_before + array_temp

        for year in range(2004, 2021):
            file = bf.file_filter(folder, [str(year)])
            ds_temp = gdal.Open(file[0])
            array_temp = ds_temp.GetRasterBand(1).ReadAsArray()
            array_temp = array_temp.astype(np.float)
            array_temp[array_temp == -32768] = np.nan
            if array_after == []:
                array_after = array_temp
            else:
                array_after = array_after + array_temp

        array_before[array_before!=-32768] = array_before[array_before!=-32768]/18
        array_after[array_after!=-32768] = array_after[array_after!=-32768]/17

        bf.write_raster(ds_temp, array_before,'E:\A_Vegetation_Identification\Paper\Fig\Fig22\\', f'{sa}_before.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)
        bf.write_raster(ds_temp, array_after,'E:\A_Vegetation_Identification\Paper\Fig\Fig22\\', f'{sa}_after.TIF', raster_datatype=gdal.GDT_Float32, nodatavalue=np.nan)

def fig_23_func():

    sa_dic = {}
    sa_dic['datatype'] = []
    sa_dic['data'] = []
    sa_dic['inundated_or_not'] = []
    for sa, sa_short in zip(['baishazhou', 'nanmenzhou'], ['bsz', 'nmz']):
        folder_inundation = f'G:\Landsat\Sample123039\Landsat_{sa}_datacube\Landsat_Inundation_Condition\\{sa}_DT\\annual\\'
        folder_2016 = f'G:\Landsat\Sample123039\Landsat_{sa}_datacube\OSAVI_flood_free_phenology_metrics\\average_VI_between_max_and_flood\\annual_variation\\'
        folder_2002 = f'E:\A_Vegetation_Identification\Wuhan_Landsat_Original\Sample_123039\Backup\Landsat_{sa_short}_phenology_metrics\pheyear_OSAVI_SPL_veg_variation\well_bloom_season_ave_VI_abs_value\\'

        cut = f'G:\Landsat\Jingjiang_shp\shpfile_123\Inside\\{sa_short}.shp'

        veg_2002 = bf.file_filter(folder_2002, ['2002_2003'])
        veg_2016 = bf.file_filter(folder_2016, ['2016_2017'])
        inundation_2002 = bf.file_filter(folder_inundation, ['2002'])
        inundation_2016 = bf.file_filter(folder_inundation, ['2016'])

        gdal.Warp('/vsimem/veg_2002.tif', veg_2002[0], cutlineDSName=cut, cropToCutline=True, dstNodata=-32768, xRes=30, yRes=30)
        gdal.Warp('/vsimem/veg_2016.tif', veg_2016[0], cutlineDSName=cut, cropToCutline=True, dstNodata=-32768, xRes=30,
                  yRes=30)
        gdal.Warp('/vsimem/inu_2002.tif', inundation_2002[0], cutlineDSName=cut, cropToCutline=True, dstNodata=-32768, xRes=30,
                  yRes=30)
        gdal.Warp('/vsimem/inu_2016.tif', inundation_2016[0], cutlineDSName=cut, cropToCutline=True, dstNodata=-32768, xRes=30,
                  yRes=30)

        veg_2002_ds = gdal.Open('/vsimem/veg_2002.tif')
        veg_2016_ds = gdal.Open('/vsimem/veg_2016.tif')
        inundation_2002_ds = gdal.Open('/vsimem/inu_2002.tif')
        inundation_2016_ds = gdal.Open('/vsimem/inu_2016.tif')

        veg_2002_array = veg_2002_ds.GetRasterBand(1).ReadAsArray()
        veg_2016_array = veg_2016_ds.GetRasterBand(1).ReadAsArray()
        inundation_2002_array = inundation_2002_ds.GetRasterBand(1).ReadAsArray()
        inundation_2016_array = inundation_2016_ds.GetRasterBand(1).ReadAsArray()

        for y in range(inundation_2016_array.shape[0]):
            for x in range(inundation_2016_array.shape[1]):
                if inundation_2016_array[y, x] == 1 and -1 < veg_2016_array[y, x] < 1:
                    sa_dic['data'].append(veg_2016_array[y, x])
                    sa_dic['inundated_or_not'].append('inundated')
                    sa_dic['datatype'].append(f'2016_{sa_short}')
                elif inundation_2016_array[y, x] == 0 and -1 < veg_2016_array[y, x] < 1:
                    sa_dic['data'].append(veg_2016_array[y, x])
                    sa_dic['inundated_or_not'].append('noninundated')
                    sa_dic['datatype'].append(f'2016_{sa_short}')

        for y in range(inundation_2002_array.shape[0]):
            for x in range(inundation_2002_array.shape[1]):
                if inundation_2002_array[y, x] == 1 and -1 < veg_2002_array[y, x] < 1:
                    sa_dic['data'].append(veg_2002_array[y, x])
                    sa_dic['inundated_or_not'].append('inundated')
                    sa_dic['datatype'].append(f'2002_{sa_short}')
                elif inundation_2002_array[y, x] == 0 and -1 < veg_2002_array[y, x] < 1:
                    sa_dic['data'].append(veg_2002_array[y, x])
                    sa_dic['inundated_or_not'].append('noninundated')
                    sa_dic['datatype'].append(f'2002_{sa_short}')

    sa_dic = pd.DataFrame(sa_dic)
    sns.set_theme(style="whitegrid")
    sa_dic.to_csv('E:\A_Vegetation_Identification\Paper\Fig\Fig25\\1.csv')
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    fig11 = plt.figure(figsize=(10, 10), tight_layout=True)
    ax1 = fig11.add_subplot()
    sns.violinplot(data=sa_dic, x="datatype", y="data", hue='inundated_or_not',
                   split=True, inner="quart", linewidth=1.5, cut=0,
                   palette={"inundated": "b", "noninundated": ".85"})
    ax1.set_ylim(-0.3,0.3)
    plt.savefig('E:\A_Vegetation_Identification\Paper\Fig\Fig25\\Fig25.png',dpi=300)

fig11_new_func()

