import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.gridspec as gridspec


def seven_para_logistic_function(x, m1, m2, m3, m4, m5, m6, m7):
    return m1 + (m2 - m7 * x) * ((1 / (1 + np.exp((m3 - x) / m4))) - (1 / (1 + np.exp((m5 - x) / m6))))


def mark_plotter(ax, data1, data2, param_dict):
    out = ax.plot(data1, data2, **param_dict)
    return out


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

    fig, ax = plt.subplots(figsize=(20, 6), constrained_layout=True)
    ax.set_axis_on()
    ax.set_xlim(0, 730)
    ax.set_ylim(0, 0.7)
    ax.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax.plot(fig3_dic['DOY'], fig3_dic['OSAVI'], linewidth=10, markersize=24, **{'marker': 'o', 'color': 'b'})
    ax.plot(array_temp[0, 0: a], array_temp[1, 0: a], linewidth=10, markersize=24, **{'marker': 'o', 'color': 'r'})
    ax.plot(array_temp[0, :], array_temp[1, :], linewidth=10, markersize=24, **{'ls': '--', 'marker': 'o', 'color': 'r'})

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

    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6]), linewidth=10, color=(0/256, 109/256, 44/256))
    fig4_dic['DOY'] = fig4_dic['DOY'][1:]
    fig4_dic['OSAVI'] = fig4_dic['OSAVI'][1:]
    fig4_df = pd.DataFrame.from_dict(fig4_dic)
    # ax4.plot(array_temp[0, :], array_temp[1, :], linewidth=4, markersize=12, **{'ls': '--', 'marker': 'o', 'color': 'b'})
    ax4.fill_between(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.203, 0.523, 88, 9, 330, 12, 0.00069), seven_para_logistic_function(np.linspace(0, 365, 366), 0.05, 0.53, 102, 8, 330, 12, 0.00125), color=(0.1, 0.1, 0.1), alpha=0.1)
    ax4.scatter(fig4_dic['DOY'], fig4_dic['OSAVI'], s=12**2, color="none", edgecolor=(160/256, 196/256, 160/256), linewidth=3)
    # ax4.fill_between(np.linspace(560, 650, 100), np.linspace(0, 0, 100), np.linspace(1, 1, 100), color=(0, 197/255, 1), alpha=1)
    # ax4.plot(np.linspace(365, 365, 100), np.linspace(0, 1, 100), linewidth=4, **{'ls': '--', 'color': (0, 0, 0)})
    ax4.set_xlabel('DOY', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.set_ylabel('OSAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax4.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.203, 0.523, 88, 9, 330, 12, 0.00069), linewidth=2, color=(0 / 256, 109 / 256, 44 / 256), **{'ls': '--'})
    ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), 0.05, 0.53, 102, 8, 330, 12, 0.00125), linewidth=2, color=(0 / 256, 109 / 256, 44 / 256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_min, paras2_min, paras3_min, paras4_max, paras5_min, paras6_max, paras7_max), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    # ax4.plot(np.linspace(0, 365, 366), seven_para_logistic_function(np.linspace(0, 365, 366), paras1_max, paras2_max, paras3_max, paras4_min, paras5_max, paras6_min, paras7_min), linewidth=2, color=(0/256, 109/256, 44/256), **{'ls': '--'})
    predicted_y_data = seven_para_logistic_function(np.array(fig4_dic['DOY']), paras[0], paras[1], paras[2], paras[3], paras[4], paras[5], paras[6])
    r_square = (1 - np.sum((predicted_y_data - np.array(fig4_dic['OSAVI'])) ** 2) / np.sum((np.array(fig4_dic['OSAVI']) - np.mean(np.array(fig4_dic['OSAVI']))) ** 2))
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
    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig4\\Figure_4.png', dpi=1000)
    plt.show()
    print(r_square)


def fig5_func():
    # Generate npy
    # a = self.vi_sa_array_for_phenology[120:150, 253:277, :]
    # b = self.doy_array_for_phenology
    # c = np.array([b, np.nanmean(a, axis=(0, 1))])
    # np.save('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\bsz_NDVI.npy', c)
    # Create fig5
    VI_curve_fitting = {'para_ori': [0.01, 0.01, 50, 2, 300, 2, 0.01], 'para_boundary': ([0, 0, 50, 0, 300, 0, 0], [0.5, 1, 100, 15, 330, 15, 0.03])}
    bsz_EVI = np.load('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\bsz_EVI.npy')
    bsz_NDVI = np.load('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\bsz_NDVI.npy')
    bsz_OSAVI = np.load('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\bsz_OSAVI.npy')

    i = 0
    while i < bsz_EVI.shape[1]:
        if np.isnan(bsz_EVI[1, i]) or bsz_EVI[0, i] // 1000 < 2000 or bsz_EVI[0, i] // 1000 in [2002, 2203, 2007, 2010, 2012, 2016, 2017, 2020]:
            bsz_EVI = np.delete(bsz_EVI, i, axis=1)
            i -= 1
        i += 1

    i = 0
    while i < bsz_NDVI.shape[1]:
        if np.isnan(bsz_NDVI[1, i]) or bsz_NDVI[0, i] // 1000 < 2000 or bsz_NDVI[0, i] // 1000 in [2002, 2203, 2007, 2010, 2012, 2016, 2017, 2020]:
            bsz_NDVI = np.delete(bsz_NDVI, i, axis=1)
            i -= 1
        i += 1
    i = 0
    while i < bsz_OSAVI.shape[1]:
        if np.isnan(bsz_OSAVI[1, i]) or bsz_OSAVI[0, i] // 1000 < 2000 or bsz_OSAVI[0, i] // 1000 in [2002, 2203, 2007, 2010, 2012, 2016, 2017, 2020]:
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
    ax2 = fig5.add_subplot(gs[0, 1])
    ax3 = fig5.add_subplot(gs[0, 2])
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

    bsz_OSAVI_error = (bsz_OSAVI[1, :] - predicted_y_data_OSAVI) / predicted_y_data_OSAVI
    bsz_NDVI_error = (bsz_NDVI[1, :] - predicted_y_data_NDVI) / predicted_y_data_NDVI
    bsz_EVI_error = (bsz_EVI[1, :] - predicted_y_data_EVI) / predicted_y_data_EVI

    death_array_NDVI = np.array([])
    greenup_array_NDVI = np.array([])
    well_boom_array_NDVI = np.array([])
    i = 0
    while i < bsz_NDVI.shape[1]:
        if bsz_NDVI[0, i] < 65 or bsz_NDVI[0, i] > 350:
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
        if bsz_OSAVI[0, i] < 65 or bsz_OSAVI[0, i] > 350:
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
        if bsz_EVI[0, i] < 65 or bsz_EVI[0, i] > 350:
            death_array_EVI = np.append(death_array_EVI, bsz_EVI_error[i])
        if 125 < bsz_EVI[0, i] < 320:
            well_boom_array_EVI = np.append(well_boom_array_EVI, bsz_EVI_error[i])
        if 75 < bsz_EVI[0, i] < 125:
            greenup_array_EVI = np.append(greenup_array_EVI, bsz_EVI_error[i])
        i += 1

    box1 = ax1_box.boxplot([death_array_NDVI, death_array_OSAVI, death_array_EVI], labels=['NDVI', 'OSAVI', 'EVI'], sym='', notch=True, widths=0.45, patch_artist=True, whis=(5, 95))
    plt.setp(box1['boxes'], linewidth=1.5)
    plt.setp(box1['whiskers'], linewidth=1.5)
    plt.setp(box1['medians'], linewidth=1.5)
    plt.setp(box1['caps'], linewidth=1.5)
    ax1_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax1_box.set_yticks(b)
    ax1_box.set_yticklabels(d, fontname='Times New Roman', fontsize=16)
    ax1_box.set_xlabel('Senescence phase', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax1_box.set_ylabel('Fractional uncertainty', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax1_box.set_ylim(-0.6, 0.6)
    ax1_box.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))

    box2 = ax2_box.boxplot([well_boom_array_NDVI, well_boom_array_OSAVI, well_boom_array_EVI], labels=['NDVI', 'OSAVI', 'EVI'], sym='', notch=True, widths=0.45, patch_artist=True, whis=(5, 95))
    plt.setp(box2['boxes'], linewidth=1.5)
    plt.setp(box2['whiskers'], linewidth=1.5)
    plt.setp(box2['medians'], linewidth=1.5)
    plt.setp(box2['caps'], linewidth=1.5)
    ax2_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax2_box.set_yticks(b)
    ax2_box.set_yticklabels(d, fontname='Times New Roman', fontsize=16)
    ax2_box.set_xlabel('Flourish phase', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax2_box.set_ylim(-0.6, 0.6)
    ax2_box.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))

    box3 = ax3_box.boxplot([bsz_NDVI_error[:], bsz_OSAVI_error[:], bsz_EVI_error[:]], labels=['NDVI', 'OSAVI', 'EVI'], sym='', notch=True, widths=0.45, patch_artist=True, whis=(5, 95))
    plt.setp(box3['boxes'], linewidth=1.5)
    plt.setp(box3['whiskers'], linewidth=1.5)
    plt.setp(box3['medians'], linewidth=1.5)
    plt.setp(box3['caps'], linewidth=1.5)
    ax3_box.set_xticklabels(['NDVI', 'OSAVI', 'EVI'], fontname='Times New Roman', fontsize=20, fontweight='bold')
    ax3_box.set_yticks(b)
    ax3_box.set_yticklabels(d, fontname='Times New Roman', fontsize=16)
    ax3_box.set_xlabel('The entire period', fontname='Times New Roman', fontsize=24, fontweight='bold')
    ax3_box.set_ylim(-0.6, 0.6)
    ax3_box.grid(b=True, axis='y', color=(240/256, 240/256, 240/256))

    colors = [(196/256, 120/256, 120/256), (100/256, 196/256, 70/256), (120/256, 120/256, 196/256)]
    for patch, colort in zip(box1['boxes'], colors):
        patch.set(facecolor=colort)
    for patch, colort in zip(box2['boxes'], colors):
        patch.set(facecolor=colort)
    for patch, colort in zip(box3['boxes'], colors):
        patch.set(facecolor=colort)

    plt.savefig('E:\\A_Vegetation_Identification\\Paper\\Fig\\Fig5\\Figure_5.png', dpi=1000)
    plt.show()


fig4_func()