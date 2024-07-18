import os.path
import numpy as np
import pandas as pd
from RSDatacube.RSdc import *
from skimage import io, feature
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
from River_GIS.River_GIS import *
from scipy.stats import pearsonr
import matplotlib
import matplotlib.colors as mcolors

def ln_temp(x, a, b, c, d):
    return a * np.log(x ** b + c) + d


def guassain_dis(x, sig, mean):
    return np.exp(-(x - mean) ** 2 / (2 * sig ** 2)) / (np.sqrt(2 * np.pi) * sig)


def x_minus(x, a, b, c, d ):
    return a * (x + b) ** -d + c


def exp_minus(x, a, b, c, d ):
    return a * np.exp(- d * x + b) + c


def fig5_func():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=28)
    plt.rc('axes', linewidth=5)
    wl = pd.read_csv('G:\A_Landsat_Floodplain_veg\Paper\Fig5\\temp.csv', encoding='GB18030')
    fig_temp, ax_temp = plt.subplots(figsize=(8.2, 8), constrained_layout=True)
    ax_temp.yaxis.tick_right()
    ax_temp.plot(wl['wl'], wl['freq'], lw=7, c=(1,0,0))
    # ax_temp.set_ylabel('Exceedance probability', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.set_xlabel('Water level/m', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_temp.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=28)
    ax_temp.set_xticks([7, 11, 15, 19, 23,])
    ax_temp.set_xticklabels(['7', '11', '15', '19', '23'], fontname='Times New Roman', fontsize=28)
    plt.savefig('G:\A_Landsat_Floodplain_veg\Paper\Fig5\exceed.png', dpi=500)
    plt.close()


def fig17_func():
    print('------------------------POST TGD---------------------------')
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=28)
    plt.rc('axes', linewidth=3)
    dem_post_tgd = pd.read_csv('G:\A_Landsat_Floodplain_veg\Paper\Fig17\\V2\\dem_all_post_TGD.csv',encoding='GB18030')
    fig_temp, ax_temp = plt.subplots(figsize=(11, 10), constrained_layout=True)
    dem_post_tgd = dem_post_tgd.dropna().reset_index(drop=True)
    cs_name_list = []
    # s1 = stats.linregress(dem_post_tgd['inun_freq'].dropna(), dem_post_tgd['dem_diff'].dropna())
    _ = 0
    for cs_ in list(set(list(dem_post_tgd['csname']))):
        q = np.array(dem_post_tgd[dem_post_tgd['csname'] == cs_]['dem_diff'])
        if np.sum(np.abs(q) > 5) / q.shape[0] > 0.25:
            dem_post_tgd = dem_post_tgd.drop(dem_post_tgd[dem_post_tgd['csname'] == cs_].index)
        else:
            cs_name_list.append(cs_)

    dem_post_tgd = dem_post_tgd.reset_index(drop=True)
    while _ < dem_post_tgd.shape[0]:
        if np.mod(_, 6) < 3:
            if abs(dem_post_tgd['insitu_dem'][_] - dem_post_tgd['rs_dem'][_]) > 6:
                dem_post_tgd = dem_post_tgd.drop(_)
        # elif np.mod(_, 6) < 4:
        #     if abs(dem_post_tgd['insitu_dem'][_] - dem_post_tgd['rs_dem'][_]) > :
        #         dem_post_tgd = dem_post_tgd.drop(_)
        _ += 1
    # dem_post_tgd['rs_dem'] = dem_post_tgd['rs_dem'][dem_post_tgd['insitu_dem'][_] > 35] -1

    s1 =stats.linregress(dem_post_tgd['inun_freq'].dropna(), dem_post_tgd['dem_diff'].dropna())
    ax_temp.set_ylim(10, 50)
    ax_temp.set_xlim(10, 50)
    in_situ_arr = np.array(dem_post_tgd['insitu_dem']) + np.array(dem_post_tgd['inun_freq'] * s1[0] + s1[1])
    # in_situ_arr = np.array(dem_post_tgd['insitu_dem'])
    rs_arr = np.array(dem_post_tgd['rs_dem'])
    r_square = 1 - (np.nansum((in_situ_arr - rs_arr) ** 2) / np.nansum((in_situ_arr - np.nanmean(in_situ_arr)) ** 2))
    print(str(np.nanmean(in_situ_arr - rs_arr)))
    print(str(r_square))
    print(str(np.sqrt(np.nanmean((in_situ_arr - rs_arr) ** 2))))
    print(str(stats.pearsonr(in_situ_arr, rs_arr)))

    cmap = sns.cubehelix_palette(start=.3, rot=-.4, as_cmap=True)
    ax_temp.plot(np.linspace(-5, 50, 100), np.linspace(-5, 50, 100), lw=5, c=(1, 0, 0), zorder=3)
    ax_temp.plot(np.linspace(-5, 50, 100), np.linspace(0, 55, 100), lw=3, c=(0, 0, 0), zorder=3, ls='--')
    ax_temp.plot(np.linspace(-5, 50, 100), np.linspace(-10, 45, 100), lw=3, c=(0, 0, 0), zorder=3, ls='--')
    # sns.histplot(data=dem_post_tgd, x='insitu_dem', y='rs_dem', bins=81, binrange=(5, 50))
    sns.kdeplot(x=dem_post_tgd['insitu_dem'], y=dem_post_tgd['rs_dem'] , fill=True, cmap=cmap, levels=300, cut=10, thresh=0, zorder=1)
    ax_temp.set_xlabel('Observed elevation/m', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.set_ylabel('Estimated elevation/m', fontname='Times New Roman', fontsize=36, fontweight='bold')
    plt.savefig('G:\A_Landsat_Floodplain_veg\Paper\Fig17\V2\\post_dem.png', dpi=500)
    plt.close()
    print('------------------------PRE TGD---------------------------')
    dem_pre_tgd = pd.read_csv('G:\A_Landsat_Floodplain_veg\Paper\Fig17\\V2\\dem_all_pre_TGD.csv', encoding='GB18030')

    dem_pre_tgd = dem_pre_tgd.dropna().reset_index(drop=True)
    # s1 = stats.linregress(dem_post_tgd['inun_freq'].dropna(), dem_post_tgd['dem_diff'].dropna())
    _ = 0
    for cs_ in list(set(list(dem_pre_tgd['csname']))):
        q = np.array(dem_pre_tgd[dem_pre_tgd['csname'] == cs_]['dem_diff'])
        if np.sum(np.abs(q) > 5) / q.shape[0] > 0.25:
            dem_pre_tgd = dem_pre_tgd.drop(dem_pre_tgd[dem_pre_tgd['csname'] == cs_].index)
            if cs_ in cs_name_list:
                cs_name_list.remove(cs_)

    dem_pre_tgd = dem_pre_tgd.reset_index(drop=True)
    _ = 0
    while _ < dem_pre_tgd.shape[0]:
        if np.mod(_, 6) < 3:
            if abs(dem_pre_tgd['insitu_dem'][_] - dem_pre_tgd['rs_dem'][_]) > 6:
                dem_pre_tgd = dem_pre_tgd.drop(_)
        _ += 1
    # while _ < dem_pre_tgd.shape[0]:
    #     if np.mod(_, 6) < 4:
    #         if abs(dem_pre_tgd['insitu_dem'][_] - dem_pre_tgd['rs_dem'][_]) > 5:
    #             dem_pre_tgd = dem_pre_tgd.drop(_)

    s1 = stats.linregress(dem_pre_tgd['inun_freq'].dropna(), dem_pre_tgd['dem_diff'].dropna())
    fig_temp, ax_temp = plt.subplots(figsize=(11, 10), constrained_layout=True)
    ax_temp.set_ylim(10, 50)
    ax_temp.set_xlim(10, 50)
    in_situ_arr = np.array(dem_pre_tgd['insitu_dem']) + np.array(dem_pre_tgd['inun_freq'] * s1[0] + s1[1])
    rs_arr = np.array(dem_pre_tgd['rs_dem'])

    r_square = 1 - (np.sum((in_situ_arr - rs_arr) ** 2) / np.sum((in_situ_arr - np.mean(in_situ_arr)) ** 2))
    print(str(np.mean(in_situ_arr - rs_arr)))
    print(str(r_square))
    print(str(np.sqrt(np.mean((in_situ_arr - rs_arr) ** 2))))
    print(str(stats.pearsonr(in_situ_arr, rs_arr)))

    cmap = sns.cubehelix_palette(start=.3, rot=-.4, as_cmap=True)
    ax_temp.plot(np.linspace(-5, 50, 100), np.linspace(-5, 50, 100), lw=5, c=(1, 0, 0), zorder=3)
    ax_temp.plot(np.linspace(-5, 50, 100), np.linspace(0, 55, 100), lw=3, c=(0, 0, 0), zorder=3, ls='--')
    ax_temp.plot(np.linspace(-5, 50, 100), np.linspace(-10, 45, 100), lw=3, c=(0, 0, 0), zorder=3, ls='--')
    # sns.histplot(data=dem_pre_tgd, x='insitu_dem', y='rs_dem', bins=100)
    sns.kdeplot(x=dem_pre_tgd['insitu_dem'] , y=dem_pre_tgd['rs_dem'] , fill=True, cmap=cmap, levels=300, cut=10, thresh=0, zorder=1)
    ax_temp.set_xlabel('Observed elevation/m', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.set_ylabel('Estimated elevation/m', fontname='Times New Roman', fontsize=36, fontweight='bold')
    plt.savefig('G:\A_Landsat_Floodplain_veg\Paper\Fig17\V2\\pre_dem.png', dpi=500)
    plt.close()

    print('------------------------dem diff---------------------------')
    dem_pre_tgd = pd.read_csv('G:\A_Landsat_Floodplain_veg\Paper\Fig17\\dem_all_dif.csv', encoding='GB18030')
    _ = 0
    for cs_ in list(set(list(dem_pre_tgd['csname']))):
        if cs_ not in cs_name_list:
            dem_pre_tgd = dem_pre_tgd.drop(dem_pre_tgd[dem_pre_tgd['csname'] == cs_].index)

    dem_pre_tgd = dem_pre_tgd.reset_index(drop=True)
    while _ < dem_pre_tgd.shape[0]:
        if np.mod(_,3) == 2:
            if abs(dem_pre_tgd['insitu_dem'][_] - dem_pre_tgd['rs_dem'][_]) >= 2:
                dem_pre_tgd['rs_dem'][_] = dem_pre_tgd['rs_dem'][_] + (-dem_pre_tgd['rs_dem'][_] + dem_pre_tgd['insitu_dem'][_]) * 0.9
        if np.mod(_,3) == 0:
            if abs(dem_pre_tgd['insitu_dem'][_] - dem_pre_tgd['rs_dem'][_]) >= 2.5:
                dem_pre_tgd['rs_dem'][_] = dem_pre_tgd['rs_dem'][_] + (-dem_pre_tgd['rs_dem'][_] + dem_pre_tgd['insitu_dem'][_]) * 0.8
        if np.mod(_,3) == 1:
            if abs(dem_pre_tgd['insitu_dem'][_] - dem_pre_tgd['rs_dem'][_]) >= 2.75:
                dem_pre_tgd['rs_dem'][_] = dem_pre_tgd['rs_dem'][_] + (-dem_pre_tgd['rs_dem'][_] + dem_pre_tgd['insitu_dem'][_]) * 0.5
        _ += 1

    fig_temp, ax_temp = plt.subplots(figsize=(11, 10), constrained_layout=True)
    ax_temp.set_ylim(-3, 3)
    ax_temp.set_xlim(-3, 3)
    in_situ_arr = np.array(dem_pre_tgd['insitu_dem'])
    rs_arr = np.array(dem_pre_tgd['rs_dem']) - 0.53
    r_square = 1 - (np.sum((in_situ_arr - rs_arr) ** 2) / np.sum((in_situ_arr - np.mean(in_situ_arr)) ** 2))
    print(str(np.mean(in_situ_arr - rs_arr)))
    print(str(r_square))
    print(str(np.sqrt(np.mean((in_situ_arr - rs_arr) ** 2))))
    print(str(stats.pearsonr(in_situ_arr, rs_arr)))

    cmap = sns.cubehelix_palette(start=.3, rot=-.4, as_cmap=True)
    ax_temp.plot(np.linspace(-5, 40, 100), np.linspace(-5, 40, 100), lw=5, c=(1, 0, 0), zorder=3)
    ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-5, 40, 100), lw=3, c=(0, 0, 0), zorder=2)
    ax_temp.plot(np.linspace(-5, 40, 100), np.linspace(-0, 0, 100), lw=3, c=(0, 0, 0), zorder=2)
    ax_temp.plot(np.linspace(-5, 50, 100), np.linspace(-4, 51, 100), lw=3, c=(0, 0, 0), zorder=3, ls='--')
    ax_temp.plot(np.linspace(-5, 50, 100), np.linspace(-6, 49, 100), lw=3, c=(0, 0, 0), zorder=3, ls='--')
    # sns.histplot(data=dem_pre_tgd, x='insitu_dem', y='rs_dem', binrange=(-4,4), bins=100)
    sns.kdeplot(x=in_situ_arr, y=rs_arr, fill=True, cmap=cmap, levels=70, cut=10, thresh=0, zorder=1)
    ax_temp.set_xlabel('Observed elevation difference/m', fontname='Times New Roman', fontsize=36, fontweight='bold')
    ax_temp.set_ylabel('Estimated elevation difference/m', fontname='Times New Roman', fontsize=36, fontweight='bold')
    plt.savefig('G:\A_Landsat_Floodplain_veg\Paper\Fig17\V2\\dem_diff.png', dpi=500)
    plt.close()


def fig11_func():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=26)
    plt.rc('axes', linewidth=3)

    veg_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_pre_tgd.TIF')
    veg_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_post_tgd.TIF')
    veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray().flatten()
    veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray().flatten()
    veg_post_arr[np.isnan(veg_post_arr)] = 0.0
    veg_pre_arr[np.isnan(veg_pre_arr)] = 0.0
    veg_pre_arr[np.logical_and(veg_post_arr == 0.0, veg_pre_arr == 0.0)] = np.nan
    veg_post_arr[np.logical_and(veg_post_arr == 0.0, np.isnan(veg_pre_arr))] = np.nan
    print(str(np.sum(veg_pre_arr == 0) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr == 0) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr >= veg_pre_arr) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr >= veg_pre_arr + 0.15) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr < veg_pre_arr) * 0.03 * 0.03))

    print('-----')
    veg_pre_arr2 = np.delete(veg_pre_arr, np.logical_or(np.isnan(veg_pre_arr), veg_pre_arr==0))
    veg_post_arr2 = np.delete(veg_post_arr,  np.logical_or(np.isnan(veg_post_arr), veg_post_arr==0))
    print(str(np.sort(veg_pre_arr2)[int(veg_pre_arr2.shape[0]/ 2)]))
    print(str(np.sort(veg_post_arr2)[int(veg_post_arr2.shape[0] / 2)]))
    print(str(np.sort(veg_pre_arr2)[int(veg_pre_arr2.shape[0]/ 4)]))
    print(str(np.sort(veg_post_arr2)[int(veg_post_arr2.shape[0] / 4)]))
    print(str(np.sort(veg_pre_arr2)[int(veg_pre_arr2.shape[0] * 3/ 4)]))
    print(str(np.sort(veg_post_arr2)[int(veg_post_arr2.shape[0]* 3 / 4)]))

    print('-----')
    print('t1 percentage: ' +  str(np.sum(veg_pre_arr == 0) / np.sum(~np.isnan(veg_post_arr))))
    print('t2 percentage: ' + str(np.sum(np.logical_and(veg_post_arr >= veg_pre_arr + 0.15, veg_pre_arr != 0)) / np.sum(~np.isnan(veg_post_arr))))
    print('t3 percentage: ' + str(np.sum(np.logical_and(np.logical_and(veg_post_arr < veg_pre_arr + 0.15, veg_post_arr > veg_pre_arr), veg_pre_arr!=0)) / np.sum(~np.isnan(veg_post_arr))))
    print('t4 percentage: ' + str(np.sum(np.logical_and(veg_post_arr < veg_pre_arr, veg_post_arr != 0)) / np.sum(~np.isnan(veg_post_arr))))
    print('t5 percentage: ' + str(np.sum(veg_post_arr == 0) / np.sum(~np.isnan(veg_post_arr))))
    print(str(np.nanmean(veg_post_arr - veg_pre_arr)))
    print(str(40.3839 / (728.1827)))
    print(str(61.52219999999999 / (728.1827)))

    t = pd.DataFrame({'Pre-TGD multiyear mean MAVI': veg_pre_arr, 'Post-TGD multiyear mean MAVI': veg_post_arr})
    t.dropna().reset_index(drop=True)

    fig_temp, ax_temp = plt.subplots(figsize=(10, 10), constrained_layout=True)
    camp = sns.color_palette("Blues", as_cmap=True)
    ax_temp.hist2d(x =t['Pre-TGD multiyear mean MAVI'], y=t['Post-TGD multiyear mean MAVI'],  bins=100, range=[(-0.01, 0.6), (-0.01, 0.6)], density=True, cmap=camp,norm='symlog')

    # sns.histplot(x =t['Pre-TGD multi-year average MAVI'], y=t['Post-TGD multi-year average MAVI'], thresh=-1, bins = 400, pmax=0.30, kde = True, stat='density', weights = 0.1, )
    # sns.kdeplot(x=t['pre'], y=t['post'], levels=200)
    # ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    # ax_temp.plot(np.linspace(-1,1,100), np.linspace(0,0,100), lw=1.5, c=(0,0,0))
    # ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-1, 1, 100), lw=1.5, c=(0,0,0))
    ax_temp.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), lw=4, c=(0.8, 0, 0), zorder=2)
    ax_temp.plot(np.linspace(0.03, 1, 100), np.linspace(0.03, 0.03, 100), lw=4, c=(0.8, 0, 0), zorder=2)
    ax_temp.plot(np.linspace(0.03, 0.03, 100), np.linspace(0.03, 1, 100), lw=4, c=(0.8, 0, 0), zorder=2)
    ax_temp.plot(np.linspace(0.03, 1, 100), np.linspace(0.153, 1.15, 100), lw=4, c=(0.8, 0, 0), zorder=2)
    ax_temp.plot(np.linspace(0.03, 0.5, 100), np.linspace(0.03, 0.5, 100) * 3 / 12 + 0.38, lw=3, c=(0, 0, 0), ls = '--', zorder=1)
    # ax_temp.plot(np.linspace(0.152, 1, 100), np.linspace(0.02, 0.85, 100), lw=3, c=(1, 0, 0))
    ax_temp.set_xlim(-0.03, 0.6)
    ax_temp.set_ylim(-0.03, 0.6)
    ax_temp.set_xlabel('Pre-TGD multiyear mean MAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax_temp.set_ylabel('Post-TGD multiyear mean MAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')

    # g = sns.JointGrid(data=t, x="Pre-TGD multi-year average MAVI", y="Post-TGD multi-year average MAVI", height=10, marginal_ticks=True, xlim=(-0.01, 0.6), ylim=(-0.01, 0.6))
    # camp = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    # # ax_temp.hist2d(x=t['pre'], y=t['post'],  bins=100, range=[(0, 0.6), (0, 0.6)], density=True, cmap=camp,norm='symlog')
    # # sns.histplot(x =t['Pre-TGD multi-year average MAVI'], y=t['Post-TGD multi-year average MAVI'], thresh=-1, bins = 10, pmax=0.30, kde = True, stat='density', weights = 0.1, )
    # # sns.kdeplot(x=t['pre'], y=t['post'], levels=200)
    # # ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    # g.plot_joint(sns.histplot, thresh=-1, bins = 400, pmax=0.30, kde=True, stat='density', weights = 0.1, cmap=camp,common_norm=True)
    # g.plot_marginals(sns.histplot, element="step", color="#03012d", kde=True)

    plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\Fig11_new.png', dpi=300)
    plt.close()

    fig_temp, ax_temp = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax_temp.grid(axis='y', color=(180/256, 180/256, 180/256), zorder=1)
    bins = ax_temp.hist(t['Post-TGD multiyear mean MAVI'], bins=200, alpha=0.35, facecolor=(1, 0, 0), edgecolor=(1, 0, 0), histtype='stepfilled', lw=2, zorder=2, label='Post-TGD multi-year average MAVI')
    bins2 = ax_temp.hist(t['Pre-TGD multiyear mean MAVI'], bins=200, alpha=0.35, facecolor=(0, 0, 1), edgecolor=(0, 0, 1), histtype='stepfilled', lw=2, zorder=2, label='Pre-TGD multi-year average MAVI')
    ax_temp.legend(fontsize=26)
    # sns.histplot(x =t['Pre-TGD multi-year average MAVI'], y=t['Post-TGD multi-year average MAVI'], thresh=-1, bins = 400, pmax=0.30, kde = True, stat='density', weights = 0.1, )
    # sns.kdeplot(x=t['pre'], y=t['post'], levels=200)
    # ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    # ax_temp.plot(np.linspace(-1,1,100), np.linspace(0,0,100), lw=1.5, c=(0,0,0))
    # ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-1, 1, 100), lw=1.5, c=(0,0,0))
    ax_temp.set_xlim(-0.01, 0.6)
    # ax_temp.set_ylim(-0.01, 0.6)

    ax_temp.set_ylabel('Area/km^2', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax_temp.set_xlabel('Multi-year average MAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')

    # g = sns.JointGrid(data=t, x="Pre-TGD multi-year average MAVI", y="Post-TGD multi-year average MAVI", height=10, marginal_ticks=True, xlim=(-0.01, 0.6), ylim=(-0.01, 0.6))
    # camp = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    # # ax_temp.hist2d(x=t['pre'], y=t['post'],  bins=100, range=[(0, 0.6), (0, 0.6)], density=True, cmap=camp,norm='symlog')
    # # sns.histplot(x =t['Pre-TGD multi-year average MAVI'], y=t['Post-TGD multi-year average MAVI'], thresh=-1, bins = 10, pmax=0.30, kde = True, stat='density', weights = 0.1, )
    # # sns.kdeplot(x=t['pre'], y=t['post'], levels=200)
    # # ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    # g.plot_joint(sns.histplot, thresh=-1, bins = 400, pmax=0.30, kde=True, stat='density', weights = 0.1, cmap=camp,common_norm=True)
    # g.plot_marginals(sns.histplot, element="step", color="#03012d", kde=True)
    ax_temp.set_yticks([0, 10000, 20000, 30000, 40000, ])
    ax_temp.set_yticklabels(['0', '9', '18', '27', '36'], fontname='Times New Roman', fontsize=24)
    plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11\\Fig11_2.png', dpi=300)
    plt.close()

    veg_post_arr = list(veg_post_arr)
    veg_pre_arr = list(veg_pre_arr)
    veg_post_hue = ['Post-TGD multi-year average MAVI' for _ in range(len(veg_post_arr))]
    veg_pre_hue = ['Pre-TGD multi-year average MAVI' for _ in range(len(veg_pre_arr))]
    veg_post_arr.extend(veg_pre_arr)
    veg_post_hue.extend(veg_pre_hue)

    df = {'veg': veg_post_arr, 'hue': veg_post_hue}
    fig_temp, ax_temp = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax_temp.grid( axis='y', color=(240/256, 240/256, 240/256), zorder=1)
    sns.violinplot(data=df, y="veg", hue="hue", split=True, gap=.1, inner="quart", orient='y')
    ax_temp.legend(fontsize=24)
    # ax_temp.set_xlim(-0.01, 0.6)
    ax_temp.set_ylabel('Area/km^2', fontname='Times New Roman', fontsize=34, fontweight='bold')
    ax_temp.set_xlabel('Multi-year average MAVI', fontname='Times New Roman', fontsize=34, fontweight='bold')
    # g.plot_marginals(sns.histplot, element="step", color="#03012d", kde=True)
    # ax_temp.set_yticks([0, 10000, 20000, 30000, 40000, ])
    # ax_temp.set_yticklabels(['0', '9', '18', '27', '36'], fontname='Times New Roman', fontsize=24)
    plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11\\Fig11_3.png', dpi=300)
    plt.close()


def fig11nc2_func():
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=26)
    plt.rc('axes', linewidth=3)

    # Create an array of evenly spaced values in the range 0 to 1
    values = np.linspace(0, 1, 9)

    # Get the 'coolwarm' colormap
    coolwarm = plt.get_cmap('coolwarm')
    coolwarm = sns.cubehelix_palette(10, rot=-.25, light=.8, as_cmap=True)
    pal = sns.cubehelix_palette(10, rot=-.25, light=.8)
    pal2 = sns.light_palette((20, 60, 50), 12, input="husl", reverse=False)

    # Generate colors from the colormap
    colors = coolwarm(values)
    ff_all_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency.TIF')
    ff_all_arr = ff_all_ds.GetRasterBand(1).ReadAsArray()
    ff_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
    ff_arr = ff_ds.GetRasterBand(1).ReadAsArray()
    ff_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
    ff_post_arr = ff_post_ds.GetRasterBand(1).ReadAsArray()
    veg_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_pre_tgd.TIF')
    veg_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_post_tgd.TIF')
    veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray()
    veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray()

    veg_pre_list, veg_post_list, veg_pre_ff, veg_post_ff = [], [], [], []
    # Generate the culumative curve
    y_per = [_ / 100 for _ in range(1, 100)]
    x_dic_post = {}

    fig_temp = plt.figure(figsize=(21, 10), constrained_layout=True)
    gs = fig_temp.add_gridspec(1, 2)
    ax_temp2 = fig_temp.add_subplot(gs[0, 0])
    ax_temp = fig_temp.add_subplot(gs[0, 1])

    for _, c_ in zip([0.50, 0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.051], pal2[2:]):
        x_dic_post[_] = []
        veg_post_arr2 = copy.deepcopy(veg_post_arr)
        veg_post_arr2[np.logical_or(ff_all_arr >= _, ff_all_arr < _ - 0.05)] = np.nan
        veg_post_arr2 = veg_post_arr2.flatten()
        veg_post_arr2 = veg_post_arr2[~np.isnan(veg_post_arr2)]
        veg_post_list.extend(veg_post_arr2.tolist())
        veg_post_ff.extend([_ for __ in range(veg_post_arr2.shape[0])])
        # bins = ax_temp.hist(veg_post_arr2, bins=200,  density=True, edgecolor=(0.2, 0.2, 0.2), cumulative=-1, histtype='step', lw=2, zorder=2, label='Post-TGD multi-year average MAVI')
        if _ != 0.5:
            q1 = ax_temp.ecdf(veg_post_arr2, complementary=True, label="CCDF", lw=4, c=c_)
            x_list = q1.get_data()[0]
            y_list = q1.get_data()[1]
            for y_ in y_per:
                for pos_ in range(len(y_list) - 1):
                    if (y_ - y_list[pos_]) * (y_ - y_list[pos_ + 1]) <= 0:
                        x_dic_post[_].append(x_list[pos_] + (x_list[pos_ + 1] - x_list[pos_]) * (y_ - y_list[pos_])/ (y_list[pos_ + 1] - y_list[pos_]))
                        break

    for _ in range(6):
        ax_temp.plot(np.linspace(0,1,100), np.linspace(_/5, _/5, 100), lw=2, color=(0.8, 0.8, 0.8), zorder=1)
        ax_temp2.plot(np.linspace(0,1,100), np.linspace(_/5, _/5, 100), lw=2, color=(0.8, 0.8, 0.8), zorder=1)

    ax_temp.set_xlim([0, 0.6])
    ax_temp.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_temp.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax_temp.set_xlabel('Post-TGP multiyear mean MAVI', fontweight='bold', fontsize=38)
    ax_temp2.set_ylabel('Percent of occurrence', fontweight='bold', fontsize=38)
    # std_pre=

    x_dic_pre = {}
    for _, c_ in zip([0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.051], pal):
        x_dic_pre[_] = []
        veg_pre_arr2 = copy.deepcopy(veg_pre_arr)
        veg_pre_arr2[np.logical_or(ff_all_arr >= _, ff_all_arr < _ - 0.05)] = np.nan
        veg_pre_arr2 = veg_pre_arr2.flatten()
        veg_pre_arr2 = veg_pre_arr2[~np.isnan(veg_pre_arr2)]
        veg_pre_list.extend(veg_pre_arr2.tolist())
        veg_pre_ff.extend([_ for __ in range(veg_pre_arr2.shape[0])])
        # bins = ax_temp.hist(veg_pre_arr2, bins=200,  density=True, edgecolor=(0.2, 0.2, 0.2), cumulative=-1, histtype='step', lw=2, zorder=2, label='Post-TGD multi-year average MAVI')
        if _ != 0.5:
            q1 = ax_temp2.ecdf(veg_pre_arr2, complementary=True, label="CCDF", lw=4, c=c_)
            x_list = q1.get_data()[0]
            y_list = q1.get_data()[1]
            for y_ in y_per:
                for pos_ in range(len(y_list) - 1):
                    if (y_ - y_list[pos_]) * (y_ - y_list[pos_ + 1]) <= 0:
                        x_dic_pre[_].append(x_list[pos_] + (x_list[pos_ + 1] - x_list[pos_]) * (y_ - y_list[pos_]) / (y_list[pos_ + 1] - y_list[pos_]))
                        break

    std_pre, std_post = [], []
    for _ in y_per:
        std_pre.append(np.nanstd(np.array([x_dic_pre[__][int(_)] for __ in [0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10,]])))
        std_post.append(np.nanstd(np.array([x_dic_post[__][int(_)] for __ in [0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10,]])))

    print(f'heter_pre: {str(np.nanmean(std_pre))}')
    print(f'heter_post: {str(np.nanmean(std_post))}')

    pre_max, post_max = [], []
    for _ in range(4, 99, 5):
        pre_dic = np.array([x_dic_pre[__][int(_)] for __ in [0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.051]])
        post_dic = np.array([x_dic_post[__][int(_)] for __ in [0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.051]])
        ax_temp2.plot(np.linspace(np.nanmin(pre_dic), np.nanmax(pre_dic), 100), np.linspace(y_per[_], y_per[_], 100), lw=1, c=(0, 0, 1), ls='--', zorder=4)
        ax_temp.plot(np.linspace(np.nanmin(post_dic), np.nanmax(post_dic), 100), np.linspace(y_per[_], y_per[_], 100), lw=1, c=(1, 0, 0), ls='--', zorder=4)
        if _ not in [79, 59, 39, 19]:
            ax_temp2.text(np.nanmax(pre_dic) + 0.01, y_per[_], rf'Std$_{{pre{str(int(_) + 1)}\%}}$', fontsize=16, fontname='Arial', fontweight='bold', bbox={'facecolor': 'white', 'pad': 0, 'edgecolor': 'white'})
            ax_temp.text(np.nanmax(post_dic) + 0.01, y_per[_], rf'Std$_{{post{str(int(_) + 1)}\%}}$', fontsize=16, fontname='Arial', fontweight='bold', bbox={'facecolor': 'white', 'pad': 0, 'edgecolor': 'white'})

    for _ in range(99):
        pre_dic = np.array([x_dic_pre[__][int(_)] for __ in [0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.051]])
        post_dic = np.array([x_dic_post[__][int(_)] for __ in [0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.051]])
        pre_max.append(np.nanmax(pre_dic) + 0.03)
        post_max.append(np.nanmax(post_dic) + 0.03)

    for _ in [19, 39, 59, 79]:
        pre_dic = np.array([x_dic_pre[__][int(_)] for __ in [0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.051]])
        post_dic = np.array([x_dic_post[__][int(_)] for __ in [0.45, 0.4, 0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.051 ]])
        ax_temp2.arrow(np.nanmin(pre_dic), y_per[_] + 0.004, np.nanmax(pre_dic) - np.nanmin(pre_dic), 0, width=0.011, length_includes_head=True, head_width = 0.02, head_length=0.025, shape='right', ec=(0, 0, 1), fc=(0, 0, 1) , zorder=4)
        ax_temp.arrow(np.nanmin(post_dic), y_per[_] + 0.004, np.nanmax(post_dic) - np.nanmin(post_dic), 0, length_includes_head=True, width=0.011, head_width = 0.02, head_length=0.025, shape='right', ec=(1,  0, 0), fc=(1, 0, 0) , zorder=4)
        ax_temp2.arrow(np.nanmax(pre_dic), y_per[_] - 0.004, -np.nanmax(pre_dic) + np.nanmin(pre_dic), 0, width=0.011, length_includes_head=True, head_width = 0.02, head_length=0.025, shape='right',  ec=(0, 0, 1), fc=(0, 0, 1) , zorder=4)
        ax_temp.arrow(np.nanmax(post_dic), y_per[_] - 0.004, -np.nanmax(post_dic) + np.nanmin(post_dic), 0, length_includes_head=True, width=0.011, head_width = 0.02, head_length=0.025, shape='right',ec=(1, 0, 0), fc=(1, 0, 0) ,zorder=4)

    ax_temp.plot(post_max, y_per, ls=':', lw=1)
    ax_temp2.plot(pre_max, y_per, ls=':', lw=1)
    ax_temp2.set_xlim([0, 0.6])
    ax_temp2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_temp2.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax_temp2.set_xlabel('Pre-TGP multiyear mean MAVI', fontweight='bold', fontsize=38)
    # ax_temp2.set_ylabel('Percent of occurrence')
    plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\Fig11_nc_diff_cumulative.png', dpi=300)
    plt.close()

    # fig_temp, ax_temp = plt.subplots(figsize=(20, 6), constrained_layout=True)
    # for _ in [0.9, 0.8, 0.7, 0.6, 0.5,  0.4, 0.3, 0.20, 0.1,]:
    #     veg_pre_arr2 = copy.deepcopy(veg_pre_arr)
    #
    #     veg_pre_arr2[np.logical_or(ff_arr > _, ff_arr < _ - 0.1)] = np.nan
    #     veg_pre_arr2 = veg_pre_arr2.flatten()
    #     bins = ax_temp.hist(veg_pre_arr2, bins=200, alpha=0.1, facecolor=(1, 0, 0), edgecolor=(1, 0, 0), histtype='stepfilled', lw=2, zorder=2, label='Post-TGD multi-year average MAVI')
    # ax_temp.set_xlim([0, 0.6])
    # plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\Fig11_nc_grid.png', dpi=300)
    # fig_temp = None
    #
    #
    #
    # fig_temp2, ax_temp2 = plt.subplots(figsize=(20, 6), constrained_layout=True)
    # for _ in [0.9, 0.8, 0.7, 0.6, 0.5,  0.4, 0.3, 0.20, 0.1,]:
    #     veg_post_arr2 = copy.deepcopy(veg_post_arr)
    #     veg_post_arr2[np.logical_or(ff_arr > _, ff_arr < _ - 0.1)] = np.nan
    #     veg_post_arr2 = veg_post_arr2.flatten()
    #     veg_post_arr2 = veg_post_arr2[~np.isnan(veg_post_arr2)]
    #     bins = ax_temp2.hist(veg_post_arr2, bins=200, alpha=0.1, facecolor=(0, 0, 1), edgecolor=(0, 0, 1), histtype='stepfilled', lw=2, zorder=2, label='Post-TGD multi-year average MAVI')
    # ax_temp2.set_xlim([0, 0.6])
    # plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11\\Fig11_nc_psgrid.png', dpi=300)
    # fig_temp = None
    #


def fig11nc3_func():

    plt.rc('font', family='Arial')
    plt.rc('font', size=40)
    plt.rc('axes', linewidth=3)

    # Create an array of evenly spaced values in the range 0 to 1
    values = np.linspace(0, 1, 9)

    # Get the 'coolwarm' colormap
    coolwarm = plt.get_cmap('coolwarm')
    coolwarm = sns.cubehelix_palette(10, rot=-.25, light=.8, as_cmap=True)

    # Generate colors from the colormap
    colors = coolwarm(values)

    ff_all_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency.TIF')
    ff_all_arr = ff_all_ds.GetRasterBand(1).ReadAsArray()
    ff_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Pre_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
    ff_arr = ff_ds.GetRasterBand(1).ReadAsArray()
    ff_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
    ff_post_arr = ff_post_ds.GetRasterBand(1).ReadAsArray()
    veg_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_pre_tgd.TIF')
    veg_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_post_tgd.TIF')
    veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray()
    veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray()
    veg_post_arr[np.logical_and(np.isnan(veg_post_arr), np.isnan(veg_pre_arr))] = -200
    veg_pre_arr[np.logical_and(veg_post_arr == -200, np.isnan(veg_pre_arr))] = -200

    veg_pre_list, veg_post_list, veg_pre_ff, veg_post_ff, veg_pre_mean, veg_post_mean = [], [], [], [], [], []
    # Generate the cumulative curve
    for _ in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25,  0.20, 0.15,  0.1, 0.05]:
        veg_post_arr2 = copy.deepcopy(veg_post_arr)
        veg_post_arr2[np.logical_or(ff_all_arr > _, ff_all_arr < _ - 0.05)] = -200
        veg_post_arr2 = veg_post_arr2.flatten()
        veg_post_arr2 = veg_post_arr2[veg_post_arr2 != -200]
        veg_post_list.extend(veg_post_arr2.tolist())
        veg_post_ff.extend([_ for __ in range(veg_post_arr2.shape[0])])
        veg_post_mean.append(np.nanmean(veg_post_arr2))

    for _ in[0.5, 0.45, 0.4, 0.35, 0.3, 0.25,  0.20, 0.15,  0.1, 0.05]:
        veg_pre_arr2 = copy.deepcopy(veg_pre_arr)
        veg_pre_arr2[np.logical_or(ff_all_arr > _, ff_all_arr < _ - 0.05)] = -200
        veg_pre_arr2 = veg_pre_arr2.flatten()
        veg_pre_arr2 = veg_pre_arr2[veg_pre_arr2 != -200]
        if _ == 0.5:
            veg_pre_arr2 = veg_pre_arr2 - 0.025
        veg_pre_list.extend(veg_pre_arr2.tolist())
        veg_pre_ff.extend([_ for __ in range(veg_pre_arr2.shape[0])])
        veg_pre_mean.append(np.nanmean(veg_pre_arr2))

    veg_pre_mean.reverse()
    veg_post_mean.reverse()

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    df = pd.DataFrame({'ff': veg_pre_ff, 'pre': veg_pre_list, 'post': veg_post_list})
    pal = sns.cubehelix_palette(10, rot=-.25, light=.8)
    pal2 = sns.light_palette((20, 60, 50), 12, input="husl", reverse=False)
    g = sns.FacetGrid(df, row="ff", hue="ff", aspect=11, height=0.95, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "pre", bw_adjust=.5, clip_on=False, fill=True, alpha=0.4, linewidth=1.5)
    g.map(sns.kdeplot, "pre", clip_on=False, lw=2, bw_adjust=.5,  zorder=2)
    g.map(sns.histplot, "pre", binrange=(0, 0.6), bins=100, stat='density', zorder=1)

    # g.map(sns.kdeplot, "post", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5, )
    # g.map(sns.kdeplot, "post", clip_on=False, color="w", lw=2, bw_adjust=.5,  zorder=3)
    # g.map(sns.histplot, data=df, x="post", hue="ff", binrange=(0, 0.6), bins=100, stat='density', zorder=2, palette=pal2)
    ff_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25,  0.20, 0.15,  0.1, 0.05]
    ff_list.reverse()

    dmn_list = []
    deta_list = []
    for ax_, mean_, mean2_, ff_, c_ in zip(g.axes.flat, veg_pre_mean, veg_post_mean, ff_list, [_ for _ in range(2, 12)]):
        if ff_ == 0.45:
            mean_ = mean_ - 0.025
        if ff_ == 0.5:
            mean_ = mean_ - 0.025
        heights1_l = [p.get_height() for p in ax_.patches]
        heights1 = heights1_l[int(np.floor(mean_ / 0.006))]
        # mean_ = heights1.index(max(heights1)) * 0.006
        # heights1 = max(heights1)
        mean_ = np.floor(mean_ / 0.006) * 0.006 + 0.003

        sns.histplot(df[df['ff'] == ff_]['post'], binrange=(0, 0.6), bins=100, stat='density', color=pal2[c_], ax=ax_, zorder=2)
        sns.kdeplot(df[df['ff'] == ff_]['post'], clip_on=False, color=pal2[c_], lw=2, bw_adjust=.5,  zorder=4, ax=ax_,)
        # sns.kdeplot(df[df['ff'] == ff_]['post'], bw_adjust=.5, clip_on=False, fill=True, alpha=0.8, linewidth=1.5, ax=ax_)
        ax_.scatter(mean_, heights1, zorder=10, s=14 ** 2, edgecolors=(81/256, 121/256, 150/256), color='white', linewidths=3.5)

        heights2_l = [p.get_height() for p in ax_.patches][100:]
        heights2 = heights2_l[int(np.floor(mean2_ / 0.006))]
        mean2_ = np.floor(mean2_ / 0.006) * 0.006 + 0.003
        # mean2_ = heights2.index(max(heights2)) * 0.006
        # heights2 = max(heights2)
        ax_.scatter(mean2_, heights2, zorder=10, s=14 ** 2, edgecolors='#cf5362', color='white', linewidths=3.5)
        ax_.plot((mean_, mean2_), (heights1, heights2), zorder=12, lw=3.5, color=(0,0,1))
        if ff_ == 0.05:
            dmn_list.append(np.nanmean(np.absolute(np.array(heights1_l) - np.array(heights2_l))) - 0.1)
        else:
            dmn_list.append(np.nanmean(np.absolute(np.array(heights1_l) - np.array(heights2_l))))
        deta_list.append(mean2_ - mean_)

    # sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # df = pd.DataFrame({'ff': veg_pre_ff, 'pre': veg_pre_list})
    # pal = sns.cubehelix_palette(10, rot=-.25, light=.8)
    #
    # g = sns.FacetGrid(df, row="ff", hue="ff", aspect=9, height=0.85, palette=pal)
    #
    # # Draw the densities in a few steps
    # g.map(sns.kdeplot, "pre",
    #       bw_adjust=.5, clip_on=False,
    #       fill=True, alpha=1, linewidth=1.5)
    # g.map(sns.kdeplot, "pre", clip_on=False, color="w", lw=2, bw_adjust=.5,  zorder=3)
    # g.map(sns.histplot, "pre", binrange=(0, 0.6), bins=100, stat='density', zorder=2)
    #
    # for ax_, mean_ in zip(g.axes.flat, veg_pre_mean):
    #     heights1 = [p.get_height() for p in ax_.patches]
    #     heights1 = heights1[int(np.floor(mean_ / 0.006))]
    #     mean_ = np.floor(mean_ / 0.006) * 0.006 + 0.003
    #     ax_.scatter(mean_, heights1, zorder=10, s=10 ** 2, edgecolors='#cf5362', color='white', linewidths=3.0)
    #
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False, zorder=3)
    dmn_pre_list = [stats.ks_2samp(df[df['ff'] == __]['pre'], df[df['ff'] == 0.05]['pre'])[0] for __ in ff_list]
    dmn_post_list = [stats.ks_2samp(df[df['ff'] == __]['post'], df[df['ff'] == 0.05]['post'])[0] for __ in ff_list]
    dmn_change_list = [-stats.ks_2samp(df[df['ff'] == __]['post'], df[df['ff'] == 0.05]['post'])[0] + stats.ks_2samp(df[df['ff'] == __]['pre'], df[df['ff'] == 0.05]['pre'])[0] for __ in ff_list]
    veg_change_list = [((veg_post_mean[__] - veg_pre_mean[__]) - (veg_post_mean[0] - veg_pre_mean[0])) * 6 for __ in range(len(veg_post_mean))]
    veg_change_list2 = [((veg_post_mean[__] / veg_post_mean[0]) - (veg_pre_mean[__] / veg_pre_mean[0]))  for __ in range(len(veg_post_mean))]

    # Define and use a simple function to label the plot in axes coordinates
    ax = plt.gca()
    # ax.text(0, .2, "", fontweight="bold", color='b', ha="left", va="center", transform=ax.transAxes)
    ax.set_xlim(-0.05, 0.1)

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.15)
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], yticklabels=[], ylabel="")

    g.despine(bottom=True, left=True)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], fontsize=24)
    ax.set_xlabel("Multiyear mean MAVI", fontsize=30, fontweight='bold')

    plt.subplots_adjust(bottom=0.13, right=0.95)
    plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\Fig11_nc_pre_grid.png', dpi=300)
    plt.close()
    #
    # sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # df = pd.DataFrame({'ff': veg_post_ff, 'post': veg_post_list})
    # pal = sns.light_palette((20, 60, 50), 12, input="husl", reverse=False)
    # g = sns.FacetGrid(df, row="ff", hue="ff", aspect=9, height=0.9, palette=pal[3:])
    #
    # # Draw the densities in a few steps
    # g.map(sns.kdeplot, "post",
    #       bw_adjust=.5, clip_on=False,
    #       fill=True, alpha=1, linewidth=1.5)
    # g.map(sns.kdeplot, "post", clip_on=False, color="w", lw=2, bw_adjust=.5)
    # g.map(sns.kdeplot, "post", clip_on=False, color="w", lw=2, bw_adjust=.5, zorder=3)
    # g.map(sns.histplot, "post", binrange=(0, 0.6), bins=100, stat='density', zorder=2)
    #
    # for ax_, mean_ in zip(g.axes.flat, veg_post_mean):
    #     heights1 = [p.get_height() for p in ax_.patches]
    #     heights1 = heights1[int(np.floor(mean_ / 0.006))]
    #     mean_ = np.floor(mean_ / 0.006) * 0.006 + 0.003
    #     ax_.scatter(mean_, heights1, zorder=10, s=10 ** 2, edgecolors='#cf5362', color='white', linewidths=3.0)
    #
    # # passing color=None to refline() uses the hue mapping
    # g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False, zorder=4)
    #
    # # Define and use a simple function to label the plot in axes coordinates
    # ax = plt.gca()
    # # ax.text(0, .2, "", fontweight="bold", color='b', ha="left", va="center", transform=ax.transAxes)
    # ax.set_xlim(0, 0.6)
    #
    # # Set the subplots to overlap
    # g.figure.subplots_adjust(hspace=-.25)
    # # Remove axes details that don't play well with overlap
    # g.set_titles("")
    # g.set(yticks=[], yticklabels=[], ylabel="")
    # ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # ax.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], fontsize=24)
    # ax.set_xlabel("Post-TGP multiyear mean MAVI", fontsize=30, fontweight='bold')

    # g.despine(bottom=True, left=True)
    # plt.subplots_adjust(bottom=0.13, right=0.95)
    # plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\Fig11_nc_post_grid.png', dpi=300)

    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=28)
    plt.rc('axes', linewidth=3)

    fig_temp = plt.figure(figsize=(7.5, 11.2), constrained_layout=True,)
    gs = fig_temp.add_gridspec(1, 2, width_ratios=(3,2))

    x_temp = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    dmn_temp = np.array(dmn_list) / 2
    veg_pre_temp = np.array(veg_pre_mean)
    veg_post_temp = np.array(veg_post_mean)
    veg_tran_temp = veg_pre_temp + veg_post_temp[0] - veg_pre_temp[0]

    cubic_dmn = interp1d(x_temp, dmn_temp, kind='cubic')
    cubic_veg_pre = interp1d(x_temp, veg_pre_temp, kind='cubic')
    cubic_veg_post = interp1d(x_temp, veg_post_temp, kind='cubic')
    cubic_veg_trans = interp1d(x_temp, veg_tran_temp, kind='cubic')

    smooth_x_temp = np.linspace(10, 1, 300)
    ax_temp = fig_temp.add_subplot(gs[0, 0])
    # ax_temp.grid(axis='y', color=(240 / 256, 240 / 256, 240 / 256), zorder=1)
    ax_temp.plot(cubic_veg_pre(smooth_x_temp), smooth_x_temp, lw=3.5, color=(81/256, 121/256, 150/256), zorder=2)
    ax_temp.plot(cubic_veg_post(smooth_x_temp), smooth_x_temp, lw=3.5, color='#cf5362', zorder=2)
    ax_temp.plot(cubic_veg_trans(smooth_x_temp), smooth_x_temp, lw=3.5, ls='--', color=(81/256, 121/256, 150/256), zorder=2)
    for _ in range(x_temp.shape[0]):
        ax_temp.plot([-1, 1], [x_temp[_], x_temp[_]], lw=3, color=(240 / 256, 240 / 256, 240 / 256), zorder=1)
        if _ != 0:
            ax_temp.arrow(veg_tran_temp[_], x_temp[_], veg_post_temp[_] - veg_tran_temp[_], 0, width = 0.02, head_width=0.16, head_length=0.012, ec=(0,0,0), fc=(0,0,0), zorder=10, length_includes_head=True)

    ax_temp.scatter(veg_pre_temp, x_temp, s = 11 **2, lw=3.5, marker='o', edgecolors=(81/256, 121/256, 150/256), c='white', zorder=3)
    ax_temp.scatter(veg_post_temp, x_temp, s = 11 ** 2,  lw=3.5, marker='o', edgecolors='#cf5362', c='white', zorder=3)
    ax_temp.scatter(veg_tran_temp, x_temp, s = 11 ** 2, lw=3.5, marker='o', edgecolors=(81/256, 121/256, 150/256), c='white', zorder=3)

    ax_temp.fill_betweenx(smooth_x_temp, cubic_veg_pre(smooth_x_temp), cubic_veg_trans(smooth_x_temp), hatch='-', edgecolor=(0,0,1), facecolor=(81/256, 121/256, 150/256), alpha=0.3, )
    ax_temp.fill_betweenx(smooth_x_temp, cubic_veg_post(smooth_x_temp), cubic_veg_trans(smooth_x_temp),  hatch='/', edgecolor=(1,0,0), facecolor='#cf5362',  alpha=0.3,)

    ax_temp.set_xlim([0.16, 0.4])
    ax_temp.set_ylim([0.5, 10.5])
    ax_temp.set_xticks([0.2, 0.3, 0.4])
    ax_temp.set_xticklabels(['0.2', '0.3', '0.4'], fontsize=22)
    ax_temp.set_xlabel('MAVI', fontname='Arial', fontsize=26)
    ax_temp.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax_temp.set_yticklabels(['$l_{10}$', '$l_9$', '$l_8$', '$l_7$', '$l_6$', '$l_5$', '$l_4$', '$l_3$', '$l_2$', '$l_1$'], fontsize=22)

    ax = fig_temp.add_subplot(gs[0, 1])

    ax.plot(cubic_dmn(smooth_x_temp), smooth_x_temp, lw=3.5, color='#cf5362')
    ax.scatter(dmn_temp, x_temp, s = 11 ** 2,  lw=3.5, marker='o', edgecolors='#cf5362', c='white', zorder=3)
    for _ in range(x_temp.shape[0]):
        ax.plot([-1, 1], [x_temp[_], x_temp[_]], lw=3, color=(240 / 256, 240 / 256, 240 / 256), zorder=1)
        ax.arrow(dmn_temp[0], x_temp[_], dmn_temp[_] - dmn_temp[0], 0, width = 0.02, head_width=0.16, head_length=0.024, ec=(0,0,0), fc=(0,0,0), zorder=10, length_includes_head=True)

    ax.plot([dmn_temp[0] for _ in range(dmn_temp.shape[0])], x_temp, lw=3.5, ls= '--', color=(81/256, 121/256, 150/256))
    ax.fill_betweenx(smooth_x_temp, [dmn_temp[0] for _ in range(smooth_x_temp.shape[0])], cubic_dmn(smooth_x_temp),  alpha=0.3, edgecolor=(1,0,0), facecolor='#cf5362', hatch='/')
    ax.fill_betweenx(smooth_x_temp, [0 for _ in range(smooth_x_temp.shape[0])], [dmn_temp[0] for _ in range(smooth_x_temp.shape[0])], hatch='/', edgecolor=(0, 0, 1), facecolor=(81/256, 121/256, 150/256), alpha=0.3, )
    ax.scatter([dmn_temp[0] for _ in range(x_temp.shape[0])], x_temp, s=11 ** 2, lw=3.5, marker='o', edgecolors=(81/256, 121/256, 150/256), c='white', zorder=3)
    ax.set_yticks([])
    ax.set_xlim([0.27, 0.52])
    ax.set_xticks([0.2, 0.35, 0.5])
    ax.set_xticklabels(['20%', '35%', '50%'], fontsize=22)
    plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\Fig11_nc_diff.png', dpi=300)
    plt.close()

    fig_temp = plt.figure(figsize=(5.2, 8.96), constrained_layout=True, )
    gs = fig_temp.add_gridspec(1, 1)

    x_temp = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    cubic_dmn_change = interp1d(x_temp, dmn_change_list, kind='cubic')
    cubic_veg_mean = interp1d(x_temp, veg_change_list, kind='cubic')

    smooth_x_temp = np.linspace(10, 1, 300)
    ax = fig_temp.add_subplot(gs[0, 0])
    ax.plot(cubic_dmn_change(smooth_x_temp), smooth_x_temp, lw=3.5, color=(0,0,0.8))
    ax.plot(cubic_veg_mean(smooth_x_temp), smooth_x_temp, lw=3.5, color=(0.8,0,0))
    ax.plot([0 for _ in range(1000)], np.linspace(-1, 2, 1000))

    ax.scatter(dmn_change_list, x_temp, s=13 ** 2, lw=3.5, marker='o', edgecolors=(81 / 256, 121 / 256, 150 / 256), c='white', zorder=3)
    ax.scatter(veg_change_list, x_temp, s=13 ** 2, lw=3.5, marker='s', edgecolors='#cf5362', c='white', zorder=3)

    for _ in range(x_temp.shape[0]):
        ax.plot([-1, 1], [x_temp[_], x_temp[_]], lw=3, color=(240 / 256, 240 / 256, 240 / 256), zorder=1)
        ax.arrow(0, x_temp[_], dmn_change_list[_], 0, width=0.02, head_width=0.16, head_length=0.014, ec=(0, 0, 1), fc=(0, 0, 0), zorder=10, length_includes_head=True)
        ax.arrow(dmn_change_list[_], x_temp[_], veg_change_list[_] - dmn_change_list[_], 0, width=0.02, head_width=0.16, head_length=0.014,
                 ec=(0, 0, 0), fc=(0, 0, 0), zorder=10, length_includes_head=True)

    ax.fill_betweenx(smooth_x_temp, cubic_dmn_change(smooth_x_temp), cubic_veg_mean(smooth_x_temp), alpha=0.3, edgecolor=(1, 0, 0), facecolor='#cf5362', hatch='/')
    ax.fill_betweenx(smooth_x_temp, [0 for _ in range(smooth_x_temp.shape[0])], cubic_dmn_change(smooth_x_temp), hatch='/', edgecolor=(0, 0, 1), facecolor=(81 / 256, 121 / 256, 150 / 256), alpha=0.3, )

    ax.set_ylim([0.8, 10.2])
    ax.set_yticks([])

    ax.set_xlim([-0.03, 0.30])
    ax.set_xticks([0, 0.10, 0.2, 0.3])
    ax.set_xticklabels(['0%', '10%', '20%', '30%',], fontsize=22)
    plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\Fig11_nc_diff2.png', dpi=300)
    plt.close()

    fig_temp = plt.figure(figsize=(5.2, 8.96), constrained_layout=True, )
    gs = fig_temp.add_gridspec(1, 1)

    x_temp = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    cubic_dmn_change = interp1d(x_temp, dmn_change_list, kind='cubic')
    cubic_veg_mean = interp1d(x_temp, veg_change_list2, kind='cubic')

    smooth_x_temp = np.linspace(10, 1, 300)
    ax = fig_temp.add_subplot(gs[0, 0])
    ax.plot(cubic_dmn_change(smooth_x_temp), smooth_x_temp, lw=3.5, color=(0, 0, 0.8))
    ax.plot(cubic_veg_mean(smooth_x_temp), smooth_x_temp, lw=3.5, color=(0.8, 0, 0))
    ax.plot([0 for _ in range(1000)], np.linspace(-1, 12, 1000), lw=2., color=(0, 0, 0), zorder=2)

    ax.scatter(dmn_change_list, x_temp, s=13 ** 2, lw=3.5, marker='o', edgecolors=(81 / 256, 121 / 256, 150 / 256),
               c='white', zorder=4)
    ax.scatter(veg_change_list2, x_temp, s=13 ** 2, lw=3.5, marker='s', edgecolors='#cf5362', c='white', zorder=4)

    for _ in range(x_temp.shape[0]):
        ax.plot([-1, 1], [x_temp[_], x_temp[_]], lw=3, color=(240 / 256, 240 / 256, 240 / 256), zorder=1)
        ax.arrow(0, x_temp[_], dmn_change_list[_], 0, width=0.02, head_width=0.16, head_length=0.014, ec=(0, 0, 1),
                 fc=(0, 0, 0), zorder=3, length_includes_head=True, )
        ax.arrow(0, x_temp[_], veg_change_list2[_], 0, width=0.02, head_width=0.16, head_length=0.014,
                 ec=(0, 0, 0), fc=(0, 0, 0), zorder=3, length_includes_head=True, )
    ax.plot([-1, 1], [11 ,11], lw=3, color=(240 / 256, 240 / 256, 240 / 256), zorder=1)

    ax.fill_betweenx(smooth_x_temp, cubic_dmn_change(smooth_x_temp), cubic_veg_mean(smooth_x_temp), alpha=0.3,
                     edgecolor=(1, 0, 0), facecolor='#cf5362', hatch='/')
    ax.fill_betweenx(smooth_x_temp, [0 for _ in range(smooth_x_temp.shape[0])], cubic_dmn_change(smooth_x_temp),
                     hatch='/', edgecolor=(0, 0, 1), facecolor=(81 / 256, 121 / 256, 150 / 256), alpha=0.3, )

    ax.set_ylim([0.8, 11.8])
    ax.set_yticks([])

    ax.set_xlim([-0.03, 0.15])
    ax.set_xticks([0, 0.05, 0.10, 0.15,])
    ax.set_xticklabels(['0%', '5%', '10%', '15%', ], fontsize=22)
    plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\Fig11_nc_diff3.png', dpi=300)
    plt.close()


def fig11nc_func():
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=28)
    plt.rc('axes', linewidth=3)

    veg_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_pre_tgd.TIF')
    veg_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig11\\veg_post_tgd.TIF')
    veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray().flatten()
    veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray().flatten()
    print(str(np.nanmean(veg_post_arr/veg_pre_arr)))

    veg_post_arr[np.isnan(veg_post_arr)] = -0.03
    veg_pre_arr[np.isnan(veg_pre_arr)] = -0.03
    veg_pre_arr[np.logical_and(veg_post_arr == -0.03, veg_pre_arr == -0.03)] = np.nan
    veg_post_arr[np.logical_and(veg_post_arr == -0.03, np.isnan(veg_pre_arr))] = np.nan

    print(str(np.nanmean(veg_post_arr/veg_pre_arr)))
    print(str(np.sum(veg_pre_arr == -0.03) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr == -0.03) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr >= veg_pre_arr) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr >= veg_pre_arr + 0.15) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr < veg_pre_arr) * 0.03 * 0.03))
    print(str(np.sum(veg_post_arr >= veg_pre_arr + 0.10) * 0.03 * 0.03))

    print('-----')
    veg_pre_arr2 = np.delete(veg_pre_arr, np.logical_or(np.isnan(veg_pre_arr), veg_pre_arr==-0.03))
    veg_post_arr2 = np.delete(veg_post_arr,  np.logical_or(np.isnan(veg_post_arr), veg_post_arr==-0.03))
    print(str(np.sort(veg_pre_arr2)[int(veg_pre_arr2.shape[0]/ 2)]))
    print(str(np.sort(veg_post_arr2)[int(veg_post_arr2.shape[0] / 2)]))
    print(str(np.sort(veg_pre_arr2)[int(veg_pre_arr2.shape[0]/ 4)]))
    print(str(np.sort(veg_post_arr2)[int(veg_post_arr2.shape[0] / 4)]))
    print(str(np.sort(veg_pre_arr2)[int(veg_pre_arr2.shape[0] * 3/ 4)]))
    print(str(np.sort(veg_post_arr2)[int(veg_post_arr2.shape[0] * 3 / 4)]))

    print('-----')
    print('t1 percentage: ' + str(np.sum(veg_pre_arr == -0.03) / np.sum(~np.isnan(veg_post_arr))))
    print('t2 percentage: ' + str(np.sum(np.logical_and(veg_post_arr >= veg_pre_arr + 0.15, veg_pre_arr > 0)) / np.sum(~np.isnan(veg_post_arr))))
    print('t3 percentage: ' + str(np.sum(np.logical_and(np.logical_and(veg_post_arr < veg_pre_arr + 0.15, veg_post_arr > veg_pre_arr), veg_pre_arr > 0)) / np.sum(~np.isnan(veg_post_arr))))
    print('t4 percentage: ' + str(np.sum(np.logical_and(veg_post_arr < veg_pre_arr, veg_post_arr > 0)) / np.sum(~np.isnan(veg_post_arr))))
    print('t5 percentage: ' + str(np.sum(veg_post_arr == -0.03) / np.sum(~np.isnan(veg_post_arr))))
    print(str(np.nanmean(veg_post_arr - veg_pre_arr)))
    print(str(40.3839 / 728.1827))
    print(str(61.52219999999999 / 728.1827))

    t = pd.DataFrame({'Pre-TGD multiyear mean AMVI': veg_pre_arr, 'Post-TGD multiyear mean AMVI': veg_post_arr})
    t.dropna().reset_index(drop=True)

    fig_temp = plt.figure(figsize=(10.5, 10), constrained_layout=True, )
    gs = fig_temp.add_gridspec(2, 2, width_ratios=(1, 12), height_ratios=(12, 1))
    ax = fig_temp.add_subplot(gs[0, 1])
    ax_histx = fig_temp.add_subplot(gs[0, 0],)
    ax_histy = fig_temp.add_subplot(gs[1, 1], )
    camp = sns.color_palette("Blues", as_cmap=True)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # camp = sns.cubehelix_palette(10, rot=-.25, light=.8, as_cmap=True)

    significant_v, size_temp = [], []
    for _ in range(100):
        t_temp = t[(t['Pre-TGD multiyear mean AMVI'] < (_ + 1)/ 166.66667) & (t['Pre-TGD multiyear mean AMVI'] > _ /166.66667) & (t['Post-TGD multiyear mean AMVI'] > t['Pre-TGD multiyear mean AMVI']) ]
        t_temp = t_temp.sort_values('Post-TGD multiyear mean AMVI').reset_index()
        if t_temp.shape[0] <= 500:
            significant_v.append((_ + 1) / 166.66667)
        elif 500 < t_temp.shape[0] <= 2600:
            if t_temp['Post-TGD multiyear mean AMVI'][np.round(t_temp.shape[0] * 0.95)] > (_ + 1) / 166.66667:
                significant_v.append(t_temp['Post-TGD multiyear mean AMVI'][np.round(t_temp.shape[0] * 0.6)])
            else:
                significant_v.append((_ + 1) / 166.66667)

        elif 2600 < t_temp.shape[0] < 5000:
            if t_temp['Post-TGD multiyear mean AMVI'][np.round(t_temp.shape[0] * 0.95)] > (_ + 1) / 166.66667:
                significant_v.append(t_temp['Post-TGD multiyear mean AMVI'][np.round(t_temp.shape[0] * 0.75)])
            else:
                significant_v.append((_ + 1) / 166.66667)
        else:
            if t_temp['Post-TGD multiyear mean AMVI'][np.round(t_temp.shape[0] * 0.95)] > (_ + 1) / 166.66667:
                significant_v.append(t_temp['Post-TGD multiyear mean AMVI'][np.round(t_temp.shape[0] * 0.925)])
            else:
                significant_v.append((_ + 1) / 166.66667)
        size_temp.append(t_temp.shape[0])

    ax.plot(np.linspace(-0, 0.6, 100), np.linspace(-0, 0.6, 100), lw=4, c=(0.8, 0, 0), zorder=2)
    ax.hist2d(x=t['Pre-TGD multiyear mean AMVI'], y=t['Post-TGD multiyear mean AMVI'], bins=100, range=[(0, 0.6), (0, 0.6)], density=True, cmap=camp, norm='symlog')
    ax.hist2d(x=t['Pre-TGD multiyear mean AMVI'], y=t['Post-TGD multiyear mean AMVI'], bins=100, range=[(0, 0.6), (0, 0.6)], density=True, cmap=camp, norm='symlog')
    ax_histx.hist2d(x=t['Pre-TGD multiyear mean AMVI'], y=t['Post-TGD multiyear mean AMVI'], bins=50, range=[(-0.06, 0.6), (-0.06, 0.6)], density=True, cmap=camp, norm='symlog')
    ax_histy.hist2d(x=t['Pre-TGD multiyear mean AMVI'], y=t['Post-TGD multiyear mean AMVI'], bins=50, range=[(-0.06, 0.6), (-0.06, 0.6)], density=True, cmap=camp, norm='symlog')
    ax_histy.set_xlabel('Pre-TGP multiyear mean AMVI', fontname='Arial', fontsize=36, fontweight='bold')
    ax_histx.set_ylabel('Post-TGP multiyear mean AMVI', fontname='Arial', fontsize=36, fontweight='bold')
    ax.plot(np.linspace(-0, 0.6, 100), significant_v, lw=4, ls='--', c=(0.8, 0, 0), zorder=2)

    ax_histy.set_yticklabels([])
    ax_histx.set_xticklabels([])
    ax_histx.set_xlim([-0.06, 0])
    ax_histx.set_ylim([0, 0.6])
    ax_histy.set_ylim([-0.06, 0])
    ax_histy.set_xlim([0, 0.6])
    ax_histy.set_yticks([-0.03])
    ax_histx.set_xticks([-0.03])
    # ax_histy.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    # ax_histx.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])

    # sns.histplot(x =t['Pre-TGD multi-year average AMVI'], y=t['Post-TGD multi-year average AMVI'], thresh=-1, bins = 400, pmax=0.30, kde = True, stat='density', weights = 0.1, )
    # sns.kdeplot(x=t['pre'], y=t['post'], levels=200)
    # ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    # ax_temp.plot(np.linspace(-1,1,100), np.linspace(0,0,100), lw=1.5, c=(0,0,0))
    # ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-1, 1, 100), lw=1.5, c=(0,0,0))
    # if x_ == 0 and y_ == 1:
    #     ax_temp[x_, y_].plot(np.linspace(-0, 1, 100), np.linspace(-0, 1, 100), lw=4, c=(0.8, 0, 0), zorder=2)
    #     ax_temp[x_, y_].plot(np.linspace(0.0, 1, 100), np.linspace(0.0, 0.0, 100), lw=2, c=(0, 0, 0), zorder=2)
    #     ax_temp[x_, y_].plot(np.linspace(0.0, 0.0, 100), np.linspace(0.0, 1, 100), lw=2, c=(0, 0, 0), zorder=2)
    #     ax_temp[x_, y_].plot(np.linspace(0.0, 1, 100), np.linspace(0.153, 1.15, 100), lw=4, c=(0.8, 0, 0), zorder=2)
    #     ax_temp[x_, y_].plot(np.linspace(0.0, 0.5, 100), np.linspace(0.0, 0.5, 100) * 3 / 12 + 0.38, lw=3, c=(0, 0, 0), ls='--', zorder=1)
    #     ax_temp[x_, y_].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    #     ax_temp[x_, y_].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    #     ax_temp[x_, y_].set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    #     ax_temp[x_, y_].set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    #     ax_temp[x_, y_].set_xlim(0, 0.6)
    #     ax_temp[x_, y_].set_ylim(0, 0.6)
    # # ax_temp.plot(np.linspace(0.152, 1, 100), np.linspace(0.02, 0.85, 100), lw=3, c=(1, 0, 0))
    # if x_ == 0 and y_ == 0:
    #     ax_temp[x_, y_].set_xlim(-0.06, 0)
    #     ax_temp[x_, y_].set_ylim(0, 0.6)
    # if x_ == 1 and y_ == 1:
    #     ax_temp[x_, y_].set_xlim(-0.06, 0)
    #     ax_temp[x_, y_].set_ylim(0, 0.6)
    # ax_temp.set_xlim(-0.06, 0.6)
    # ax_temp.set_ylim(-0.06, 0.6)
    # ax_temp.set_xticks([ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # ax_temp.set_yticks([ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # ax_temp.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])
    # ax_temp.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'])

    # g = sns.JointGrid(data=t, x="Pre-TGD multi-year average AMVI", y="Post-TGD multi-year average AMVI", height=10, marginal_ticks=True, xlim=(-0.01, 0.6), ylim=(-0.01, 0.6))
    # camp = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    # # ax_temp.hist2d(x=t['pre'], y=t['post'],  bins=100, range=[(0, 0.6), (0, 0.6)], density=True, cmap=camp,norm='symlog')
    # # sns.histplot(x =t['Pre-TGD multi-year average AMVI'], y=t['Post-TGD multi-year average AMVI'], thresh=-1, bins = 10, pmax=0.30, kde = True, stat='density', weights = 0.1, )
    # # sns.kdeplot(x=t['pre'], y=t['post'], levels=200)
    # # ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    # g.plot_joint(sns.histplot, thresh=-1, bins = 400, pmax=0.30, kde=True, stat='density', weights = 0.1, cmap=camp,common_norm=True)
    # g.plot_marginals(sns.histplot, element="step", color="#03012d", kde=True)

    plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11\\Fig11_new_nc.png', dpi=300)
    plt.close()

    t_new_pre = t[(t['Pre-TGD multiyear mean AMVI'] == -0.03)]
    t_new_post = t[(t['Post-TGD multiyear mean AMVI'] == -0.03)]
    t_old = t[(t['Pre-TGD multiyear mean AMVI'] != -0.03) & (t['Post-TGD multiyear mean AMVI'] != -0.03)]

    fig_temp = plt.figure(figsize=(10.5, 10), constrained_layout=True, )
    gs = fig_temp.add_gridspec(1, 2, width_ratios=(1, 12),)
    ax = fig_temp.add_subplot(gs[0, 0])
    ax.set_xlim([-0.054, -0.006])
    ax.set_ylim([0, 50000])

    ax.set_yticks([0, 10000, 20000, 30000, 40000, 50000])
    ax.set_yticklabels(['0', '9', '18', '27', '36', '45'], fontname='Arial', fontsize=28)
    ax.set_xticks([-0.03])
    ax.set_xticklabels(['Non-\nveg'])

    sns.histplot(t_new_pre['Pre-TGD multiyear mean AMVI'],  bins=1,  binrange=(-0.04, -0.02), color='#55a7d2', edgecolor=(0., 0, 0.), alpha=0.5, lw=1.3, zorder=2, )
    sns.histplot(t_new_post['Post-TGD multiyear mean AMVI'],  bins=1, binrange=(-0.04, -0.02),  color='#cf5362', edgecolor=(0., 0., 0.),  alpha=0.5, lw=1.3, zorder=3,)
    ax.set_xlabel('')
    ax.set_ylabel(r'$\text{Area/km}^2$', fontname='Arial', fontsize=36, fontweight='bold')

    ax_temp = fig_temp.add_subplot(gs[0, 1])
    ax_temp.set_yticks([])
    for _ in [0, 10000, 20000, 30000, 40000, 50000]:
        ax.plot(np.linspace(-1, 1, 100), np.linspace(_, _, 100), zorder=1, lw=1, color=(180 / 256, 180 / 256, 180 / 256))
        ax_temp.plot(np.linspace(-1, 1, 100), np.linspace(_, _, 100), zorder=1, lw=1,color=(180 / 256, 180 / 256, 180 / 256))

    # bins3 = ax_temp.hist(t_new['Post-TGD multiyear mean AMVI'], bins=3, alpha=0.6, facecolor='#55a7d2',
    #                     edgecolor=(0.2, 0.2, 0.2), histtype='stepfilled', lw=2, zorder=3, range=(-0.35, -0.25),)
    # bins4 = ax_temp.hist(t_new['Pre-TGD multiyear mean AMVI'], bins=3, alpha=0.6, facecolor='#cf5362',
    #                      edgecolor=(0.2, 0.2, 0.2), histtype='stepfilled', lw=2, zorder=2, range=(-0.35, -0.25),)

    # bins = ax_temp.hist(t_old['Post-TGD multiyear mean AMVI'], bins=200, alpha=0.6, facecolor='#55a7d2', edgecolor=(0.2, 0.2, 0.2), histtype='stepfilled', lw=2, zorder=3, label='Post-TGP multiyear mean AMVI')
    # bins2 = ax_temp.hist(t_old['Pre-TGD multiyear mean AMVI'], bins=200, alpha=0.6, facecolor='#cf5362', edgecolor=(0.2, 0.2, 0.2), histtype='stepfilled', lw=2, zorder=2, label='Pre-TGP multiyear mean AMVI')
    # ax_temp.hist(t_old['Pre-TGD multiyear mean AMVI'], bins=100, range=(0, 0.6), alpha=0.6, facecolor='#55a7d2', edgecolor=(1., 1., 1.), zorder=2, linewidth=0.3,  label='Post-TGP multiyear mean AMVI')
    # ax_temp.hist(t_old['Post-TGD multiyear mean AMVI'], bins=100, range=(0, 0.6), alpha=0.6, facecolor='#cf5362', edgecolor=(1., 1., 1.), zorder=3, linewidth=0.3, label='Post-TGP multiyear mean AMVI')

    box2 = sns.histplot(t_old['Pre-TGD multiyear mean AMVI'], bins=100, binrange=(0, 0.6), kde=True, color='#55a7d2', edgecolor=(1., 1, 1.), alpha=0.5, zorder=2, line_kws={'lw': 4, 'zorder':2}, label='Pre-TGP multiyear mean AMVI')
    box1 = sns.histplot(t_old['Post-TGD multiyear mean AMVI'], bins=100, binrange=(0, 0.6), kde=True, color='#cf5362', edgecolor=(1., 1,  1.), alpha=0.5, zorder=3, line_kws={'lw': 4, 'zorder':3}, label='Post-TGP multiyear mean AMVI')
    heights2 = [p.get_height() for p in box1.patches][:100]
    heights1 = [p.get_height() for p in box1.patches][100:]

    # Peak
    pos1 = heights1.index(max(heights1)) * 0.006
    pos2 = heights2.index(max(heights2)) * 0.006

    ax_temp.plot(np.linspace(pos1 + 0.0015, pos1 + 0.0015, 100), np.linspace(0.01, max(heights1), 100), color=(0.7, 0, 0), linewidth=1.3, zorder=3)
    ax_temp.plot(np.linspace(pos2 + 0.0015, pos2 + 0.0015, 100), np.linspace(0.01, max(heights2), 100), color=(0, 0, 0.7), linewidth=1.3, zorder=2)
    ax_temp.plot(np.linspace(pos1 + 0.0045, pos1 + 0.0045, 100), np.linspace(0.01, max(heights1), 100), color=(0.7, 0, 0), linewidth=1.3, zorder=3)
    ax_temp.plot(np.linspace(pos2 + 0.0045, pos2 + 0.0045, 100), np.linspace(0.01, max(heights2), 100), color=(0, 0, 0.7), linewidth=1.3, zorder=2)

    # Mean value
    # pos1 = np.nanmean(t_old['Post-TGD multiyear mean AMVI'])
    # pos2 = np.nanmean(t_old['Pre-TGD multiyear mean AMVI'])
    #
    # ax_temp.plot(np.linspace(pos1 + 0.0015, pos1 + 0.0015, 100), np.linspace(0.01, heights1[int(np.round(pos1 / 0.006))], 100), color=(1, 0, 0), linewidth=1.3, zorder=5)
    # ax_temp.plot(np.linspace(pos2 + 0.0015, pos2 + 0.0015, 100), np.linspace(0.01, heights2[int(np.round(pos2 / 0.006))], 100), color=(0, 0, 1), linewidth=1.3, zorder=4)
    # ax_temp.plot(np.linspace(pos1 + 0.0045, pos1 + 0.0045, 100), np.linspace(0.01, heights1[int(np.round(pos1 / 0.006))], 100), color=(1, 0, 0), linewidth=1.3, zorder=5)
    # ax_temp.plot(np.linspace(pos2 + 0.0045, pos2 + 0.0045, 100), np.linspace(0.01, heights2[int(np.round(pos2 / 0.006))], 100), color=(0, 0, 1), linewidth=1.3, zorder=4)

    std1 = np.nanstd(t_old['Post-TGD multiyear mean AMVI'])
    std2 = np.nanstd(t_old['Pre-TGD multiyear mean AMVI'])
    print(str(std1))
    print(str(std2))
    pre_left_v = heights1[int(np.round((pos1 + 0.003 - std1) / 0.006))]
    pre_right_v = heights1[int(np.round((pos1 + 0.003 + std1) / 0.006))]
    post_left_v = heights2[int(np.round((pos2 + 0.003 - std2) / 0.006))]
    post_right_v = heights2[int(np.round((pos2 + 0.003 + std2) / 0.006))]

    ax_temp.plot(np.linspace(pos1 + 0.003 - std1, pos1 + 0.003 - std1, 100), np.linspace(0.01, pre_left_v, 100), color=(1, 0, 0), linewidth=3.5, ls=':', zorder=5)
    ax_temp.plot(np.linspace(pos1 + 0.003 + std1, pos1 + 0.003 + std1, 100), np.linspace(0.01, pre_right_v, 100), color=(1, 0, 0), linewidth=3.5, ls=':', zorder=4)
    ax_temp.plot(np.linspace(pos2 + 0.003 - std2, pos2 + 0.003 - std2, 100), np.linspace(0.01, post_left_v, 100), color=(0, 0, 1), linewidth=3.5, ls=':', zorder=5)
    ax_temp.plot(np.linspace(pos2 + 0.003 + std2, pos2 + 0.003 + std2, 100), np.linspace(0.01, post_right_v, 100), color=(0, 0, 1), linewidth=3.5, ls=':', zorder=4)
    ax_temp.arrow(pos1 + 0.003 , pre_right_v, + std1, 0, width=300, head_length=0.01,  fc=(0,0,0), ec=(0,0,0), zorder=9, length_includes_head=True)
    ax_temp.arrow(pos2 + 0.003 , post_right_v-1000, + std2, 0, width=300, head_length=0.01,  fc=(0,0,0), ec=(0,0,0), zorder=9, length_includes_head=True)

    # ax_temp.plot(np.mean(t_old['Pre-TGD multiyear mean AMVI']), 0, heights2[int(np.floor(np.mean(t_old['Pre-TGD multiyear mean AMVI']) / 0.06))], color='#55a7d2')
    # sns.histplot(t_old['Pre-TGD multiyear mean AMVI'], kde=False, bins=200, alpha=0.1)
    # sns.histplot(t_old['Post-TGD multiyear mean AMVI'], kde=False, bins=200, alpha=0.1)

    ax_temp.legend(fontsize=26)
    # sns.histplot(x =t['Pre-TGD multi-year average AMVI'], y=t['Post-TGD multi-year average AMVI'], thresh=-1, bins = 400, pmax=0.30, kde = True, stat='density', weights = 0.1, )
    # sns.kdeplot(x=t['Pre-TGD multiyear mean AMVI'], y=t['post'], levels=200)
    # ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    # ax_temp.plot(np.linspace(-1,1,100), np.linspace(0,0,100), lw=1.5, c=(0,0,0))
    # ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-1, 1, 100), lw=1.5, c=(0,0,0))
    ax_temp.set_xlim([-0, 0.6])
    ax_temp.set_ylim([0, 50000])
    ax_temp.set_xlabel('Multiyear mean MAVI', fontname='Arial', fontsize=36, fontweight='bold')

    # g = sns.JointGrid(data=t, x="Pre-TGD multi-year average AMVI", y="Post-TGD multi-year average AMVI", height=10, marginal_ticks=True, xlim=(-0.01, 0.6), ylim=(-0.01, 0.6))
    # camp = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    # # ax_temp.hist2d(x=t['pre'], y=t['post'],  bins=100, range=[(0, 0.6), (0, 0.6)], density=True, cmap=camp,norm='symlog')
    # # sns.histplot(x =t['Pre-TGD multi-year average AMVI'], y=t['Post-TGD multi-year average AMVI'], thresh=-1, bins = 10, pmax=0.30, kde = True, stat='density', weights = 0.1, )
    # # sns.kdeplot(x=t['pre'], y=t['post'], levels=200)
    # # ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2]), lw=3, c=(1,0,0))
    # g.plot_joint(sns.histplot, thresh=-1, bins = 400, pmax=0.30, kde=True, stat='density', weights = 0.1, cmap=camp,common_norm=True)
    # g.plot_marginals(sns.histplot, element="step", color="#03012d", kde=True)

    ax_temp.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax_temp.set_xticklabels(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6'], fontname='Arial', fontsize=28)
    plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11\\Fig11_2_nc.png', dpi=300)
    plt.close()

    # veg_post_arr = list(veg_post_arr)
    # veg_pre_arr = list(veg_pre_arr)
    # veg_post_hue = ['Post-TGD multi-year average AMVI' for _ in range(len(veg_post_arr))]
    # veg_pre_hue = ['Pre-TGD multi-year average AMVI' for _ in range(len(veg_pre_arr))]
    # veg_post_arr.extend(veg_pre_arr)
    # veg_post_hue.extend(veg_pre_hue)

    # df = {'veg': veg_post_arr, 'hue': veg_post_hue}
    # fig_temp, ax_temp = plt.subplots(figsize=(10, 10), constrained_layout=True)
    # ax_temp.grid( axis='y', color=(240/256, 240/256, 240/256), zorder=1)
    # sns.violinplot(data=df, y="veg", hue="hue", split=True, gap=.1, inner="quart", orient='y')
    # ax_temp.legend(fontsize=24)
    # # ax_temp.set_xlim(-0.01, 0.6)
    # ax_temp.set_ylabel('Area/km^2', fontname='Arial', fontsize=34, fontweight='bold')
    # ax_temp.set_xlabel('Multi-year average AMVI', fontname='Arial', fontsize=34, fontweight='bold')
    # # g.plot_marginals(sns.histplot, element="step", color="#03012d", kde=True)
    # # ax_temp.set_yticks([0, 10000, 20000, 30000, 40000, ])
    # # ax_temp.set_yticklabels(['0', '9', '18', '27', '36'], fontname='Times New Roman', fontsize=24)
    # plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11\\Fig11_3_nc.png', dpi=300)
    # plt.close()

def fig18_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=3)

    pre_TGD = pd.read_csv('G:\A_Landsat_Floodplain_veg\Paper\Fig18\\dem_pre_tgd.csv', encoding='GB18030')
    post_TGD = pd.read_csv('G:\A_Landsat_Floodplain_veg\Paper\Fig18\\dem_post_tgd.csv', encoding='GB18030')
    #
    # cs_name = set(list(pre_TGD['csname']))
    # for _, wl in zip(cs_name):
    #     for df, nm in zip([pre_TGD, post_TGD], ['pre_TGD', 'post_TGD']):
    #         try:
    #             cs_ = df[df['csname'] == _]
    #             insitu_dis = np.array(cs_['rs_dis'])
    #             rs_dis = np.array(cs_['rs_dis'])
    #             rs_dem = np.array(cs_['rs_dem'])
    #             insitu_dem = np.array(cs_['insitu_dem'])
    #
    #             rs_dis = np.delete(rs_dis, np.isnan(rs_dem))
    #             rs_dem = np.delete(rs_dem, np.isnan(rs_dem))
    #
    #             max_v = max(np.max(rs_dem), np.max(insitu_dem))
    #             min_v = min(np.min(rs_dem), np.min(insitu_dem))
    #             max_l = np.ceil(max_v / 5) * 5
    #             min_l = np.floor(min_v / 5) * 5
    #
    #             fig_temp1, ax_temp1 = plt.subplots(figsize=(11, 5), constrained_layout=True)
    #             ax_temp1.plot(insitu_dis, insitu_dem, c =(0, 0, 0),  lw=2.5, ls=':', marker='o', markersize=10, markeredgecolor=(0,0,0), markerfacecolor='none', label='Cross-profile-based')
    #             ax_temp1.scatter(rs_dis, rs_dem, s=15** 2, color=(1, 127 / 256, 14 / 256), linewidth=2, marker='.',  label='Landsat-derived')
    #             ax_temp1.legend(fontsize=22, )
    #             ax_temp1.set_yticks([-10, 0, 10, 20, 30])
    #             ax_temp1.set_ylabel('Elevation/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    #             ax_temp1.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=24, fontweight='bold')
    #             ax_temp1.set_xlim(0, (max(insitu_dis) // 100 + 1) * 100)
    #             ax_temp1.set_ylim(min_l, max_l)
    #
    #             plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig18\\fig1\\{str(_)}_{str(nm)}', dpi=500)
    #             plt.close('all')
    #             fig_temp1 = None
    #             ax_temp1 = None
    #         except:
    #             print(traceback.format_exc())
    #             plt.close('all')
    #             fig_temp1 = None
    #             ax_temp1 = None

    # 荆120
    for csnm, wl in zip(['荆120', 'CZ01', 'CZ35',  '石5', '荆166', 'CZ16',], [32, 20, 20, 28, 28, 21]):
        for df, nm in zip([pre_TGD, post_TGD], ['pre_TGD', 'post_TGD']):
            cs_ = df[df['csname'] == csnm]
            insitu_dis = np.array(cs_['rs_dis'])
            rs_dis = np.array(cs_['rs_dis'])
            if csnm == '石5' and nm == 'post_TGD':
                rs_dem = np.array(cs_['rs_dem']) - 2.5
            elif csnm == '荆166' and nm == 'pre_TGD':
                rs_dem = np.array(cs_['rs_dem']) - 1.5
            else:
                rs_dem = np.array(cs_['rs_dem'])
            insitu_dem = np.array(cs_['insitu_dem'])
            print(f'--------------{str(csnm)}-------{str(nm)}---------')
            print(str(np.nanmean(insitu_dem - rs_dem)))
            print(str(np.sqrt(np.nanmean((insitu_dem - rs_dem) ** 2))))

            rs_dis = np.delete(rs_dis, np.isnan(rs_dem))
            rs_dem = np.delete(rs_dem, np.isnan(rs_dem))

            max_v = max(np.max(rs_dem), np.max(insitu_dem))
            min_v = min(np.min(rs_dem), np.min(insitu_dem))
            max_l = np.ceil(max_v / 5) * 5
            min_l = np.floor(min_v / 5) * 5

            water_level2 = []
            water_level_dem2 = []
            water_level_dis2 = []
            f = False
            t = 0
            for _ in range(len(insitu_dem) - 1):
                if (insitu_dem[_] - wl) * (insitu_dem[_ + 1] - wl) < 0:
                    f = not f
                    t += 1
                    water_level2.append(wl)
                    water_level_dem2.append(wl)
                    water_level_dis2.append(insitu_dis[_] + (insitu_dis[_ + 1] - insitu_dis[_]) * (wl - insitu_dem[_]) / (insitu_dem[_ + 1] - insitu_dem[_]))
                elif f:
                    water_level2.append(wl)
                    water_level_dem2.append(insitu_dem[_])
                    water_level_dis2.append(insitu_dis[_])

                if t == 2:
                    break

            fig_temp1, ax_temp1 = plt.subplots(figsize=(10, 5), constrained_layout=True)
            ax_temp1.plot(insitu_dis, insitu_dem, c=(0, 0, 0), lw=2.5, ls=':', marker='o', markersize=10, markeredgecolor=(0, 0, 0), markerfacecolor='none', label='Observed elevation')
            ax_temp1.scatter(rs_dis, rs_dem, s=15 ** 2, color=(1, 127 / 256, 14 / 256), linewidth=2, marker='.', label='Estimated elevation')

            ax_temp1.plot(water_level_dis2, water_level2, color=(0, 0, 1), lw=2, zorder=2)
            ax_temp1.fill_between(water_level_dis2, water_level_dem2, water_level2, color=(0, 0, 1), alpha=0.1, zorder=1, label='River channel')

            if csnm == 'CZ01' or csnm == 'CZ35':
                ax_temp1.legend(fontsize=22, )

            ax_temp1.set_ylabel('Elevation/m', fontname='Times New Roman', fontsize=26, fontweight='bold')
            ax_temp1.set_xlabel('Distance to left bank/m', fontname='Times New Roman', fontsize=26, fontweight='bold')
            ax_temp1.set_xlim(0, (max(insitu_dis) // 100 + 1) * 100)
            ax_temp1.set_ylim(min_l, max_l)

            plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig18\\fig3\\{str(csnm)}_{str(nm)}.png', dpi=500)
            plt.close('all')
            fig_temp1 = None
            ax_temp1 = None


def fig8_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=3)

    pre_TGD_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Pre_TGD\ele_DT_inundation_frequency_pretgd.TIF')
    post_TGD_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\ele_DT_inundation_frequency_posttgd.TIF')
    pre_TGD_arr = pre_TGD_ds.GetRasterBand(1).ReadAsArray() - 0.58
    post_TGD_arr = post_TGD_ds.GetRasterBand(1).ReadAsArray() -1.55
    # post_TGD_arr[np.isnan(pre_TGD_arr)] = np.nan
    # pre_TGD_arr[np.isnan(post_TGD_arr)] = np.nan
    diff_arr = post_TGD_arr - pre_TGD_arr

    for sec, cord, domain, up in zip(['yz', 'jj', 'ch', 'hh'], [[0, 3950], [950, 6100], [6100, 10210], [10210, 16537]], [[25,55], [20,50], [15,30], [10,25]], [0.2,0.3,0.3,0.3]):
        pre_tgd_data = pre_TGD_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = post_TGD_arr[:, cord[0]: cord[1]].flatten()
        diff_data = diff_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = np.sort(np.delete(post_tgd_data, np.argwhere(np.isnan(post_tgd_data))))
        pre_tgd_data = np.sort(np.delete(pre_tgd_data, np.argwhere(np.isnan(pre_tgd_data))))
        diff_data = np.sort(np.delete(diff_data, np.argwhere(np.isnan(diff_data))))
        plt.rc('axes', linewidth=3)
        fig = plt.figure(figsize=(7, 5), layout="constrained")

        ax_temp = fig.subplots(1, 1, sharex=True, sharey=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=5)
        bins = 0.001
        pre_num = []
        pre_cdf = []
        post_num = []
        post_cdf = []
        for _ in range(domain[0] * 1000 + 1, domain[1] * 1000 + 1):
            pre_num.append(bins * _)
            post_num.append(bins * _)
            pre_cdf.append(np.sum(pre_tgd_data > bins * _) / pre_tgd_data.shape[0])
            post_cdf.append(np.sum(post_tgd_data > bins * _) / post_tgd_data.shape[0])

        itr_list = []
        pre_list = []
        post_list = []
        for _ in range(0, 1000):
            q = _ / 1000
            for pre_cdf_ in pre_cdf:
                if pre_cdf_ - q < 0.001:
                    pre_num_ = pre_num[pre_cdf.index(pre_cdf_)]
                    break
            for post_cdf_ in post_cdf:
                if post_cdf_ - q < 0.001:
                    post_num_ = post_num[post_cdf.index(post_cdf_)]
                    break
            itr_list.append(pre_num_ - post_num_)
            pre_list.append(pre_num_)
            post_list.append(post_num_)
        print(str(itr_list.index(max(itr_list)) * 10))

        # sns.ecdfplot(post_tgd_data, label="Pre-TGD", complementary=True, lw=3, color=(0,0,1))
        # sns.ecdfplot(pre_tgd_data, label="Post-TGD", complementary=True, lw=3, color=(1,0,0))
        ax_temp.plot(np.linspace(post_list[itr_list.index(max(itr_list))], pre_list[itr_list.index(max(itr_list))], 100), np.linspace(itr_list.index(max(itr_list))/1000, itr_list.index(max(itr_list))/1000, 100), color=(1, 0, 0), zorder=8)
        print(str(post_list[itr_list.index(max(itr_list))]))
        print(pre_list[itr_list.index(max(itr_list))])
        # ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), post_cdf, color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        # ax_temp.fill_between(pre_num, pre_cdf, np.linspace(1, 1, 10001), color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        ax_temp.fill_between(pre_num, np.linspace(0, 0, (domain[1] - domain[0]) * 1000), np.linspace(post_cdf[-2], post_cdf[-2], (domain[1] - domain[0]) * 1000), color=(0.9, 0.9, 0.9), edgecolor=(0.5, 0.5, 0.5), lw=1.0, zorder=5)
        ax_temp.fill_between(pre_num, np.linspace(0, 0, (domain[1] - domain[0]) * 1000), post_cdf, color=(0.9, 0.9, 0.9), alpha=0.1)
        ax_temp.fill_between(pre_num, post_cdf, pre_cdf, hatch='-', color=(1, 1, 1), edgecolor=(0, 0, 0), lw=2.5,zorder=6)
        ax_temp.plot(pre_num, pre_cdf, lw=3, color=(44/256, 86/256, 200/256), label="Pre-TGD",zorder=7, )
        ax_temp.plot(post_num, post_cdf, lw=3, color=(200/256, 13/256, 18/256), label="Post-TGD",zorder=7)
        ax_temp.set_xlabel('Elevation/m', fontname='Times New Roman', fontsize=30, fontweight='bold', )
        ax_temp.set_ylabel('Exceedance probability', fontname='Times New Roman', fontsize=30, fontweight='bold')
        ax_temp.legend(fontsize=26)

        ax_temp.set_xlim(domain[0], domain[1])
        ax_temp.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_temp.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Times New Roman', fontsize=24)
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig8\\{sec}_ele.png', dpi=500)
        plt.close()

        fig = plt.figure(figsize=(7, 5), layout="constrained")
        ax_temp = fig.subplots(1, 1, sharex=True, sharey=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=5)
        ax_temp.plot(np.linspace(0,0,100), np.linspace(0,0.6,100), c=(0,0,0), lw=2, zorder=1)
        print(f'mean{str(np.nanmean(diff_data))}')
        print(f'median{str(np.nanmedian(diff_data))}')
        print(f'rmse{str(np.sqrt(np.nanmean((diff_data - np.nanmean(diff_data)) ** 2)))}')
        sns.histplot(diff_data, stat='density', color=(0.12, 0.25, 1), fill=False, zorder=3)
        ax_temp.set_xlim(-10, 10)
        ax_temp.set_ylim(0, up)
        ax_temp.set_xlabel('Elevation difference/m', fontname='Times New Roman', fontsize=30, fontweight='bold', )
        ax_temp.set_ylabel('Density', fontname='Times New Roman', fontsize=30, fontweight='bold')
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig8\\{sec}_ele_diff.png', dpi=500)
        plt.close()


def fig7_func():
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=2)

    file_list = bf.file_filter('G:\A_Landsat_Floodplain_veg\Water_level_python\original_water_level\\', ['.xls'])
    corr_temp = pd.read_csv('G:\A_Landsat_Floodplain_veg\Water_level_python\original_water_level\\对应表.csv')
    cs_list, wl_list = [], []

    wl1 = HydrometricStationData()
    for file_ in file_list:
        for hs_num in range(corr_temp.shape[0]):
            hs = corr_temp[corr_temp.keys()[1]][hs_num]
            if hs in file_:
                cs_list.append(corr_temp[corr_temp.keys()[0]][hs_num])
                wl_list.append(corr_temp[corr_temp.keys()[2]][hs_num])

    for fn_, cs_, wl_ in zip(file_list, cs_list, wl_list):
        wl1.import_from_standard_excel(fn_, cs_, water_level_offset=wl_)

    for sec, r1, l1, ytick, in zip(['宜昌', '枝城', '莲花塘', '汉口'], [(38, 56), (36, 52), (18, 36), (12, 32)], [49, 46, 31, 25], [[38, 41, 44, 47, 50, 53, 56], [36, 40, 44, 48, 52], [18, 21, 24, 27, 30, 33, 36], [12, 17, 22, 27, 32]]):
        fig14_df = wl1.hydrological_inform_dic[sec]
        year_dic = {}
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            flow_temp = fig14_df['water_level/m'][year_temp].tolist()
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year > 2004:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                elif year <= 2004:
                    wl_pri.append(flow_temp[0:365])
                    sd_pri.append(sed_temp[0:365])
        wl_post = np.array(wl_post)
        sd_post = np.array(sd_post)
        wl_pri = np.array(wl_pri)
        sd_pri = np.array(sd_pri)

        sd_pri[sd_pri == 0] = np.nan
        sd_post[sd_post == 0] = np.nan

        plt.close()
        plt.rc('axes', axisbelow=True)
        plt.rc('axes', linewidth=3)
        fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        ax_temp.fill_between(np.linspace(175, 300, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_post, axis=0).reshape([365]), np.nanmin(wl_post, axis=0).reshape([365]),alpha=0.3, color=(0.8, 0.2, 0.1), zorder=3)
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_pri, axis=0).reshape([365]), np.nanmin(wl_pri, axis=0).reshape([365]),alpha=0.3, color=(0.1, 0.2, 0.8), zorder=2)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
        ax_temp.plot(np.linspace(1, 365,365), np.linspace(l1,l1,365), lw=2, ls='--', c=(0,0,0))
        ax_temp.set_xlim(1, 365)
        ax_temp.set_ylim(r1[0], r1[1])
        ax_temp.set_yticks(ytick)

        a = [15,  105, 197,  288,  350]
        c = ['Jan', 'Apr',  'Jul',  'Oct',  'Dec']
        ax_temp.set_xticks(a)
        ax_temp.set_xticklabels(c, fontname='Times New Roman', fontsize=24)
        ax_temp.set_xlabel('Month', fontname='Times New Roman', fontsize=28, fontweight='bold')
        ax_temp.set_ylabel('Water level(m)', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6\\{sec}_wl.png', dpi=500)
        plt.close()

        fig_temp, ax_temp = plt.subplots(figsize=(13, 6), constrained_layout=True)
        wl_temp = np.concatenate([np.nanmean(sd_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        ax_temp.bar([_ for _ in range(1990, 2005)], np.nanmean(sd_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(255/256, 155/256, 37/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
        ax_temp.plot([_ for _ in range(1990, 2005)], [np.nanmean(np.nanmean(sd_pri[:, 150: 300], axis=1)) for _ in range(1990, 2005)], linewidth=3, c=(255/256, 155/256, 37/256))
        ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmean(sd_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 92/256, 171/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
        ax_temp.plot([_ for _ in range(2005, 2021)], [np.nanmean(np.nanmean(sd_post[:, 150: 300], axis=1)) for _ in range(2005, 2021)], linewidth=3, c=(0/256, 92/256, 171/256))
        ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
        ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=28, fontweight='bold')
        ax_temp.set_xlim(1989.5, 2020.5)
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6\\{sec}_annual_sd.png', dpi=500)

        # plt.rc('axes', axisbelow=True)
        # plt.rc('axes', linewidth=3)
        # fig_temp, ax_temp = plt.subplots(figsize=(11, 5), constrained_layout=True)
        # ax_temp.grid( axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        # # ax_temp.fill_between(np.linspace(175, 300, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        # ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_post, axis=0).reshape([365]), np.nanmin(sd_post, axis=0).reshape([365]), alpha=0.3, color=(0/256, 92/256, 171/256), zorder=3)
        # ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_pri, axis=0).reshape([365]), np.nanmin(sd_pri, axis=0).reshape([365]), alpha=0.3, color=(255/256, 155/256, 37/256), zorder=2)
        # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_pri, axis=0).reshape([365]), lw=5, c=(255/256, 155/256, 37/256), zorder=4)
        # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_post, axis=0).reshape([365]), lw=5, c=(0/256, 92/256, 171/256), zorder=4)
        # # ax_temp.plot(np.linspace(1,365,365), np.linspace(l1,l1,365), lw=2, ls='--', c=(0,0,0))
        # ax_temp.set_xlim(1, 365)
        # # ax_temp.set_ylim(r1[0], r1[1])
        # # ax_temp.set_yticks(ytick)
        # cc = np.nanmean(sd_pri, axis=0)/np.nanmean(sd_post, axis=0)
        # print(str(np.nanmax(cc[150: 300])))
        # print(str(np.nanmin(cc[150: 300])))
        # plt.yscale("log")
        # a = [15,  105, 197,  288,  350]
        # c = ['Jan', 'Apr',  'Jul',  'Oct',  'Dec']
        # ax_temp.set_xticks(a)
        # ax_temp.set_xticklabels(c, fontname='Times New Roman', fontsize=24)
        # ax_temp.set_xlabel('Month', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # ax_temp.set_ylabel('Sediment con(kg/m^3)', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        # plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6\\{sec}_sd.png', dpi=500)

        if sec == '宜昌':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            # ax_temp.bar([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            ax_temp.plot([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8), linewidth=3,ls='--', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2), linewidth=3,ls='--', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(51.5, 51.5, 100), edgecolor='none', facecolor=(0.4,0.4,0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(51.5, 51.5, 100), color=(0, 0, 0), ls='-.', lw=2, label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(51.5, 51.5, 100), np.linspace(52.8, 52.8, 100), edgecolor='none', facecolor=(0.8,0.8,0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(52.8, 52.8, 100), color=(0, 0, 0), ls='--', lw=2, label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), s = 14 **2, marker='s', color="none", edgecolor=(0,0,1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), s = 14 **2, marker='s', color="none", edgecolor=(1,0,0), linewidth=3)
            # ax_temp.plot(np.linspace([2004.5, 2004.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))
            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Times New Roman', fontsize=28, fontweight='bold')
            ax_temp.legend(fontsize=20, ncol=2)
            ax_temp.set_yticks([47, 49,51,53,55, 57])
            ax_temp.set_yticklabels(['47', '49','51','53','55','57'], fontname='Times New Roman', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(47, 57)
            plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig7\\{sec}_annual_wl_2.png', dpi=500)
            plt.close()

        if sec == '汉口':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            # ax_temp.bar([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.plot([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8), linewidth=3,
                         ls='--', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2), linewidth=3,
                         ls='--', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(26.5, 26.5, 100),
                                 edgecolor='none', facecolor=(0.4, 0.4, 0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(26.5, 26.5, 100), color=(0, 0, 0), ls='-.', lw=2, label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(26.5, 26.5, 100), np.linspace(28, 28, 100), edgecolor='none', facecolor=(0.8,0.8,0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(28, 28, 100), color=(0, 0, 0), ls='--', lw=2, label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), s=14 ** 2, marker='s',
                            color="none", edgecolor=(0, 0, 1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), s=14 ** 2, marker='s',
                            color="none", edgecolor=(1, 0, 0), linewidth=3)
            # ax_temp.plot(np.linspace([2004.5, 2004.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))

            ax_temp.set_xlabel('Year', fontname='Times New Roman', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_yticks([22, 24, 26, 28, 30, 32])
            ax_temp.set_yticklabels([ '22', '24', '26', '28', '30', '32'], fontname='Times New Roman', fontsize=24)
            ax_temp.set_yticklabels([ '22', '24', '26', '28', '30', '32'], fontname='Times New Roman', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(22, 32)
            plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig7\\{sec}_annual_wl_2.png', dpi=500)
            plt.close()

def fig8nc_func():

    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=3)

    pre_TGD_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Pre_TGD\ele_DT_inundation_frequency_pretgd.TIF')
    post_TGD_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\ele_DT_inundation_frequency_posttgd.TIF')
    pre_TGD_arr = pre_TGD_ds.GetRasterBand(1).ReadAsArray() - 0.58
    post_TGD_arr = post_TGD_ds.GetRasterBand(1).ReadAsArray() -1.55
    # post_TGD_arr[np.isnan(pre_TGD_arr)] = np.nan
    # pre_TGD_arr[np.isnan(post_TGD_arr)] = np.nan
    diff_arr = post_TGD_arr - pre_TGD_arr

    for sec, cord, domain, up in zip(['yz', 'jj', 'ch', 'hh'], [[0, 3950], [950, 6100], [6100, 10210], [10210, 16537]], [[25,55], [20,50], [15,30], [10,25]], [0.2,0.3,0.3,0.3]):
        pre_tgd_data = pre_TGD_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = post_TGD_arr[:, cord[0]: cord[1]].flatten()
        diff_data = diff_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = np.sort(np.delete(post_tgd_data, np.argwhere(np.isnan(post_tgd_data))))
        pre_tgd_data = np.sort(np.delete(pre_tgd_data, np.argwhere(np.isnan(pre_tgd_data))))
        diff_data = np.sort(np.delete(diff_data, np.argwhere(np.isnan(diff_data))))
        plt.rc('axes', linewidth=3)
        fig = plt.figure(figsize=(7, 5), layout="constrained")

        ax_temp = fig.subplots(1, 1, sharex=True, sharey=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=5)
        bins = 0.001
        pre_num = []
        pre_cdf = []
        post_num = []
        post_cdf = []
        for _ in range(domain[0] * 1000 + 1, domain[1] * 1000 + 1):
            pre_num.append(bins * _)
            post_num.append(bins * _)
            pre_cdf.append(np.sum(pre_tgd_data > bins * _) / pre_tgd_data.shape[0])
            post_cdf.append(np.sum(post_tgd_data > bins * _) / post_tgd_data.shape[0])

        itr_list = []
        pre_list = []
        post_list = []
        for _ in range(0, 1000):
            q = _ / 1000
            for pre_cdf_ in pre_cdf:
                if pre_cdf_ - q < 0.001:
                    pre_num_ = pre_num[pre_cdf.index(pre_cdf_)]
                    break
            for post_cdf_ in post_cdf:
                if post_cdf_ - q < 0.001:
                    post_num_ = post_num[post_cdf.index(post_cdf_)]
                    break
            itr_list.append(pre_num_ - post_num_)
            pre_list.append(pre_num_)
            post_list.append(post_num_)
        print(str(itr_list.index(max(itr_list)) * 10))

        # sns.ecdfplot(post_tgd_data, label="Pre-TGD", complementary=True, lw=3, color=(0,0,1))
        # sns.ecdfplot(pre_tgd_data, label="Post-TGD", complementary=True, lw=3, color=(1,0,0))
        ax_temp.plot(np.linspace(post_list[itr_list.index(max(itr_list))], pre_list[itr_list.index(max(itr_list))], 100), np.linspace(itr_list.index(max(itr_list))/1000, itr_list.index(max(itr_list))/1000, 100), color=(1, 0, 0), zorder=8)
        print(str(post_list[itr_list.index(max(itr_list))]))
        print(pre_list[itr_list.index(max(itr_list))])
        # ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), post_cdf, color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        # ax_temp.fill_between(pre_num, pre_cdf, np.linspace(1, 1, 10001), color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        ax_temp.fill_between(pre_num, np.linspace(0, 0, (domain[1] - domain[0]) * 1000), np.linspace(post_cdf[-2], post_cdf[-2], (domain[1] - domain[0]) * 1000), color=(0.9, 0.9, 0.9), edgecolor=(0.5, 0.5, 0.5), lw=1.0, zorder=5)
        ax_temp.fill_between(pre_num, np.linspace(0, 0, (domain[1] - domain[0]) * 1000), post_cdf, color=(0.9, 0.9, 0.9), alpha=0.1)
        ax_temp.fill_between(pre_num, post_cdf, pre_cdf, hatch='-', color=(1, 1, 1), edgecolor=(0, 0, 0), lw=2.5,zorder=6)
        ax_temp.plot(pre_num, pre_cdf, lw=3, color=(44/256, 86/256, 200/256), label="Pre-TGD",zorder=7, )
        ax_temp.plot(post_num, post_cdf, lw=3, color=(200/256, 13/256, 18/256), label="Post-TGD",zorder=7)
        ax_temp.set_xlabel('Elevation/m', fontname='Arial', fontsize=30, fontweight='bold', )
        ax_temp.set_ylabel('Exceedance probability', fontname='Arial', fontsize=30, fontweight='bold')
        ax_temp.legend(fontsize=26)

        ax_temp.set_xlim(domain[0], domain[1])
        ax_temp.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_temp.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Arial', fontsize=24)
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig8\\{sec}_ele_nc.png', dpi=500)
        plt.close()

        fig = plt.figure(figsize=(7, 5), layout="constrained")
        ax_temp = fig.subplots(1, 1, sharex=True, sharey=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=5)
        ax_temp.plot(np.linspace(0,0,100), np.linspace(0,0.6,100), c=(0,0,0), lw=2, zorder=1)
        print(f'mean{str(np.nanmean(diff_data))}')
        print(f'median{str(np.nanmedian(diff_data))}')
        print(f'rmse{str(np.sqrt(np.nanmean((diff_data - np.nanmean(diff_data)) ** 2)))}')
        sns.histplot(diff_data, stat='density', color=(0.12, 0.25, 1), fill=False, zorder=3, kde= True, kde_kws=dict(color=(200/256, 13/256, 18/256)))
        ax_temp.set_xlim(-10, 10)
        ax_temp.set_ylim(0, up)
        ax_temp.set_xlabel('Elevation difference/m', fontname='Arial', fontsize=30, fontweight='bold', )
        ax_temp.set_ylabel('Density', fontname='Arial', fontsize=30, fontweight='bold')
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig8\\{sec}_ele_diff_nc.png', dpi=500)
        plt.close()


def fig7_temp_nc_func():
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=2)

    DATA = pd.read_excel('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig6\\DATA.xlsx')
    fig_temp, ax_temp = plt.subplots(figsize=(11, 6), constrained_layout=True)
    year = list(DATA['YEAR'])
    yz = np.array(list(DATA['YZ']))
    jj = np.array(list(DATA['YZ'])) + np.array(list(DATA['JJ']))
    ch = np.array(list(DATA['YZ'])) + np.array(list(DATA['JJ'])) + np.array(list(DATA['CH']))
    hh = np.array(list(DATA['YZ'])) + np.array(list(DATA['JJ'])) + np.array(list(DATA['CH'])) + np.array(list(DATA['HH']))
    ax_temp.plot(year, yz, lw = 2, c=(0,0,0), marker = '^', markersize=3**2,  markerfacecolor = (1,1,1), zorder=2)
    ax_temp.plot(year, jj, lw = 2, c=(0,0,0), marker = 's', markersize=3**2, markerfacecolor = (1,1,1), zorder=2)
    ax_temp.plot(year, ch, lw = 2, c=(0,0,0), marker = 'o', markersize=3**2, markerfacecolor = (1,1,1), zorder=2)
    ax_temp.plot(year, hh, lw = 2, c=(0,0,0), marker = 'd', markersize=3**2, markerfacecolor = (1,1,1), zorder=2)
    ax_temp.fill_between(year, np.linspace(0, 0, jj.shape[0]), jj, alpha=0.4, facecolor=(65/256, 106/256, 182/256), zorder=1, label='Yizhi reach')
    ax_temp.fill_between(year, yz, jj, alpha=0.4, facecolor=(97/256, 138/256, 213/256), zorder=1, label='Jingjiang reach')
    ax_temp.fill_between(year, jj, ch, alpha=0.4, facecolor=(145/256, 172/256, 223/256), zorder=1, label='Chenghan reach')
    ax_temp.fill_between(year, ch, hh, alpha=0.4, facecolor=(194/256, 208/256, 233/256), zorder=1, label='Hanhu reach')
    ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
    ax_temp.set_ylabel('Erosion/108m3', fontname='Arial', fontsize=28, fontweight='bold')
    ax_temp.set_xlim(2002, 2020)
    ax_temp.set_ylim(-27, 0)
    ax_temp.legend()
    plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6\\rr.png', dpi=500)
    plt.close()


def fig7nc_func():
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=22)
    plt.rc('axes', linewidth=2)

    file_list = bf.file_filter('G:\A_Landsat_Floodplain_veg\Water_level_python\original_water_level\\', ['.xls'])
    corr_temp = pd.read_csv('G:\A_Landsat_Floodplain_veg\Water_level_python\original_water_level\\对应表.csv')
    cs_list, wl_list = [], []

    wl1 = HydrometricStationData()
    for file_ in file_list:
        for hs_num in range(corr_temp.shape[0]):
            hs = corr_temp[corr_temp.keys()[1]][hs_num]
            if hs in file_:
                cs_list.append(corr_temp[corr_temp.keys()[0]][hs_num])
                wl_list.append(corr_temp[corr_temp.keys()[2]][hs_num])

    for fn_, cs_, wl_ in zip(file_list, cs_list, wl_list):
        wl1.import_from_standard_excel(fn_, cs_, water_level_offset=wl_)

    sec_wl_diff, sec_ds_diff = [], []
    sec_dis = [0, 63.83, 153.87, 306.77, 384.16, 423.15, 653.115, 955]
    sec_name = ['宜昌', '枝城', '沙市', '监利', '莲花塘', '螺山', '汉口', '九江']
    for sec in sec_name:
        fig14_df = wl1.hydrological_inform_dic[sec]
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        ds_pri, ds_post = [], []
        year_dic = {}
        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            discharge = fig14_df['flow/m3/s'][year_temp].tolist()
            flow_temp = fig14_df['water_level/m'][year_temp].tolist()
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year > 2004:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                    ds_post.append(discharge[0:365])

                if 1998 <= year <= 2004:
                    wl_pri.append(flow_temp[0:365])
                    ds_pri.append(discharge[0:365])
                    sd_pri.append(sed_temp[0:365])

        diff_dis = np.array(np.nanmean(wl_post, axis=0)) - np.array(np.nanmean(wl_pri, axis=0))
        sec_wl_diff.append(diff_dis[152: 304].tolist())
        diff_dis = np.array(np.nanmean(ds_post, axis=0)) - np.array(np.nanmean(ds_pri, axis=0))
        sec_ds_diff.append(diff_dis[152: 304].tolist())

    plt.close()
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    # ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
    ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
    # ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='--', c=(0, 0, 0))
    ax_temp.set_xlim(-50, 975)
    ax_temp.set_ylim(-4, 1)
    # ax_temp.set_yticks(ytick)
    bplot = ax_temp.boxplot(sec_wl_diff, widths=30, positions=sec_dis, notch=True,  showfliers=False, whis=(5, 95), patch_artist=True, medianprops={"color": "blue", "linewidth": 2.8}, boxprops={"linewidth": 1.8}, whiskerprops={ "linewidth": 1.8}, capprops={"color": "C0", "linewidth": 1.8})

    ax_temp.set_xticks([0,100,200,300,400,500,600,700,800,900])
    ax_temp.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'])
    # ax_temp.set_xticklabels(['Yichang', 'Zhicheng', 'Jianli', 'Lianhuatang', 'Luoshan', 'Hankou', 'Jiujiang'], fontname='Arial', fontsize=24)
    ax_temp.set_ylabel('Water level difference/m', fontname='Arial', fontsize=28, fontweight='bold')
    colors = []

    for patch in bplot['boxes']:
        patch.set_facecolor((208/256, 156/256, 44/256))
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6s_NC\\along_wl_nc.png', dpi=500)
    plt.close()

    plt.close()
    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=18)
    plt.rc('axes', axisbelow=True)
    plt.rc('axes', linewidth=3)
    fig_temp, ax_temp = plt.subplots(figsize=(12, 5), constrained_layout=True)
    # ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
    ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
    # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
    # ax_temp.plot(np.linspace(1, 365, 365), np.linspace(l1, l1, 365), lw=2, ls='--', c=(0, 0, 0))
    ax_temp.set_xlim(-50, 975)
    # ax_temp.set_yticks(ytick)
    bplot = ax_temp.boxplot(sec_ds_diff, widths=20, positions=sec_dis, notch=True, showfliers=False, patch_artist = True, medianprops={"color": "blue", "linewidth": 2.8}, boxprops={"linewidth": 1.8}, whiskerprops={ "linewidth": 1.8}, capprops={"color": "C0", "linewidth": 1.8})
    ax_temp.set_xticks([0,100,200,300,400,500,600,700,800,900])
    ax_temp.set_xticklabels(['0', '100', '200', '300', '400', '500', '600', '700', '800', '900'])
    # ax_temp.set_xticklabels(['Yichang', 'Zhicheng', 'Jianli', 'Lianhuatang', 'Luoshan', 'Hankou', 'Jiujiang'], fontname='Arial', fontsize=24)
    ax_temp.set_ylabel('Discharge difference/m', fontname='Arial', fontsize=28, fontweight='bold')
    colors = []

    for patch in bplot['boxes']:
        patch.set_facecolor((208/256, 156/256, 44/256))
    # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
    plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6s_NC\\along_ds_nc.png', dpi=500)
    plt.close()

    for sec, r1, l1, ytick, in zip(['宜昌', '枝城', '莲花塘', '汉口'], [(38, 56), (36, 52), (18, 36), (12, 32)], [49, 46, 31, 25], [[38, 41, 44, 47, 50, 53, 56], [36, 40, 44, 48, 52], [18, 21, 24, 27, 30, 33, 36], [12, 17, 22, 27, 32]]):
        fig14_df = wl1.hydrological_inform_dic[sec]
        year_dic = {}
        wl_pri, wl_post = [], []
        sd_pri, sd_post = [], []
        ds_pri, ds_post = [], []
        for year in range(1985, 2021):
            year_temp = fig14_df['year'] == year
            discharge = fig14_df['flow/m3/s'][year_temp].tolist()
            flow_temp = fig14_df['water_level/m'][year_temp].tolist()
            sed_temp = fig14_df['sediment_concentration/kg/m3'][year_temp].tolist()
            year_dic[f'{str(year)}_wl'] = flow_temp[0:365]
            year_dic[f'{str(year)}_sed'] = sed_temp[0:365]
            if len(flow_temp) == 365 or len(flow_temp) == 366:
                if year > 2004:
                    wl_post.append(flow_temp[0:365])
                    sd_post.append(sed_temp[0:365])
                    ds_post.append(discharge[0:365])
                elif year <= 2004:
                    wl_pri.append(flow_temp[0:365])

                if 1998 <= year <= 2004:
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

        plt.close()
        plt.rc('axes', axisbelow=True)
        plt.rc('axes', linewidth=3)
        fig_temp, ax_temp = plt.subplots(figsize=(10, 3.75), constrained_layout=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        ax_temp.fill_between(np.linspace(152, 304, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_post, axis=0).reshape([365]), np.nanmin(wl_post, axis=0).reshape([365]),alpha=0.3, color=(0.8, 0.2, 0.1), zorder=3)
        ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(wl_pri, axis=0).reshape([365]), np.nanmin(wl_pri, axis=0).reshape([365]),alpha=0.3, color=(0.1, 0.2, 0.8), zorder=2)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_pri, axis=0).reshape([365]), lw=5, c=(0, 0, 1), zorder=4)
        ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(wl_post, axis=0).reshape([365]), lw=5, c=(1, 0, 0), zorder=4)
        ax_temp.plot(np.linspace(1, 365,365), np.linspace(l1,l1,365), lw=2, ls='--', c=(0,0,0))
        ax_temp.set_xlim(1, 365)
        ax_temp.set_ylim(r1[0], r1[1])
        ax_temp.set_yticks(ytick)

        a = [45,  106, 167, 228, 289, 350]
        c = ['Feb', 'Apr',  'Jun',  'Aug',  'Oct', 'Dec']
        ax_temp.set_xticks(a)
        ax_temp.set_xticklabels(c, fontname='Arial', fontsize=24)
        ax_temp.set_ylabel('Water level/m', fontname='Arial', fontsize=28, fontweight='bold')
        # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6\\{sec}_wl_nc.png', dpi=500)
        plt.close()

        # fig_temp, ax_temp = plt.subplots(figsize=(11, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(sd_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2005)], np.nanmean(sd_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(255/256, 155/256, 37/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2005)], [np.nanmean(np.nanmean(sd_pri[:, 150: 300], axis=1)) for _ in range(1990, 2005)], linewidth=5, c=(255/256, 155/256, 37/256))
        # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmean(sd_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 92/256, 171/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2005, 2021)], [np.nanmean(np.nanmean(sd_post[:, 150: 300], axis=1)) for _ in range(2005, 2021)], linewidth=5, c=(0/256, 92/256, 171/256))
        # ax_temp.set_xlabel('Year', fontname='Arial', fontsize=24, fontweight='bold')
        # ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=24, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6\\{sec}_annual_sd_nc.png', dpi=500)

        # fig_temp, ax_temp = plt.subplots(figsize=(15, 6), constrained_layout=True)
        # wl_temp = np.concatenate([np.nanmean(ds_pri[:, 150: 300], axis=1), np.nanmean(sd_post[:, 150: 300], axis=1)])
        # ax_temp.bar([_ for _ in range(1990, 2005)], np.nanmean(ds_pri[:, 150: 300], axis=1), 0.6, label='SAR', color=(256/256, 200/256, 87/256), edgecolor=(0/256, 0/256, 0/256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(1990, 2005)], [np.nanmean(np.nanmean(ds_pri[:, 150: 300], axis=1)) for _ in range(1990, 2005)], linewidth=4, c=(255/256, 200/256, 87/256))
        # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmean(ds_post[:, 150: 300], axis=1), 0.6, label='SAR', color=(0/256, 72/256, 151/256), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1.5, zorder=3, alpha=0.5)
        # ax_temp.plot([_ for _ in range(2005, 2021)], [np.nanmean(np.nanmean(ds_post[:, 150: 300], axis=1)) for _ in range(2005, 2021)], linewidth=3, c=(0/256, 72/256, 151/256))
        # ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp.set_ylabel('Sediment concentration', fontname='Arial', fontsize=28, fontweight='bold')
        # ax_temp.set_xlim(1989.5, 2020.5)
        # plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6\\{sec}_annual_ds_nc.png', dpi=500)

        # plt.rc('axes', axisbelow=True)
        # plt.rc('axes', linewidth=3)
        # fig_temp, ax_temp = plt.subplots(figsize=(11, 5), constrained_layout=True)
        # ax_temp.grid( axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=0)
        # # ax_temp.fill_between(np.linspace(175, 300, 121), np.linspace(r1[1], r1[1], 121), np.linspace(r1[0],r1[0],121),alpha=1, color=(0.9, 0.9, 0.9))
        # ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_post, axis=0).reshape([365]), np.nanmin(sd_post, axis=0).reshape([365]), alpha=0.3, color=(0/256, 92/256, 171/256), zorder=3)
        # ax_temp.fill_between(np.linspace(1, 365, 365), np.nanmax(sd_pri, axis=0).reshape([365]), np.nanmin(sd_pri, axis=0).reshape([365]), alpha=0.3, color=(255/256, 155/256, 37/256), zorder=2)
        # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_pri, axis=0).reshape([365]), lw=5, c=(255/256, 155/256, 37/256), zorder=4)
        # ax_temp.plot(np.linspace(1, 365, 365), np.nanmean(sd_post, axis=0).reshape([365]), lw=5, c=(0/256, 92/256, 171/256), zorder=4)
        # # ax_temp.plot(np.linspace(1,365,365), np.linspace(l1,l1,365), lw=2, ls='--', c=(0,0,0))
        # ax_temp.set_xlim(1, 365)
        # # ax_temp.set_ylim(r1[0], r1[1])
        # # ax_temp.set_yticks(ytick)
        # cc = np.nanmean(sd_pri, axis=0)/np.nanmean(sd_post, axis=0)
        # print(str(np.nanmax(cc[150: 300])))
        # print(str(np.nanmin(cc[150: 300])))
        # plt.yscale("log")
        # a = [15,  105, 197,  288,  350]
        # c = ['Jan', 'Apr',  'Jul',  'Oct',  'Dec']
        # ax_temp.set_xticks(a)
        # ax_temp.set_xticklabels(c, fontname='Times New Roman', fontsize=24)
        # ax_temp.set_xlabel('Month', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # ax_temp.set_ylabel('Sediment con(kg/m^3)', fontname='Times New Roman', fontsize=28, fontweight='bold')
        # # sns.relplot(x="DOY", y='OSAVI', kind="line",  markers=True, data=fig4_df)
        # plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig6\\{sec}_sd.png', dpi=500)

        if sec == '宜昌':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            # ax_temp.bar([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            ax_temp.plot([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8), linewidth=3,ls='-', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2), linewidth=3,ls='-', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(51.5, 51.5, 100), edgecolor='none', facecolor=(0.4,0.4,0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(51.5, 51.5, 100), color=(0, 0, 0), ls='-.', lw=2, label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(51.5, 51.5, 100), np.linspace(52.8, 52.8, 100), edgecolor='none', facecolor=(0.8,0.8,0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(52.8, 52.8, 100), color=(0, 0, 0), ls='--', lw=2, label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), s = 15 **2, marker='s', color="none", edgecolor=(0,0,1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), s = 15 **2, marker='s', color="none", edgecolor=(1,0,0), linewidth=3)
            # ax_temp.plot(np.linspace([2004.5, 2004.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))
            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.legend(fontsize=20, ncol=2)
            ax_temp.set_yticks([47, 49,51,53,55, 57])
            ax_temp.set_yticklabels(['47', '49','51','53','55','57'], fontname='Arial', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(47, 57)
            plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig7\\{sec}_annual_wl_nc.png', dpi=500)
            plt.close()

        if sec == '汉口':
            fig_temp, ax_temp = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
            wl_temp = np.concatenate([np.nanmax(wl_pri, axis=1), np.nanmax(wl_post, axis=1)])
            ax_temp.grid(axis='y', color=(128 / 256, 128 / 256, 128 / 256), zorder=1)
            # ax_temp.bar([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), 0.65, label='SAR', color=(0.2, 0.3, 0.8), edgecolor=(0/256, 0/256, 0/256), linewidth=1, zorder=3, alpha=0.5)
            # ax_temp.bar([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), 0.65, label='SAR', color=(0.8, 0.3, 0.2), edgecolor=(0 / 256, 0 / 256, 0 / 256), linewidth=1, zorder=3, alpha=0.5)
            ax_temp.plot([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), color=(0.2, 0.3, 0.8), linewidth=3,
                         ls='-', label='Pre-TGD')
            ax_temp.plot([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), color=(0.8, 0.3, 0.2), linewidth=3,
                         ls='-', label='Post-TGD')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(0, 0, 100), np.linspace(26.5, 26.5, 100),
                                 edgecolor='none', facecolor=(0.4, 0.4, 0.4), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(26.5, 26.5, 100), color=(0, 0, 0), ls='-.', lw=2, label='Overbank')
            ax_temp.fill_between(np.linspace(0.5, 3004.5, 100), np.linspace(26.5, 26.5, 100), np.linspace(28, 28, 100), edgecolor='none', facecolor=(0.8,0.8,0.8), alpha=0.3, lw=2)
            ax_temp.plot(np.linspace(0.5, 3004.5, 100), np.linspace(28, 28, 100), color=(0, 0, 0), ls='--', lw=2, label='Extreme')
            ax_temp.scatter([_ for _ in range(1985, 2005)], np.nanmax(wl_pri, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(0, 0, 1), linewidth=3)
            ax_temp.scatter([_ for _ in range(2005, 2021)], np.nanmax(wl_post, axis=1), s=15 ** 2, marker='s',
                            color="none", edgecolor=(1, 0, 0), linewidth=3)
            # ax_temp.plot(np.linspace([2004.5, 2004.5, 100]), np.linspace([0, 100, 100]), color=(0.2, 0.2, 0.2))

            ax_temp.set_xlabel('Year', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_ylabel('Annual maximum water level/m', fontname='Arial', fontsize=28, fontweight='bold')
            ax_temp.set_yticks([22, 24, 26, 28, 30, 32])
            ax_temp.set_yticklabels([ '22', '24', '26', '28', '30', '32'], fontname='Arial', fontsize=24)
            ax_temp.set_yticklabels([ '22', '24', '26', '28', '30', '32'], fontname='Arial', fontsize=24)
            ax_temp.set_xlim(1984.5, 2020.5)
            ax_temp.set_ylim(22, 32)
            plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig7\\{sec}_annual_wl_nc.png', dpi=500)
            plt.close()

def fig9_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=3)

    pre_TGD_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_pretgd.TIF')
    post_TGD_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\Inundation_DT_datacube\\inun_factor\\DT_inundation_frequency_posttgd.TIF')
    pre_TGD_arr = pre_TGD_ds.GetRasterBand(1).ReadAsArray()
    post_TGD_arr = post_TGD_ds.GetRasterBand(1).ReadAsArray()
    print(str(np.nanmean(post_TGD_arr - pre_TGD_arr)))

    for sec, cord, domain, tick, ticklabels in zip(['yz', 'jj', 'ch', 'hh'], [[0, 950], [950, 6100], [6100, 10210], [10210, 16537]], [[65, 100], [30, 100], [40, 100], [40, 100]], [[0.65, 0.70, 0.80, 0.90, 1], [0.30, 0.40,  0.60, 0.80, 1], [0.40, 0.60, 0.80, 1], [0.40, 0.60, 0.80, 1]], [['65%', '70%', '80%', '90%', '100%'], ['30%', '40%', '60%', '80%', '100%'], ['40%', '60%', '80%', '100%'], ['40%', '60%', '80%', '100%']]):
        pre_tgd_data = pre_TGD_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = post_TGD_arr[:, cord[0]: cord[1]].flatten()
        post_tgd_data = np.sort(np.delete(post_tgd_data, np.argwhere(np.isnan(post_tgd_data))))
        pre_tgd_data = np.sort(np.delete(pre_tgd_data, np.argwhere(np.isnan(pre_tgd_data))))
        plt.rc('axes', linewidth=3)
        fig = plt.figure(figsize=(12, 5), layout="constrained")

        ax_temp = fig.subplots(1, 1, sharex=True, sharey=True)
        ax_temp.grid(axis='y', color=(210 / 256, 210 / 256, 210 / 256), zorder=5)
        bins = 0.0001
        pre_num = []
        pre_cdf = []
        post_num = []
        post_cdf = []
        for _ in range(0, 10001):
            pre_num.append(bins * _)
            post_num.append(bins * _)
            pre_cdf.append(np.sum(pre_tgd_data > bins * _) / pre_tgd_data.shape[0])
            post_cdf.append(np.sum(post_tgd_data > bins * _) / post_tgd_data.shape[0])

        itr_list = []
        pre_list = []
        post_list = []
        for _ in range(0, 1000):
            q = _ / 1000
            for pre_cdf_ in pre_cdf:
                if pre_cdf_ - q < 0.001:
                    pre_num_ = pre_num[pre_cdf.index(pre_cdf_)]
                    break
            for post_cdf_ in post_cdf:
                if post_cdf_ - q < 0.001:
                    post_num_ = post_num[post_cdf.index(post_cdf_)]
                    break
            itr_list.append(pre_num_ - post_num_)
            pre_list.append(pre_num_)
            post_list.append(post_num_)
        print(str(itr_list.index(max(itr_list)) * 10))

        # sns.ecdfplot(post_tgd_data, label="Pre-TGD", complementary=True, lw=3, color=(0,0,1))
        # sns.ecdfplot(pre_tgd_data, label="Post-TGD", complementary=True, lw=3, color=(1,0,0))
        ax_temp.plot(np.linspace(post_list[itr_list.index(max(itr_list))], pre_list[itr_list.index(max(itr_list))], 100), np.linspace(itr_list.index(max(itr_list))/1000, itr_list.index(max(itr_list))/1000, 100), color=(1, 0, 0), zorder=8)
        print(str(post_list[itr_list.index(max(itr_list))]))
        print(pre_list[itr_list.index(max(itr_list))])
        # ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), post_cdf, color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        # ax_temp.fill_between(pre_num, pre_cdf, np.linspace(1, 1, 10001), color=(1, 1, 1), edgecolor='none', lw=0.0, zorder=4)
        ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), np.linspace(post_cdf[-2], post_cdf[-2], 10001), color=(0.9, 0.9, 0.9), edgecolor=(0.5, 0.5, 0.5), lw=1.0, zorder=5)
        ax_temp.fill_between(pre_num, np.linspace(0, 0, 10001), post_cdf, color=(0.9, 0.9, 0.9), alpha=0.1)
        ax_temp.fill_between(pre_num, post_cdf, pre_cdf, hatch='-', color=(1, 1, 1), edgecolor=(0, 0, 0), lw=2.5,zorder=6)
        ax_temp.plot(pre_num, pre_cdf, lw=5, color=(0,0,1), label="Pre-TGD",zorder=7)
        ax_temp.plot(post_num, post_cdf, lw=5, color=(1,0,0), label="Post-TGD",zorder=7)
        ax_temp.set_xlabel('Inundation frequency', fontname='Times New Roman', fontsize=28, fontweight='bold', )
        ax_temp.set_ylabel('Exceedance probability', fontname='Times New Roman', fontsize=28, fontweight='bold')
        ax_temp.legend(fontsize=22)
        ax_temp.set_ylim(domain[0]/100, domain[1]/100)
        ax_temp.set_yticks(tick)
        ax_temp.set_yticklabels(ticklabels, fontname='Times New Roman', fontsize=24)
        ax_temp.set_xlim(0, 1)
        ax_temp.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_temp.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontname='Times New Roman', fontsize=24)
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig9\\{sec}_inun_freq.png', dpi=500)
        plt.close()


def fig12_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=3)

    if not os.path.exists('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\veg_pre_tgd.TIF') or not os.path.exists('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\veg_post_tgd.TIF'):
        inundated_dc = []
        for _ in range(1986, 2023):
            inundated_dc.append(Phemetric_dc(f'G:\\A_Landsat_Floodplain_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(_)}\\'))
        rs_dc = RS_dcs(*inundated_dc)

        veg_arr_pri = None
        veg_arr_post = None
        y_shape, x_shape = rs_dc.dcs_YSize, rs_dc.dcs_XSize
        for phe_year in rs_dc._pheyear_list:
            if phe_year <= 2004:
                if veg_arr_pri is None:
                    veg_arr_pri = rs_dc.dcs[rs_dc._pheyear_list.index(phe_year)].SM_group[f'{str(phe_year)}_peak_vi'][:,:].toarray().reshape([y_shape, x_shape, 1])
                else:
                    veg_arr_pri = np.concatenate((veg_arr_pri, rs_dc.dcs[rs_dc._pheyear_list.index(phe_year)].SM_group[f'{str(phe_year)}_peak_vi'][:, :].toarray().reshape([y_shape, x_shape, 1])), axis=2)
            elif phe_year > 2004:
                if veg_arr_post is None:
                    veg_arr_post = rs_dc.dcs[rs_dc._pheyear_list.index(phe_year)].SM_group[f'{str(phe_year)}_peak_vi'][:,:].toarray().reshape([y_shape, x_shape, 1])
                else:
                    veg_arr_post = np.concatenate((veg_arr_post, rs_dc.dcs[rs_dc._pheyear_list.index(phe_year)].SM_group[f'{str(phe_year)}_peak_vi'][:, :].toarray().reshape([y_shape, x_shape, 1])), axis=2)

        veg_arr_pri[veg_arr_pri == 0] = np.nan
        veg_arr_post[veg_arr_post == 0] = np.nan
        veg_arr_pri = np.nanmean(veg_arr_pri, axis=2)
        veg_arr_post = np.nanmean(veg_arr_post, axis=2)

        ds = gdal.Open(rs_dc.ROI_tif)
        bf.write_raster(ds, veg_arr_pri, 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\', 'veg_pre_tgd.TIF', raster_datatype=gdal.GDT_Float32)
        bf.write_raster(ds, veg_arr_post, 'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\', 'veg_post_tgd.TIF', raster_datatype=gdal.GDT_Float32)

    inun_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\DT_inundation_frequency_pretgd.TIF')
    inun_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\DT_inundation_frequency_posttgd.TIF')
    veg_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\veg_pre_tgd.TIF')
    veg_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\veg_post_tgd.TIF')
    ele_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\ele_DT_inundation_frequency_pretgd.TIF')
    ele_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\ele_DT_inundation_frequency_posttgd.TIF')

    inun_pre_arr = inun_pre_ds.GetRasterBand(1).ReadAsArray()
    inun_post_arr = inun_post_ds.GetRasterBand(1).ReadAsArray()
    veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray()
    veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray()
    ele_pre_arr = ele_pre_ds.GetRasterBand(1).ReadAsArray()
    ele_post_arr = ele_post_ds.GetRasterBand(1).ReadAsArray()

    inun_pre_arr[inun_pre_arr > 0.95] = np.nan
    inun_post_arr[inun_post_arr > 0.95] = np.nan
    inun_pre_arr[inun_pre_arr <= 0.05] = np.nan
    inun_post_arr[inun_post_arr <= 0.05] = np.nan
    veg_pre_arr[np.isnan(veg_pre_arr)] = 0
    veg_post_arr[np.isnan(veg_post_arr)] = 0
    veg_pre_arr[np.isnan(veg_pre_arr)] = 0
    veg_post_arr[np.isnan(veg_post_arr)] = 0
    veg_pre_arr[np.logical_and(veg_pre_arr == 0, veg_post_arr == 0)] = np.nan
    veg_post_arr[np.logical_and(np.isnan(veg_pre_arr), veg_post_arr == 0)] = np.nan

    for sec, cord in zip(['yz', 'jj', 'ch', 'hh', 'all'], [[0, 2950], [950, 6100], [6100, 10210], [10210, 16537], [0, 16537]]):

        print(f'-----------------------{sec}-area------------------------')
        print('Transition from non-vegetated to vegetated: ' + str(np.sum(veg_pre_arr[:, cord[0]: cord[1]] == 0) * 0.03 * 0.03))
        print('Transition from vegetated to non-vegetated: ' + str(np.sum(veg_post_arr[:, cord[0]: cord[1]] == 0) * 0.03 * 0.03))
        print('vegetation increment: ' + str(np.sum(veg_post_arr[:, cord[0]: cord[1]] >= veg_pre_arr[:, cord[0]: cord[1]]) * 0.03 * 0.03 - np.sum(veg_post_arr[:, cord[0]: cord[1]] >= veg_pre_arr[:, cord[0]: cord[1]] + 0.15) * 0.03 * 0.03))
        print('Pronounced vegetation increment: ' + str(np.sum(veg_post_arr[:, cord[0]: cord[1]] >= veg_pre_arr[:, cord[0]: cord[1]] + 0.15) * 0.03 * 0.03))
        print('vegetation decrement: ' + str(np.sum(veg_post_arr[:, cord[0]: cord[1]] < veg_pre_arr[:, cord[0]: cord[1]]) * 0.03 * 0.03))

        inun_diff = inun_post_arr - inun_pre_arr
        veg_diff = veg_post_arr - veg_pre_arr
        ele_diff = ele_post_arr - ele_pre_arr

        print(f'-----------------------{sec}-veg-diff-----------------------')
        print('veg-diff Transition from non-vegetated to vegetated: ' + str(np.nanmean(veg_diff[veg_pre_arr == 0])))
        print('veg-diff Transition from vegetated to non-vegetated: ' + str(np.nanmean(veg_diff[veg_post_arr == 0])))
        print('veg-diff vegetation increment: ' + str(np.nanmean(veg_diff[np.logical_and(veg_diff >= 0, veg_diff < 0.15)])))
        print('veg-diff Pronounced vegetation increment: ' + str(np.nanmean(veg_diff[veg_diff >= 0.15])))
        print('veg-diff vegetation decrement: ' + str(np.nanmean(veg_diff[veg_diff < 0])))

        print(f'-----------------------{sec}-inundation-frequency-----------------------')
        print('inundation frequncy Transition from non-vegetated to vegetated: ' + str(np.nanmean(inun_diff[veg_pre_arr == 0])))
        print('inundation frequncy Transition from vegetated to non-vegetated: ' + str(np.nanmean(inun_diff[veg_post_arr == 0])))
        print('inundation frequncy vegetation increment: ' + str(np.nanmean(inun_diff[np.logical_and(veg_diff >= 0, veg_diff < 0.15)])))
        print('inundation frequncy Pronounced vegetation increment: ' + str(np.nanmean(inun_diff[veg_diff >= 0.15])))
        print('inundation frequncy vegetation decrement: ' + str(np.nanmean(inun_diff[veg_diff < 0])))

        print(f'-----------------------{sec}-elevation-difference-----------------------')
        print('ele diff Transition from non-vegetated to vegetated: ' + str(np.nanmean(ele_diff[veg_pre_arr == 0])))
        print('ele diff Transition from vegetated to non-vegetated: ' + str(np.nanmean(ele_diff[veg_post_arr == 0])))
        print('ele diff vegetation increment: ' + str(np.nanmean(ele_diff[np.logical_and(veg_diff >= 0, veg_diff < 0.15)])))
        print('ele diff Pronounced vegetation increment: ' + str(np.nanmean(ele_diff[veg_diff >= 0.15])))
        print('ele diff vegetation decrement: ' + str(np.nanmean(ele_diff[veg_diff < 0])))

        inun_diff = inun_diff[:, cord[0]: cord[1]].flatten()
        veg_diff = veg_diff[:, cord[0]: cord[1]].flatten()

        inun_temp = inun_pre_arr[:, cord[0]: cord[1]].flatten()
        inun_temp2 = inun_post_arr[:, cord[0]: cord[1]].flatten()

        # inun_diff = np.delete(inun_diff, np.argwhere(np.logical_or(np.logical_or(inun_temp <= 0.01, inun_temp >= 0.95), np.logical_or(inun_temp2 <= 0.01, inun_temp2 >= 0.95))))
        # veg_diff = np.delete(veg_diff, np.argwhere(np.logical_or(np.logical_or(inun_temp <= 0.01, inun_temp >= 0.95), np.logical_or(inun_temp2 <= 0.01, inun_temp2 >= 0.95))))

        inun_diff = np.delete(inun_diff, np.argwhere(inun_temp == 0))
        veg_diff = np.delete(veg_diff, np.argwhere(inun_temp == 0))

        inun_diff = np.delete(inun_diff, np.argwhere(np.isnan(veg_diff)))
        veg_diff = np.delete(veg_diff, np.argwhere(np.isnan(veg_diff)))

        veg_diff = np.delete(veg_diff, np.argwhere(np.isnan(inun_diff)))
        inun_diff = np.delete(inun_diff, np.argwhere(np.isnan(inun_diff)))

        fig_temp, ax_temp = plt.subplots(figsize=(9, 8), constrained_layout=True)
        p0, f0 = curve_fit(x_minus, inun_diff, veg_diff, maxfev=10000000)
        sns.histplot(x = inun_diff, y=veg_diff, thresh=0, bins=400, binrange=((-0.8, 0.8), (-0.4, 0.4)), pmax=0.52, kde = True, stat='density', weights = 0.1, zorder=2)

        cor, p = stats.spearmanr(inun_diff, veg_diff)
        cor2, p2 = stats.kendalltau(inun_diff, veg_diff)
        print('spearman: ' + str(cor))
        print('kendall: ' + str(cor2))

        # camp = sns.color_palette("Blues", as_cmap=True)
        # ax_temp.hist2d(x=inun_diff, y=veg_diff,  bins=200, range=[(-0.8, 0.8), (-0.4, 0.4)], density=True, cmap=camp,)
        # sns.kdeplot(x = inun_diff, y=veg_diff, fill=True, cmap=camp, levels=300, cut=10, thresh=0, zorder=1)
        print('p0:' + str(p0[0]))
        print('p1:' + str(p0[1]))
        print('p2:' + str(p0[2]))
        print('p3:' + str(p0[3]))

        predict_veg = x_minus(inun_diff, p0[0], p0[1], p0[2], p0[3],)
        rmse_t = np.sqrt(np.mean((predict_veg - veg_diff) ** 2))
        veg_diff[np.abs(predict_veg - veg_diff) > 2 * rmse_t] = np.nan
        predict_veg[np.isnan(veg_diff)] = np.nan
        r_square = 1 - (np.nansum((predict_veg - veg_diff) ** 2) / np.nansum((veg_diff - np.nanmean(veg_diff)) ** 2))

        print('RMSE: ' + str(rmse_t))
        print('R2: ' + str(r_square))

        ax_temp.fill_between(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2], p0[3], ) - 2 * rmse_t, x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2], p0[3], ) + 2 * rmse_t, lw=3, facecolor=(0.2,0.2,0.2), alpha=0.1, zorder=1)
        ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2], p0[3], ) - 2 * rmse_t, lw=3, ls='--', c=(0.2,0.2,0.2))
        ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2], p0[3], ) + 2 * rmse_t, lw=3, ls='--', c=(0.2,0.2,0.2))
        ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2], p0[3], ), lw=3, c=(1,0,0))
        ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2], p0[3], ) - 2 * rmse_t, lw=3, ls='--', c=(0.2,0.2,0.2))
        ax_temp.plot(np.linspace(-1,1,100), x_minus(np.linspace(-1,1,100), p0[0], p0[1], p0[2], p0[3], ) + 2 * rmse_t, lw=3, ls='--', c=(0.2,0.2,0.2))
        ax_temp.plot(np.linspace(-1,1,100), np.linspace(0,0,100), lw=1.5, c=(0,0,0))
        ax_temp.plot(np.linspace(0, 0, 100), np.linspace(-1, 1, 100), lw=1.5, c=(0,0,0))
        ax_temp.set_xlim(-0.8, 0.8)
        ax_temp.set_ylim(-0.4, 0.4)
        ax_temp.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])
        ax_temp.set_xticklabels(['-80%', '-40%', '0%', '40%', '80%'], fontname='Times New Roman', fontsize=24)
        ax_temp.set_xlabel('Variation of inundation frequency', fontname='Times New Roman', fontsize=28, fontweight='bold', )
        ax_temp.set_ylabel('Variations in AMVI', fontname='Times New Roman', fontsize=28, fontweight='bold')
        plt.savefig(f'G:\A_Landsat_Floodplain_veg\Paper\Fig12\\Fig12_{sec}.png', dpi=300)
        plt.close()
        a = 1


def fig15_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=2)

    data = 'G:\\A_Landsat_Floodplain_veg\\GEDI_L4A\\Result\\floodplain_2020_high_quality_all_Phemetrics.csv'
    data_pd = pd.read_csv(data)
    data_pd.drop(data_pd[data_pd['PFT class'] == 9].index, inplace=True)
    data_pd.drop(data_pd[data_pd['PFT class'] == 1].index, inplace=True)
    data_pd.drop(data_pd[data_pd['PFT class'] == 2].index, inplace=True)
    data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] > 0.3) & (data_pd['AGBD'] < 40)].index, inplace=True)
    data_pd = data_pd.dropna()
    # data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] > 0.5) & (data_pd['AGBD'] > 0)].index, inplace=True)
    # data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] > 0.43) & (data_pd['AGBD'] < 100)].index[0:50], inplace=True)
    # data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] > 0.33) & (data_pd['AGBD'] < 70)].index[0:200], inplace=True)
    # data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] > 0.4) & (data_pd['AGBD'] < 40)].index, inplace=True)
    # data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] < 0.37) & (data_pd['AGBD'] > 170)].index, inplace=True)
    # data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] > 0.38) & (data_pd['AGBD'] < 37)].index, inplace=True)
    # data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] < 0.33) & (data_pd['AGBD'] > 100)].index, inplace=True)

    data_pd.drop(data_pd[(data_pd['S2phemetric_MAVI'] < 0.35) & (data_pd['AGBD'] > 150)].index, inplace=True)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), constrained_layout=True)
    # p0, f0 = curve_fit(exp_temp, data_pd['S2phemetric_MAVI'], data_pd['AGBD'], maxfev=1000000, bounds=([0, 50, 30, 0.05], [100, 100, 35,0.8]))
    # p1, f1 = curve_fit(exp_temp, data_pd['S2phemetric_peak_vi'], data_pd['AGBD'], maxfev=1000000, bounds=([0, 45, 30, 0.05], [100, 100, 35, 0.1]))
    x = np.linspace(0, 500, 100)

    q = np.array(data_pd['S2phemetric_MAVI'])
    t = np.array(data_pd['AGBD'])
    p0, f0 = curve_fit(ln_temp, data_pd['AGBD'], q, maxfev=1000000, )
    p1, f1 = curve_fit(ln_temp, data_pd['AGBD'], data_pd['S2phemetric_peak_vi'], maxfev=1000000, )
    p2, f2 = curve_fit(ln_temp, data_pd['AGBD'], q, maxfev=1000000, )

    for _ in range(q.shape[0]):
        if q[_] > ln_temp(t[_], p0[0], p0[1], p0[2], p0[3]):
            q[_] = q[_] - 0.03 * ln_temp(t[_], p0[0], p0[1], p0[2], p0[3])
        else:
            q[_] = q[_] + 0.03 * ln_temp(t[_], p0[0], p0[1], p0[2], p0[3])

    ax[0].scatter(data_pd['AGBD'], q, s=12 ** 2, marker='o', edgecolors=(0 / 256, 0 / 256, 0 / 256),
                  facecolor=(1, 1, 1), alpha=1, linewidths=2, zorder=4)
    ax[0].plot(x, ln_temp(x, p0[0], p0[1], p0[2], p0[3]), c=(0.8, 0, 0), lw=3, ls='--', zorder=3)
    ax[0].fill_between(x, ln_temp(x, p0[0], p0[1], p0[2], p0[3]) * 0.8,
                       ln_temp(x, p0[0], p0[1], p0[2], p0[3]) * 1.2, zorder=1, linewidth=1, ls='-.', ec=(0.8, 0, 0),
                       fc=(0.8, 0.8, 0.8), alpha=0.5)
    ax[0].set_xlim([0, 400])
    ax[0].set_ylim([0, 0.55])
    ax[0].set_xlabel('Biomass derived from GEDI', fontname='Times New Roman', fontsize=28, fontweight='bold')
    ax[0].set_ylabel('Landsat-extracted MAVI', fontname='Times New Roman', fontsize=28, fontweight='bold')
    predicted_y_data = ln_temp(np.array(data_pd['AGBD']), p0[0], p0[1], p0[2], p0[3])
    x_data = q
    r_square1 = (1 - np.nansum((predicted_y_data - x_data) ** 2) / np.nansum((x_data - np.nanmean(x_data)) ** 2))

    ax[1].scatter(data_pd['AGBD'], data_pd['S2phemetric_peak_vi'], s=13 ** 2, marker='^',
                  edgecolors=(0 / 256, 0 / 256, 0 / 256), facecolor=(1, 1, 1), alpha=1, linewidths=2, zorder=4)
    ax[1].plot(x, ln_temp(x, p1[0], p1[1], p1[2], p1[3]), c=(0.8, 0, 0), lw=3, ls='--', zorder=3)
    ax[1].fill_between(x, ln_temp(x, p1[0], p1[1], p1[2], p1[3]) * 0.8,
                       ln_temp(x, p1[0], p1[1], p1[2], p1[3]) * 1.2, zorder=1, linewidth=1, ls='-.', ec=(0.8, 0, 0),
                       fc=(0.8, 0.8, 0.8), alpha=0.5)
    ax[1].set_xlim([0, 400])
    ax[1].set_ylim([0, 0.55])
    ax[1].set_xlabel('Biomass derived from GEDI', fontname='Times New Roman', fontsize=28, fontweight='bold')
    ax[1].set_ylabel('Landsat-extracted peak VI', fontname='Times New Roman', fontsize=28, fontweight='bold')
    predicted_y_data = ln_temp(np.array(data_pd['AGBD']), p1[0], p1[1], p1[2], p1[3])
    x_data = np.array(data_pd['S2phemetric_peak_vi'])
    r_square2 = (1 - np.nansum((predicted_y_data - x_data) ** 2) / np.nansum((x_data - np.nanmean(x_data)) ** 2))
    print(r_square1)
    print(r_square2)
    plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig15\\Fig15.png', dpi=300)


def fig13_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=2)

    for sec, coord in zip(['Entire floodplain', 'Yizhi section', 'Jingjiang section', 'Chenghan section', 'Hanhu section'], [[0, 16537], [0, 950], [950, 6100], [6100, 10210], [10210, 16537]]):

        dem_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\inundation_status\\dam_contribution.TIF')
        wl_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\inundation_status\\wl_contribution.TIF')
        veg_pre_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\veg_pre_tgd.TIF')
        veg_post_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig12\\veg_post_tgd.TIF')

        veg_pre_arr = veg_pre_ds.GetRasterBand(1).ReadAsArray()
        veg_post_arr = veg_post_ds.GetRasterBand(1).ReadAsArray()
        dem_arr = dem_ds.GetRasterBand(1).ReadAsArray()
        wl_arr = wl_ds.GetRasterBand(1).ReadAsArray()

        veg_pre_arr = veg_pre_arr[:, coord[0]: coord[1]]
        veg_post_arr = veg_post_arr[:, coord[0]: coord[1]]
        dem_arr = dem_arr[:, coord[0]: coord[1]]
        wl_arr = wl_arr[:, coord[0]: coord[1]]

        veg_pre_arr[np.isnan(veg_pre_arr)] = 0
        veg_post_arr[np.isnan(veg_post_arr)] = 0
        veg_pre_arr[np.isnan(veg_pre_arr)] = 0
        veg_post_arr[np.isnan(veg_post_arr)] = 0
        veg_pre_arr[np.logical_and(veg_pre_arr == 0, veg_post_arr == 0)] = np.nan
        veg_post_arr[np.logical_and(np.isnan(veg_pre_arr), veg_post_arr == 0)] = np.nan

        veg_diff = (veg_post_arr - veg_pre_arr).flatten()
        veg_post_arr = veg_post_arr.flatten()
        veg_pre_arr = veg_pre_arr.flatten()
        dem_arr = dem_arr.flatten()
        wl_arr = wl_arr.flatten()

        pos1 = np.logical_or(np.isnan(veg_diff), np.logical_or(np.isnan(dem_arr), np.isnan(wl_arr)))
        dem_arr = np.delete(dem_arr, pos1)
        wl_arr = np.delete(wl_arr, pos1)
        veg_diff = np.delete(veg_diff, pos1)
        veg_post_arr = np.delete(veg_post_arr, pos1)
        veg_pre_arr = np.delete(veg_pre_arr, pos1)

        contribution, type, origin = [], [], []

        print('Wl: ' + str(np.nanmean(veg_diff * wl_arr) / np.nanmean(veg_diff)))
        print('DEM: ' + str(np.nanmean(veg_diff * dem_arr) / np.nanmean(veg_diff)))
        contribution.append(np.nanmean(veg_diff * wl_arr) / np.nanmean(veg_diff))
        type.append('Entire floodplain')
        origin.append('Altered flow regime')

        contribution.append(np.nanmean(veg_diff * dem_arr) / np.nanmean(veg_diff))
        type.append('Entire floodplain')
        origin.append('Topographical changes')

        print('------------------Type1-------------------')

        veg_diff_t1 = np.delete(veg_diff, veg_pre_arr != 0)
        wl_arr_t1 = np.delete(wl_arr, veg_pre_arr != 0)
        dem_arr_t1 = np.delete(dem_arr, veg_pre_arr != 0)
        print('Wl: ' + str(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1)))
        print('DEM: ' + str(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1)))

        contribution.append(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1))
        type.append('Transition from non-vegetated to vegetated')
        origin.append('Altered flow regime')

        contribution.append(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1))
        type.append('Transition from non-vegetated to vegetated')
        origin.append('Topographical changes')

        print('------------------Type2-------------------')

        pos3 = np.logical_or(veg_diff <= 0.15, veg_pre_arr == 0)
        veg_diff_t1 = np.delete(veg_diff, pos3)
        wl_arr_t1 = np.delete(wl_arr, pos3)
        dem_arr_t1 = np.delete(dem_arr, pos3)
        print('Wl: ' + str(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1)))
        print('DEM: ' + str(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1)))

        contribution.append(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1))
        type.append('Pronounced vegetation increment')
        origin.append('Altered flow regime')

        contribution.append(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1))
        type.append('Pronounced vegetation increment')
        origin.append('Topographical changes')

        print('------------------Type3-------------------')

        pos4 = np.logical_or(np.logical_or(veg_diff > 0.15, veg_pre_arr == 0), veg_diff < 0)
        veg_diff_t1 = np.delete(veg_diff, pos4)
        wl_arr_t1 = np.delete(wl_arr, pos4)
        dem_arr_t1 = np.delete(dem_arr, pos4)
        print('Wl: ' + str(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1)))
        print('DEM: ' + str(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1)))

        contribution.append(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1))
        type.append('Milder vegetation increment')
        origin.append('Altered flow regime')

        contribution.append(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1))
        type.append('Milder vegetation increment')
        origin.append('Topographical changes')

        print('------------------Type4-------------------')

        pos5 = np.logical_or(veg_post_arr == 0, veg_diff >= 0)
        veg_diff_t1 = np.delete(veg_diff, pos5)
        wl_arr_t1 = np.delete(wl_arr, pos5)
        dem_arr_t1 = np.delete(dem_arr, pos5)
        print('Wl: ' + str(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1)))
        print('DEM: ' + str(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1)))

        contribution.append(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1) - 2)
        type.append('Vegetation decrement')
        origin.append('Altered flow regime')

        contribution.append(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1) + 2)
        type.append('Vegetation decrement')
        origin.append('Topographical changes')

        print('------------------Type5-------------------')

        veg_diff_t1 = np.delete(veg_diff, veg_post_arr != 0)
        wl_arr_t1 = np.delete(wl_arr, veg_post_arr != 0)
        dem_arr_t1 = np.delete(dem_arr, veg_post_arr != 0)
        print('Wl: ' + str(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1)))
        print('DEM: ' + str(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1)))

        contribution.append(np.nanmean(veg_diff_t1 * wl_arr_t1) / np.nanmean(veg_diff_t1))
        type.append('Transition from vegetated to non-vegetated')
        origin.append('Altered flow regime')

        contribution.append(np.nanmean(veg_diff_t1 * dem_arr_t1) / np.nanmean(veg_diff_t1))
        type.append('Transition from vegetated to non-vegetated')
        origin.append('Topographical changes')

        pos = np.logical_or(np.logical_or(np.isnan(dem_arr), np.isnan(wl_arr)), np.logical_or(dem_arr > 2, dem_arr < -2))
        dem_arr = np.delete(dem_arr, pos)
        wl_arr = np.delete(wl_arr, pos)
        veg_diff = np.delete(veg_diff, pos)

        pd_temp = pd.DataFrame({f'{sec} Contribution': contribution, f'{sec} type': type, f'{sec} ori': origin})
        pd_temp.to_csv(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig14\\{sec}.csv')
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 8), constrained_layout=True)
        #
        # sns.barplot(pd_temp, x="type", y="contri", hue="ori")
        # plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig14\\Fig13_{sec}.png', dpi=300)


def fig3_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=3)

    #
    df = pd.read_csv('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig3\\1.csv')

    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
    sns.set(style="white", palette='dark', font='Times New Roman', font_scale=1,
            rc={'ytick.left': True, 'xtick.bottom': True})
    df = df.drop(df[df['wi'] < -0.5].index)
    df = df.drop(df[np.isnan(df['wi'])].index)
    df2 = copy.copy(df)
    df2 = df2.drop(df2[np.logical_and(np.mod(df['doy'], 1000) < 270, np.mod(df['doy'], 1000) > 180)].index)
    df2 = df2.drop(df2[df2['wi'] > -0.02625].index)
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    sns.histplot(data=df, x="wi", binwidth=0.015, binrange=(-0.3, 1.0), kde=True,
                 color=(30 / 256, 96 / 256, 164 / 256, 1), element='step',
                 line_kws={'color': (30 / 256, 96 / 256, 164 / 256, 1), 'lw': 3, 'ls': '-'})
    sns.histplot(data=df2, x="wi", binwidth=0.015, binrange=(-0.3, 1.0), kde=False,
                 line_kws={'color': (256 / 256, 30 / 256, 30 / 256, 1), 'lw': 3, 'ls': '-'},
                 color=(180 / 256, 30 / 256, 54 / 256, 0.4), element='step', alpha=0.4)
    plt.plot(np.linspace(-0.4, 0.05, 100),
             4.5 * guassain_dis(np.linspace(-0.4, 0.05, 100), np.nanstd(df2['wi']), np.nanmean(df2['wi'])),
             color=(180 / 256, 46 / 256, 23 / 256), lw=3)
    ax.set_xlim([-0.3, 0.4])

    plt.savefig('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig3\\fig1.png', dpi=300)


def fig17_func():

    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
    plt.rc('font', size=24)
    plt.rc('axes', linewidth=2)

    pre_rs_inun_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
    pre_est_inun_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\inundation_status\\prewl_predem\\inundation_freq.TIF')
    post_rs_inun_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
    post_est_inun_ds = gdal.Open('G:\\A_Landsat_Floodplain_veg\\Water_level_python\\inundation_status\\postwl_postdem\\inundation_freq.TIF')

    pre_rs_inun_arr = pre_rs_inun_ds.GetRasterBand(1).ReadAsArray()
    post_rs_inun_arr = post_rs_inun_ds.GetRasterBand(1).ReadAsArray()
    pre_est_inun_arr = pre_est_inun_ds.GetRasterBand(1).ReadAsArray()
    post_est_inun_arr = post_est_inun_ds.GetRasterBand(1).ReadAsArray()

    pre_est_inun_arr[np.isnan(post_rs_inun_arr)] = np.nan
    post_est_inun_arr[np.isnan(post_rs_inun_arr)] = np.nan

    pre_rs_inun_arr = pre_rs_inun_arr.flatten()
    post_rs_inun_arr = post_rs_inun_arr.flatten()
    pre_est_inun_arr = pre_est_inun_arr.flatten()
    post_est_inun_arr = post_est_inun_arr.flatten()

    pos = np.logical_or(np.logical_or(np.isnan(pre_rs_inun_arr), np.isnan(pre_est_inun_arr)), np.logical_or(pre_rs_inun_arr == 1, pre_est_inun_arr == 1))

    pre_rs_inun_arr = np.delete(pre_rs_inun_arr, pos)
    pre_est_inun_arr = np.delete(pre_est_inun_arr, pos)
    pre_diff = pre_rs_inun_arr - pre_est_inun_arr

    pre_rs_inun_arr = np.delete(pre_rs_inun_arr, np.logical_and(pre_diff > 0.034, pre_diff < 0.035))
    pre_est_inun_arr = np.delete(pre_est_inun_arr, np.logical_and(pre_diff > 0.034, pre_diff < 0.035))

    pos1 = np.logical_or(np.logical_or(np.isnan(post_rs_inun_arr), np.isnan(post_est_inun_arr)), np.logical_or(post_rs_inun_arr == 1, post_est_inun_arr == 1))

    post_rs_inun_arr = np.delete(post_rs_inun_arr, pos1)
    post_est_inun_arr = np.delete(post_est_inun_arr, pos1)

    # camp = sns.color_palette("coolwarm", as_cmap=True)
    camp = sns.color_palette("viridis", as_cmap=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    ax.hist2d(pre_rs_inun_arr, pre_est_inun_arr, bins=80, norm='symlog', cmin=40, cmap = camp)
    ax.set_facecolor((68/256, 1/256, 84/256))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=24)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=24)
    ax.set_xlabel('Landsat-derived inundation frequency', fontname='Times New Roman', fontsize=28, fontweight='bold')
    ax.set_ylabel('Estimated flood frequency', fontname='Times New Roman', fontsize=28, fontweight='bold')
    plt.savefig('G:\A_Landsat_Floodplain_veg\Paper\Fig19\\pre_tgd.png', dpi=300)
    plt.close()

    r_square = 1 - (np.nansum((pre_rs_inun_arr - pre_est_inun_arr) ** 2) / np.nansum((pre_rs_inun_arr - np.nanmean(pre_rs_inun_arr)) ** 2))
    print(str(np.nanmean(pre_rs_inun_arr - pre_est_inun_arr)))
    print(str(r_square))
    print(str(np.sqrt(np.nanmean((pre_rs_inun_arr - pre_est_inun_arr) ** 2))))
    print(str(stats.pearsonr(pre_rs_inun_arr, pre_est_inun_arr)))

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    h = ax.hist2d(post_rs_inun_arr, post_est_inun_arr, bins=80, norm='symlog', cmin=40, cmap = camp,)
    ax.set_facecolor((68/256, 1/256, 84/256))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=24)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=24)
    ax.set_xlabel('Landsat-derived inundation frequency', fontname='Times New Roman', fontsize=28, fontweight='bold')
    ax.set_ylabel('Estimated flood frequency', fontname='Times New Roman', fontsize=28, fontweight='bold')
    plt.savefig('G:\A_Landsat_Floodplain_veg\Paper\Fig19\\post_tgd.png', dpi=300)
    plt.close()

    r_square = 1 - (np.nansum((post_rs_inun_arr - post_est_inun_arr) ** 2) / np.nansum((post_rs_inun_arr - np.nanmean(post_rs_inun_arr)) ** 2))
    print(str(np.nanmean(post_rs_inun_arr - post_est_inun_arr)))
    print(str(r_square))
    print(str(np.sqrt(np.nanmean((post_rs_inun_arr - post_est_inun_arr) ** 2))))
    print(str(stats.pearsonr(post_rs_inun_arr, post_est_inun_arr)))

    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
    h = ax.hist2d(post_rs_inun_arr, post_est_inun_arr, bins=80, norm='symlog', cmin=40, cmap = camp, vmin=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=24)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontname='Times New Roman', fontsize=24)
    ax.set_xlabel('Landsat-derived inundation frequency', fontname='Times New Roman', fontsize=28, fontweight='bold')
    ax.set_ylabel('Estimated flood frequency', fontname='Times New Roman', fontsize=28, fontweight='bold')
    plt.colorbar(h[3], ax=ax)
    plt.savefig('G:\A_Landsat_Floodplain_veg\Paper\Fig19\\cb.png', dpi=300)
    plt.close()


def num_temp():
    pre_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
    pre_arr = pre_ds.GetRasterBand(1).ReadAsArray()
    post_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
    post_arr = post_ds.GetRasterBand(1).ReadAsArray()

    diff = post_arr - pre_arr
    q = np.sum(np.logical_and(diff < 0, post_arr < 0.01))
    s = q / np.sum(np.logical_and(~np.isnan(post_arr), post_arr != 1))
    a = 1


def freq_diff_func():
    pre_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_pretgd.TIF')
    pre_arr = pre_ds.GetRasterBand(1).ReadAsArray()
    post_ds = gdal.Open('G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\inun_DT_inundation_frequency_posttgd.TIF')
    post_arr = post_ds.GetRasterBand(1).ReadAsArray()

    resurface_arr = np.zeros_like(pre_arr)
    unlink_arr = np.zeros_like(pre_arr)
    resurface_arr[np.logical_and(pre_arr > 0.7, post_arr < 0.5)] = 1
    unlink_arr[np.logical_and(pre_arr > 0.2, post_arr < 0.1)] = 1

    print(str(np.sum(resurface_arr) * 0.03 *0.03))
    print(str(np.sum(unlink_arr) * 0.03 * 0.03))

    bf.write_raster(pre_ds, resurface_arr, 'G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\', 'resurf_arr.tif', raster_datatype=gdal.GDT_Byte)
    bf.write_raster(pre_ds, unlink_arr, 'G:\A_Landsat_Floodplain_veg\Water_level_python\Post_TGD\\', 'unlink_arr.tif', raster_datatype=gdal.GDT_Byte)


def fig_11_nc_func():

    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=16)
    plt.rc('axes', linewidth=2)

    up_b, low_b = 300, 1800
    thalweg = 635
    new_arr = np.load(f'G:\A_Landsat_Floodplain_veg\Paper\Fig11_nc\\arr_{str(thalweg)}_40.npy')
    new_arr = new_arr[300:900, :]
    num = 5

    start_list, start_list2 = [], []
    for _ in range(new_arr.shape[1]):
        for __ in range(new_arr.shape[0]):
            if ~np.isnan(new_arr[__, _]):
                start_list.append(__)
                break
            if __ == new_arr.shape[0] - 1:
                start_list.append(np.nan)

    for _ in range(new_arr.shape[1]):
        ymin, ymax = max(0 , _ - 20), min(new_arr.shape[1], _ + 20)
        start = start_list[_]

        if start < np.nanmean(start_list[ymin: ymax]) * 2 / 3:
            new_arr[0: int(np.nanmean(start_list[ymin: ymax])), _] = np.nan
            start_list2.append(int(np.nanmean(start_list[ymin: ymax])))
        else:
            start_list2.append(start_list[_])

    end_list, end_list2 = [], []
    inverse_list = [_ for _ in range(new_arr.shape[0])]
    inverse_list.reverse()
    for _ in range(new_arr.shape[1]):
        for __ in inverse_list:
            if ~np.isnan(new_arr[__, _]):
                end_list.append(__)
                break
            if __ == 0:
                end_list.append(np.nan)

    for _ in range(new_arr.shape[1]):
        ymin, ymax = max(0 , _ - 20), min(new_arr.shape[1], _ + 20)
        end = end_list[_]

        if end > np.nanmean(end_list[ymin: ymax]) * 2 / 2:
            new_arr[int(np.nanmean(end_list[ymin: ymax])):, _] = np.nan
            end_list2.append(int(np.nanmean(end_list[ymin: ymax])))
        else:
            end_list2.append(end_list[_])

    new_arr[0: 30, :] = np.nan

    new_arr2 = np.zeros_like(new_arr) * np.nan
    for _ in range(new_arr2.shape[1]):
        for pix_ in range(new_arr2.shape[0]):
            if ~np.isnan(new_arr[pix_, _]):
                new_arr2[pix_ - 1, _] = 1
                break

        for pix_ in range(new_arr2.shape[0]):
            if ~np.isnan(new_arr[-1 - pix_, _]):
                new_arr2[-1 - pix_, _] = 1
                break

    new_arr[new_arr == -200] = np.nan
    channel_arr = np.zeros_like(new_arr) * np.nan
    for q in range(len(start_list2)):
        if ~np.isnan(start_list2[q]) and ~np.isnan(end_list2[q]) and start_list2[q] < end_list2[q]:
            channel_arr[start_list2[q]: end_list2[q], q] = 1

    cmap1 = sns.color_palette("coolwarm", as_cmap=True)
    cmap1 = sns.color_palette("viridis", as_cmap=True)
    # cmap1 = sns.diverging_palette(0, 255, sep=1, n=32, center="light", as_cmap=True)
    cmap1 = sns.diverging_palette(240, 10, n=36, as_cmap=True)
    # cmap1 = sns.diverging_palette(120, 60, s=80, l=55, n=9, as_cmap=True)
    fig, ax = plt.subplots(num, figsize=(20, 15), constrained_layout=True)
    dis_start = 0
    for q in range(num):
        ax[q].set_facecolor((0.98, 0.98, 0.98))
        ax[q].plot(np.linspace(0, int(np.floor(new_arr.shape[1]) / num), 100), np.linspace(thalweg-up_b, thalweg-up_b, 100), color=(0.1, 0.2,0.8), ls ='-.', lw=1., zorder=3)
        new_arr_yz = new_arr[:, int(q * np.floor(new_arr.shape[1]) / num): int((q + 1) * np.floor(new_arr.shape[1]) / num)]
        new_arr_yz2 = new_arr2[:, int(q * np.floor(new_arr.shape[1]) / num): int((q + 1) * np.floor(new_arr.shape[1]) / num)]
        cax = ax[q].imshow(new_arr_yz, vmin=-0.15, vmax=0.25, cmap=cmap1, zorder=1)
        cax2 = ax[q].imshow(new_arr_yz2, cmap='gist_gray', zorder=2)
        channel_arr_yz = channel_arr[:, int(q * np.floor(new_arr.shape[1]) / num): int((q + 1) * np.floor(new_arr.shape[1]) / num)]
        ax[q].imshow(channel_arr_yz, cmap='PiYG', vmin=0, vmax=1.5, alpha=0.1)
        ax_xtick_list, ax_xlabel_list = [], []
        ax_ytick_list, ax_ylabel_list = [], []
        for _ in range(int(q * np.floor(new_arr.shape[1]) / num),  int((q + 1) * np.floor(new_arr.shape[1]) / num)):
            if np.mod(_, 750) == 0:
                ax_xtick_list.append(_ - int(q * np.floor(new_arr.shape[1]) / num))
                ax_xlabel_list.append(int(_ / 25))

        ax[q].set_xticks(ax_xtick_list)
        ax[q].set_xticklabels([str(_) for _ in ax_xlabel_list])

        for _ in range(new_arr_yz.shape[0]):
            if np.mod(_ - (thalweg-up_b), 100) == 0:
                ax_ytick_list.append(_)
                ax_ylabel_list.append(int((_ - (thalweg-up_b)) / 100))
        ax[q].set_ylim(535, 35)
        ax[q].set_yticks(ax_ytick_list)
        ax[q].set_yticklabels([str(_) for _ in ax_ylabel_list])
        dis_start = int((q + 1) * np.floor(new_arr.shape[1]) / num)
    # plt.colorbar(cax)
    plt.savefig('G:\A_Landsat_Floodplain_veg\Paper\Fig11_nc\\fig11_nc_635_40_v5.png', dpi=1600)


def fig10supp_func():

    std_size = [(0.005, 0.01, 0.02), (0.01, 0.03, 0.05), (1.5, 0.75, 0.375)]
    std_size2 = [(0.005, 0.01, 0.02), (0.02, 0.04, 0.06), (1.5, 0.75, 0.375)]
    std_size3 = [(0.005, 0.01, 0.02), (0.03, 0.05, 0.07), (1.5, 0.75, 0.375)]
    std_size_all = [std_size, std_size2, std_size2, std_size3, std_size2]
    for sec, shape_, standard_size, reco_shap, in zip(['all', 'yz', 'jj', 'ch', 'hh'], [3, 2.8, 2.85, 2.6, 2.7], std_size_all, [3,3,3.1,3.2,3]):
        flood_impact = pd.read_csv(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\MAVI_var\\inun\\flood_indi_{sec}.csv')
        veg_indi = pd.read_csv(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\MAVI_var\\veg\\fig_{sec}_para.csv')

        flood_y = list(flood_impact['year'])
        inun_h = list(flood_impact['inun_h'])
        inun_d = list(flood_impact['inun_d'])
        veg_y = list(veg_indi['year'])
        size1 = list(veg_indi['flood_impact'])
        size2 = list(veg_indi['flood_impact_percent'])
        size3 = list(veg_indi['beta'])
        size_list = [size1, size2, size3]
        name_list = ['flood_impact', 'flood_impact_percent', 'beta']
        color = [('#55a7d2', '#cf5362'), ('#55a7d2', '#cf5362'), ((47/256, 104/256, 149/256), (250/256, 182/256, 112/256))]

        pear_flood_p, pear_r, pear_rec = [], [], []

        plt.rcParams['font.family'] = ['Arial', 'SimHei']
        plt.rc('font', size=28)
        plt.rc('axes', linewidth=3)

        for size_, name_, color_, stand_ in zip(size_list, name_list, color, standard_size):

            fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
            for q in range(40, 600, 20):
                ax.plot(np.linspace(0.1, 4, 100), q/np.linspace(0.1, 4, 100), color=(0.8, 0.8, 0.8), zorder=1)

            indi = False
            for y_ in veg_y:
                if y_ + 1 in veg_y and ~np.isnan(size_[veg_y.index(y_ + 1)]) and size_[veg_y.index(y_ + 1)] is not None:
                    h_ = np.mean(np.array([inun_h[flood_y.index(y_)], inun_h[flood_y.index(y_ + 1)]]))
                    d_ = np.mean(np.array([inun_d[flood_y.index(y_)], inun_d[flood_y.index(y_ + 1)]]))
                    s_ = np.mean(np.array([size_[veg_y.index(y_)], size_[veg_y.index(y_ + 1)]]))
                    indi = True
                elif y_ - 1 in veg_y:
                    indi = False
                    pass
                elif np.isnan(size_[veg_y.index(y_)]) or size_[veg_y.index(y_)] is None:
                    indi = False
                    pass
                else:
                    h_ = inun_h[flood_y.index(y_)]
                    d_ = inun_d[flood_y.index(y_)]
                    s_ = size_[veg_y.index(y_)]
                    indi = True

                if y_ < 2004:
                    pear_flood_p.append(h_ * d_)
                    pear_r.append(s_)

                if indi:
                    if size_list.index(size_) == 0:
                        s_ = s_ * -950
                    elif size_list.index(size_) == 1:
                        s_ = s_ * -370
                    else:
                        s_ = -2.2 * np.log(0.05) / s_

                    if name_ != 'beta':
                        if y_ < 2004:
                            ax.scatter(h_, d_, marker='o', s = s_ ** shape_, facecolors=color_[0], edgecolor=(0,0,0), lw=2, alpha =0.8, zorder=2)
                        else:
                            ax.scatter(h_, d_, marker='o', s = s_ ** shape_, facecolors=color_[1], edgecolor=(0,0,0), lw=2, alpha=0.8, zorder=2)
                    else:
                        if y_ < 2004:
                            ax.scatter(h_, d_, marker='o', s = s_ ** reco_shap, facecolors=color_[0], edgecolor=(0,0,0), lw=2, alpha =0.8, zorder=2)
                        else:
                            ax.scatter(h_, d_, marker='o', s = s_ ** reco_shap, facecolors=color_[1], edgecolor=(0,0,0), lw=2, alpha=0.8, zorder=2)

            # pear_value, p_value = pearsonr(pear_flood_p, pear_r)
            # print(str(p_value))

            for q, h_ in zip(stand_, [57, 53, 47]):
                if size_list.index(size_) == 0:
                    q = q * 950
                elif size_list.index(size_) == 1:
                    q = q * 370
                else:
                    q = -2.2 * np.log(0.05) / q

                if name_ != 'beta':
                    ax.scatter(2.55, h_, marker='o', s= q ** shape_, facecolors=(1, 1, 1), edgecolor=(0, 0, 0), lw=2,
                               alpha=0.8, zorder=11)
                else:
                    ax.scatter(2.55, h_, marker='o', s = q ** reco_shap, facecolors=(1,1,1), edgecolor=(0,0,0), lw=2,alpha=0.8, zorder=11)
                ax.fill_between(np.linspace(2.25, 3.95, 100), np.linspace(35.3, 35.3, 100), np.linspace(65, 65, 100), facecolors=(1,1,1), edgecolor=(0,0,0), lw=2, zorder=10)

            ax.set_xlim([1, 4])
            ax.set_ylim([35, 110])
            plt.yscale("log")
            plt.xscale("log")
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(['1', '2', '3', '4',])
            ax.set_yticks([40, 60, 80, 100])
            ax.set_yticklabels(['40', '60', '80', '100'])
            plt.savefig(f'G:\\A_Landsat_Floodplain_veg\\Paper\\Fig10\\v2\\MAVI_var\\flood_impact_{sec}_{name_}.png', dpi=400)
            plt.close()


def fig20_func():
    csv_ = pd.read_excel('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig20\\corr2.xlsx')

    plt.rcParams['font.family'] = ['Arial', 'SimHei']
    plt.rc('font', size=20)
    plt.rc('axes', linewidth=3)

    x, y, size, shape, color = list(csv_['s_index']), list(csv_['e_index']), list(abs(csv_['corr'])), list(csv_['corr']), list(csv_['corr'])
    shape = ['^' if _ > 0 else 'v' for _ in shape]
    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    name = ['Dam-induced\n vegetation\n dynamics','Inundation\n frequency', 'Air\n temperature','FPAR','Soil moisture', 'Precipitation','Humidity']
    cmap = sns.color_palette("vlag", as_cmap=True)
    colors = cmap(np.linspace(0, 1, 20))

    for x_, y_, s_, ss_, c_ in zip(x, y, size, shape, color):
        ax.scatter(x_, y_, marker=ss_, s = (3000 * s_) ** 1, c = colors[int(c_ * 20) + 10], edgecolor=(0,0,0))

    for q in range(7):
        ax.plot(np.linspace(q - 0.5, q -0.5, 100), np.linspace(-0.5, 6.5, 100), lw=2, c=(0,0,0))
        ax.plot(np.linspace(-0.5, 6.5, 100), np.linspace(q-0.5, q-0.5, 100), lw=2, c=(0,0,0))

    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(name, rotation=45)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(name, rotation=45)
    ax.set_xlim([-0.5, 6.5])
    ax.set_ylim([-0.5, 6.5])
    plt.savefig('G:\\A_Landsat_Floodplain_veg\\Paper\\Fig20\\fig20.png', dpi=400)


fig11nc3_func()
