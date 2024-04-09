import os.path
import pandas as pd
from RSDatacube.RSdc import *
from skimage import io, feature
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
from River_GIS.River_GIS import *


def fig8_func():
    for year in range(1989, 2021):
        for sec, content in zip(['yz', 'jj', 'ch', 'hh', 'all'], [(0, 4827, 0, 3950), (0, 4827, 950, 6100), (0, 4827, 6100, 10210), (0, 4827, 10210, 16537), None]):
            if not os.path.exists(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\{str(year)}_{sec}.csv'):
                if sec != 'all':
                    pheme_pri = Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(year)}\\')
                    pheme_post = Phemetric_dc(f'G:\\A_Landsat_veg\\Landsat_floodplain_2020_datacube\\OSAVI_noninun_curfit_datacube\\floodplain_2020_Phemetric_datacube\\{str(year + 1)}\\')

                    pheme_pri_arr = pheme_pri.dc.SM_group[f'{str(year)}_peak_vi'].toarray() if isinstance(pheme_pri.dc, NDSparseMatrix) else pheme_pri.dc[:,:,]
                    pheme_post_arr = pheme_post.dc.SM_group[f'{str(year + 1)}_peak_vi'].toarray() if isinstance(pheme_pri.dc, NDSparseMatrix) else pheme_pri.dc[:,:,]
                    pheme_pri_arr_MAVI = pheme_pri.dc.SM_group[f'{str(year)}_MAVI'].toarray() if isinstance(pheme_pri.dc, NDSparseMatrix) else pheme_pri.dc[:,:,]
                    pheme_post_arr_MAVI = pheme_post.dc.SM_group[f'{str(year + 1)}_MAVI'].toarray() if isinstance(pheme_pri.dc, NDSparseMatrix) else pheme_pri.dc[:,:,]

                    inunh_ds = gdal.Open(f'G:\\A_Landsat_veg\\Water_level_python\\Inundation_indicator\\{str(year)}\\inundation_factor\\{str(year)}_inun_mean_wl.tif')
                    inunh_arr = inunh_ds.GetRasterBand(1).ReadAsArray()

                    inund_ds = gdal.Open(f'G:\\A_Landsat_veg\\Water_level_python\\Inundation_indicator\\{str(year)}\\inundation_factor\\{str(year)}_inun_duration.tif')
                    inund_arr = inund_ds.GetRasterBand(1).ReadAsArray()

                    arr_pri = pheme_pri_arr[content[0]: content[1], content[2]: content[3]]
                    arr_post = pheme_post_arr[content[0]: content[1], content[2]: content[3]]
                    mavi_pri = pheme_pri_arr_MAVI[content[0]: content[1], content[2]: content[3]]
                    mavi_post = pheme_post_arr_MAVI[content[0]: content[1], content[2]: content[3]]
                    arr_mean_inunhei = inunh_arr[content[0]: content[1], content[2]: content[3]]
                    arr_inundur = inund_arr[content[0]: content[1], content[2]: content[3]]
            #
                    if arr_pri.shape[0] != arr_mean_inunhei.shape[0] or arr_pri.shape[1] != arr_mean_inunhei.shape[1]:
                        raise Exception('Error')
            #
                    dic_temp = {'peak_vi_pri': [], 'peak_vi_post': [], 'mavi_pri': [], 'mavi_post': [], 'mean_inun_h': [], 'inun_d': []}
                    for y in range(arr_pri.shape[0]):
                        for x in range(arr_pri.shape[1]):
                            if ~np.isnan(arr_pri[y, x]) and arr_pri[y, x] > 0.2 and ~np.isnan(arr_post[y, x]) and arr_post[y, x] > 0.2 and ~np.isnan(arr_inundur[y, x]):
                                dic_temp['mavi_pri'].append(mavi_pri[y, x])
                                dic_temp['mavi_post'].append(mavi_post[y, x])
                                dic_temp['peak_vi_pri'].append(arr_pri[y, x])
                                dic_temp['peak_vi_post'].append(arr_post[y, x])
                                dic_temp['mean_inun_h'].append(arr_mean_inunhei[y, x])
                                dic_temp['inun_d'].append(arr_inundur[y, x])
                    pd_temp = pd.DataFrame(dic_temp)
                    pd_temp.to_csv(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\{str(year)}_{sec}.csv')
                else:
                    for _ in ['yz', 'jj', 'ch', 'hh']:
                        pd_temp_ = pd.read_csv(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\{str(year)}_{_}.csv')
                        if _ != 'yz':
                            pd_all = pd.concat((pd_all, pd_temp_))
                        else:
                            pd_all = copy.deepcopy(pd_temp_)
                    pd_all.to_csv(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\{str(year)}_{sec}.csv')

            # sec
            pd_temp = pd.read_csv(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\{str(year)}_{sec}.csv')
            pd_temp['mavi_diff'] = pd_temp['mavi_post'] - pd_temp['mavi_pri']
            pd_temp['mavi_perc'] = (pd_temp['mavi_post'] - pd_temp['mavi_pri']) / pd_temp['mavi_pri']
            pd_temp['peak_vi_diff'] = pd_temp['peak_vi_post'] - pd_temp['peak_vi_pri']
            pd_temp['peak_vi_perc'] = (pd_temp['peak_vi_post'] - pd_temp['peak_vi_pri']) / pd_temp['peak_vi_pri']
            pd_temp_ = pd_temp[['mavi_diff', 'mavi_perc', 'peak_vi_diff', 'peak_vi_perc', 'mean_inun_h', 'inun_d']]
            pd_temp__ = pd_temp_.dropna()

            for _ in ['mavi', 'peak_vi']:
                box_list_inund_diff, mid_list_inund_diff = [], []
                box_list_inunh_diff, mid_list_inunh_diff = [], []
                box_list_inund_perc, mid_list_inund_perc = [], []
                box_list_inunh_perc, mid_list_inunh_perc = [], []

                inun_durlist = []
                mean_inun_h_list = []

                for q in range(140):
                    mean_inun_h_list.append(q)
                    box_temp = pd_temp__[(pd_temp__['mean_inun_h'] >= q / 20) & (pd_temp__['mean_inun_h'] < (q + 1) / 20)][f'{_}_diff']
                    box_list_inunh_diff.append(box_temp.sort_values()[int(box_temp.shape[0] * 0.05): int(box_temp.shape[0] * 0.95)])
                    mid_list_inunh_diff.append(np.nanmean(box_temp) if box_temp.shape[0] != 0 else np.nan)

                    box_temp = pd_temp__[(pd_temp__['mean_inun_h'] >= q / 20) & (pd_temp__['mean_inun_h'] < (q + 1) / 20)][f'{_}_perc']
                    box_list_inunh_perc.append(box_temp.sort_values()[int(box_temp.shape[0] * 0.05): int(box_temp.shape[0] * 0.95)])
                    mid_list_inunh_perc.append(np.nanmean(box_temp) if box_temp.shape[0] != 0 else np.nan)

                for q in range(90):
                    inun_durlist.append(q)
                    box_temp = pd_temp__[pd_temp__['inun_d'] == q][f'{_}_diff']
                    box_list_inund_diff.append(box_temp.sort_values()[int(box_temp.shape[0] * 0.05): int(box_temp.shape[0] * 0.95)])
                    mid_list_inund_diff.append(np.nanmean(box_temp) if box_temp.shape[0] != 0 else np.nan)

                    box_temp = pd_temp__[pd_temp__['inun_d'] == q][f'{_}_perc']
                    box_list_inund_perc.append(box_temp.sort_values()[int(box_temp.shape[0] * 0.05): int(box_temp.shape[0] * 0.95)])
                    mid_list_inund_perc.append(np.nanmean(box_temp) if box_temp.shape[0] != 0 else np.nan)

                # inund_perc
                fig1, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
                box_temp = ax1.boxplot(box_list_inund_perc, vert=True, notch=False, widths=0.98, patch_artist=True, whis=(15, 85), showfliers=False, zorder=4, )

                for patch in box_temp['boxes']:
                    patch.set_facecolor((72 / 256, 127 / 256, 166 / 256))
                    patch.set_alpha(0.5)
                    patch.set_linewidth(0.2)

                for median in box_temp['medians']:
                    median.set_lw(0.8)
                    median.set_color((255 / 256, 128 / 256, 64 / 256))

                bf.create_folder(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\Fig\\{_}\\inund_perc\\')
                bf.create_folder(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\Fig\\{_}\\inund_diff\\')
                bf.create_folder(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\Fig\\{_}\\inunh_perc\\')
                bf.create_folder(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\Fig\\{_}\\inunh_diff\\')

                ax1.plot(np.linspace(0, 90, 100), np.linspace(0, 0, 100), lw=2.5, c=(0, 0, 0), ls='-', zorder=5)
                ax1.scatter(inun_durlist, mid_list_inund_perc, marker='^', s=10 ** 2, facecolor=(0.8, 0, 0), zorder=7, edgecolor=(0.0, 0.0, 0.0), linewidth=0.2)
                ax1.plot(np.linspace(0, 90, 90), np.linspace(mid_list_inund_perc[0], mid_list_inund_perc[0], 90), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
                ax1.set_ylim(-0.1, 0.1)
                ax1.set_xlim([0.5, 61.5])
                ax1.set_yticks([-0.4, -0.2, 0, 0.2, 0.5])
                ax1.set_yticklabels(['-40%', '-20%', '0%', '20%', '40%'])
                ax1.set_xticks([1, 11, 21, 31, 41, 51, 61])
                ax1.set_xticklabels(['0', '10', '20', '30', '40', '50', '60'])

                plt.savefig(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\Fig\\{_}\\inund_perc\\{sec}_{str(year)}_inund_perc.png', dpi=300)
                plt.close()
                fig1, ax1 = None, None

                # inund_diff
                fig2, ax2 = plt.subplots(figsize=(10, 6), constrained_layout=True)
                box_temp = ax2.boxplot(box_list_inund_diff, vert=True, notch=False, widths=0.98, patch_artist=True, whis=(15, 85), showfliers=False, zorder=4, )

                for patch in box_temp['boxes']:
                    patch.set_facecolor((72 / 256, 127 / 256, 166 / 256))
                    patch.set_alpha(0.5)
                    patch.set_linewidth(0.2)

                for median in box_temp['medians']:
                    median.set_lw(0.8)
                    median.set_color((255 / 256, 128 / 256, 64 / 256))

                ax2.plot(np.linspace(0, 140, 100), np.linspace(0, 0, 100), lw=2.5, c=(0, 0, 0), ls='-', zorder=5)
                ax2.scatter(inun_durlist, mid_list_inund_diff, marker='^', s=10 ** 2, facecolor=(0.8, 0, 0), zorder=7, edgecolor=(0.0, 0.0, 0.0), linewidth=0.2)
                ax2.plot(np.linspace(0, 140, 140), np.linspace(mid_list_inund_diff[0], mid_list_inund_diff[0], 140), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
                ax2.set_ylim(-0.1, 0.1)
                ax2.set_xlim([0.5, 61.5])
                ax2.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
                ax2.set_xticks([1, 11, 21, 31, 41, 51, 61])
                plt.savefig(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\Fig\\{_}\\inund_diff\\{sec}_{str(year)}_inund_diff.png', dpi=300)
                plt.close()
                fig2, ax2 = None, None

                # inunh_perc
                fig3, ax3 = plt.subplots(figsize=(10, 6), constrained_layout=True)
                box_temp = ax3.boxplot(box_list_inunh_perc, vert=True, notch=False, widths=0.98, patch_artist=True, whis=(15, 85), showfliers=False, zorder=4, )

                for patch in box_temp['boxes']:
                    patch.set_facecolor((72 / 256, 127 / 256, 166 / 256))
                    patch.set_alpha(0.5)
                    patch.set_linewidth(0.2)

                for median in box_temp['medians']:
                    median.set_lw(0.8)
                    median.set_color((255 / 256, 128 / 256, 64 / 256))

                ax3.plot(np.linspace(0, 140, 100), np.linspace(0, 0, 100), lw=2.5, c=(0, 0, 0), ls='-', zorder=5)
                ax3.scatter(mean_inun_h_list, mid_list_inunh_perc, marker='^', s=10 ** 2, facecolor=(0.8, 0, 0), zorder=7, edgecolor=(0.0, 0.0, 0.0), linewidth=0.2)
                ax3.plot(np.linspace(0, 140, 140), np.linspace(mid_list_inunh_perc[0], mid_list_inunh_perc[0], 140), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
                ax3.set_ylim(-0.2, 0.2)
                ax3.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
                ax3.set_yticklabels(['-40%', '-20%', '0%', '20%', '40%'])
                ax3.set_xticks([1, 40, 80, 120, 140])
                ax3.set_xticklabels(['0', '2', '4', '6', '7'])
                ax3.set_xlim([0.5, 140.5])
                plt.savefig(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\Fig\\{_}\\inunh_perc\\{sec}_{str(year)}_inunh_perc.png', dpi=300)
                plt.close()
                fig3, ax3 = None, None

                # inunh_diff
                fig4, ax4 = plt.subplots(figsize=(10, 6), constrained_layout=True)
                box_temp = ax4.boxplot(box_list_inunh_diff, vert=True, notch=False, widths=0.19, patch_artist=True, whis=(15, 85), showfliers=False, zorder=4, )

                for patch in box_temp['boxes']:
                    patch.set_facecolor((72 / 256, 127 / 256, 166 / 256))
                    patch.set_alpha(0.5)
                    patch.set_linewidth(0.2)

                for median in box_temp['medians']:
                    median.set_lw(0.8)
                    median.set_color((255 / 256, 128 / 256, 64 / 256))

                ax4.plot(np.linspace(0, 20, 100), np.linspace(0, 0, 100), lw=2.5, c=(0, 0, 0), ls='-', zorder=5)
                ax4.scatter(mean_inun_h_list, mid_list_inunh_diff, marker='^', s=10 ** 2, facecolor=(0.8, 0, 0), zorder=7, edgecolor=(0.0, 0.0, 0.0), linewidth=0.2)
                ax4.plot(np.linspace(0, 20, 90), np.linspace(mid_list_inunh_diff[0], mid_list_inunh_diff[0], 90), lw=2, ls='--', c=(0.8, 0, 0), zorder=8)
                ax4.set_ylim(-0.1, 0.1)
                ax4.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
                ax4.set_xticks([1, 40, 80, 120, 140])
                ax4.set_xticklabels(['0', '2', '4', '6', '7'])
                ax4.set_xlim([0.5, 140.5])
                plt.savefig(f'G:\\A_Landsat_veg\\Paper2\\Fig8\\Fig\\{_}\\inunh_diff\\{sec}_{str(year)}_inunh_diff.png', dpi=300)
                plt.close()
                fig4, ax4 = None, None

fig8_func()