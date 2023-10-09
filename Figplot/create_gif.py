import os
from matplotlib.colors import ListedColormap
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2


def file_filter(file_path_temp, containing_word_list):
    file_list = os.listdir(file_path_temp)
    filter_list = []
    for file in file_list:
        for containing_word in containing_word_list:
            if containing_word in file:
                filter_list.append(file_path_temp + file)
            else:
                break
    return filter_list


def plot_ndvi_image(file_path):
    containing_word = ['.tif']
    vals = np.ones((200, 4))
    vals[0:47, 0] = 156 / 256
    vals[0:47, 1] = 156 / 256
    vals[0:47, 2] = 156 / 256
    vals[47:51, 0] = 176 / 256
    vals[47:51, 1] = 176 / 256
    vals[47:51, 2] = 164 / 256
    vals[51:78, 0] = 199 / 256
    vals[51:78, 1] = 198 / 256
    vals[51:78, 2] = 173 / 256
    vals[78:112, 0] = 219 / 256
    vals[78:112, 1] = 219 / 256
    vals[78:112, 2] = 178 / 256
    vals[112:133, 0] = 242 / 256
    vals[112:133, 1] = 242 / 256
    vals[112:133, 2] = 187 / 256
    vals[133:151, 0] = 227 / 256
    vals[133:151, 1] = 240 / 256
    vals[133:151, 2] = 168 / 256
    vals[151:166, 0] = 171 / 256
    vals[151:166, 1] = 209 / 256
    vals[151:166, 2] = 125 / 256
    vals[166:176, 0] = 122 / 256
    vals[166:176, 1] = 181 / 256
    vals[166:176, 2] = 85 / 256
    vals[176:183, 0] = 72 / 256
    vals[176:183, 1] = 150 / 256
    vals[176:183, 2] = 48 / 256
    vals[183:200, 0] = 20 / 256
    vals[183:200, 1] = 122 / 256
    vals[183:200, 2] = 11 / 256
    newcmp = ListedColormap(vals)
    i_t = 1
    filenames = []
    for i in file_filter(file_path, containing_word):
        ds_temp = gdal.Open(i)
        band = ds_temp.GetRasterBand(1).ReadAsArray()
        pic = plt.figure()
        plt.imshow(band, cmap=newcmp)
        plt.clim(-1, 1)
        plt.axis('off')

        date = i[i.find('\\20') + 1: i.find('\\20') + 9]
        date = date[0:4] + '/' + date[4:6] + '/' + date[6:8]
        pic.text(0.155, 0.81, date, fontsize=25, fontweight='bold', fontname='Times New Roman')
        pic.text(0.755, 0.895, 'NDVI', fontsize=15, fontweight='bold', fontname='Times New Roman')
        cbar = plt.colorbar()
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(10)
            t.set_fontname('Times New Roman')
        plt.savefig(file_path + str(i_t) + '.png', dpi=600)
        filenames.append(file_path + str(i_t) + '.png')
        i_t += 1
        plt.close()

    with imageio.get_writer(file_path + 'NDVI_motion.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)


def plot_ndwi_image(file_path):
    containing_word = ['.tif']
    vals = np.ones((200, 4))
    # vals[140:200, 0] = 12 / 256
    # vals[140:200, 1] = 16 / 256
    # vals[140:200, 2] = 120 / 256
    # vals[118:140, 0] = 26 / 256
    # vals[118:140, 1] = 65 / 256
    # vals[118:140, 2] = 143 / 256
    # vals[103:118, 0] = 15 / 256
    # vals[103:118, 1] = 119 / 256
    # vals[103:118, 2] = 163 / 256
    # vals[86:103, 0] = 0 / 256
    # vals[86:103, 1] = 148 / 256
    # vals[86:103, 2] = 69 / 256
    # vals[75:86, 0] = 67 / 256
    # vals[75:86, 1] = 125 / 256
    # vals[75:86, 2] = 0 / 256
    # vals[65:75, 0] = 120 / 256
    # vals[65:75, 1] = 109 / 256
    # vals[65:75, 2] = 25 / 256
    # vals[53:65, 0] = 130 / 256
    # vals[53:65, 1] = 98 / 256
    # vals[53:65, 2] = 51 / 256
    # vals[20:53, 0] = 148 / 256
    # vals[20:53, 1] = 103 / 256
    # vals[20:53, 2] = 83 / 256
    # vals[10:20, 0] = 186 / 256
    # vals[10:20, 1] = 159 / 256
    # vals[10:20, 2] = 147 / 256
    # vals[0:10, 0] = 224 / 256
    # vals[0:10, 1] = 224 / 256
    # vals[0:10, 2] = 224 / 256

    vals[180:200, 0] = 12 / 256
    vals[180:200, 1] = 16 / 256
    vals[180:200, 2] = 120 / 256
    vals[160:180, 0] = 26 / 256
    vals[160:180, 1] = 65 / 256
    vals[160:180, 2] = 143 / 256
    vals[120:160, 0] = 15 / 256
    vals[120:160, 1] = 119 / 256
    vals[120:160, 2] = 163 / 256
    vals[100:120, 0] = 0 / 256
    vals[100:120, 1] = 148 / 256
    vals[100:120, 2] = 256 / 256
    # vals[125:135, 0] = 67 / 256
    # vals[125:135, 1] = 125 / 256
    # vals[125:135, 2] = 0 / 256
    # vals[104:125, 0] = 120 / 256
    # vals[104:125, 1] = 109 / 256
    # vals[104:125, 2] = 25 / 256
    vals[87:115, 0] = 130 / 256
    vals[87:115, 1] = 98 / 256
    vals[87:115, 2] = 51 / 256
    vals[72:87, 0] = 148 / 256
    vals[72:87, 1] = 103 / 256
    vals[72:87, 2] = 83 / 256
    vals[60:72, 0] = 186 / 256
    vals[60:72, 1] = 159 / 256
    vals[60:72, 2] = 147 / 256
    vals[0:60, 0] = 224 / 256
    vals[0:60, 1] = 224 / 256
    vals[0:60, 2] = 224 / 256
    newcmp = ListedColormap(vals)
    i_t = 1
    filenames = []
    for i in file_filter(file_path, containing_word):
        ds_temp = gdal.Open(i)
        band = ds_temp.GetRasterBand(1).ReadAsArray()
        pic = plt.figure()
        plt.imshow(band, cmap=newcmp)
        plt.clim(-1, 1)
        plt.axis('off')
        date = i[i.find('\\20') + 1: i.find('\\20') + 9]
        date = date[0:4] + '/' + date[4:6] + '/' + date[6:8]
        pic.text(0.155, 0.81, date, fontsize=25, fontweight='bold', fontname='Times New Roman')
        pic.text(0.755, 0.895, 'NDWI', fontsize=15, fontweight='bold', fontname='Times New Roman')
        cbar = plt.colorbar()
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(10)
            t.set_fontname('Times New Roman')
        plt.savefig(file_path + str(i_t) + '.png', dpi=600)
        filenames.append(file_path + str(i_t) + '.png')
        i_t += 1
        plt.close()

    with imageio.get_writer(file_path + 'NDWI_motion.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)
            writer.append_data(image)


# ndvi_file_path = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\Sentinel2_L2A_output\\NDVI_gif\\'
# plot_ndvi_image(ndvi_file_path)
ndwi_file_path = 'D:\\A_Vegetation_Identification\\Wuhan_Sentinel_L2_Original\\Sentinel2_L2A_output\\NDWI_gif\\'
plot_ndwi_image(ndwi_file_path)
