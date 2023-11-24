# coding=utf-8
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from Aborted_codes import Landsat_main_v1
import os
import gdal
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.figure as fg
import seaborn as sns
from sklearn.model_selection import KFold
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimSun'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False
plt.rc('font', size=20)


def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i:  ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    # print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid, y_valid


def R_sqaure(predict_y, ori_y):
    if type(predict_y) is not np.ndarray:
        try:
            predict_y = np.array(predict_y).astype(np.float)
        except:
            print('Unknown datatype')
            return

    if type(ori_y) is not np.ndarray:
        try:
            ori_y = np.array(ori_y).astype(np.float)
        except:
            print('Unknown datatype')
            return
    return 1 - (np.nansum(np.square(predict_y-ori_y)) / np.nansum(np.square(ori_y - np.mean(ori_y))))



def RMSE(predict_y, ori_y):
    if type(predict_y) is not np.ndarray:
        try:
            predict_y = np.array(predict_y).astype(np.float)
        except:
            print('Unknown datatype')
            return

    if type(ori_y) is not np.ndarray:
        try:
            ori_y = np.array(ori_y).astype(np.float)
        except:
            print('Unknown datatype')
            return

    return np.sqrt(np.nanmean(np.square(predict_y-ori_y)))


def linear_function(x, a, b):
    return a * x + b


def high_order_linear_fucntion(x, a, b, c):
    return (a * x + b) ** c


def polynomial_function(x, a, b, c):
    return a * x ** 2 + b * x + c


#搭建MLP回归模型
class MLPregression(nn.Module):

    def __init__(self):
        super(MLPregression,self).__init__()
        self.hidden1=nn.Linear(in_features=11,out_features=26,bias=True)
        self.hidden2=nn.Linear(26,26) #100*100
        self.hidden3 = nn.Linear(26, 13)#100*100
        #回归预测层
        self.predict=nn.Linear(13,1)

    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        x=F.relu(self.hidden3(x))
        output=self.predict(x)
        if len(output.shape) == 1:
            return output[0]
        else:
            return output[:,0]


class Scatterplot(fg.Figure):

    def __init__(self, x_data, y_data, xlabel, y_axis=0):
        super().__init__(figsize=(10, 9), linewidth=5)
        plt.rc('axes', linewidth=3)
        plt.yticks(fontsize=20, fontname='TImes New Roman')
        plt.xticks(fontsize=20, fontname='TImes New Roman')
        self.ax1 = self.add_subplot()
        self.ydata = y_data
        self.xdata = x_data
        self.ylim = [0, 50]
        self.xlim = [min(x_data), max(x_data)]
        self.ax1.scatter(x_data, y_data, marker='o', s=10 ** 2, color='none', edgecolor=(196/256, 30/256, 30/256), linewidth=3)
        self.ax1.plot(np.linspace(y_axis, y_axis, 100), np.linspace(0, 50, 100), lw=4, **{'ls':':','color': (0, 0, 0)})
        self.ax1.plot(np.linspace(-10, 10, 100), np.linspace(5, 5, 100), lw=4, **{'ls':':', 'color': (0, 0, 0)})
        # thalweg_temp.ax1.text(thalweg_temp.xlim[0] + 9*(thalweg_temp.xlim[1] - thalweg_temp.xlim[0])/10, 5 + (thalweg_temp.ylim[1] - thalweg_temp.ylim[0])/20, 'Division', fontname='Times New Roman', fontsize=14, fontweight='bold')
        self.ax1.set_xlim(min(x_data), max(x_data))
        self.ax1.set_ylim(0, 50)
        self.ax1.set_ylabel('Chl-a concentration/ug/L', fontname='Times New Roman', fontsize=25, fontweight='bold')
        self.ax1.set_xlabel(xlabel, fontname='Times New Roman', fontsize=25, fontweight='bold')
        # thalweg_temp.ax1.set_yticks([0,10,20,30,40,50])
        # thalweg_temp.ax1.set_yticklabels([0,10,20,30,40,50], fontname='Times New Roman', fontsize=18)


    def save(self, fname):

        Basic_function.path_check(fname[0:fname.rindex('\\')])
        self.savefig(fname, dpi=500)


    def plot(self, func, para):
        try:
            if func.__code__.co_argcount != len(para) + 1:
                print('Please make sure the length of the para')
                return
        except:
            print('Please make sure the length of the para')
            return
        self.ax1.plot(np.linspace(self.xlim[0], self.xlim[1], 100), func(np.linspace(self.xlim[0], self.xlim[1], 100), *para), lw=4,  color=(0/256, 9/256, 200/256), **{'ls': '--'})
        R2 = R_sqaure(func(self.xdata, *para), self.ydata)
        r = RMSE(func(self.xdata, *para), self.ydata)
        # for x_temp in np.linspace(thalweg_temp.xlim[0] + (thalweg_temp.xlim[1] - thalweg_temp.xlim[0])/4, thalweg_temp.xlim[1] - (thalweg_temp.xlim[1] - thalweg_temp.xlim[0])/4, 1000):
        #     if thalweg_temp.ylim[0] + 4 * (thalweg_temp.ylim[1]-thalweg_temp.ylim[0]) / 5 < func(x_temp, *para) < thalweg_temp.ylim[0] + 5 * (thalweg_temp.ylim[1]-thalweg_temp.ylim[0]) / 6:
        #         break
        # thalweg_temp.ax1.text(x_temp + (thalweg_temp.xlim[1]-thalweg_temp.xlim[0])/20, func(x_temp, *para), r'R^2 = ' + str(R2)[0:5] + "\n" + 'RMSE = ' + str(r)[0:5], fontname='Times New Roman', fontsize=16, fontweight='bold')
        print(R2)
        print(r)


    def regression(self,function, xrange=None, yrange=None):
        if xrange is None:
            xrange = self.xlim
        if yrange is None:
            yrange = self.ylim
        if type(self.ydata) is not np.ndarray:
            try:
                y_temp2 = np.array(self.ydata).astype(np.float)
            except:
                print('Unknown datatype')
                return
        else:
            y_temp2 = self.ydata
        if type(self.xdata) is not np.ndarray:
            try:
                x_temp2 = np.array(self.xdata).astype(np.float)
            except:
                print('Unknown datatype')
                return
        else:
            x_temp2 = self.xdata
        size_temp = 0
        while size_temp < y_temp2.shape[0]:
            if y_temp2[size_temp] > yrange[1] or y_temp2[size_temp] < yrange[0]:
                y_temp2 = np.delete(y_temp2, size_temp)
                x_temp2 = np.delete(x_temp2, size_temp)
                size_temp -= 1
            size_temp += 1

        size_temp = 0
        while size_temp < x_temp2.shape[0]:
            if x_temp2[size_temp] > xrange[1] or x_temp2[size_temp] < xrange[0]:
                y_temp2 = np.delete(y_temp2, size_temp)
                x_temp2 = np.delete(x_temp2, size_temp)
                size_temp -= 1
            size_temp += 1

        paras, extra = curve_fit(function, x_temp2, y_temp2, maxfev=500000)
        self.ax1.plot(np.linspace(self.xlim[0], self.xlim[1], 100), function(np.linspace(self.xlim[0], self.xlim[1], 100), *paras), lw=4, color=(196 / 256, 30 / 256, 30 / 256), **{'ls': '--'})
        R2 = R_sqaure(function(x_temp2, *paras), y_temp2)
        r = RMSE(function(x_temp2, *paras), y_temp2)
        # for x_t in np.linspace(thalweg_temp.xlim[0] + (thalweg_temp.xlim[1] - thalweg_temp.xlim[0])/5, thalweg_temp.xlim[1] - (thalweg_temp.xlim[1] - thalweg_temp.xlim[0])/5, 1000):
        #     if thalweg_temp.ylim[0] + 4 * (thalweg_temp.ylim[1]-thalweg_temp.ylim[0]) / 5 < function(x_t, *paras) < thalweg_temp.ylim[0] + 5 * (thalweg_temp.ylim[1]-thalweg_temp.ylim[0]) / 6:
        #         break
        # thalweg_temp.ax1.text(x_t + (thalweg_temp.xlim[1] - thalweg_temp.xlim[0]) / 20, function(x_t, *paras),
        #               r'R^2 = ' + str(R2)[0:5] + "\n" + 'RMSE = ' + str(r)[0:5], fontname='Times New Roman',
        #               fontsize=16, fontweight='bold')
        print(R2)
        print(r)
        print(paras)


    def close(self):
        plt.close()
        plt.cla()
        plt.clf()


for i in Basic_function.file_filter('E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\xjb\\', ['.tif']):
    if not os.path.exists('E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\xjb_nm\\' +  i[i.find('xjb') + 4:]):
        date = i[i.find('S20') + 1: i.find('S20') + 9]
        tile = i[i.find('_48R') + 3:i.find('_48R') + 6]
        mndwi_file = Basic_function.file_filter('E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\NDWI\\xjb\\', ['.tif', str(date), str(tile)], and_or_factor='and')
        mndwi_raster = Basic_function.file2raster(mndwi_file[0])
        temp_raster = Basic_function.file2raster(i)
        temp_raster[mndwi_raster <= -0.03] = 0
        ori_ds1 = gdal.Open(i)
        Basic_function.write_raster(ori_ds1, temp_raster,'E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\xjb_nm\\', i[i.find('xjb') + 4:], raster_datatype=gdal.GDT_UInt16)

s2_fig = 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\S2_fig.xlsx'
sa_fig_df = pd.read_excel(s2_fig)
sa_fig_ds = np.array(sa_fig_df)[:, 0:9]
plt.rc('axes', linewidth=3)

fig1, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
for q in range(1,sa_fig_ds.shape[0]):
    if q == 40:
        ax.plot(sa_fig_ds[0, :], sa_fig_ds[q, :], lw=5, color=(256 / 256, 30 / 256, 30 / 256))
    else:
        ax.plot(sa_fig_ds[0,:], sa_fig_ds[q,:], lw=1.5, color=(120/256,120/256,120/256),**{'ls':'--'})
ax.set_ylim([0, 0.07])
ax.set_xlim([400,1000])
plt.yticks(fontsize=20, fontname='TImes New Roman')
plt.xticks(fontsize=20, fontname='TImes New Roman')
plt.savefig('E:\\A_Chl-a\\Fig\\s2_band.jpg', dpi=500)
plt.close()

data_file = 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Original_insitu_data.xlsx'
SRF_file = 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\S2-SRF.xlsx'

SRF_list = {}
data_list = pd.read_excel(data_file)
SRF_list['S2A'] = pd.read_excel(SRF_file, sheet_name='Spectral Responses (S2A)')
SRF_list['S2B'] = pd.read_excel(SRF_file, sheet_name='Spectral Responses (S2B)')
output_pd = data_list[['No', 'Site', 'Date',  'X', 'Y', 'Chl-a 0.1m', 'Chl-a 1m']]
output_pd['Lat'] = [float(lat_temp[0:2]) + float(lat_temp[3:5]) / 60 + float(lat_temp[6:11]) / 3600 for lat_temp in data_list['Lat']]
output_pd['Lon'] = [float(lon_temp[0:3]) + float(lon_temp[4:6]) / 60 + float(lon_temp[7:12]) / 3600 for lon_temp in data_list['Lon']]

if not os.path.exists('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\S2_Simulated_Band.xlsx'):
    for i in ['S2A', 'S2B']:
        for col in SRF_list[i].columns.tolist():
            if col != 'SR_WL':
                SR_list = []
                for no in output_pd['No'].tolist():
                    all_temp = np.nansum(np.array(SRF_list[i][col].tolist()))
                    SR_temp = 0
                    SRF_list_temp = np.array(SRF_list[i][['SR_WL', col]])
                    in_situ_list = data_list[data_list["No"] == no]
                    for band in range(SRF_list_temp.shape[0]):
                        if SRF_list_temp[band, 1] != 0:
                            SR_temp += in_situ_list[int(SRF_list_temp[band, 0])][no - 1] * (SRF_list_temp[band, 1] / all_temp)
                    SR_list.append(SR_temp)
                output_pd['Simulated_' + col[0:3] + '_' + col[-2:]] = SR_list
    output_pd.to_excel('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\S2_Simulated_Band.xlsx')
in_situ_df = pd.read_excel('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\S2_Simulated_Band.xlsx')
in_situ_df['D05'] = ((in_situ_df['Simulated_S2A_B6'] + in_situ_df['Simulated_S2B_B6'])/2) * (2/(in_situ_df['Simulated_S2A_B4'] + in_situ_df['Simulated_S2B_B4']) - 2/(in_situ_df['Simulated_S2A_B5'] + in_situ_df['Simulated_S2B_B5']))
in_situ_df['NIR_RED'] = (in_situ_df['Simulated_S2A_B5'] + in_situ_df['Simulated_S2B_B5'])/(in_situ_df['Simulated_S2A_B4'] + in_situ_df['Simulated_S2B_B4'])
in_situ_df['BLUE_GREEN'] = (in_situ_df['Simulated_S2A_B3'] + in_situ_df['Simulated_S2B_B3'])/(in_situ_df['Simulated_S2A_B2'] + in_situ_df['Simulated_S2B_B2'])
in_situ_df['NDCI'] = ((in_situ_df['Simulated_S2A_B5'] + in_situ_df['Simulated_S2B_B5']) - (in_situ_df['Simulated_S2A_B4'] + in_situ_df['Simulated_S2B_B4'])) / ((in_situ_df['Simulated_S2A_B5'] + in_situ_df['Simulated_S2B_B5']) + (in_situ_df['Simulated_S2A_B4'] + in_situ_df['Simulated_S2B_B4']))
in_situ_df['FLH'] = (in_situ_df['Simulated_S2A_B5'] + in_situ_df['Simulated_S2B_B5'])/2 - ((in_situ_df['Simulated_S2A_B4'] + in_situ_df['Simulated_S2B_B4']) + (in_situ_df['Simulated_S2A_B6'] + in_situ_df['Simulated_S2B_B6']))/4

for band_i in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']:
    in_situ_df[band_i] = (in_situ_df['Simulated_S2A_' + band_i] + in_situ_df['Simulated_S2B_' + band_i])/2

for var in ['D05', 'NIR_RED', 'BLUE_GREEN', 'NDCI', 'FLH']:
    if var == 'D05':
        fig_temp = Scatterplot(in_situ_df[var], in_situ_df['Chl-a 0.1m'], var)
        fig_temp.plot(high_order_linear_fucntion, [113.36, 16.45, 1.124])
        fig_temp.regression(high_order_linear_fucntion)
    elif var == 'FLH':
        fig_temp = Scatterplot(in_situ_df[var], in_situ_df['Chl-a 0.1m'], var,y_axis=1)
        fig_temp.plot(linear_function, [2231.29, 12.7])
        fig_temp.regression(linear_function)
    elif var == 'NDCI':
        fig_temp = Scatterplot(in_situ_df[var], in_situ_df['Chl-a 0.1m'], var, y_axis=0)
        fig_temp.plot(polynomial_function, [194.325, 86.115, 14.039])
        fig_temp.regression(polynomial_function)
    elif var == 'NIR_RED':
        fig_temp = Scatterplot(in_situ_df[var], in_situ_df['Chl-a 0.1m'], var, y_axis=1)
        fig_temp.regression(linear_function, yrange=[0, 5])
        fig_temp.regression(linear_function, yrange=[5, 50])
    elif var == 'BLUE_GREEN':
        fig_temp = Scatterplot(in_situ_df[var], in_situ_df['Chl-a 0.1m'], var, y_axis=1)
        fig_temp.regression(linear_function, yrange=[0, 5])
        fig_temp.regression(linear_function, yrange=[5, 50])
    fig_temp.save('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Fig\\' + var + '.jpg')
    fig_temp.close()

output_df = in_situ_df[['No', 'Site', 'Date', 'Chl-a 0.1m', 'D05', 'NIR_RED', 'BLUE_GREEN', 'NDCI', 'FLH', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']]
output_df.to_excel('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\S2_Simulated_Index.xlsx')

#可视化训练数据的相关系数热力图 自变量以及目标之间的相关系数
sr_df = in_situ_df[['D05', 'NIR_RED', 'BLUE_GREEN', 'NDCI', 'FLH', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','Chl-a 0.1m']]
datacor=np.corrcoef(sr_df.values,rowvar=0)
datacor=pd.DataFrame(data=datacor,columns=sr_df.columns,index=sr_df.columns)
plt.figure(figsize=(10,8))
ax=sns.heatmap(datacor,square=True,annot=True,fmt=".3f",linewidths=5,cmap="rocket",cbar_kws={"fraction":0.046,"pad":0.03})
plt.savefig('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Fig\\Heatmap.jpg')

#导入数据
sr_data = np.array(in_situ_df[['D05', 'NIR_RED', 'BLUE_GREEN', 'FLH', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']])
chla_target = np.array(in_situ_df[['Chl-a 0.1m']]).reshape([-1])

# if True:
#     high_conc = np.sum(chla_target >= 10)
#     low_conc = np.sum(chla_target < 10)
#     ratio = low_conc//high_conc - 1
#     sr_data_new = copy.copy(sr_data)
#     chla_target_new = copy.copy(chla_target)
#     i = 0
#     while i < sr_data.shape[0]:
#         if chla_target[i] >= 10:
#             chla_con = np.ones([ratio]) * chla_target[i]
#             sr_con = np.ones([ratio, sr_data.shape[1]])
#             for q in range(ratio):
#                 sr_con[q,:] = sr_data[i, :]
#             chla_target_new = np.concatenate([chla_target_new, chla_con], axis=0)
#             sr_data_new = np.concatenate([sr_data_new, sr_con], axis=0)
#         i += 1
#     sr_data = sr_data_new
#     chla_target = chla_target_new

#
# sr_data = np.delete(sr_data, axis=0)
# chla_target = np.delete(chla_target, axis=0)
#划分测试集和训练集
x_train, x_test, y_train, y_test=train_test_split(sr_data, chla_target, test_size=0.3, random_state=41, shuffle=True)
#标准化处理
# scale=StandardScaler()
# x_train=scale.fit_transform(x_train)
# x_test=scale.fit_transform(x_test)

mlpreg=MLPregression()
print(mlpreg)

#将数据集转化为张量 并处理为PyTorch网络使用的数据
train_xt=torch.from_numpy(x_train.astype(np.float32))
train_yt=torch.from_numpy(y_train.astype(np.float32))
test_xt=torch.from_numpy(x_test.astype(np.float32))
test_yt=torch.from_numpy(y_test.astype(np.float32))
#将数据处理为数据加载器
train_data=Data.TensorDataset(train_xt,train_yt)
test_data=Data.TensorDataset(test_xt,test_yt)
train_loader=Data.DataLoader(dataset=train_data,batch_size=200,shuffle=True,num_workers=0)

#定义优化器
optimizer=torch.optim.SGD(mlpreg.parameters(),lr=0.001)
loss_func=nn.MSELoss()
train_loss_all=[]
rmse_list=[]
for epoch in range(2000):
    train_loss=0
    train_num=0
    rmse_all=0
    for step,(b_x,b_y) in enumerate(train_loader):
        output=mlpreg(b_x)
        loss=loss_func(output,b_y)
        rmse_all += RMSE(output.detach().numpy(), b_y.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*b_x.size(0)
        train_num+=b_x.size(0)
    print("epoch " + str(epoch) + ',loss ' + str(train_loss/train_num) + ',train acc ' + str(rmse_all))
    train_loss_all.append(train_loss/train_num)
    rmse_list.append(rmse_all* 5)


fig_temp, ax_temp = plt.subplots(figsize=(10, 6), constrained_layout=True)
plt.plot(train_loss_all,"ro-",label="训练损失函数",lw=1)
plt.plot(rmse_list,"bo-",label="训练集RMSE",lw=1)
plt.legend(loc="upper right", fontsize=18)
plt.grid()
ax_temp.set_ylim(0,250)
ax_temp.set_xlim(0,500)
plt.xlabel("回合数", fontsize=24, fontweight='bold')
plt.ylabel("损失量", fontsize=24, fontweight='bold')
plt.savefig('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Fig\\mlp.jpg', dpi=500)
plt.close()
plt.clf()

#预测
pre_y=mlpreg(test_xt)
pre_y=pre_y.data.numpy()
mae=mean_absolute_error(y_test,pre_y)

index=np.argsort(y_test)
plt.figure(figsize=(6,6))
plt.plot(np.arange(len(y_test)),y_test[index],lw=4,c="r",label="实测叶绿素a浓度")
plt.scatter(np.arange(len(pre_y)),pre_y[index],s=5**2,c="b",label="反演叶绿素a浓度")
print(RMSE(y_test[index],pre_y[index]))
plt.legend(loc="upper left", fontsize=18)

plt.xlabel("Series", fontname='Times New Roman', fontsize=24, fontweight='bold')
plt.ylabel("Chl-a concentration", fontname='Times New Roman', fontsize=24, fontweight='bold')
plt.savefig('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Fig\\predict.jpg', dpi=500)

for xls_file in Basic_function.file_filter('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Sentinel_2_output\\', containing_word_list=['.xlsx']):
    if not os.path.exists(xls_file[0:xls_file.find('.xlsx')] + '_pre.xlsx') and '_pre' not in xls_file:
        df_temp =pd.read_excel(xls_file)
        arr_temp = np.array(df_temp[['D05', 'NIR_RED', 'BLUE_GREEN', 'FLH', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']])
        ten_temp = torch.from_numpy(arr_temp.astype(np.float32))
        pre_y = mlpreg(ten_temp)
        df_temp['pre_chl_a'] = pre_y.detach().numpy()
        df_temp.to_excel(xls_file[0:xls_file.find('.xlsx')] + '_pre.xlsx')

# for xls_file in Basic_function.file_filter()
date = []
for i in Basic_function.file_filter('E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\xjb_nm\\', containing_word_list=['.tif']):
    date.append(i[i.find('48R'): i.find('S20') + 9])
date = np.unique(np.array(date))

# date = ['48RUS20210804', '48RVS20210804']
for date_temp in date:
    if '2020' in date_temp:
        raster_file = Basic_function.file_filter('E:\\A_Chl-a\\Sample_XJB\\Sentinel2_L2A_output\\all_band\\xjb_nm\\', containing_word_list=[str(date_temp)])
        band_list = ['B1_', 'B2_', 'B3', 'B4', 'B5', 'B6', 'B7']
        band_dic = {}
        for raster in raster_file:
            for band_temp in band_list:
                if band_temp in raster:
                    ds_temp = gdal.Open(raster)
                    band_dic[band_temp[0:2]] = ds_temp.GetRasterBand(1).ReadAsArray() / 10000
        band_dic['D05'] = band_dic['B6']*(1/band_dic['B4']-1/band_dic['B5'])
        band_dic['NIR_RED'] = band_dic['B5'] / band_dic['B4']
        band_dic['BLUE_GREEN'] = band_dic['B3'] / band_dic['B2']
        band_dic['NDCI'] = (band_dic['B5'] - band_dic['B4']) / (band_dic['B5'] + band_dic['B4'])
        band_dic['FLH'] = band_dic['B5'] - (band_dic['B4'] + band_dic['B6']) / 2
        chl_a_map = np.zeros_like(band_dic['B1']) * np.nan
        for y_temp in range(band_dic['B1'].shape[0]):
            for x_temp in range(band_dic['B1'].shape[1]):
                if not np.isnan(band_dic['D05'][y_temp,x_temp]) and band_dic['B1'][y_temp,x_temp] != 0:
                    arr_temp = np.array([band_dic['D05'][y_temp,x_temp],band_dic['NIR_RED'][y_temp,x_temp],band_dic['BLUE_GREEN'][y_temp,x_temp],band_dic['FLH'][y_temp,x_temp],band_dic['B1'][y_temp,x_temp],band_dic['B2'][y_temp,x_temp],band_dic['B3'][y_temp,x_temp],band_dic['B4'][y_temp,x_temp],band_dic['B5'][y_temp,x_temp],band_dic['B6'][y_temp,x_temp],band_dic['B7'][y_temp,x_temp]])
                    ten_temp = torch.from_numpy(arr_temp.astype(np.float32))
                    pre_y = mlpreg(ten_temp)
                    chl_a_map[y_temp,x_temp] = pre_y.detach().numpy()
        Landsat_main_v1.write_raster(ds_temp, chl_a_map, 'E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Chl_a\\', date_temp + '_chl_a.tif', nodatavalue=np.nan, raster_datatype=gdal.GDT_Float32)


# k_fold
RMSE_NUM = []
for data_num in range(35,36):
    RMSE_VALID = []
    for t in range(10000):
        x_train, x_test, y_train, y_test=train_test_split(sr_data,chla_target,test_size=(1-data_num/50),random_state=42,shuffle=True)
        kf = KFold(n_splits=5,shuffle=True)
        rmse_train_list = []
        rmse_valid_list = []
        for train_index, valid_index in kf.split(x_train):
            print("TRAIN:", train_index, "TEST:", valid_index)
            x_train_temp, x_valid_temp = x_train[train_index], x_train[valid_index]
            y_train_temp, y_valid_temp = y_train[train_index], y_train[valid_index]
            train_x_ten = torch.from_numpy(x_train_temp.astype(np.float32))
            train_y_ten = torch.from_numpy(y_train_temp.astype(np.float32))
            valid_x_ten = torch.from_numpy(x_valid_temp.astype(np.float32))
            valid_y_ten = torch.from_numpy(y_valid_temp.astype(np.float32))

            train_data = Data.TensorDataset(train_x_ten, train_y_ten)
            valid_data = Data.TensorDataset(valid_x_ten, valid_y_ten)
            train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True, num_workers=0)
            mlpreg=MLPregression()
            optimizer = torch.optim.SGD(mlpreg.parameters(), lr=0.001)
            loss_func = nn.MSELoss()
            train_loss_all = []

            for epoch in range(400):
                train_loss = 0
                train_num = 0
                rmse_train = 0
                for step, (b_x, b_y) in enumerate(train_loader):
                    output = mlpreg(b_x)
                    loss = loss_func(output, b_y)
                    rmse_train += RMSE(output.detach().numpy(), b_y.detach().numpy())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * b_x.size(0)
                    train_num += b_x.size(0)
                train_loss_all.append(train_loss / train_num)
                rmse_train_list.append(rmse_train)
            pre_y = mlpreg(valid_x_ten)
            pre_y = pre_y.data.numpy()
            rmse_valid_list.append(RMSE(valid_y_ten, pre_y))
        print(rmse_valid_list)
        RMSE_VALID.append(np.nanmean(np.array(rmse_valid_list)))
    RMSE_NUM.append(RMSE_VALID)
    np.save('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Ds\\RMSE_' + str(data_num) + '.npy', RMSE_VALID)
np.save('E:\\A_Chl-a\\Sample_XJB\\MLP_Sentinel\\Ds\\RMSE_all.npy', RMSE_NUM)
print(mlpreg)




#