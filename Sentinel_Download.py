from subprocess import call
from sentinelsat import SentinelAPI
import time
from tqdm import tqdm
from xml.dom.minidom import parse

# This Program is mainly used to batch download Sentinel-2 image using IDM
# Several things you need to make sure before the download
# (1) Successfully download and install IDM as well as sentinelsat package
# (2) Add username and password for the scihub url in the setting interface of IDM
IDM = "C:\\Program Files (x86)\\Internet Download Manager\\IDMan.exe"
DownPath = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\Original_Zipfile\\'

api = SentinelAPI('shixi2ng', 'shixi2nG', 'https://scihub.copernicus.eu/dhus')

url_file = 'E:\\A_PhD_Main_stuff\\2022_04_22_Mid_Yangtze\\Sample_Sentinel\\Original_Zipfile\\products.meta4'
data = parse(url_file).documentElement
linklist = [i.childNodes[0].nodeValue for i in data.getElementsByTagName('url')]

print('Download started')
n = 0
while linklist:
    print('---------------------------------------------------')
    n = n + 1
    print('Download the ' + str(n) + 'file ' + '\n')
    # if n % 2 == 1:
    #     for i in tqdm(range(int(1200)), ncols=100):
    #         time.sleep(2)

    id = linklist[0].split('\'')[1]
    link = linklist[0]
    product_info = api.get_product_odata(id)
    print('The metadata of the file：')
    print('The ID：' + id)
    print('Document Name:：' + product_info['title'] + '\n')
    '''
    IDM参数解释：
    /d URL  #根据URL下载文件
    /s      #开始下载队列中的任务
    /p      #定义文件要存储在本地的地址
    /f      #定义文件存储在本地的文件名
    /q      #下载成功后IDM将退出。
    /h      #下载成功后IDM将挂起你的链接
    /n      #当IDM不出问题时启动静默模式
    /a      #添加指定文件到/d的下载队列，但是不进行下载	 	
    '''
    if product_info['Online']:
        print(product_info['title'] + 'is online')
        call([IDM, '/d', link, '/p', DownPath, '/f', product_info['title'] + '.zip', '/a'])  # 将连接添加到任务。/a表示先不下载
        linklist.remove(link)
        call([IDM, '/s'])  # 开启下载
        print("Start download")

