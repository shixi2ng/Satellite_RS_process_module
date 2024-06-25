import os.path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import traceback
import random
import ast
import datetime
import pytz
import time


def worm_waterlevel_runoff(output_folder):

    # Define station name
    jj_station_name = ['沙市', '枝城', '监利']
    myr_station_name = ['长江_莲花塘', '长江_螺山', '长江_石矶头', '长江_汉口', '长江_黄石港', '长江_码头镇', '松滋河(西支)_三岔河', '(null)_梅田湖', '藕池河(北支)_南县(罗文窖)', '洞庭湖_注滋口', '澧水_石龟山', '澧水_蒿子港', '松滋河(西支)_官垸', '(null)_三不管', '松滋河(中支)_自治局(三)', '(null)_张九台', '(null)_大湖口', '(null)_小望角', '松虎合流_安乡', '松虎合流_白蚌口', '沅江_牛鼻滩', '沅江_周文庙', '西洞庭湖湖口(北端)_南咀', '目平湖_沙湾', '西洞庭湖湖口(南端)_小河咀', '草尾河_草尾', '草尾河_黄茅洲', '南洞庭湖_东南湖', '万子湖_沅江（二）', '资水_沙头(二)', '资水(西支)_甘溪港', '资水（东支）_杨堤(二)', '资水(东支)_白马寺', '南洞庭湖_杨柳潭', '湘江(东支)_湘阴', '横岭湖_营田', '东洞庭湖_鹿角', '洞庭湖湖口_岳阳', '洞庭湖_城陵矶(七)', '陆水_崇阳', '陆水_毛家桥(二)', '陆水_洪下', '陆水_陆水水库坝下', '陆水_蒲圻', '陆水_车埠', '陆水_石坑', '陆水_白云潭', '陆水_浪口', '陆水_毛家桥', '陆水_小港', '陆水_南渠', '陆水_北渠', '大河_白霓桥(二)', '汉江_皇庄', '汉江_大同', '汉江_沙洋(三)', '汉江_兴隆', '汉江_泽口', '汉江_岳口', '汉江_仙桃(二)', '汉江_汉川', '东荆河_潜江', '陆水_北渠开度', '陆水_南渠开度']
    lyr_station_name = ['九江', '八里江', '彭泽', '安庆', '江口', '大通', '南京', '南京潮位', '湖口', '襄河口闸上', '襄河口闸下', '晓桥', '滁州', '水口闸']

    # Define url
    jj_url = 'http://jj.cjh.com.cn/'
    myr_url = 'http://zy.cjh.com.cn/'
    lyr_url = 'http://xy.cjh.com.cn/'

    # Define the output dic
    jj_dic,myr_dic, lyr_dic = {}, {}, {}

    while True:
        wait_time = 3600

        # Read the csv
        try:
            for station_name_ in jj_station_name:
                if not os.path.exists(os.path.join(output_folder, f'{station_name_}.csv')):
                    jj_dic[station_name_] = pd.DataFrame(
                        columns=['Station_name', 'Time', 'Water level(m)', 'Runoff(m3/s)'])
                else:
                    jj_dic[station_name_] = pd.read_csv(os.path.join(output_folder, f'{station_name_}.csv'))
                    if list(jj_dic[station_name_].columns) != ['Station_name', 'Time', 'Water level(m)', 'Runoff(m3/s)']:
                        raise Exception(f'Column is not consistent for {station_name_}!')
        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to read or construct the JJ csv file")

        # Request the MYR inform
        try:
            response = requests.get(jj_url)
            if response.status_code == 200:
                page_content = response.text
                content = str(page_content).split('sssq')
                for content_ in content:
                    if [_ in content_ for _ in jj_station_name].count(True) == len(jj_station_name):
                        content_station = content_.split('[')[1].split(']')[0].split('{')
                        content_station = [_.split('}')[0] for _ in content_station if '}' in _]
                        content_station = [ast.literal_eval('{' + _.replace(" ", '') + '}') for _ in content_station]
                        break

                for station_name in jj_station_name:
                    num_ = 0
                    while num_ in range(len(content_station)):
                        if len(content_station[num_].keys()) > 0 and 'stnm' in content_station[num_].keys() and content_station[num_]['stnm'] == station_name:
                            stamp_time = datetime.datetime.utcfromtimestamp(content_station[num_]['tm'] / 1000)
                            target_time_zone = pytz.timezone('Asia/Shanghai')  # UTC+8
                            local_date_time = stamp_time.replace(tzinfo=pytz.utc).astimezone(target_time_zone)
                            formatted_local_date_time = local_date_time.strftime('%Y-%m-%d %H:%M:%S')
                            water_level = content_station[num_]['z'] if 'z' in content_station[num_].keys() else -9999
                            runoff = content_station[num_]['q'] if 'q' in content_station[num_].keys() else -9999
                            temp_dic = {'Station_name': [station_name],
                                        'Time': [formatted_local_date_time],
                                        'Water level(m)': [water_level],
                                        'Runoff(m3/s)': [runoff]}
                            new_pd = pd.DataFrame(temp_dic)

                            if new_pd['Time'][0] not in list(jj_dic[station_name]['Time']):
                                jj_dic[station_name] = pd.concat([jj_dic[station_name], new_pd], axis=0, ignore_index=True)
                                wait_time = min(wait_time, random.randint(3540, 3600))
                            else:
                                wait_time = min(wait_time, random.randint(840, 960))
                            break
                        num_ += 1
        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to retrieve the information from jj page")

        # Update the MYR files
        try:
            for station_name_ in jj_station_name:
                jj_dic[station_name_].to_csv(os.path.join(output_folder, f'{station_name_}.csv'), encoding='utf-8-sig', index=False)
        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to save the csv file!")

        # Read the csv
        try:
            for station_name_ in myr_station_name:
                if not os.path.exists(os.path.join(output_folder, f'{station_name_}.csv')):
                    myr_dic[station_name_] = pd.DataFrame(columns=['Station_name', 'Time', 'Water level(m)', 'Runoff(m3/s)'])
                else:
                    myr_dic[station_name_] = pd.read_csv(os.path.join(output_folder, f'{station_name_}.csv'))
                    if list(myr_dic[station_name_].columns) != ['Station_name', 'Time', 'Water level(m)', 'Runoff(m3/s)']:
                        raise Exception(f'Column is not consistent for {station_name_}!')
        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to read or construct the MYR_CSV file")

        # Request the MYR inform
        try:
            response = requests.get(myr_url)
            if response.status_code == 200:
                page_content = response.text
                # 使用BeautifulSoup解析页面内容
                soup = BeautifulSoup(page_content, 'html.parser')
                content = str(soup).split('sssq')
                myr_station_temp = [_.split('_')[1] for _ in myr_station_name]
                for content_ in content:
                    if [_ in content_ for _ in myr_station_temp].count(True) > len(myr_station_temp) - 3:
                        content_station = content_.split('{')
                        content_station = [_.split('},')[0] for _ in content_station if '},' in _]
                        content_station = [ast.literal_eval('{' + _.replace(" ", '') + '}') for _ in content_station]
                        break

                for station_name in myr_station_name:
                    num_ = 0
                    while num_ in range(len(content_station)):
                        if len(content_station[num_].keys()) > 0 and 'stnm' in content_station[num_].keys() and 'rvnm' in content_station[num_].keys() and content_station[num_]['rvnm'] + '_' + content_station[num_]['stnm'] == station_name:
                            stamp_time = datetime.datetime.utcfromtimestamp(content_station[num_]['tm'] / 1000)
                            target_time_zone = pytz.timezone('Asia/Shanghai')  # UTC+8
                            local_date_time = stamp_time.replace(tzinfo=pytz.utc).astimezone(target_time_zone)
                            formatted_local_date_time = local_date_time.strftime('%Y-%m-%d %H:%M:%S')
                            water_level = content_station[num_]['z'] if 'z' in content_station[num_].keys() else -9999
                            runoff = content_station[num_]['q'] if 'q' in content_station[num_].keys() else -9999
                            temp_dic = {'Station_name': [station_name],
                                        'Time': [formatted_local_date_time],
                                        'Water level(m)': [water_level],
                                        'Runoff(m3/s)': [runoff]}
                            new_pd = pd.DataFrame(temp_dic)

                            if new_pd['Time'][0] not in list(myr_dic[station_name]['Time']):
                                myr_dic[station_name] = pd.concat([myr_dic[station_name], new_pd], axis=0, ignore_index=True)
                                wait_time = min(wait_time, random.randint(3540, 3600))
                            else:
                                wait_time = min(wait_time, random.randint(840, 960))
                            break
                        num_ += 1
        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to retrieve the information from myr page")

        # Update the MYR files
        try:
            for station_name_ in myr_station_name:
                myr_dic[station_name_].to_csv(os.path.join(output_folder, f'{station_name_}.csv'), encoding='utf-8-sig', index=False)
        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to save the csv file!")

        # Read the csv
        try:
            for station_name_ in lyr_station_name:
                if not os.path.exists(os.path.join(output_folder, f'{station_name_}.csv')):
                    lyr_dic[station_name_] = pd.DataFrame(columns=['Station_name', 'Time', 'Water level(m)', 'Runoff(m3/s)'])
                else:
                    lyr_dic[station_name_] = pd.read_csv(os.path.join(output_folder, f'{station_name_}.csv'))
                    if list(lyr_dic[station_name_].columns) != ['Station_name', 'Time', 'Water level(m)', 'Runoff(m3/s)']:
                        raise Exception(f'Column is not consistent for {station_name_}!')

        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to read or construct the LYR_CSV file")

        # Request the LYR inform
        try:
            response = requests.get(lyr_url)
            if response.status_code == 200:
                page_content = response.text
                # 使用BeautifulSoup解析页面内容
                soup = BeautifulSoup(page_content, 'html.parser')
                content = str(soup).split('sssq')
                for content_ in content:
                    if [_ in content_ for _ in lyr_station_name].count(True) > len(lyr_station_name) / 2:
                        content_station = content_.split('{')
                        content_station = [_.split('},')[0] for _ in content_station if '},' in _]
                        content_station = [ast.literal_eval('{' + _.replace(" ", '') + '}') for _ in content_station]
                        break

                for station_name in lyr_station_name:
                    num_ = 0
                    while num_ in range(len(content_station)):
                        if 'stnm' in content_station[num_].keys() and content_station[num_]['stnm'] == station_name:
                            stamp_time = datetime.datetime.utcfromtimestamp(content_station[num_]['TM'] / 1000)
                            target_time_zone = pytz.timezone('Asia/Shanghai')  # UTC+8
                            local_date_time = stamp_time.replace(tzinfo=pytz.utc).astimezone(target_time_zone)
                            formatted_local_date_time = local_date_time.strftime('%Y-%m-%d %H:%M:%S')
                            water_level = content_station[num_]['Z'] if 'Z' in content_station[num_].keys() else -9999
                            runoff = content_station[num_]['Q'] if 'Q' in content_station[num_].keys() else -9999
                            temp_dic = {'Station_name': [station_name],
                                        'Time': [formatted_local_date_time],
                                        'Water level(m)': [water_level],
                                        'Runoff(m3/s)': [runoff]}
                            new_pd = pd.DataFrame(temp_dic)

                            if new_pd['Time'][0] not in list(lyr_dic[station_name]['Time']):
                                lyr_dic[station_name] = pd.concat([lyr_dic[station_name], new_pd], axis=0, ignore_index=True)
                                wait_time = min(wait_time, random.randint(3540, 3600))
                            else:
                                wait_time = min(wait_time, random.randint(840, 960))
                            break
                        num_ += 1
        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to retrieve the information from lyr page")

        # Update the LYR files
        try:
            for station_name_ in lyr_station_name:
                lyr_dic[station_name_].to_csv(os.path.join(output_folder, f'{station_name_}.csv'), encoding='utf-8-sig', index=False)
        except:
            print(traceback.format_exc())
            raise Exception(f"Failed to save the csv file!")

        time.sleep(wait_time)


if __name__ == '__main__':
    worm_waterlevel_runoff('G:\\WL\\')