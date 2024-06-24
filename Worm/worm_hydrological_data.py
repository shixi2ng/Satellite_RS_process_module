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
    myr_station_name = ['莲花塘', '螺山', '石矶头', '汉口', '黄石港', '码头镇', '三岔河', '南县(罗文窖)', '石龟山']
    lyr_station_name = ['九江', '八里江', '彭泽', '安庆', '江口', '大通', '南京', '南京潮位', '湖口', '襄河口闸上', '襄河口闸下', '晓桥', '滁州', '水口闸']

    # Define url
    myr_url = 'http://zy.cjh.com.cn/'
    lyr_url = 'http://xy.cjh.com.cn/'

    # Define the output dic
    myr_dic, lyr_dic = {}, {}

    while True:
        wait_time = 3600
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
                content = str(soup).split('td')

                for station_name in myr_station_name:
                    num_ = 0
                    while num_ in range(len(content)):
                        if content[num_].startswith(f'>{station_name}'):
                            temp_dic = {'Station_name': [content[num_].split('>')[-1].split('<')[0]],
                                        'Time': [content[num_ + 2].split('>')[-1].split('<')[0]],
                                        'Water level(m)': [content[num_ + 4].split('>')[-1].split('<')[0]],
                                        'Runoff(m3/s)': [content[num_ + 6].split('>')[-1].split('<')[0]]}
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
                            formatted_local_date_time = local_date_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')
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