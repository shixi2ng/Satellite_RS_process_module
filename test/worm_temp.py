import json
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
from selenium import webdriver
import time


def worm_whpch(folder: str, list_range=282):

    if not folder.endswith('\\'):
        folder = folder + '\\'

    if not os.path.exists(f'{folder}weihuapincangchu.xlsx'):
        temp_dic = {'danwei':[], 'faren':[], 'dizhi': [], 'fanwei':[], 'zhengshu': []}
        with tqdm(total=list_range, desc=f'Worm the WHPCH', bar_format='{l_bar}{bar:24}{r_bar}{bar:-24b}') as pbar:
            for list_temp in range(1, list_range + 1):
                page_url = f"http://yjglj.ningbo.gov.cn/col/col1229107419/index.html?uid=5867528&pageNum={str(list_temp)}"
                try:
                    browser = webdriver.Chrome()
                    browser.get(page_url)
                    time.sleep(6)
                    url_str = browser.page_source
                    url_str_list = url_str.split('</td>')
                    __ = 0
                    for _ in url_str_list:
                        if '<td class="biaoge"' in _ or '<td class="biaoge text-tag"' in _:
                            q = _.split('>')[-1]
                            if __ == 0:
                                if q != '\r\n' and q != '单位名称 ':
                                    temp_dic['danwei'].append(q)
                            elif __ == 1:
                                if q != '\r\n' and q != '法人':
                                    temp_dic['faren'].append(q)
                            elif __ == 2:
                                if q != '\r\n' and q != '地址':
                                    temp_dic['dizhi'].append(q)
                            elif __ == 3:
                                if q != '\r\n' and q != '经营范围':
                                    temp_dic['fanwei'].append(q)
                            elif __ == 4:
                                if q != '\r\n' and q != '证书编号':
                                    temp_dic['zhengshu'].append(q)
                                __ = -1
                            __ = __ + 1
                    browser.quit()
                except:
                    print(f'The {str(list_temp)} cannot be request!')
                pbar.update()
        df_temp = pd.DataFrame(temp_dic)
        df_temp.to_excel(f'{folder}weihuapincangchu.xlsx')


worm_whpch('G:\\')