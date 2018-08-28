# Use OCR to extract tables from pdf file
## Requirement:
`python3` and the following package:

`skimage`

`textract`

`pdf2image`

`baiduaip`

## Example:
```
import os
import glob
def pdf2csv(pdf_path, output_path_folder=None, table_keyword='接待时间'):
    pages = getPDFpages(pdf_path, keyword=table_keyword)
    if pages == []:
        return -1 
    ims = pdf2image(pdf_path, pages)
    i = 0
    d = {}
    for im_ in ims:
        im = im_[5:-5,5:-5]
        im2csv = Image2Csv_CL(im)
        im2csv.set_tables_dict()
        if im2csv.tables_p_dict == {}:
            continue
        else:
            for _, p in im2csv.tables_p_dict.items():
                img = im[p[0]:p[-1],:]
                im2csv_new = Image2Csv_CL(img)
                #im2csv_new.plot()
                #im2csv_new.write_to_csv(filename.split('.png')[0] + "-" + str(i) + '.csv')
                i += 1
                df = im2csv_new.image2df()
                d[i] = df

    for idx, df in d.items():
        s = df.astype('str').applymap(lambda cell: '接待时间' in cell).sum().sum()
        if s>=1:
            if not output_path_folder:
                output_path = pdf_path.replace('.pdf',str(idx)) + '.csv'
            else:
                output_path = os.path.join(output_path_folder, pdf_path.split('/')[-1].replace('.pdf', str(idx))) + '.csv'
            df.to_csv(output_path, index=None, header=False) 
            print(f'Success, write to file:{output_path}')
    return 0
#pdf2csv('/Users/sfi/Desktop/pdfs/000518 四环生物-2011.pdf', '/Users/sfi/Desktop')
```
## Example 2:
```
from baidu import Image2Csv_CL, rgb2gray, pdf2image#,getPDFpages
from bs4 import BeautifulSoup, NavigableString, Tag
import sys
import glob
import textract
import requests
source = requests.get('http://www.mca.gov.cn/article/sj/tjbz/a/2018/201803131439.html')
soup = BeautifulSoup(source.text, 'lxml')
cities = []
for i in soup.find('div').findAll('tr')[3:-10]:
    cities.append(i.findAll('td')[2].text[:-1])
re_cities = ['^'+ i + '$|' for i in cities]
re_ct = ''.join(re_cities)[:-1]

time_Re = re.compile(r"(\d{4}.?\d{0,1}|\d{0,1}.?\d{4}|\d{4}年|报告期|.{1,2}月.{1,2}日|日常时间|\d{1}\.\d{2}|\d{2}-\d{2}-\d{2})") # 接待时间
how_Re = re.compile(r"(实地调研|电话沟通|面谈|实地采访|电话|电话会|座谈|书面问询|实地联合调研|口头|公司现场|现场调研|策略会|项目调研|书面问讯|$集体调研|见面会$|一对多会议)") #接待方式
content_Re = re.compile(r"(发展前景|经营(?!公司)|发展|前景|状况|资料|披露|谈论|报告(?!期)|公告|提供|事项|情况|现状|未来|行业|趋势|形势|生产|投产|计划|转型|方向|进度|进展|市场|价格|竞争|产业|品牌|重组|信息|说明|转型|看法|销售|项目|材料|问题|原因|回答|基本面|限售|告知|问及|同上)") #谈论内容及提供资料
who_Re = re.compile(r'(调研员|女士|先生|券商|证券|基金|资产管理有限公司|信托|投资者$|组织|客户|股东$|^股东|个人|研究员|分析师|分析员|摩根大通|摩根斯坦利|申银万国|人寿|银行|Gartmore|景顺|中金|保险|UBS|养老|Invesco|New silk road|Jefferies|JP Morgan|BNP|太保|投信|Asian Century Quest|Manulife|信澳达|TPG Axon Capital|凯基亚洲|Tufton Ocean|太和|易方达)(?!(?:活动|推荐会|推介会))',re.IGNORECASE) #接待对象

import re
kw_re = re.compile(r'(接待时间|接待地点|接待对象|接待方式|谈论的内容及提供的资料)')
def getPDFpages(path, keyword_re=kw_re):
    text = textract.process(path).decode('utf-8')
    text_all_pages = text.split('\x0c')[:-1]
    pages = []
    for page, text in enumerate(text_all_pages):
        if len(keyword_re.findall(text))>=2:
            pages.append(page)
    #print(pages)
    if pages == []:
        return pages
    for page in pages[:]:
        for n in range(1,8):
            next_n_page = page + n
            text_n = text_all_pages[next_n_page]
            #print(time_Re.findall(text_n) , '!!!',loc_Re.findall(text_n) ,'!!!', how_Re.findall(text_n) , '!!!',who_Re.findall(text_n))
            #print("-"*60)
            if time_Re.findall(text_n) and how_Re.findall(text_n) and who_Re.findall(text_n):
                pages.append(next_n_page)
            else:
                break
    return sorted(list(set(pages)))

import os
import glob

def is_desire_table(text, keyword_re=kw_re):
    if keyword_re.findall(text):
        return True
    else:
        return False

def list_distance(l1, l2):
    try:
        diff = [abs(i-j) for i,j in zip(l1,l2)]
        s = sum(diff)
        return s
    except:
        return 10000
    
def pdf2csv(pdf_path, output_path_folder=None):
    pages = getPDFpages(pdf_path, keyword_re=kw_re)
    if pages == []:
        return -1 
    ims = pdf2image(pdf_path, pages)
    i = 0
    d = {}
    col_p = []
    for im_ in ims:
        im = im_[5:-5,5:-5]
        im2csv = Image2Csv_CL(im)
        im2csv.set_tables_dict()
        if im2csv.tables_p_dict == {}:
            continue
        else:
            for _, p in im2csv.tables_p_dict.items():
                img = im[p[0]:p[-1],:]
                im2csv_new = Image2Csv_CL(img)
                #im2csv_new.plot()
                #cropped =im2csv_new.crop_image()
#                 plt.imshow(cropped[0])
#                 plt.show()
#                 plt.imshow(cropped[1])
#                 plt.show()
                #im2csv_new.write_to_csv(filename.split('.png')[0] + "-" + str(i) + '.csv')
                df = im2csv_new.image2df()
                d[i] = df
                i += 1
                col_p.append(im2csv_new.col_position)
    for idx, df in d.items():
        s = df.astype('str').applymap(lambda cell: is_desire_table(cell)).sum().sum()
        if s>=1:
            keys = list(range(idx, i))
            break
    if len(keys)==1:
        df_final = d[keys[0]]
    else:
        col_position = col_p[keys[0]:]
        ref_position = col_position[0]
        count = 0
        for p in col_position[1:]:
            if list_distance(ref_position, p) < 50:
                count += 1
            else:
                break 
        keys = keys[:count+1]
        for key in keys:
            df_final = pd.concat([d[i] for i in keys])
    
    if not output_path_folder:
        output_path = pdf_path.replace('.pdf','.csv')
    else:
        output_path = os.path.join(output_path_folder, pdf_path.split('/')[-1].replace('.pdf', '.csv')) 
    df_final.to_csv(output_path, index=None, header=False) 
    print(f'Success, write to file:{output_path}')
    
    return 0
#pdf2csv('/Users/sfi/Desktop/返回值/LEVEL4/REST/000826桑德环境-2011.pdf','/Users/sfi/Desktop/')
```
