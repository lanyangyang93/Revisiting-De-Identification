# -*- coding: utf-8 -*-
# @Time    : 2022/12/14 8:55
# @Author  : LYY
# @Role    :
# @FileName: utril.py
# @Project: 20服务器
import pandas as pd
import random
import string
from faker import Faker
import re
from collections import defaultdict
from datetime import datetime, timedelta


fake = Faker(["zh_CN"])

metrics_re = {
                'style0': '%Y-%m-%d %H',
                'style1': '%Y-%m-%d%H:%M',
                'style2': '%Y-%m-%d',
                'style3': '%Y-%m',
                'style4': '%m-%d',
                'style5': '%Y-%m-%d%H:%M',
                'style6': '%Y-%m-%d%H:%M:%S',

}
def get_arecode(length):
    '''区号生成'''
    number = [str(random.randint(1,9)) for _ in range(length-1)]
    arecode = '0'+ ''.join(number)
    return arecode

def get_telephone(length):
    '''固定电话号码生成'''
    number = [str(random.randint(0,9)) for _ in range(length-1)]
    telephone = str(random.randint(1,9))+''.join(number)
    return telephone

def get_wechat(wechatID):
    '''wechat 生成'''
    wechatcode = ''
    for char in wechatID:
        if char.isdigit():
            wechatcode += str(random.randint(0, 9))
        elif char.islower():
            wechatcode += random.choice(string.ascii_lowercase)
        elif char.isupper():
            wechatcode += random.choice(string.ascii_uppercase)
        else:
            wechatcode += char
    return wechatcode

#生成随机姓名
def get_familyname_givenname(name):
    familyname = pd.read_csv('../Genrate_Name/data-csv/familyname.csv')
    # givenname = pd.read_csv('data-csv/givenname.csv')
    double_familyname = familyname[familyname['surname'].map(len) == 2]
    # single_familyname = familyname[familyname['surname'].map(len) == 1]
    # first_name
    if name[:2] in double_familyname['surname'].values:
        last_name = fake.last_name(2)
        name_length = len(name) - len(name[:2])
        # first_name = fake.first_name(name_length)
    else:
        last_name = fake.last_name(1)
        name_length = len(name) - 1
    first_name = fake.first_name(name_length)
    while len(first_name) != name_length:
        first_name = fake.first_name(name_length)
    name = last_name + first_name
    return name

    # familyname = pd.read_csv('../Genrate_Name/data-csv/familyname.csv')
    # # givenname = pd.read_csv('data-csv/givenname.csv')
    # double_familyname = familyname[familyname['surname'].map(len) == 2]
    # # single_familyname = familyname[familyname['surname'].map(len) == 1]
    #
    # if name[:2] in double_familyname['surname'].values:
    #     last_name = fake.last_name(2)
    #     # name_length = len(name)-len(name[:2])
    #     first_name = fake.first_name()
    # else:
    #     last_name = fake.last_name(1)
    #     first_name = fake.first_name()
    # name = last_name+first_name
    # return name

#生成随机ID
def get_rand_ID(old_num):
    year_list = []
    num = ''
    id_str = re.findall('\d{4}', old_num)[0]
    if id_str <= '2022' and id_str >= '1960':
        year = str(fake.date_between(start_date='-60y', end_date='today').year)
        year_list = [_ for _ in year]
    for char in old_num:
        if char.isdigit():
            if not year_list:
                num += str(random.randint(0, 9))
            else:
                num += year_list.pop(0)
        else:
            num += char
    #get_wechat(old_num)
    return num

#生成随机contact
def get_rand_contact(contact):
    if contact.isdigit() and len(contact) == 11:
        num_contact = fake.phone_number()
    elif re.search('\d{3,4}-\d+$', contact):
        areacode = get_arecode(len(contact.split('-')[0]))
        phonecode = get_telephone(len(contact.split('-')[1]))
        num_contact = areacode + '-' + phonecode
    else:
        num_contact = get_wechat(contact)

    return num_contact

#生成随机年龄
def get_rand_age(age):
    agecode = ''
    for char in age:
        if char.isdigit():
            agecode += str(random.randint(1, 9))
        else:
            agecode += char
    # if not '岁' in agecode and random.Random(42).random() < 0.5:
    #     agecode += '岁'
    return agecode

#生成随机职业
def get_rand_profession(profession):
    profession_length_dict = defaultdict(list)
    with open('../Genrate_Name/data-csv/simple_profession.txt',encoding='utf-8') as f:
        profession_list = [work.strip() for work in f]
        # for work in f:
        #     work = work.strip()
        #     profession_length_dict[len(work)].append(work)
    # if profession in profession_length_dict[len(profession)]:
    #     profession_length_dict[len(profession)].remove(profession)
    # profession_list = profession_length_dict[len(profession)]
    profession_list.remove(profession)
    if profession_list:
        profession = random.choice(profession_list)
    return profession

##生成随机时间
def get_rand_datetime(days,datetimes):
    date_list = re.compile('\d+').findall(datetimes)
    # if '上旬' in datetimes or '19年3月11日' in datetimes or '209-3-26' in datetimes or '02:34:24' in datetimes or '者020-11-17 10:22' in datetimes or \
    #         '19年9 月 10日' in datetimes or '19.10' in datetimes or '207.03' in datetimes or '20201-1' in datetimes:
    #     return datetimes
    for i,j in enumerate(date_list):
        if i != 0 and len(j) > 2:
            return datetimes
    mark_list = re.compile('\s*[^\d+]\s*').findall(datetimes)
    dates = re.sub('年|月|\.|/', '-',datetimes)
    dates = re.sub('时|：', ':', dates)
    dates = re.sub('日|分|\n| |\r', '', dates)
    try:
        if len(date_list) >3:
            if len(date_list) == 4:
                if len(date_list) == len(mark_list):
                    dates = '-'.join(date_list[:-1])+' '+date_list[3]
                    old_date = datetime.strptime(dates, metrics_re['style0'])
                else:
                    return datetimes
            elif len(date_list) == 5:
                old_date = datetime.strptime(dates, metrics_re['style5'])
            elif len(date_list) == 6:
                old_date = datetime.strptime(dates, metrics_re['style6'])
            else:
                old_date = datetime.strptime(dates, metrics_re['style1'])
        elif len(date_list) == 3:
            if ':' in mark_list or '：' in mark_list:
                return datetimes
            old_date = datetime.strptime(dates, metrics_re['style2'])
        elif len(date_list) == 2 and not ':' in mark_list and not '：' in mark_list:
            if len(date_list[0]) == 4:
                if len(date_list) == len(mark_list):
                    dates = dates[:-1]
                old_date = datetime.strptime(dates, metrics_re['style3'])
            else:
                old_date = datetime.strptime(dates, metrics_re['style4'])
        else:
            return datetimes
    except:
        return  datetimes

    new_datetime = old_date - timedelta(days)
    new_date_list = re.compile('\d+').findall(str(new_datetime))
    news1 = ''
    news_char = new_date_list.pop(0)
    for old_char in date_list:
        while len(news_char) - len(old_char) >=2:
            news_char = new_date_list.pop(0)
        if len(old_char) != 4 and len(old_char) != len(news_char) and news_char.startswith('0'):
            news1 += news_char[1:]
        elif len(old_char) != 4 and len(old_char) != len(news_char) and old_char.startswith('0'):
            news_char = '0' + news_char
            news1 += news_char
        elif len(old_char) != 4 and len(old_char) != len(news_char) and len(old_char) == 1 and len(news_char) == 2:
            news1 += str(random.randint(1, 9))
        else:
            news1 += news_char
        if mark_list:
            mark = mark_list.pop(0)
            news1 += mark
        if new_date_list:
            news_char = new_date_list.pop(0)
    return news1

#生成随机地址
def get_rand_local(location):
    #{1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 21, 23, 28, 29, 31}
    df = pd.read_csv('../Genrate_Name/data-csv/HwaMei_Privacy_v20221026/reidentified_location.csv')
    local_list = [char for char in df['reidentified_location']]
    # location.remove(location)
    location = random.choice(local_list)
    return location

def get_rand_hospital(hospital):
    old_to_new_dict = {} #老新对应
    hospital_length_dict = defaultdict(list)  #长度对应医院名
    with open('../Genrate_Name/data-csv/hospital.txt', encoding='utf-8') as f:
        for work in f:
            work = work.strip()
            hospital_length_dict[len(work)].append(work)
    if hospital in old_to_new_dict.keys():
        new_hospital = old_to_new_dict[hospital]
    else:
        if hospital in hospital_length_dict[len(hospital)]:
            hospital_length_dict[len(hospital)].remove(hospital) #移出old 医院名
        hospital_list = hospital_length_dict[len(hospital)]
        if hospital_list:
            new_hospital = random.choice(hospital_list)
            old_to_new_dict[hospital] = new_hospital

    return new_hospital

def get_rand_type(entity_type,entity_chunk,days=None):
    type = entity_type.split('_')[-1]
    if type == 'Datetime':
        return get_rand_datetime(days,entity_chunk)
    if type == 'Location':
        return get_rand_local(entity_chunk)
        # return entity_chunk
    if type == 'Profession':
        return get_rand_profession(entity_chunk)
    if type == 'Age':
        return get_rand_age(entity_chunk)
    if type == 'Name':
        return get_familyname_givenname(entity_chunk)
    if type == 'ID':
        return get_rand_ID(entity_chunk)
    if type == 'Contact':
        return get_rand_contact(entity_chunk)
    if type == 'Hospital':
        return get_rand_hospital(entity_chunk)


if __name__ == '__main__':
    print(get_rand_datetime(3,'05：46'))