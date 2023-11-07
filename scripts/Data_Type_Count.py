# -*- coding: utf-8 -*-
# @Time    : 2023/10/8 16:07
# @Author  : LYY
# @Role    :
# @FileName: Data_Type_Count.py
# @Project: 20服务器
import json
a1,a2,a3,a4,a5,a6,a7,a8 = 0,0,0,0,0,0,0,0
with open("data/HwaMei_Privacy/privacy-test-deid.json",encoding='utf-8') as f:
    train_data = [json.loads(line) for line in f if len(line.strip()) > 0]
    for ids,datas in enumerate(train_data):
        for dataset in datas['entities']:
            entity_type = dataset['type']
            if entity_type == 'Priv_Name':
                a1 += 1
            elif entity_type == 'Priv_Location':
                a2 += 1
            elif entity_type == 'Priv_Hospital':
                a3 += 1
            elif entity_type == 'Priv_Datetime':
                a4 += 1
            elif entity_type == 'Priv_ID':
                a5 += 1
            elif entity_type == 'Priv_Contact':
                a6 += 1
            elif entity_type == 'Priv_Age':
                a7 += 1
            elif entity_type == 'Priv_Profession':
                a8 += 1
print(ids+1)
print(a1,a2,a3,a4,a5,a6,a7,a8)
