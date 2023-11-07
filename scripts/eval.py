# -*- coding: utf-8 -*-
# @Time    : 2022/12/12 16:00
# @Author  : LYY
# @Role    :
# @FileName: eval.py
# @Project: 20服务器
import os
import sys
import argparse
import datetime
import re
import logging
import pprint
import json
import pandas as pd
import glob
import torch
from eznlp.metrics import precision_recall_f1_report
from eznlp import auto_device
from eznlp.dataset import Dataset,LableDataset
    #LableDataset
import copy
from eznlp.training import Trainer, count_params, evaluate_entity_recognition
from torchstat import stat
from tqdm import tqdm
from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)

    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='conll2003',
                            help="dataset name")
    group_data.add_argument('--doc_level', default=False, action='store_true',
                            help="whether to load data at document level")
    group_data.add_argument('--corrupt_rate', type=float, default=0.0,
                            help="boundary corrupt rate")
    group_data.add_argument('--save_preds', default=False, action='store_true',
                            help="whether to save predictions on the test split (typically in case without ground truth)")
    group_data.add_argument('--pipeline', default=False, action='store_true',
                            help="whether to save predictions on all splits for pipeline modeling")
    group_data.add_argument('--random_replace_ratio', default=1.0, type=float,
                            help="Random Replacement Ratio")
    group_data.add_argument('--overfit', default=False, action='store_true',
                            help="whether overfit")
    group_data.add_argument('--coefficient', default=0, type=float,
                            help="word2vec coefficient (dataset=HwaMei_Privacy_10)")
    group_data.add_argument('--use_lable', default=False, action='store_true',
                            help="whether use_lable ")

    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--ck_decoder', type=str, default='sequence_tagging',
                               help="chunk decoding method",
                               choices=['sequence_tagging', 'span_classification', 'boundary_selection'])
    # Loss
    group_decoder.add_argument('--fl_gamma', type=float, default=0.0,
                               help="Focal Loss gamma")
    group_decoder.add_argument('--sl_epsilon', type=float, default=0.0,
                               help="Label smoothing loss epsilon")

    # Sequence tagging
    group_decoder.add_argument('--scheme', type=str, default='BIOES',
                               help="sequence tagging scheme", choices=['BIOES', 'BIO2'])
    group_decoder.add_argument('--no_crf', dest='use_crf', default=True, action='store_false',
                               help="whether to use CRF")

    group_decoder.add_argument('--post_processing', default=False, action='store_false',
                               help="whether to use post processing in the test-Evaluating")

    group_decoder.add_argument('--use_source', default=False, action='store_false',
                               help="whether to use source")

    # Span-based
    group_decoder.add_argument('--agg_mode', type=str, default='max_pooling',
                               help="aggregating mode")
    group_decoder.add_argument('--num_neg_chunks', type=int, default=100,
                               help="number of sampling negative chunks")
    group_decoder.add_argument('--max_span_size', type=int, default=10,
                               help="maximum span size")
    group_decoder.add_argument('--ck_size_emb_dim', type=int, default=25,
                               help="span size embedding dim")

    # Boundary selection
    group_decoder.add_argument('--no_biaffine', dest='use_biaffine', default=True, action='store_false',
                               help="whether to use biaffine")
    group_decoder.add_argument('--affine_arch', type=str, default='FFN',
                               help="affine encoder architecture")
    group_decoder.add_argument('--neg_sampling_rate', type=float, default=1.0,
                               help="Negative sampling rate")
    group_decoder.add_argument('--hard_neg_sampling_rate', type=float, default=1.0,
                               help="Hard negative sampling rate")
    group_decoder.add_argument('--hard_neg_sampling_size', type=int, default=5,
                               help="Hard negative sampling window size")
    group_decoder.add_argument('--sb_epsilon', type=float, default=0.0,
                               help="Boundary smoothing loss epsilon")
    group_decoder.add_argument('--sb_size', type=int, default=1,
                               help="Boundary smoothing window size")
    group_decoder.add_argument('--sb_adj_factor', type=float, default=1.0,
                               help="Boundary smoothing probability adjust factor")
    return parse_to_args(parser)


#返回位置信息
def entity_info(account,entities,lable):
    lable_dict = {'ID':'Priv_ID', 'phone':'Priv_Contact'}
    entities_info = []
    entity = '|'.join(list(set(entities)))
    informations = re.finditer(entity, account)
    for m in informations:
        lable_entity = account[m.span()[0]-5:m.span()[1]]
        if '电话' in lable_entity and lable == 'ID':
            continue
        entities_info.append((lable_dict[lable],m.span()[0],m.span()[1]))
    return entities_info

# 正则匹配手机号
def judge_phone_number(account):
    phone = re.findall('(13\d{9}|14[5|7]\d{8}|15\d{9}|166{\d{8}|17[3|6|7]{\d{8}|18\d{9}$)', account)
    if phone:
        return entity_info(account, phone, 'phone')
    return phone

# 正则匹配固定电话 与ID冲突
def judge_fixed_phone_number(account):
    telephone = re.findall('0\d{2,3}—\d{7,9}|0\d{2,3}-\d{7,9}|(?<=[^\d])\d{7,9}(?=[^\d])', account)
    if telephone:
        return entity_info(account, telephone, 'phone')
    return telephone

def judge_fixed_phone_number_2(account):
    telephone = re.findall('0\d{2,3}—\d{7,9}|0\d{2,3}-\d{7,9}', account)
    if telephone:
        return entity_info(account, telephone, 'phone')
    return telephone
# 正则匹配微信号
def judge_wechat_number(account):
    if not '微信' in account:
        return []
    wechat = re.findall('([a-z]{1}[-_a-zA-Z0-9]{5,19})', account)
    if wechat:
        return entity_info(account, wechat, 'phone')
    return wechat

# 正则匹配ID
def judge_ID_number(account):
    # account = re.sub('0\d{2,3}—\d{7,9}|0\d{2,3}-\d{7,9}', '', account)
    ID = re.findall('\d{8}[A-Z]{0,5}\d+|[A-Z]{0,2}\d{0,6}\-{0,1}\d{5,6}(?!\s*.*/)', account)
    if ID:
        return entity_info(account, ID,'ID')
    return ID


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    args = parse_arguments(parser)
    exp_results = []
    Regular = False
    device = auto_device()
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device)
        temp = torch.randn(100).to(device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    path = 'cache/HwaMei_Privacy-ER-无数据增强'
    dict_fns = glob.glob(f"{path}/*")

    train_data, dev_data, test_data = load_data(args)
    args.language = dataset2language[args.dataset]

    for file_path in tqdm(dict_fns):
        if not '20230404-093412-105082' in file_path:
            continue
        exp_res = {}

        with open(f'{file_path}/training.log') as f:
            log_text = f.read()
        F1 = re.compile("(?<=Micro F1-score: )\d+\.\d+(?=%)").findall(log_text)  #判断程序是否执行完成
        word2vec = re.compile("(?<=OOV tokens: )\d+").findall(log_text)          #判断程序是否使用w2v
        # radio = re.compile("(?<='random_replace_ratio': ).+").findall(log_text)[0]
        # dropout_rate = re.compile("(?<='weight_decay': ).+").findall(log_text)[0] #判断程序是否修改weight_decay
        if not F1:
            print(f'Failed to parse {file_path}')
            continue
        # 旧版
        model_path_list = glob.glob(f"{file_path}/*.pth")
        for model_path in model_path_list:
            if 'config' in model_path:
                config = torch.load(f'{model_path}')
            else:
                if word2vec:
                    exp_res['filepath'] = model_path.split('/')[-1][:-4]+f'_word2vec'
                else:
                    exp_res['filepath'] = model_path.split('/')[-1][:-4]
                model = torch.load(f'{model_path}', map_location=device)


        # model_path_list = glob.glob(f"{file_path}/*_best.pth")
        # config_path = model_path_list[0].rsplit('_',2)[0]
        # config = torch.load(f'{config_path}-config.pth')
        # for model_path in model_path_list:
        #     if word2vec:
        #         exp_res['filepath'] = model_path.split('/')[-1][:-4]+f'_{radio}'
        #     else:
        #         exp_res['filepath'] = model_path.split('/')[-1][:-4]+f'_{radio}'
        # model = torch.load(f'{model_path}', map_location=device)
        test_set = Dataset(test_data, config, training=False)
        trainer = Trainer(model, device=device)
        # evaluate_entity_recognition(trainer, test_set, batch_size=args.batch_size)
        #

        scores, ave_scores = evaluate_entity_recognition(trainer, test_set, batch_size=args.batch_size,
                                                             save_preds=args.save_preds)

        # print(scores, ave_scores)
    #     exp_res['Priv_Location'] = scores['Priv_Location']['f1'] if 'Priv_Location' in scores.keys() else 0
    #     exp_res['Priv_Hospital'] = scores['Priv_Hospital']['f1'] if 'Priv_Hospital' in scores.keys() else 0
    #     exp_res['Priv_Name'] = scores['Priv_Name']['f1'] if 'Priv_Name' in scores.keys() else 0
    #     exp_res['Priv_Age'] = scores['Priv_Age']['f1'] if 'Priv_Age' in scores.keys() else 0
    #     exp_res['Priv_Profession'] = scores['Priv_Profession']['f1'] if 'Priv_Profession' in scores.keys() else 0
    #     exp_res['Priv_Datetime'] = scores['Priv_Datetime']['f1'] if 'Priv_Datetime'in scores.keys() else 0
    #     exp_res['Priv_ID'] = scores['Priv_ID']['f1'] if 'Priv_ID' in scores.keys() else 0
    #     exp_res['Priv_Contact'] = scores['Priv_Contact']['f1'] if 'Priv_Contact' in scores.keys() else 0
    #     p = float(ave_scores['micro']['precision']*100)
    #     r = float(ave_scores['micro']['recall']*100)
    #     f = float(ave_scores['micro']['f1']*100)
    #     exp_res['mp'] = p
    #     exp_res['mr'] = r
    #     exp_res['mf'] = f
    #     exp_results.append(copy.deepcopy(exp_res))
    # # print(exp_results)
    # df = pd.DataFrame(exp_results)
    # filter_cols = ['filepath', 'Priv_Location','Priv_Hospital','Priv_Name','Priv_Age','Priv_Profession','Priv_Datetime','Priv_ID','Priv_Contact','mp', 'mr', 'mf']
    # # df = df.iloc[:, ~df.columns.isin(filter_cols)]
    # df.to_excel(f"cache/{args.dataset}-{timestamp}.xlsx", index=False)


    # with open('cache/HwaMei_Privacy-ER/20221212-013707-442362/macro.json','w',encoding='utf-8') as f:
    #     bJson = json.dump(scores, f, ensure_ascii=False, indent=2)

# if Regular:
#     set_y_pred = trainer.predict(test_set, batch_size=args.batch_size)
#     set_new_y_pred = []
#     for ex, chunks_pred in zip(test_set.data, set_y_pred):
#         # entity_list = []
#         doc_name = ex['doc_id']
#         tokens = [str(word) for word in ex['tokens']]
#         phone = judge_phone_number(''.join(tokens))
#         if 'ChuYuanYiZhu' in doc_name:
#             number = judge_fixed_phone_number(''.join(tokens))
#             ID = []
#         else:
#             number = judge_fixed_phone_number_2(''.join(tokens))
#             ID = judge_ID_number(''.join(tokens))
#         # ID = judge_ID_number(''.join(tokens))
#         # number = []
#         wechat = judge_wechat_number(''.join(tokens))
#         pred_set = set(phone + ID + number + wechat)
#         # if pred_set:
#         #     print(pred_set)
#         new_chunks_pred = [p for p in chunks_pred if p[0] != 'Priv_ID' and p[0] != 'Priv_Contact']
#
#         chunks_pred = list(set(new_chunks_pred) | pred_set)
#         set_new_y_pred.append(chunks_pred)
#     set_y_pred = set_new_y_pred
#     set_y_gold = [ex['chunks'] for ex in test_set.data]
#     scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    # print(scores, ave_scores)





    # a = []
    # for ex, chunks_pred in zip(test_set.data, set_y_pred):
    #     ex['chunks_pred'] = chunks_pred
    # torch.save(test_data, f"{path}/test-data-with-preds.pth")
    #
    # if args.pipeline:
    #     # Replace gold chunks with predicted chunks for pipeline
    #     if args.train_with_dev:
    #         # Retrieve the original splits
    #         train_set = Dataset(train_data, train_set.config, training=True)
    #         dev_set = Dataset(dev_data, train_set.config, training=False)
    #
    #     train_set_chunks_pred = trainer.predict(train_set, batch_size=args.batch_size)
    #     for ex, chunks_pred in zip(train_data, train_set_chunks_pred):
    #         ex['chunks'] = ex['chunks'] + [ck for ck in chunks_pred if ck not in ex['chunks']]
    #
    #     dev_set_chunks_pred = trainer.predict(dev_set, batch_size=args.batch_size)
    #     for ex, chunks_pred in zip(dev_data, dev_set_chunks_pred):
    #         ex['chunks'] = chunks_pred
    #
    #     test_set_chunks_pred = trainer.predict(test_set, batch_size=args.batch_size)
    #     for ex, chunks_pred in zip(test_data, test_set_chunks_pred):
    #         ex['chunks'] = chunks_pred


#[{'filepath': 'text-BERT_hm-15m-BIO2-CE_epoch_best_0.01}', 'Priv_Location': 0.3212669683257918, 'Priv_Hospital': 0.7985611510791366, 'Priv_Name': 0.9761904761904762, 'Priv_Age': 0.9941927990708478, 'Priv_Profession': 0.5, 'Priv_Datetime': 0.9158469945355192, 'Priv_ID': 0.9411764705882353, 'Priv_Contact': 0, 'mp': 79.4961511546536, 'mr': 86.06060606060606, 'mf': 82.64823572208077}]
#[{'filepath': 'text-BERT_hm-15m-BIO2-CE_epoch_best_0.01}', 'Priv_Location': 0.19203747072599534, 'Priv_Hospital': 0.6016260162601625, 'Priv_Name': 0.9268292682926829, 'Priv_Age': 0.9883449883449883, 'Priv_Profession': 0.37499999999999994, 'Priv_Datetime': 0.9080962800875273, 'Priv_ID': 0.7619047619047619, 'Priv_Contact': 0, 'mp': 73.19378798109386, 'mr': 82.12121212121211, 'mf': 77.40092823991432}]
