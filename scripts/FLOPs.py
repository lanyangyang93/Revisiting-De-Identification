# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 16:05
# @Author  : LYY
# @Role    :
# @FileName: FLOPs.py
# @Project: 20服务器
# -*- coding: utf-8 -*-
# @Time    : 2022/12/12 16:00
# @Author  : LYY
# @Role    : Calculate FLOPs
# @FileName: FLOPs.py
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
# from metrics import precision_recall_f1_report
from eznlp import auto_device
from eznlp.dataset import Dataset
    #LableDataset

from eznlp.training import Trainer, count_params, evaluate_entity_recognition
from thop import profile
from tqdm import tqdm
from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language
from torchstat import stat

def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)

    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='conll2003',
                            help="dataset name")

    return parse_to_args(parser)


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

    dict_fns = ['cache/HwaMei_Privacy-ER/20230823-154035-367389']
    for file_path in tqdm(dict_fns):
        exp_res = {}
        with open(f'{file_path}/training.log') as f:
            log_text = f.read()
        F1 = re.compile("(?<=Micro F1-score: )\d+\.\d+(?=%)").findall(log_text)
        if not F1:
            print(f'Failed to parse {file_path}')
            continue
        model_path_list = glob.glob(f"{file_path}/*.pth")
        for model_path in model_path_list:
            if 'config' in model_path:
                config = torch.load(model_path)
            else:
                exp_res['filepath'] = model_path.split('/')[-1][:-4]
                model = torch.load(model_path, map_location=device)

        train_data, dev_data, test_data = load_data(args)
        args.language = dataset2language[args.dataset]
        test_set = Dataset(test_data, config, training=False)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                                 collate_fn=test_set.collate)
        for batch in dataloader:
            batch = batch.to(device, non_blocking=False)
            macs, params = profile(model, inputs=(batch,), verbose=False)
            print(model_path)
            print('FLOPs = ' + str(macs * 2 / 1000 ** 3) + 'G')
            break

