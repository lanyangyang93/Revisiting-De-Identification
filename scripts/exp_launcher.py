# -*- coding: utf-8 -*-
import argparse
import time
import subprocess
import multiprocessing
import logging
import numpy

from eznlp.training import OptionSampler
from utils import dataset2language


def call_command(command: str):
    logger.warning(f"Starting: {command}")
    subprocess.check_call(command.split())
    logger.warning(f"Ending: {command}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='shaoyang_Privacy',
                        help="dataset name")
    # parser.add_argument('--seed', type=int, default=515,
    #                     help="random seed")
    parser.add_argument('--use_bert', default=False, action='store_true', 
                        help="whether to use bert-like")
    parser.add_argument('--num_exps', type=int, default=-1, 
                        help="number of experiments to run")
    parser.add_argument('--num_workers', type=int ,default=5,
                        help="number of processes to run")
    args = parser.parse_args()
    args.language = dataset2language[args.dataset]
    args.num_exps = None if args.num_exps < 0 else args.num_exps
    
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    COMMAND = f"python identification_recognition.py  --dataset {args.dataset} --scheme BIO2 "
    if args.num_workers > 0:
        COMMAND = " ".join([COMMAND, "--no_log_terminal"])

    if not args.use_bert:
        sampler = OptionSampler(
                                optimizer=['AdamW'],
                                lr=[1e-3],
                                num_epochs=100,
                                seed=[302, 32, 42, 100, 502],
                                use_interm2=[True],
                                enc_arch = ['Conv'],
                                ck_decoder='sequence_tagging',
                                no_crf = [True],
                                # drop_rate = [0.25, 0.75, 0.8],
                                # weight_decay=[0.05, 0.1],
                                # coefficient=[1, 0.5, 0.3, 0.1],
                                kernel_size = 15,
                                batch_size=32,
                                random_replace_ratio = [0, 0.5]
                                )
    else:
        sampler = OptionSampler(num_epochs=100,
                                lr=[1e-3],
                                finetune_lr=[2e-5],
                                batch_size=32,
                                seed=[302, 32, 42, 100, 502],
                                # random_replace_ratio=[1, 0.5, 0.75],
                                use_interm2=[False],
                                ck_decoder='sequence_tagging',
                                # weight_decay=[0.05, 0.1],
                                # drop_rate = [0.25, 0.75, 0.8],
                                bert_drop_rate=0.2,
                                no_crf = [True],
                                use_word2vec = False,
                                use_lable = False,
                                # bert_arch=['BERT_hm-15m','RoBERTa_base','BERT_hm-30m','BERT_hm-wmm-15m','BERT_wwm']
                                bert_arch=['BERT_mc']
                                )
    
    
    options = sampler.sample(args.num_exps)
    commands = [" ".join([COMMAND, *option]) for option in options]
    logger.warning(f"There are {len(commands)} experiments to run...")
    
    if args.num_workers <= 0:
        logger.warning("Starting a single process to run...")
        for curr_command in commands:
            call_command(curr_command)
    else:
        logger.warning(f"Starting {args.num_workers} processes to run...")
        pool = multiprocessing.Pool(processes=args.num_workers)
        for curr_command in commands:
            pool.apply_async(call_command, (curr_command, ))
            # Ensure auto-device allocated before the next process starts...
            time.sleep(60)
        pool.close()
        pool.join()
