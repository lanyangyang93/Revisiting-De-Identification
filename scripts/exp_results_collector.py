# -*- coding: utf-8 -*-
import os
import glob
import re
import argparse
import logging
import datetime
import zipfile
import pandas


dict_re = re.compile("\{[^\{\}]+\}")
metrics_re = {
              # 'micro_prec': re.compile("(?<=precision: )\d+\.\d+(?=\/)"),
              # 'micro_rec': re.compile("(?<=recall: )\d+\.\d+(?=\/)"),
              # 'micro_f1': re.compile("(?<=F1: )\d+\.\d+(?=/)"),
              # 'Kernal_size': re.compile("(?<=kernel size: )\d+\.\d+(?=/)")
              'acc': re.compile("(?<=Accuracy: )\d+\.\d+(?=%)"),
              'micro_prec': re.compile("(?<=Micro Precision: )\d+\.\d+(?=%)"),
              'micro_rec': re.compile("(?<=Micro Recall: )\d+\.\d+(?=%)"),
              'micro_f1': re.compile("(?<=Micro F1-score: )\d+\.\d+(?=%)"),
              'bleu4': re.compile("(?<=BLEU-4: )\d+\.\d+(?=%)")
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='test',
                        help="dataset name")
    parser.add_argument('--no_test_split', dest='has_test_split', default=True, action='store_false', 
                        help="whether exists test split")
    parser.add_argument('--from_date', type=str, default='None', 
                        help="from date (yyyymmdd)")
    parser.add_argument('--to_date', type=str, default='None', 
                        help="to date (yyyymmdd)")
    parser.add_argument('--format', type=str, default='zip',
                        help="output format", choices=['xlsx', 'zip'])
    args = parser.parse_args()
    
    
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    logging_fns = glob.glob(f"cache/{args.dataset}-ER/*/training.log")
    if args.from_date != 'None':
        logging_fns = [fn for fn in logging_fns if int(fn.split('/')[-2].split('-')[0]) >= int(args.from_date)]
    if args.to_date != 'None':
        logging_fns = [fn for fn in logging_fns if int(fn.split('/')[-2].split('-')[0]) <= int(args.to_date)]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    
    if args.format == 'xlsx':
        exp_results = []
        for fn in logging_fns:
            with open(fn) as f:
                log_text = f.read()
            
            try:
                exp_res = dict_re.search(log_text).group()
                exp_res = eval(exp_res)
                exp_res['logging_timestamp'] = fn.split(os.path.sep)[2]
                
                num_metrics = 0
                for m_name, m_re in metrics_re.items():
                    metric_list = m_re.findall(log_text)
                    
                    if args.has_test_split:
                        curr_num_metrics, num_res = divmod(len(metric_list), 2)
                        assert num_res == 0
                        for k in range(curr_num_metrics):
                            exp_res[f'dev_{m_name}_{k}'] = float(metric_list[k])
                            exp_res[f'test_{m_name}_{k}'] = float(metric_list[curr_num_metrics+k])
                    else:
                        curr_num_metrics = len(metric_list)
                        for k in range(curr_num_metrics):
                            exp_res[f'dev_{m_name}_{k}'] = float(metric_list[k])
                    
                    num_metrics += curr_num_metrics
                
                assert num_metrics > 0
                
            except:
                logger.warning(f"Failed to parse {fn}")
            else:
                exp_results.append(exp_res)
        
        df = pandas.DataFrame(exp_results)
        filter_cols = ['log_terminal', 'profile', 'pdb', 'pipeline', 'save_preds', 'dataset', 'use_amp', 'seed', 'fl_gamma', 'grad_clip', 'use_locked_drop', 
                       'scheme', 'num_neg_chunks', 'max_span_size', 'size_emb_dim', 'agg_mode']
        df = df.iloc[:, ~df.columns.isin(filter_cols)]
        df.to_excel(f"cache/{args.dataset}-collected-{timestamp}.xlsx", index=False)
        
    elif args.format == 'zip':
        with zipfile.ZipFile(f"cache/{args.dataset}-collected-{timestamp}.zip", 'w') as zipf:
            for fn in logging_fns:
                zipf.write(fn, fn.split('/', 1)[1])
