# -*- coding: utf-8 -*-
import logging
import nltk
import re

from ..metrics import precision_recall_f1_report
from ..dataset import Dataset
from .trainer import Trainer


logger = logging.getLogger(__name__)


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


def evaluate_text_classification(trainer: Trainer, dataset: Dataset, batch_size: int=32, save_preds: bool=False):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    if save_preds:
        for ex, label_pred in zip(dataset.data, set_y_pred):
            ex['label_pred'] = label_pred
        logger.info("TC | Predictions saved")
    else:
        set_y_gold = [ex['label'] for ex in dataset.data]
        acc = trainer.model.decoder.evaluate(set_y_gold, set_y_pred)
        logger.info(f"TC | Accuracy: {acc*100:2.3f}%")



def _disp_prf(ave_scores: dict, task: str='ER'):
    for key_text, key in zip(['Precision', 'Recall', 'F1-score'], ['precision', 'recall', 'f1']):
        logger.info(f"{task} | Micro {key_text}: {ave_scores['micro'][key]*100:2.3f}%")
    for key_text, key in zip(['Precision', 'Recall', 'F1-score'], ['precision', 'recall', 'f1']):
        logger.info(f"{task} | Macro {key_text}: {ave_scores['macro'][key]*100:2.3f}%")


def evaluate_entity_recognition(trainer: Trainer, dataset: Dataset, batch_size: int=32, save_preds: bool=True, post_processing: bool=False):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    import torch
    #post_processing 是否进行后处理（正则）
    set_new_y_pred = []
    if post_processing:
        for ex, chunks_pred in zip(dataset.data, set_y_pred):
            # entity_list = []
            doc_name = ex['doc_id']
            tokens = [str(word) for word in ex['tokens']]
            phone = judge_phone_number(''.join(tokens))
            if 'ChuYuanYiZhu' in doc_name:
                number = judge_fixed_phone_number(''.join(tokens))
                ID = []
            else:
                number = judge_fixed_phone_number_2(''.join(tokens))
                ID = judge_ID_number(''.join(tokens))
            wechat = judge_wechat_number(''.join(tokens))
            pred_set = set(phone + ID + number + wechat)
            # if pred_set:
            #     print(pred_set)
            chunks_pred = list(set(chunks_pred) | pred_set)
            set_new_y_pred.append(chunks_pred)
        set_y_pred = set_new_y_pred
    if save_preds:
        for ex, chunks_pred in zip(dataset.data, set_y_pred):
            ex['chunks_pred'] = chunks_pred
        logger.info("ER | Predictions saved")
        torch.save(dataset, f"embedding/test-data-with-preds.pth")
        return 0, 0
    else:
        set_y_gold = [ex['chunks'] for ex in dataset.data]
        scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
        _disp_prf(ave_scores, task='ER')
        return scores, ave_scores



def _eval_attr(set_y_gold, set_y_pred):
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task='AE+')
    
    set_y_gold = [[(attr_type, chunk[1:]) for attr_type, chunk in attributes] for attributes in set_y_gold]
    set_y_pred = [[(attr_type, chunk[1:]) for attr_type, chunk in attributes] for attributes in set_y_pred]
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task='AE')


def _eval_rel(set_y_gold, set_y_pred):
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task='RE+')
    
    set_y_gold = [[(rel_type, head[1:], tail[1:]) for rel_type, head, tail in relations] for relations in set_y_gold]
    set_y_pred = [[(rel_type, head[1:], tail[1:]) for rel_type, head, tail in relations] for relations in set_y_pred]
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    _disp_prf(ave_scores, task='RE')


def evaluate_attribute_extraction(trainer: Trainer, dataset: Dataset, batch_size: int=32, save_preds: bool=False):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    if save_preds:
        for ex, attrs_pred in zip(dataset.data, set_y_pred):
            ex['attributes_pred'] = attrs_pred
        logger.info("AE | Predictions saved")
    else:
        set_y_gold = [ex['attributes'] for ex in dataset.data]
        _eval_attr(set_y_gold, set_y_pred)


def evaluate_relation_extraction(trainer: Trainer, dataset: Dataset, batch_size: int=32, save_preds: bool=False):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    if save_preds:
        for ex, rels_pred in zip(dataset.data, set_y_pred):
            ex['relations_pred'] = rels_pred
        logger.info("RE | Predictions saved")
    else:
        set_y_gold = [ex['relations'] for ex in dataset.data]
        _eval_rel(set_y_gold, set_y_pred)


def evaluate_joint_extraction(trainer: Trainer, dataset: Dataset, has_attr: bool=False, has_rel: bool=True, batch_size: int=32, save_preds: bool=False):
    set_y_pred = trainer.predict(dataset, batch_size=batch_size)
    set_chunks_pred = set_y_pred[0]
    if has_attr:
        set_attrs_pred = set_y_pred[1]
    if has_rel:
        set_rels_pred = set_y_pred[2] if has_attr else set_y_pred[1]
    
    if save_preds:
        for ex, chunks_pred in zip(dataset.data, set_chunks_pred):
            ex['chunks_pred'] = chunks_pred
        if has_attr:
            for ex, attrs_pred in zip(dataset.data, set_attrs_pred):
                ex['attributes_pred'] = attrs_pred
        if has_rel:
            for ex, rels_pred in zip(dataset.data, set_rels_pred):
                ex['relations_pred'] = rels_pred
        logger.info("Joint | Predictions saved")
    else:
        set_chunks_gold = [ex['chunks'] for ex in dataset.data]
        scores, ave_scores = precision_recall_f1_report(set_chunks_gold, set_chunks_pred)
        _disp_prf(ave_scores, task='ER')
        if has_attr:
            set_attrs_gold = [ex['attributes'] for ex in dataset.data]
            _eval_attr(set_attrs_gold, set_attrs_pred)
        if has_rel:
            set_rels_gold = [ex['relations'] for ex in dataset.data]
            _eval_rel(set_rels_gold, set_rels_pred)


def evaluate_generation(trainer: Trainer, dataset: Dataset, batch_size: int=32, beam_size: int=1):
    set_trg_pred = trainer.predict(dataset, batch_size=batch_size, beam_size=beam_size)
    set_trg_gold = [[tokens.text for tokens in ex['full_trg_tokens']] for ex in dataset.data]
    
    bleu4 = nltk.translate.bleu_score.corpus_bleu(list_of_references=set_trg_gold, hypotheses=set_trg_pred)
    logger.info(f"Beam Size: {beam_size} | BLEU-4: {bleu4*100:2.3f}%")
