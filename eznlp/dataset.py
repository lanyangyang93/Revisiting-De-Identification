# -*- coding: utf-8 -*-
import bisect
from typing import List, Any, Iterable
import random
import torch
import math
import pickle
from .nn.functional import seq_lens2mask,lab_seq_lens2mask
from .wrapper import Batch
from .model.model import ModelConfigBase
from .plm import PreTrainingConfig
from .utril import get_rand_type
from .token import TokenSequence
from torch.utils.data.sampler import RandomSampler
import re

rng = random.Random(42)





#重新整合序列
def renew_list(tokens_list, re_entity_list, start_index, end_index):

    return tokens_list[:start_index] + re_entity_list + tokens_list[end_index:]

#判断替换正确否
def judgment(old_list, new_list,renew_chunk):
    test_tokens_list = new_list.copy()
    entity_length = 0
    for entity_info in renew_chunk:
        start_index = entity_info[1]
        end_index = entity_info[2]
        test_tokens_list = renew_list(test_tokens_list, [], start_index - entity_length,
                                      end_index - entity_length)
        entity_length += end_index - start_index
    assert old_list == test_tokens_list


#添加模板
def add_temple():
    contact = random.choice(['17306881795','0752-7581758','83870052'])
    contcat_data = get_rand_type('Contact',contact)
    temple_list = [("不适随诊咨询电话：contact",9), ("出现以下情况请来院就诊或与我们联系（contact）",18),
                   ("门诊预约电话：contact，提前1周可预约。",7), ("神经内一科（6号楼7层）电话：contact；",15)]
    # contcat_list = ['83870052', '17306881795', '87085742',
    #                 '838371112', '83872162','83871082', '0574-83870217',
    #                 '87085634', '15168191962', '83871030','0574-83870530', '83871092', '83872113',
    #                 '0574-83870533']
    temple_data = random.choice(temple_list)
    # contcat_data = random.choice(contcat_list)
    temple = temple_data[0].replace('contact', contcat_data)
    start_index = temple_data[1]
    end_index = start_index + len(contcat_data)
    return list(temple), start_index, end_index

#随机替换
def rand_entity(entry,random_replace_ratio):
    tokens_list = [str(word) for word in entry['tokens']]
    old_tokens_list = tokens_list.copy()
    test_old_tokens_list = tokens_list.copy()
    renew_chunk = []
    days = random.randint(1, 365)
    if rng.random() < random_replace_ratio:
        # add_list, add_start_index, add_end_index = add_temple()
        # chunks = ('Priv_Contact', len(tokens_list) + add_start_index, len(tokens_list) + add_end_index)
        # tokens_list += add_list
        # entry['chunks'].append(chunks)
        # entry['tokens'] = TokenSequence.from_tokenized_text(tokens_list)
        return entry
    else:
        chunks_list = sorted(entry['chunks'], key=lambda x: x[1])
        rember_dict = {}
        entity_length = 0
        for ids, entity_info in enumerate(chunks_list):
            entity_type = entity_info[0]
            start_index = entity_info[1]
            end_index = entity_info[2]
            test_old_tokens_list = renew_list(test_old_tokens_list, [], start_index-entity_length, end_index-entity_length)
            entity_length += end_index - start_index

            entity_str = ''.join(old_tokens_list[start_index:end_index])
            if not entity_str in rember_dict.keys():
                if entity_type == 'Priv_Datetime':
                    re_entry_list = list(get_rand_type(entity_type, entity_str, days))
                # elif entity_type == 'Priv_Name' or entity_type == 'Priv_Profession' or entity_type == 'Priv_ID':
                #     re_entry_list = list(entity_str)
                else:
                    re_entry_list = list(get_rand_type(entity_type,entity_str))  # 返回随机生成函数
                rember_dict[entity_str] = re_entry_list
            else:
                re_entry_list = rember_dict[entity_str]
            if ids == 0:
                index = len(re_entry_list) - len(entity_str)
                tokens_list = renew_list(tokens_list, re_entry_list, start_index, end_index)
                end_index = end_index + index
                renew_chunk.append((entity_type, start_index, end_index))
            else:
                start_index = start_index + index
                old_end_index = end_index + index
                tokens_list = renew_list(tokens_list, re_entry_list, start_index, old_end_index)
                index += len(re_entry_list) - len(entity_str)
                end_index = end_index + index
                renew_chunk.append((entity_type, start_index, end_index))

            #判断替换后序列是否出现实体位置错误
            assert ''.join(tokens_list[start_index:end_index]) == ''.join(re_entry_list)
        assert len(renew_chunk) == len(chunks_list)
        judgment(test_old_tokens_list, tokens_list, renew_chunk)
        re_entry = pickle.loads(pickle.dumps(entry))
        re_entry['tokens'] = TokenSequence.from_tokenized_text(tokens_list)
        re_entry['chunks'] = renew_chunk
        if len(tokens_list) > 508:
            re_entry = entry
        return re_entry




class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], config: ModelConfigBase, training: bool=True, ratio=1):
        """
        Parameters
        ----------
        data : List[dict]
            Each entry (as a dict) follows the format of:
                {'tokens': TokenSequence, 'label': str, 'chunks': List[tuple], 'relations': List[tuple], ...}
            where (1) `label` is a str (or int).  
                  (2) each `chunk` follows the format of (chunk_type, chunk_start, chunk_end). 
                  (3) each `relation` follows the format of (relation_type, head_chunk, tail_chunk), 
                      i.e., (relation_type, (head_type, head_start, head_end), (tail_type, tail_start, tail_end)). 
        """
        super().__init__()
        self.data = data
        self.config = config
        self.training = training
        self.ratio = ratio
        
    def __len__(self):
        return len(self.data)
        
    @property
    def summary(self):
        summary = []
        num_seqs = len(self.data)
        summary.append(f"The dataset consists {num_seqs:,} sequences")
        
        if 'raw_idx' in self.data[0]:
            num_raws = len({entry['raw_idx'] for entry in self.data})
            summary.append(f"\tbuilt from {num_raws:,} raw entries")
        
        if 'tokens' in self.data[0]:
            seq_lens = [len(entry['tokens']) for entry in self.data]
            ave_len, max_len = sum(seq_lens)/len(seq_lens), max(seq_lens)
            summary.extend([f"The average `tokens` length is {ave_len:,.1f}", 
                            f"The maximum `tokens` length is {max_len:,}"])
        
        if 'label' in self.data[0]:
            num_label_types = len({entry['label'] for entry in self.data})
            summary.append(f"The dataset has {num_label_types:,} categories")
        
        if 'chunks' in self.data[0]:
            num_chunks = sum(len(entry['chunks']) for entry in self.data)
            num_chunk_types = len({ck[0] for entry in self.data for ck in entry['chunks']})
            summary.append(f"The dataset has {num_chunks:,} chunks of {num_chunk_types:,} types")
        
        if 'attributes' in self.data[0]:
            num_attributes = sum(len(entry['attributes']) for entry in self.data)
            num_attr_types = len({attr[0] for entry in self.data for attr in entry['attributes']})
            summary.append(f"The dataset has {num_attributes:,} attributes of {num_attr_types:,} types")
        
        if 'relations' in self.data[0]:
            num_relations = sum(len(entry['relations']) for entry in self.data)
            num_relation_types = len({rel[0] for entry in self.data for rel in entry['relations']})
            summary.append(f"The dataset has {num_relations:,} relations of {num_relation_types:,} types")
        
        return "\n".join(summary)
        
        
    def build_vocabs_and_dims(self, *others):
        self.config.build_vocabs_and_dims(self.data, *others)


        
    def _get_entry(self, i):
        return self.data[i]
        
    def __getitem__(self, i):
        entry = self._get_entry(i)
        ## data augmentation here
        # entry['tokens']
        if self.training:
            re_entry = rand_entity(entry,self.ratio) #训练集随机替换  当self.ratio == 1.0 是 不发生替换

        else:
            re_entry = entry
        example = {}
        if 'tokens' in self.data[0]:
            example['tokenized_text'] = re_entry['tokens'].text

        re_entry['tokens'] = TokenSequence.from_tokenized_text(example['tokenized_text'])

        example.update(self.config.exemplify(re_entry, training=self.training))
        return example
        
        
    def collate(self, batch_examples: List[dict]):
        batch = {}

        if 'tokens' in self.data[0]:
            batch['tokenized_text'] = [ex['tokenized_text'] for ex in batch_examples]
            batch['seq_lens'] = torch.tensor([len(tokenized_text) for tokenized_text in batch['tokenized_text']])
            batch['mask'] = seq_lens2mask(batch['seq_lens'])
        
        batch.update(self.config.batchify(batch_examples))
        return Batch(**batch)


class LableDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], config: ModelConfigBase, training: bool = True, ratio=1):
        """
        Parameters
        ----------
        data : List[dict]
            Each entry (as a dict) follows the format of:
                {'tokens': TokenSequence, 'label': str, 'chunks': List[tuple], 'relations': List[tuple], ...}
            where (1) `label` is a str (or int).
                  (2) each `chunk` follows the format of (chunk_type, chunk_start, chunk_end).
                  (3) each `relation` follows the format of (relation_type, head_chunk, tail_chunk),
                      i.e., (relation_type, (head_type, head_start, head_end), (tail_type, tail_start, tail_end)).
        """
        super().__init__()
        self.data = data
        self.config = config
        self.training = training
        self.ratio = ratio

    def __len__(self):
        return len(self.data)

    @property
    def summary(self):
        summary = []
        num_seqs = len(self.data)
        summary.append(f"The dataset consists {num_seqs:,} sequences")

        if 'raw_idx' in self.data[0]:
            num_raws = len({entry['raw_idx'] for entry in self.data})
            summary.append(f"\tbuilt from {num_raws:,} raw entries")

        if 'tokens' in self.data[0]:
            seq_lens = [len(entry['tokens']) for entry in self.data]
            ave_len, max_len = sum(seq_lens) / len(seq_lens), max(seq_lens)
            summary.extend([f"The average `tokens` length is {ave_len:,.1f}",
                            f"The maximum `tokens` length is {max_len:,}"])

        if 'label' in self.data[0]:
            num_label_types = len({entry['label'] for entry in self.data})
            summary.append(f"The dataset has {num_label_types:,} categories")

        if 'chunks' in self.data[0]:
            num_chunks = sum(len(entry['chunks']) for entry in self.data)
            num_chunk_types = len({ck[0] for entry in self.data for ck in entry['chunks']})
            summary.append(f"The dataset has {num_chunks:,} chunks of {num_chunk_types:,} types")

        if 'attributes' in self.data[0]:
            num_attributes = sum(len(entry['attributes']) for entry in self.data)
            num_attr_types = len({attr[0] for entry in self.data for attr in entry['attributes']})
            summary.append(f"The dataset has {num_attributes:,} attributes of {num_attr_types:,} types")

        if 'relations' in self.data[0]:
            num_relations = sum(len(entry['relations']) for entry in self.data)
            num_relation_types = len({rel[0] for entry in self.data for rel in entry['relations']})
            summary.append(f"The dataset has {num_relations:,} relations of {num_relation_types:,} types")

        return "\n".join(summary)

    def build_vocabs_and_dims(self, *others):
        self.config.build_vocabs_and_dims(self.data, *others)

    def _get_entry(self, i):
        return self.data[i]

    def __getitem__(self, i):
        label_dict = {'BingChengJiLu': 'BingChengJiLu', 'BingChengShouYe': 'BingChengShouYe',
                      'ChuYuanJiLu': 'ChuYuanJiLu', 'DaBingLi': 'DaBingLi',
                      'ChaFangJiLu': 'BingChengJiLu', 'XianBingShi': 'DaBingLi', 'ZhenLiaoJingGuo': 'ChuYuanJiLu',
                      'ChuYuanQingKuang': 'ChuYuanJiLu',
                      'HuiZhenYiJian': 'BingChengJiLu', 'ShouShuJingGuo': 'BingChengJiLu', 'GeRenShi': 'DaBingLi',
                      'ShouShuJiLu': 'BingChengJiLu'}
        entry = self._get_entry(i)
        title = entry['doc_id'].split('/')[1]
        title = re.findall(r'[a-zA-Z]+', title)[0]
        title = label_dict[title].lower()
        department = entry['department'].lower()
        if self.training and self.ratio != 1:
            re_entry = rand_entity(entry, self.ratio)  # 训练集随机替换  当self.ratio ==1.0 是 不发生替换

        else:
            re_entry = entry.copy()
        example = {}
        if 'tokens' in self.data[0]:
            example['orgin_tokenized_text'] = re_entry['tokens'].text
            example['tokenized_text'] = [title,department] + re_entry['tokens'].text
        re_entry['orgin_tokenized_text'] = re_entry['tokens']
        re_entry['tokens'] = TokenSequence.from_tokenized_text(example['tokenized_text'])

        example.update(self.config.exemplify(re_entry, training=self.training))
        return example

    def collate(self, batch_examples: List[dict]):
        batch = {}

        if 'tokens' in self.data[0]:
            batch['orgin_tokenized_text'] = [ex['orgin_tokenized_text'] for ex in batch_examples]
            batch['tokenized_text'] = [ex['tokenized_text'] for ex in batch_examples]
            batch['seq_lens'] = torch.tensor([len(tokenized_text) for tokenized_text in batch['tokenized_text']])
            batch['mask'] = lab_seq_lens2mask(batch['seq_lens'])

        batch.update(self.config.batchify(batch_examples))
        return Batch(**batch)





class Word2vcDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], config: ModelConfigBase, training: bool = True):
        """
        Parameters
        ----------
        data : List[dict]
            Each entry (as a dict) follows the format of:
                {'tokens': TokenSequence, 'label': str, 'chunks': List[tuple], 'relations': List[tuple], ...}
            where (1) `label` is a str (or int).
                  (2) each `chunk` follows the format of (chunk_type, chunk_start, chunk_end).
                  (3) each `relation` follows the format of (relation_type, head_chunk, tail_chunk),
                      i.e., (relation_type, (head_type, head_start, head_end), (tail_type, tail_start, tail_end)).
        """
        super().__init__()
        self.data = data
        self.config = config
        self.training = training

    def __len__(self):
        return len(self.data)

    @property
    def summary(self):
        summary = []
        num_seqs = len(self.data)
        summary.append(f"The dataset consists {num_seqs:,} sequences")

        if 'raw_idx' in self.data[0]:
            num_raws = len({entry['raw_idx'] for entry in self.data})
            summary.append(f"\tbuilt from {num_raws:,} raw entries")

        if 'tokens' in self.data[0]:
            seq_lens = [len(entry['tokens']) for entry in self.data]
            ave_len, max_len = sum(seq_lens) / len(seq_lens), max(seq_lens)
            summary.extend([f"The average `tokens` length is {ave_len:,.1f}",
                            f"The maximum `tokens` length is {max_len:,}"])

        if 'label' in self.data[0]:
            num_label_types = len({entry['label'] for entry in self.data})
            summary.append(f"The dataset has {num_label_types:,} categories")

        if 'chunks' in self.data[0]:
            num_chunks = sum(len(entry['chunks']) for entry in self.data)
            num_chunk_types = len({ck[0] for entry in self.data for ck in entry['chunks']})
            summary.append(f"The dataset has {num_chunks:,} chunks of {num_chunk_types:,} types")

        if 'attributes' in self.data[0]:
            num_attributes = sum(len(entry['attributes']) for entry in self.data)
            num_attr_types = len({attr[0] for entry in self.data for attr in entry['attributes']})
            summary.append(f"The dataset has {num_attributes:,} attributes of {num_attr_types:,} types")

        if 'relations' in self.data[0]:
            num_relations = sum(len(entry['relations']) for entry in self.data)
            num_relation_types = len({rel[0] for entry in self.data for rel in entry['relations']})
            summary.append(f"The dataset has {num_relations:,} relations of {num_relation_types:,} types")

        return "\n".join(summary)

    def build_vocabs_and_dims(self, *others):
        self.config.build_vocabs_and_dims(self.data, *others)

    def _get_entry(self, i):
        return self.data[i]

    def get_surrounding_elements(self, lst, index, window_size=5):
        index += window_size
        lst = [0] * window_size + lst + [0] * window_size
        start = max(0, index - window_size)
        end = min(len(lst), index + window_size + 1)

        return [_ for _ in lst[start:index] + lst[index + 1:end]]

    #负采样
    def get_rank_out_elements(self, positive_word_data, neg_sample=5):
        negative_word_data = []
        vocab_list = self.config.ohots['text'].vocab.itos
        i = 0
        while i < neg_sample:
            negative_word_candidate = random.randint(0, len(vocab_list) - 1)
            if negative_word_candidate not in positive_word_data:
                negative_word_data.append([negative_word_candidate])
                i += 1
        return negative_word_data


    def __getitem__(self, i):
        entry = self._get_entry(i)
        ## data augmentation here
        example, data = {}, []
        if 'tokens' in self.data[0]:
            example['tokenized_text'] = entry['tokens'].text
        example.update(self.config.exemplify(entry, training=self.training))
        # ohots = example['ohots']['text'].numpy().tolist()
        # for ids, char in enumerate(ohots):
        #     positive_word_data = self.get_surrounding_elements(ohots, ids) #背景词
        #     data.append(positive_word_data)
        # example['skipgam'] = torch.LongTensor(data)
        return example

    def collate(self, batch_examples: List[dict]):
        batch = {}
        if 'tokens' in self.data[0]:
            batch['tokenized_text'] = [ex['tokenized_text'] for ex in batch_examples]
            batch['seq_lens'] = torch.tensor([len(tokenized_text) for tokenized_text in batch['tokenized_text']])
            batch['mask'] = seq_lens2mask(batch['seq_lens'])
            # batch['skipgam'] = [ex['skipgam'] for ex in batch_examples]
        batch.update(self.config.batchify(batch_examples))
        # if 'skipgam' in batch_examples[0].keys():
        #     batch['skipgam'] = [ex['skipgam'] for ex in batch_examples]
        return Batch(**batch)



class GenerationDataset(Dataset):
    def __init__(self, data: List[dict], config: ModelConfigBase=None, training: bool=True):
        super().__init__(data, config=config, training=training)
        if training:
            self._indexing = [(src_idx, trg_idx) for src_idx, entry in enumerate(self.data) 
                                for trg_idx, tokens in enumerate(entry['full_trg_tokens'])]
        
    @property
    def summary(self):
        summary = [super().summary]
        
        seq_lens = [len(tokens) for entry in self.data for tokens in entry['full_trg_tokens']]
        ave_len, max_len = sum(seq_lens)/len(seq_lens), max(seq_lens)
        summary.extend([f"The average `trg_tokens` length is {ave_len:,.1f}", 
                        f"The maximum `trg_tokens` length is {max_len:,}"])
        return "\n".join(summary)
        
        
    def __len__(self):
        if self.training:
            return len(self._indexing)
        else:
            return len(self.data)
        
    def _get_entry(self, i):
        if self.training:
            src_idx, trg_idx = self._indexing[i]
            entry = self.data[src_idx]
            # `trg_tokens` is a cache field
            entry['trg_tokens'] = entry['full_trg_tokens'][trg_idx]
            return entry
        else:
            return self.data[i]



class PreTrainingDataset(torch.utils.data.Dataset):
    """Dataset for Pre-training. 
    """
    def __init__(self, data: List[Any], config: PreTrainingConfig, training: bool=True, mp_rank=0, mp_world_size=0):
        super().__init__()
        # if mp_world_size > 0:
        #     assert 0 <= mp_rank < mp_world_size
            
        #     text_paths = text_paths[_slice_chunk(mp_rank, mp_world_size, len(text_paths))]
        # logger.info(f"Totally {len(text_paths)} text files in the {mp_rank}-th process")
        
        self.data = data
        self.config = config
        self.training = training
        
        
    def __len__(self):
        return len(self.data)
        
    @property
    def summary(self):
        summary = []
        num_seqs = len(self.data)
        summary.append(f"The dataset consists {num_seqs:,} sequences")
        return "\n".join(summary)
        
        
    def __getitem__(self, i):
        entry = self.data[i]
        
        if getattr(self.config, 'paired_task', 'None').lower() == 'nsp':
            paired_entry = random.choice(self.data)
        else:
            paired_entry = None


        example = self.config.exemplify(entry, paired_entry=paired_entry, training=self.training)
        return example
        
        
    def collate(self, batch_examples: List[str]):
        batch = self.config.batchify(batch_examples)
        return Batch(**batch)



class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    '''采样方式
    '''
    def __init__(self, dataset, batch_size):

        self.dataset = dataset  #数据集
        self.batch_size = batch_size  #batch大小
        self.number_of_datasets = len(dataset.datasets)  #加载数据集个数

        self.largest_dataset_size = max([len(cur_dataset.data) for cur_dataset in dataset.datasets])  #数据集最大大小

    def __len__(self):

        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):

        samplers_list = []
        samplers_iterators = []


        for dataset_idx in range(self.number_of_datasets):

            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)  #随机抽取数据集
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            samplers_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1] #索引
        step = self.batch_size #// self.number_of_datasets #* self.number_of_datasets  #每步实际大小
        samplers_to_grap = self.batch_size //2 #样例大小限制

        epoch_samples = self.largest_dataset_size * self.number_of_datasets #数据总数量

        final_samples_list = []

        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = samplers_iterators[i]
                cur_samples = []
                for _ in range(samplers_to_grap):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)   #取样本

                    except StopIteration:

                        samplers_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = samplers_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)

