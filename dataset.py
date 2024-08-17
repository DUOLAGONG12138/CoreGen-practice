import torch
import transformer.Constants as Constants
import numpy as np

class TranslationDataset(torch.utils.data.Dataset):
    """
    TranslationDataset(
            src_word2idx=data['dict']['src'], ## 修改前的代码到词向量之间的映射
            tgt_word2idx=data['dict']['tgt'], ## 修改后的代码到词向量之间的映射
            src_insts=data['train']['src'], ## 修改前的代码，为样本
            tgt_insts=data['train']['tgt']), ##修改后的代码，为目标值
    """
    def __init__(self, src_word2idx, tag_word2idx, src_insts =None, tag_insts =None):

        assert src_insts   # 当传入的src_insts为空时警告
        assert not tag_insts or (len(src_insts) == len(tag_insts))  # 当传入的tag_insts为空时警告，或样本和标签之间长度不相等时警告

        src_idx2word = {idx:word for word, idx in src_word2idx.item}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._src_insts = src_insts

        tag_idx2word = {idx:word for word, idx in tag_word2idx.item}
        self._tag_word2idx = tag_word2idx
        self._tag_idx2word = tag_idx2word
        self._tag_insts = tag_insts

    def paried_collate_fn(insts):# 将多个样本以list的形式输入后整理成每个样本长度都相等的数据，将不满最大序列长度的补全
        
        max_len = max(len(inst) for inst in insts)
        batch_seq = np.array([
            inst + [Constants.PAD]*( max_len - len(inst))  for inst in insts
        ])

        batch_pos = np.array([
            [pos_i+1 if w_i != Constants.PAD else 0
             for pos_i, w_i in enumerate(inst)]for inst in batch_seq
        ])

        return batch_seq, batch_pos

    @property
    def n_insts(self):
        return len(self.src_insts)
    
    @property
    def src_vocab_size(self):
        '''添加装饰器，将映射表的大小作为属性。方便后面获取长度信息'''
        return len(self._src_word2idx)

    @property
    def tag_vocab_size(self):
        ''' 添加装饰器，将映射表的大小作为属性。方便后面获取长度信息'''
        return len(self._tag_word2idx)
    
    @property
    def src_word2idx(self):
        ''' 添加装饰器，将映射表的大小作为属性。方便后面获取长度信息'''
        return self._src_word2idx
    @property
    def tag_word2idx(self):
        ''' 添加装饰器，将映射表的大小作为属性。方便后面获取长度信息'''
        return self._tag_word2idx
    
    @property
    def src_idx2word(self):
        ''' 添加装饰器，将映射表的大小作为属性。方便后面获取长度信息'''
        return self._src_idx2word

    @property
    def tag_idx2word(self):
        ''' 添加装饰器，将映射表的大小作为属性。方便后面获取长度信息'''
        return self._tag_idx2word


    def __len__(self):
        return self.n_insts
    
    def __getitem__(self, idx):
        if self._tag_insts:
            return self._src_insts[idx], self._tag_insts[idx]
        return self._src_insts[idx]