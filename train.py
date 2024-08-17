import argparse
import math
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformer.Models import Transformer
from dataset import TranslationDataset,paried_collate_fn
import torch.optim as optim
from safetensors import safe_open

def prepare_dataloaders(data ,opt):
    train_loader = DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'], ## 修改前的代码到词向量之间的映射
            tag_word2idx=data['dict']['tgt'], ## 修改后的代码到词向量之间的映射
            src_insts=data['train']['src'], ## 修改前的代码，为样本
            tag_insts=data['train']['tgt']), ##修改后的代码，为目标值
        num_workers= 2,
        batch_size= opt.batch_size,
        collate_fn= paried_collate_fn, ##在一个batch中，将实例的长度padding到最大序列长度
        shuffle=False)
    
    vaild_loader = DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'], ## 修改前的代码到词向量之间的映射
            tag_word2idx=data['dict']['tgt'], ## 修改后的代码到词向量之间的映射
            src_insts=data['train']['src'], ## 修改前的代码，为样本
            tag_insts=data['train']['tgt']), ##修改后的代码，为目标值
        num_workers= 2,
        batch_size= opt.batch_size,
        collate_fn= paried_collate_fn, ##在一个batch中，将实例的长度padding到最大序列长度
        shuffle=False)
    
    return train_loader, vaild_loader
    



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True) # exp/vocab/pretrain_vocab
    parser.add_argument('-epoch', type=int,default=100)
    parser.add_argument('-batch_size',type=int, default=64)
    parser.add_argument('-seed',type=int,default=30)

    parser.add_argument('-d_model',type=int, default=512)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('embs_share_weight',action='store_true')
    parser.add_argument('-model', default=None, help='Path to model .safetensors file')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    
    opt.d_word_vec = opt.d_model
    set_seed(opt.seed)

    """lodaing dataset"""
    data = torch.load(opt.data)
    print(data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len #同步data信息中setting到opt实例中
    training_data, vaild_data = prepare_dataloaders(data, opt) #从字典中将训练数据、验证数据分离开做成不同的dataset，再做成dataloader

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tag_vocab_size = training_data.dataset.tag_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight: ##如果源代码和目标代码的映射表是共享权重的话，就提示两个word2dix应该是相同的。默认不给参数
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx,\
            'The src/tgt word2idx table are different but asked to share word embedding.'
    print(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.model is None:
        transformer = Transformer(
            
        )
    
    else:
        with safe_open(opt.model, framework="pt", device = device) as f:
            model_safetensors = torch.load(f)
        model_opt = model_safetensors['setting']

        transformer = Transformer()
        transformer.load_state_dict(model_safetensors['model'])
        print('[Info] Trained model state loaded.')
    

if __name__ == "__main__":
    main()