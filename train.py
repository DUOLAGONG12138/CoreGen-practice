import argparse
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformer.Models import Transformer
from dataset import TranslationDataset,paried_collate_fn
import torch.optim as optim
from safetensors import safe_open
from transformer.Optim import ScheduledOptim
import transformer.Constants as Constants
from tqdm import tqdm

def cal_performance(pred, gold, smoothing=False):
    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def cal_loss(pred, gold,smoothing):
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss
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
    
def train_epoch(model, training_data, optimizer, device, smooting):
    model.train()

    total_loss =0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
        training_data, mininterval =2,
        desc='-training', leave=False):

        src_seq, src_pos, tag_seq, tag_pos = map(lambda x:x.to(device), batch)
        gold = tag_seq[:,1:]

        #forward
        optimizer.zero_grad()
        pred=model(src_seq, src_pos, tag_seq, tag_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smooting=smooting)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, valid_data, optimizer, device, opt):
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid,log'

        print('[info] training performance will be written to file:{} and {}',format(
            log_train_file, log_valid_file
        ))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write(str(opt)+"\n")
            log_vf.write(str(opt)+"\n")
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[Epoch',epoch_i,']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing
        )
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, valid_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                if opt.save_thres:
                    if valid_accu > opt.save_thres and valid_accu >= max(valid_accus):
                        torch.save(checkpoint, model_name)
                        print('    - [Info] The checkpoint file has been updated.')
                elif valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')
            elif opt.save_mode == 'record':
                model_name = opt.save_model + '_epoch_{0}.chkpt'.format(epoch_i)
                torch.save(checkpoint, model_name)

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True) # ./exp/vocab/pretrain_vocab
    parser.add_argument('-epoch', type=int,default=100)
    parser.add_argument('-batch_size',type=int, default=64)
    parser.add_argument('-seed',type=int,default=30)

    parser.add_argument('-d_model',type=int, default=512)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('embs_share_weight',action='store_true')
    parser.add_argument('-model', default=None, help='Path to model .safetensors file')
    parser.add_argument('-embs_share_weight',action='store_true')
    parser.add_argument('-proj_share_weight',action='store_true')
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-n_head', type=int, default=6)
    parser.add_argument('-n_layers', type=int, default=2)

    parser.add_argument('-save_model', default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best', 'record'], default='best')
    parser.add_argument('-save_thres', type=float, default=None)
    parser.add_argument('-label_smoothing', action='store_true')
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    
    opt.d_word_vec = opt.d_model
    set_seed(opt.seed)

    """loading dataset"""
    data = torch.load(opt.data)
    print(data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len #同步data信息中setting到opt实例中
    training_data, vaild_data = prepare_dataloaders(data, opt) #从字典中将训练数据、验证数据分离开做成不同的dataset，再做成dataloader

    opt.n_src_vocab = training_data.dataset.src_vocab_size
    opt.n_tag_vocab = training_data.dataset.tag_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight: ##如果源代码和目标代码的映射表是共享权重的话，就提示两个word2dix应该是相同的。默认不给参数
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx,\
            'The src/tgt word2idx table are different but asked to share word embedding.'
    print(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.model is None:
        transformer = Transformer(
            opt.n_src_vocab, 
            opt.n_tag_vocab, 
            opt.max_token_seq_len,
            tag_emb_pri_weight_sharing=opt.proj_share_weight,
            emb_src_tag_weight_sharing=opt.embs_share_weight,
            d_k = opt.d_k,
            d_v = opt.d_v,
            d_word_vec= opt.d_word_vec, 
            d_model=opt.d_model, 
            d_inner= opt.d_inner_hid,
            n_layers=opt.n_layers, 
            n_head=opt.n_head, 
            dropout=opt.dropout,
        ).to(device)
    
    else:
        with safe_open(opt.model, framework="pt", device = device) as f:
            model_safetensors = torch.load(f)
        model_opt = model_safetensors['settings']

        transformer = Transformer(
            model_opt.n_src_vocab, 
            model_opt.n_tag_vocab, 
            model_opt.max_token_seq_len,
            tag_emb_pri_weight_sharing=model_opt.proj_share_weight,
            emb_src_tag_weight_sharing=model_opt.embs_share_weight,
            d_k = model_opt.d_k,
            d_v = model_opt.d_v,
            d_word_vec= model_opt.d_word_vec, 
            d_model=model_opt.d_model, 
            d_inner= model_opt.d_inner_hid,
            n_layers=model_opt.n_layers, 
            n_head=model_opt.n_head, 
            dropout=model_opt.dropout,
        ).to(device)

        transformer.load_state_dict(model_safetensors['model'])
        print('[Info] Trained model state loaded.')
    
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98),
            eps=1e-9),
            opt.d_model,
            opt.n_warmup_steps
    )

    train(transformer, training_data, vaild_data, optimizer, device, opt)

if __name__ == "__main__":
    main()