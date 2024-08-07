import argparse
import math
import time
import random
import numpy as np
import torch



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

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    
    opt.d_word_vec = opt.d_model
    set_seed(opt.seed)

    """lodaing dataset"""
    data = torch.load(opt.data)
    print(data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len #同步data信息中setting到opt实例中
    training_data, vaild_data =

    

if __name__ == "__main__":
    main()