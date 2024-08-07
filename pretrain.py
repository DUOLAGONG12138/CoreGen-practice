import argparse
import torch
import math
import random
import numpy as np
import transformer.Constants as Constants


def read_instance_from_file(inst_file, max_sent_len, keep_case, mask_rate, in_statement_pred):
    #构建两个列表分别存储修改前句子和修改后句子
    source_insts = []
    target_insts = []
    #构建计数器
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            source_list, target_list = preprocess_for_pretrain(sent, mask_rate, in_statement_pred)
            for source, target in zip(source_list, target_list):
                source_word = source.split()
                target_word = target.split()

                if len(source_word) > max_sent_len or len(target_word) > max_sent_len:
                    trimmed_sent_count += 1
                source_inst = source_word[:max_sent_len]
                target_inst = target_word[:max_sent_len]

                if source_inst:
                    source_insts += [[Constants.BOS_WORD] + source_inst + [Constants.EOS_WORD]]
                else:
                    source_insts += [None]
                
                if target_inst:
                    target_insts += [[Constants.BOS_WORD] + target_inst + [Constants.EOS_WORD]]
                else:
                    target_insts += [None] 

    print('[info] get {} instance from {}'.format(len(source_inst),inst_file))
    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))
        
    return source_insts, target_insts


def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.MSK_WORD: Constants.MSK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]


def preprocess_for_pretrain(lines, mask_rate, in_statement_pred):
    #输入句子，前后空格去掉，按nl划分
    lines = lines.strip().split("<nl> ")

    after_commit = []
    before_commit = []
    addition_idx = []
    deletion_idx = []
    source_list = []
    target_list = []
    #将划分好的句子分别提取index和token
    for idx, line in enumerate(lines):
        if line[0] == "+":
            after_commit.append(line[1:])
            addition_idx.append(idx)
        elif line[0] == "-":
            before_commit.append(line[1:])
            deletion_idx.append(idx)
        else:
            after_commit.append(line)
            before_commit.append(line)

    if before_commit == after_commit:

        before_commit_len = [len(src.split()) for src in before_commit] # ['new file mode 100755 ', 'index 0000000 . . d125c52 ', 'binary files / dev / null and b / art / intro . png differ <nl>']
        maxlen = max(before_commit_len) # 16 [4,5,16]
        maxlen_idx = before_commit_len.index(maxlen) # 2
        #maxlen_idx = np.asarray(before_commit_len).argmax() 
        
        mask_len = math.floor(mask_rate*maxlen) #4 0.3*16=4.8 
        mask_strat = random.randint(0, maxlen-mask_len) # 0-12 取随机
        mask_end = mask_strat + mask_len # 2 6
        line_to_mask = before_commit[maxlen_idx].split() # ['binary', 'files', '/', 'dev', '/', 'null', 'and', 'b', '/', 'art', '/', 'intro', '.', 'png', 'differ', '<nl>']

        for i in range(mask_strat, mask_end):
            line_to_mask[i] = Constants.MSK_WORD
        
        before_commit[maxlen_idx] = " ".join(line_to_mask) # 将mask后的列表元素用空格连接起来并替换原来的部分

        source_list.append("<nl>".join(before_commit))
        target_list.append("<nl>".join(after_commit))

        if in_statement_pred:
            # for idx_to_mask in range(len(lines)) 
            for idx_to_mask in addition_idx + deletion_idx:
                token_mask_source = []
                token_mask_target = []
                line_to_mask = lines[idx_to_mask].split()
                mask_len = math.floor(mask_rate * len(line_to_mask))
                token_to_mask_idx = random.randint(0, len(line_to_mask))
                for idx in range(len(lines)):
                    if idx == idx_to_mask:
                        token_mask_source.append( " ".join([Constants.MSK_WORD if idx == token_to_mask_idx else token for idx, token in enumerate(line_to_mask)]) )
                        token_mask_target.append(lines[idx])
                    else:
                        token_mask_source.append( " ".join([Constants.PAD_WORD for token in lines[idx].split()]) )
                        token_mask_target.append( " ".join([Constants.PAD_WORD for token in lines[idx].split()]) )

    return source_list, target_list

"""
python pretrain.py -train_src data/cleaned.train.diff -valid_src data/cleaned.valid.diff -save_data vocab/pretrain_vocab
"""
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src',required=True)
    parser.add_argument('-valid_src',required=True)
    parser.add_argument('-vocab',default=None)
    parser.add_argument('-save_data',required=True)
    parser.add_argument('-mask_rate',type=float, default=0.3)
    parser.add_argument('-max_len','--max_word_seq_len',type=int, default=100)
    parser.add_argument('-min_word_count',type=int,default=5)
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-keep_case',action='store_true')
    parser.add_argument('-in_statement_pred', type= bool, default=False)

    opt= parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # 加入<s>and</s>

    #训练数据读取
    #-train_src 设置 "./data/cleaned.train.diff"
    #-train_tag 设置 "./data/cleaned.train.msg"
    train_src_word_insts, train_tag_word_insts = read_instance_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case,opt.mask_rate, opt.in_statement_pred)
    
    #检查训练样本和标签数据是否长度相等
    if len(train_src_word_insts) != len(train_tag_word_insts):
        print('[waring] the trianing isntances count is not equal.')
        min_insts_count = min(len(train_src_word_insts),len(train_tag_word_insts))
        train_tag_word_insts = train_tag_word_insts[:min_insts_count]
        train_src_word_insts = train_src_word_insts[:min_insts_count]

    # 移除空列表 
    # list(zip(*[...]))将过滤后的配对重新拆分成两个独立的列表，分别包含非空的源语言实例和目标语言实例。
    train_src_word_insts, train_tag_word_insts = list(zip(*[(s,t)for s,t in zip(train_src_word_insts, train_tag_word_insts) if s and t]))

    # Validation set 同样的步骤
    valid_src_word_insts, valid_tag_word_insts = read_instance_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case, opt.mask_rate, opt.in_statement_pred)

    if len(valid_src_word_insts) != len(valid_tag_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tag_word_insts = valid_tag_word_insts[:min_inst_count]
    # 移除空列表 
    # list(zip(*[...]))将过滤后的配对重新拆分成两个独立的列表，分别包含非空的源语言实例和目标语言实例。
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[(s,t)for s,t in zip(valid_src_word_insts, valid_tag_word_insts) if s and t]))
    
    #建立字典表
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data
        print('[info] pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tag_word2idx = predefined_data['dict']['tag']

    else:
        if opt.share_vocab:
            print('[info] bulid shared vocabulary for source and target')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tag_word_insts, opt.min_word_count
            )
            src_word2idx = tag_word2idx = word2idx
        else:
            print('[info] bulid vocabulary for source')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[info] bulid vocabulary for target')
            tag_word2idx = build_vocab_idx(train_tag_word_insts, opt.min_word_count)
    # src word 2 index
    print('[info] convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)            
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)   
    # tag word 2 index
    print('[Info] Convert target word instances into sequences of word index.')
    train_tag_insts = convert_instance_to_idx_seq(train_tag_word_insts, tag_word2idx)
    valid_tag_insts = convert_instance_to_idx_seq(valid_tag_word_insts, tag_word2idx)

    data = {
        'setting':opt,
        'dict':{
            'src':src_word2idx,
            'tag':tag_word2idx
        },
        'train':{
            'src':train_src_insts,
            'tag':train_tag_insts
        },
        'valid':{
            'src':valid_src_insts,
            'tag':valid_tag_insts
        }
    }
    print('[Info] save the processed data', opt.save_data)
    torch.save(data, opt.save_data)#这里save_data的文件要有，不然报错
    print('[info] finish')


    
if __name__ == "__main__":
    main()