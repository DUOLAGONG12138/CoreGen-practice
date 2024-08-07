import argparse
import torch
import transformer.Constants as Constants

def  read_instances_from_file(inst_file, max_sent_len, keep_case):
    #构建列表存储句子
    word_insts = []
    #构建计数器
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            #判断是否保留大小写，keepcase为true为保留，false不保留
            if not keep_case: 
                #保留情况默认，不保留用lower()函数全部变小写
                sent = sent.lower()
            #将句子拆分为词，按空格区分
            words = sent.split()
            #判断词的长度是否大于最大值
            if len(words) > max_sent_len:
                #大于最大值就计数器+1
                trimmed_sent_count += 1
            #截断句子，保证输出的每行句子长度相等
            word_inst = words[:max_sent_len]

            #如果word_insts不为空，就在前后加上开始符合<S>,结束符号</s>
            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]
    #打印数据来源信息和长度
    print('Info Get {} instance from{}.'.format(len(word_insts),inst_file))
    
    #若计数器不为0，打印警告信息说明多少句子被最大长度限制截断
    if trimmed_sent_count > 0:
        print('[warning]{} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))
        
    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    #将word_insts中每一个句子中每一个词提取到一个集合中，去掉重复值
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size ={}'.format(len(full_vocab)))

    word2idx = {
        Constants.BOS_WORD : Constants.BOS,
        Constants.EOS_WORD : Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.MSK_WORD: Constants.MSK
    }
    #每个token的数量
    word_count ={w:0 for w in full_vocab}
    #遍历全部训练数据，将token频数统计
    for sent in word_insts:
        for word in sent:
            word_count[word] += 1
    #统计一下多少个单词被忽略（即不满足最小词频限制
    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:#检查当前单词是否在另一个字典word2idx中。如果不在，说明这个单词还没有被处理过
            if count > min_word_count:#  min_word_count = 5
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1
    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))

    return word2idx
    
def convert_instance_to_idx_seq(word_insts, word2idx):
    return [[word2idx.get(w,Constants.UNK) for w in s]for s in word_insts]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src',required=True)
    parser.add_argument('-train_tag',required=True)
    parser.add_argument('-valid_src',required=True)
    parser.add_argument('-valid_tag',required=True)
    parser.add_argument('-save_data',required=True)
    parser.add_argument('-max_len','--max_word_seq_len',type=int, default=100)
    parser.add_argument('-min_word_count',type=int,default=5)
    parser.add_argument('-vocab',default=None)
    parser.add_argument('-keep_case',action='store_true')
    parser.add_argument('-share_vocab', action='store_true')

    opt= parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # 加入<s>and</s>

    #训练数据读取
    #-train_src 设置 "./data/cleaned.train.diff"
    #-train_tag 设置 "./data/cleaned.train.msg"
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tag_word_insts = read_instances_from_file(
        opt.train_tag, opt.max_word_seq_len, opt.keep_case)
    
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
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tag_word_insts = read_instances_from_file(
        opt.valid_tag, opt.max_word_seq_len, opt.keep_case)

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
        predefined_data = torch.loda(opt.vocab)
        assert 'dict' in predefined_data
        print('[info] pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tag_word2idx = predefined_data['dice']['tag']

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