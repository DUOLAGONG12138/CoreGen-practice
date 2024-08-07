import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants



class Transformer(nn.Moudle):
    """带有注意力机制的序列到序列的 模型"""

    def __init__(
            self,
            n_src_vocab, n_tag_vocab, len_max_seq,
            d_word_vec
        ):
    
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

        self.tag_word_prj = nn.Linear(d_model, n_tag_vocab, bias=False)
        nn.init.xavier_normal_(self.tag_word_prj.weight)
        assert d_model == d_word_vec,'to facilitate the residual connections,the dimensions of all module outputs shall be the same.'


        if tag_emb_prj_weight

