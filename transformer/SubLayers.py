import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.linear(d_model, n_head*d_k)
        self.w_ks = nn.linear(d_model, n_head*d_k)
        self.w_vs = nn.linear(d_model, n_head*d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attn = None
        self.attention = ScaleDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = q.permute(2,0,1,3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2,0,1,3).contiguous().view(-1, len_q, d_k)
        v = v.permute(2,0,1,3).contiguous().view(-1, len_q, d_v)

        mask = mask.repeat(n_head, 1, 1)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1,2,0,3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, self.attn
    
class PostionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1,2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1,2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output