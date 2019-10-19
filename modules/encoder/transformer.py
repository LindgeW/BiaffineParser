import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# PE(pos, 2i) = sin(pos/10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
def PositionEmbed(max_len, d_model, pad_idx=None):
    pe = np.asarray([[pos / np.power(10000, 2*(i//2) / d_model) for i in range(d_model)]
                     for pos in range(max_len)], dtype=np.float32)
    pe[:, 0::2] = np.sin(pe[:, 0::2])  # start : end : step
    pe[:, 1::2] = np.cos(pe[:, 1::2])

    if pad_idx is not None:
        pe[pad_idx] = 0

    return pe


class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self._dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, pad_mask=None):
        '''
        :param q: [bz, len_q, Q]
        :param k: [bz, len_k, K]
        :param v: [bz, len_v, V]
        :param pad_mask: [bz, len_q, len_k]  填充部分的mask
        more: Q==K, len_k==len_v
        :return: [bz, len_q, V]
        '''
        # [bz, len_q, Q] * [bz, K, len_k] -> [bz, len_q, len_k]
        att_weights = torch.matmul(q, k.transpose(-1, -2))
        att_weights /= math.sqrt(k.size(-1))

        if pad_mask is not None:
            att_weights.masked_fill_(pad_mask, float('-inf'))

        # [bz, len_q, len_k]
        soft_att_weights = F.softmax(att_weights, dim=-1)

        if self.training:
            soft_att_weights = self._dropout(soft_att_weights)

        # [bz, len_q, len_k] * [bz, len_v, V] -> [bz, len_q, V]
        att_out = torch.matmul(soft_att_weights, v)

        return att_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, nb_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._d_model = d_model
        self._d_k = d_k
        self._d_v = d_v
        self._nb_heads = nb_heads

        self._linear_qs = nn.Linear(in_features=d_model, out_features=d_k * nb_heads)

        self._linear_ks = nn.Linear(in_features=d_model, out_features=d_k * nb_heads)

        self._linear_vs = nn.Linear(in_features=d_model, out_features=d_v * nb_heads)

        self._linear_out = nn.Linear(in_features=d_v * nb_heads, out_features=d_model)

        self._self_attention = SelfAttention(dropout)

        self._layer_norm = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self._linear_qs.weight, mean=0, std=math.sqrt(1 / self._d_model))
        nn.init.normal_(self._linear_ks.weight, mean=0, std=math.sqrt(1 / self._d_model))
        nn.init.normal_(self._linear_vs.weight, mean=0, std=math.sqrt(1 / self._d_model))

    def forward(self, q, k, v, pad_mask=None):
        '''
        :param q: [bz, len_q, d_model]
        :param k: [bz, len_k, d_model]
        :param v: [bz, len_v, d_model]
        :param pad_mask: [bz, len_k]
        more: Q == K, len_k==len_v
        :return: [bz, len_q, d_model]
        '''
        residual = q

        bz, len_q, _ = q.size()
        bz, len_k, _ = k.size()
        bz, len_v, _ = v.size()
        # [bz, len_q, d_k * nb_heads] -> [bz, nb_heads, len_q, d_k]
        q_fc = self._linear_qs(self._layer_norm(q)).reshape(bz, len_q, self._nb_heads, -1).transpose(1, 2)
        # [bz, len_k, d_k * nb_heads] -> [bz, nb_heads, len_k, d_k]
        k_fc = self._linear_ks(self._layer_norm(k)).reshape(bz, len_k, self._nb_heads, -1).transpose(1, 2)
        # [bz, len_v, d_v * nb_heads] -> [bz, nb_heads, len_v, d_v]
        v_fc = self._linear_vs(self._layer_norm(v)).reshape(bz, len_v, self._nb_heads, -1).transpose(1, 2)

        if pad_mask is not None:
            # (bz, 1, 1, len_k)
            pad_mask = pad_mask[:, None, None, :]

        # (bz, nb_heads, len_q, d_v)
        att_out = self._self_attention(q_fc, k_fc, v_fc, pad_mask)
        att_out = att_out.transpose(1, 2).reshape(bz, len_q, -1)
        # [bz, len_q, nb_heads*d_v] -> [bz, len_q, d_model]
        multi_head = self._linear_out(att_out)

        if self.training:
            multi_head = self._dropout(multi_head)

        return residual + multi_head


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_in,
                      out_channels=d_ff,
                      kernel_size=1),  # 权重共享
            nn.ReLU(),
            nn.Conv1d(in_channels=d_ff,
                      out_channels=d_in,
                      kernel_size=1)
        )

        self._layer_norm = nn.LayerNorm(d_in)

        self._dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        '''
        :param inputs: [bz, len_q, d_model]
        :return: [bz, len_q, d_model]
        '''
        residual = inputs

        # [bz, len_q, d_model] -> [bz, d_model, len_q]
        fc_in = self._layer_norm(inputs).transpose(1, 2)

        # [bz, d_model, len_q]
        fc_out = self.ffn(fc_in)

        # [bz, len_q, d_model]
        out = fc_out.transpose(1, 2)

        if self.training:
            out = self._dropout(out)

        # return self._layer_norm(residual + out)
        return residual + out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, nb_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # multi_head self-attention
        self._multi_head_att = MultiHeadAttention(d_model=d_model,
                                                  d_k=d_k,
                                                  d_v=d_v,
                                                  nb_heads=nb_heads,
                                                  dropout=dropout)
        # feedforward
        self._pwffn = PositionwiseFeedForward(d_in=d_model,
                                              d_ff=d_ff)

    def forward(self, enc_in, pad_mask=None, non_pad_mask=None):
        '''
        :param enc_in: [bz, len_k, d_model]
        :param pad_mask: [bz, len_k] 填充部分mask
        :param non_pad_mask: [bz, len_q, 1]
        :return: [bz, len_q, d_model]
        '''
        # [bz, len_q, d_model]
        multi_head = self._multi_head_att(enc_in, enc_in, enc_in, pad_mask)
        if non_pad_mask is not None:
            multi_head *= non_pad_mask

        # [bz, len_q, d_model]
        out = self._pwffn(multi_head)
        if non_pad_mask is not None:
            out *= non_pad_mask

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()

        nb_heads = args.nb_heads
        d_model = args.wd_embed_dim + args.tag_embed_dim
        d_k = d_v = d_model // nb_heads
        assert (d_k * nb_heads == d_model)

        self._encoder_stack = nn.ModuleList([
            EncoderLayer(d_model, d_k, d_v, args.d_ff, nb_heads)
            for _ in range(args.encoder_layer)
        ])

        self.linear = nn.Linear(in_features=d_model,
                                out_features=args.hidden_size)

    def forward(self, embed_inputs, non_pad_mask):
        '''
        :param embed_inputs:  (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        pad_mask = (non_pad_mask == 0)  # 填充部分的mask(uint8类型)

        encoder_in = embed_inputs
        for encoder in self._encoder_stack:
            # [bz, len_q, d_model]
            encoder_in = encoder(encoder_in, pad_mask=pad_mask)

        # [bz, seq_len, hidden_size]
        encoder_out = self.linear(encoder_in)

        return encoder_out
