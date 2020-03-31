from .layer import *
from .encoder.transformer2 import TransformerEncoder
from .encoder.bilstm import BiLSTMEncoder
from .encoder.cnn_encoder import ConvEncoder
from .encoder.transformer import PositionEmbed
from modules.decode_alg.MST import mst_decode
from .dropout import *
import torch
'''
graph-based parser构成：
1、特征提取器(CNN / LSTM / ELMo / Transformer / Bert)
2、MLP-arc / MLP-rel
3、Biaffine
'''


def swish(x):
    return x * torch.sigmoid(x)


class ParserModel(nn.Module):
    def __init__(self, args, embed_weights=None):
        super(ParserModel, self).__init__()

        self.args = args

        self.wd_embedding = nn.Embedding(num_embeddings=args.wd_vocab_size,
                                         embedding_dim=args.wd_embed_dim,
                                         padding_idx=0)

        self.pretrained_embedding = nn.Embedding.from_pretrained(torch.from_numpy(embed_weights), freeze=True)
        # self.pretrained_embedding = nn.Embedding(num_embeddings=embed_weights.shape[0],
        #                                          embedding_dim=args.wd_embed_dim,
        #                                          padding_idx=0)
        # self.pretrained_embedding.weight.data.copy_(torch.from_numpy(embed_weights))
        # self.pretrained_embedding.weight.requires_grad = False

        # 词性embedding
        self.tag_embedding = nn.Embedding(num_embeddings=args.pos_size,
                                          embedding_dim=args.tag_embed_dim,
                                          padding_idx=0)

        self.pos_embedding = nn.Embedding.from_pretrained(torch.from_numpy(PositionEmbed(args.max_pos_embeddings, d_model=args.wd_embed_dim + args.tag_embed_dim, pad_idx=0)), freeze=True)

        in_features = args.hidden_size
        if args.enc_type == 'cnn':
            self.encoder = ConvEncoder(args)
        elif args.enc_type == 'lstm':
            self.encoder = BiLSTMEncoder(args)
        elif args.enc_type == 'transformer':
            self.encoder = TransformerEncoder(args)

        self._activation = nn.ReLU()
        # self._activation = nn.ELU()
        # self._activation = nn.LeakyReLU(0.1)

        self.mlp_arc = NonlinearMLP(in_feature=in_features,
                                    out_feature=args.arc_mlp_size * 2,
                                    activation=self._activation)
        self.mlp_lbl = NonlinearMLP(in_feature=in_features,
                                    out_feature=args.label_mlp_size * 2,
                                    activation=self._activation)

        self.arc_biaffine = Biaffine(args.arc_mlp_size,
                                     1, bias=(True, False))

        self.label_biaffine = Biaffine(args.label_mlp_size,
                                       args.rel_size, bias=(True, True))

        # self.word_fc = nn.Linear(args.wd_embed_dim, args.wd_embed_dim)
        # self.tag_fc = nn.Linear(args.tag_embed_dim, args.tag_embed_dim)
        # self.word_norm = nn.LayerNorm(args.wd_embed_dim)
        # self.tag_norm = nn.LayerNorm(args.tag_embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wd_embedding.weight)
        nn.init.xavier_uniform_(self.tag_embedding.weight)

    def forward(self, wd_inputs, extwd_inputs, tag_inputs, non_pad_mask=None):
        # (bz, seq_len, embed_dim)
        wd_embed = self.wd_embedding(wd_inputs)
        extwd_embed = self.pretrained_embedding(extwd_inputs)
        wd_embed = wd_embed + extwd_embed
        # (bz, seq_len, embed_dim)
        tag_embed = self.tag_embedding(tag_inputs)
        # wd_embed, tag_embed = self.word_fc(wd_embed), self.tag_fc(tag_embed)
        # wd_embed, tag_embed = self.word_norm(wd_embed), self.tag_norm(tag_embed)

        if self.training:
            wd_embed, tag_embed = independent_dropout(wd_embed, tag_embed, self.args.embed_drop)

        # (bz, seq_len, 2*embed_dim)
        input_embed = torch.cat((wd_embed, tag_embed), dim=-1)

        if self.args.enc_type == 'transformer':
            seq_len = wd_inputs.size(1)
            seq_range = torch.arange(seq_len, dtype=torch.long, device=wd_inputs.device)\
                .unsqueeze(dim=0).expand_as(wd_inputs)  # (1, seq_len)
            # [bz, seq_len, d_model]
            input_embed = input_embed + self.pos_embedding(seq_range)
            if self.training:
                input_embed = timestep_dropout(input_embed, self.args.embed_drop)

        enc_out = self.encoder(input_embed, non_pad_mask)

        if self.training:
            enc_out = timestep_dropout(enc_out, self.args.lstm_drop)

        arc_feat = self.mlp_arc(enc_out)
        lbl_feat = self.mlp_lbl(enc_out)
        arc_head, arc_dep = arc_feat.chunk(2, dim=-1)
        lbl_head, lbl_dep = lbl_feat.chunk(2, dim=-1)

        if self.training:
            arc_head = timestep_dropout(arc_head, self.args.arc_mlp_drop)
            arc_dep = timestep_dropout(arc_dep, self.args.arc_mlp_drop)
        arc_score = self.arc_biaffine(arc_dep, arc_head).squeeze(-1)

        if self.training:
            lbl_head = timestep_dropout(lbl_head, self.args.label_mlp_drop)
            lbl_dep = timestep_dropout(lbl_dep, self.args.label_mlp_drop)
        lbl_score = self.label_biaffine(lbl_dep, lbl_head)

        return arc_score, lbl_score

    def predict(self, wd_inputs, extwd_inputs, tag_inputs, non_pad_mask=None):
        arc_score, lbl_score = self(wd_inputs, extwd_inputs, tag_inputs, non_pad_mask)
        heads_pred = mst_decode(arc_score, non_pad_mask)
        return heads_pred
