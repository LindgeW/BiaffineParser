from .rnn_encoder import RNNEncoder
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, args):
        super(BiLSTMEncoder, self).__init__()

        self.bilstm = RNNEncoder(input_size=args.wd_embed_dim + args.tag_embed_dim,
                                 hidden_size=args.hidden_size // 2,
                                 num_layers=args.lstm_depth,
                                 dropout=args.lstm_drop,
                                 batch_first=True,
                                 bidirectional=True,
                                 rnn_type='lstm')

    def forward(self, embed_inputs, non_pad_mask=None):
        '''
        :param embed_inputs: (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        # lstm_out: (bz, seq_len, lstm_size * num_directions)
        # lstm_hidden: (num_layers, batch, lstm_size * num_directions)
        lstm_out, lstm_hidden = self.bilstm(embed_inputs, mask=non_pad_mask)

        return lstm_out
