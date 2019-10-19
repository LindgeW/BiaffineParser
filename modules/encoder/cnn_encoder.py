import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, conv_layer, dropout=0.3):
        super(ResidualConv, self).__init__()
        assert isinstance(conv_layer, nn.Conv1d)
        self.conv_layer = conv_layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return inputs + self.dropout(self.conv_layer(inputs))


class ConvEncoder(nn.Module):
    def __init__(self, args):
        super(ConvEncoder, self).__init__()

        in_features = args.wd_embed_dim + args.tag_embed_dim
        assert in_features == args.hidden_size
        # 保证卷积前后序列长度不变：kernel_size = 2 * pad + 1
        assert args.kernel_size % 2 == 1
        padding = args.kernel_size // 2

        self.conv_layers = nn.Sequential()
        for i in range(args.num_convs):
            self.conv_layers.add_module(name=f'conv_{i}', module=ResidualConv(nn.Conv1d(in_channels=in_features,
                                                                               out_channels=args.hidden_size,
                                                                               kernel_size=args.kernel_size,
                                                                               padding=padding), args.cnn_drop))
            self.conv_layers.add_module(name=f'activation_{i}', module=nn.ReLU())

    def forward(self, embed_inputs, non_pad_mask=None):
        '''
        :param embed_inputs: (bz, seq_len, embed_dim)
        :param non_pad_mask: (bz, seq_len)
        :return:
        '''
        if non_pad_mask is not None:
            # (bz, seq_len, embed_dim) * (bz, seq_len, 1)  广播
            embed_inputs *= non_pad_mask.unsqueeze(dim=-1)

        # (bz, 2*embed_dim, seq_len)
        embed = embed_inputs.transpose(1, 2)

        conv_out = self.conv_layers(embed)

        # (bz, seq_len, 2*embed_dim)
        enc_out = conv_out.transpose(1, 2)

        return enc_out
