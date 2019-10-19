import torch
import torch.nn as nn


class NonlinearMLP(nn.Module):
    def __init__(self, in_feature, out_feature, activation=nn.ReLU(), bias=True):
        super(NonlinearMLP, self).__init__()

        if activation is None:
            self.activation = lambda x: x
        else:
            assert callable(activation)
            self.activation = activation

        self.linear = nn.Linear(in_features=in_feature,
                                out_features=out_feature,
                                bias=bias)

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return self.activation(linear_out)


class Biaffine(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 bias=(True, True)  # True = 1  False = 0
                 ):
        super(Biaffine, self).__init__()
        self.in_features = in_features  # mlp_arc_size / mlp_label_size
        self.out_features = out_features  # 1 / rel_size
        self.bias = bias

        # arc / label: mlp_size + 1
        self.linear_input_size = in_features + bias[0]
        # arc: mlp_size
        # label: (mlp_size + 1) * rel_size
        self.linear_output_size = out_features * (in_features + bias[1])

        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.zeros_(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        if self.bias[0]:
            ones = input1.data.new_ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=-1)
            # dim1 += 1
        if self.bias[1]:
            ones = input2.data.new_ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=-1)
            # dim2 += 1

        # (bz, len1, dim1+1) -> (bz, len1, linear_output_size)
        affine = self.linear(input1)

        # (bz, len1 * self.out_features, dim2)
        affine = affine.reshape(batch_size, len1 * self.out_features, -1)

        # (bz, len1 * out_features, dim2) * (bz, dim2, len2)
        # -> (bz, len1 * out_features, len2) -> (bz, len2, len1 * out_features)
        biaffine = torch.bmm(affine, input2.transpose(1, 2)).transpose(1, 2)

        # (bz, len2, len1, out_features)    # out_features: 1 or rel_size
        biaffine = biaffine.reshape((batch_size, len2, len1, -1))

        return biaffine
