import os
import json
from argparse import ArgumentParser


def get_data_path(json_path):
    assert os.path.exists(json_path)
    with open(json_path, 'r', encoding='utf-8') as fr:
        data_opts = json.load(fr)

    return data_opts


def args_config():
    parse = ArgumentParser('Biaffine Parser Argument Configuration')

    parse.add_argument('--cuda', type=int, default=-1, help='training device, default on cpu')

    parse.add_argument('-lr', '--learning_rate', type=float, default=3e-3, help='learning rate of training')
    parse.add_argument('-bt1', '--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
    parse.add_argument('-bt2', '--beta2', type=float, default=0.99, help='beta2 of Adam optimizer')

    parse.add_argument('--batch_size', type=int, default=128, help='batch size')
    parse.add_argument('--epoch', type=int, default=100, help='iteration of training')
    parse.add_argument('--update_steps', type=int, default=4, help='gradient accumulation and update per x steps')

    parse.add_argument('--wd_embed_dim', type=int, default=100, help='word embedding size')
    parse.add_argument('--tag_embed_dim', type=int, default=100, help='pos_tag embedding size')
    parse.add_argument('--num_convs', type=int, default=5, help='the depth of convolutional layer')
    parse.add_argument('--kernel_size', type=int, default=5, help='the window size of convolution')
    # parse.add_argument('--lstm_size', type=int, default=400, help='lstm hidden size')
    parse.add_argument('--lstm_depth', type=int, default=3, help='the depth of lstm layer')
    parse.add_argument('--arc_mlp_size', type=int, default=500, help='arc mlp size')
    parse.add_argument('--label_mlp_size', type=int, default=100, help='label mlp size')

    parse.add_argument('-mpe', '--max_pos_embeddings', default=200, help='max sequence position embeddings')
    parse.add_argument("--d_model", type=int, default=200, help='sub-layer feature size')
    # parse.add_argument("--d_k", type=int, default=50, help='Query or Key feature size')
    # parse.add_argument("--d_v", type=int, default=50, help='Value feature size')
    parse.add_argument("--d_ff", type=int, default=512, help='pwffn inner-layer feature size')
    parse.add_argument("--nb_heads", type=int, default=8, help='sub-layer feature size')
    parse.add_argument("--encoder_layer", type=int, default=6, help='the number of encoder layer')

    parse.add_argument("--hidden_size", type=int, default=200, help='the output size of encoder layer, including lstm and transformer encoder')
    parse.add_argument("--enc_type", default='lstm', choices=['cnn', 'lstm', 'transformer'], help='encoder type, including `lstm` and `transformer`')

    parse.add_argument('--embed_drop', type=float, default=0.33, help='embedding dropout')
    parse.add_argument('--lstm_drop', type=float, default=0.33, help='LSTM dropout')
    parse.add_argument('--cnn_drop', type=float, default=0.2, help='CNN layer dropout')
    parse.add_argument('--arc_mlp_drop', type=float, default=0.33, help='Arc MLP dropout')
    parse.add_argument('--label_mlp_drop', type=float, default=0.33, help='Label MLP dropout')

    args = parse.parse_args()

    print(vars(args))

    return args
