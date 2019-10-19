'''
基于图的parser: Biaffine Attention Parser
1、读取CONLL语料：每个词语占一行，无值列用下划线'_'代替，列的分隔符为制表符'\t'，句子与句子之间用空行分隔
格式：词ID 当前词(或标点) 词性 当前词的head(ROOT为0) 当前词与head的依存关系
2、构建Dependency对象（每个词构建一个，一句话即为对象列表）
3、构建词表（词、词性、依存关系）
4、文本索引化、向量化
5、构建模型（特征提取模型+Biaffine）
6、训练
7、评估
'''

import torch
import numpy as np
from conf.config import get_data_path, args_config
from datautil.dataloader import load_dataset
from vocab.dep_vocab import create_vocab
from modules.model import ParserModel
from modules.parser import BiaffineParser


if __name__ == '__main__':
    np.random.seed(3046)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1344)

    print('cuda available:', torch.cuda.is_available())
    print('cuDnn available:', torch.backends.cudnn.enabled)
    print('GPU numbers:', torch.cuda.device_count())

    args = args_config()

    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    data_path = get_data_path("./conf/datapath.json")

    dep_vocab = create_vocab(data_path['data']['train_data'])
    embed_weights = dep_vocab.get_embedding_weights(data_path['pretrained']['word_embedding'])

    train_data = load_dataset(data_path['data']['train_data'], dep_vocab)
    dev_data = load_dataset(data_path['data']['dev_data'], dep_vocab)
    test_data = load_dataset(data_path['data']['test_data'], dep_vocab)

    args.wd_vocab_size = dep_vocab.vocab_size
    args.pos_size = dep_vocab.pos_size
    args.rel_size = dep_vocab.rel_size
    parser_model = ParserModel(args, embed_weights).to(args.device)

    biff_parser = BiaffineParser(parser_model)
    biff_parser.summary()

    biff_parser.train(train_data, dev_data, args, dep_vocab)

    test_uas, test_las = biff_parser.evaluate(test_data, args, dep_vocab)
    print('Test data -- UAS: %.3f%%, LAS: %.3f%%' % (test_uas, test_las))
