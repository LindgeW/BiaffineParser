import os
from .dependency import read_deps
import numpy as np
import torch


# Dependency对象列表的列表：[[], [], [], ...]
def load_dataset(path, vocab=None):
    assert os.path.exists(path)
    dataset = []
    with open(path, 'r', encoding='utf-8') as fr:
        for deps in read_deps(fr, vocab):
            dataset.append(deps)

    return dataset


def batch_iter(dataset: list, batch_size, dep_vocab, shuffle=False):
    if shuffle:
        np.random.shuffle(dataset)

    nb_batch = int(np.ceil(len(dataset) / batch_size))

    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        yield batch_variable(batch_data, dep_vocab)


def batch_variable(batch_data, dep_vocab):
    batch_size = len(batch_data)

    max_seq_len = max(len(deps) for deps in batch_data)

    wd_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    extwd_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    tag_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    head_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    rel_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    non_pad_mask = torch.zeros((batch_size, max_seq_len))

    for i, deps in enumerate(batch_data):
        seq_len = len(deps)
        wd_idx[i, :seq_len] = torch.tensor(dep_vocab.word2index([dep.form for dep in deps]))
        extwd_idx[i, :seq_len] = torch.tensor(dep_vocab.extwd2index([dep.form for dep in deps]))
        tag_idx[i, :seq_len] = torch.tensor(dep_vocab.pos2index([dep.pos for dep in deps]))
        head_idx[i, :seq_len] = torch.tensor([dep.head for dep in deps])
        rel_idx[i, :seq_len] = torch.tensor(dep_vocab.rel2index([dep.dep_rel for dep in deps]))
        non_pad_mask[i, :seq_len].fill_(1)

    return wd_idx, extwd_idx, tag_idx, head_idx, rel_idx, non_pad_mask
