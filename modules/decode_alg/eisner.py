# -*- coding: utf-8 -*-

import torch
from torch.nn.utils.rnn import pad_sequence


# def kmeans(x, k):
#     x = torch.tensor(x, dtype=torch.float)
#     # count the frequency of each datapoint
#     d, indices, f = x.unique(return_inverse=True, return_counts=True)
#     # calculate the sum of the values of the same datapoints
#     total = d * f
#     # initialize k centroids randomly
#     c, old = d[torch.randperm(len(d))[:k]], None
#     # assign labels to each datapoint based on centroids
#     dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
#     # make sure number of datapoints is greater than that of clusters
#     if len(d) < k:
#         raise AssertionError(f"unable to assign {len(d)} datapoints to "
#                              f"{k} clusters")
#
#     while old is None or not c.equal(old):
#         # if an empty cluster is encountered,
#         # choose the farthest datapoint from the biggest cluster
#         # and move that the empty one
#         for i in range(k):
#             if not y.eq(i).any():
#                 mask = y.eq(torch.arange(k).unsqueeze(-1))
#                 lens = mask.sum(dim=-1)
#                 biggest = mask[lens.argmax()].nonzero().view(-1)
#                 farthest = dists[biggest].argmax()
#                 y[biggest[farthest]] = i
#         mask = y.eq(torch.arange(k).unsqueeze(-1))
#         # update the centroids
#         c, old = (total * mask).sum(-1) / (f * mask).sum(-1), c
#         # re-assign all datapoints to clusters
#         dists, y = torch.abs_(d.unsqueeze(-1) - c).min(dim=-1)
#     # assign all datapoints to the new-generated clusters
#     # without considering the empty ones
#     y, assigned = y[indices], y.unique().tolist()
#     # get the centroids of the assigned clusters
#     centroids = c[assigned].tolist()
#     # map all values of datapoints to buckets
#     clusters = [torch.where(y.eq(i))[0].tolist() for i in assigned]
#
#     return centroids, clusters


def eisner(scores, mask):  # mask = mask[:, 0]
    lens = mask.sum(1)  # 不包括<root>
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i, r) + C(j, r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.permute(2, 0, 1)
        il = ilr + scores.diagonal(-w).unsqueeze(-1)
        # I(j, i) = max(C(i, r) + C(j, r+1) + S(j, i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span)
        p_i.diagonal(-w).copy_(il_path + starts)
        ir = ilr + scores.diagonal(w).unsqueeze(-1)
        # I(i, j) = max(C(i, r) + C(j, r+1) + S(i, j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span)
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j, i) = max(C(r, i) + I(j, r)), i <= r < j
        cl = stripe(s_c, n, w, dim=0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i, j) = max(I(i, r) + C(r, j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    predicts = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        # length+1的原因是第一列全部置成了0
        heads = p_c.new_ones(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads.to(mask.device))

    return pad_sequence(predicts, True)


def backtrack(p_i, p_c, heads, i, j, complete):
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r'''Returns a diagonal stripe of the tensor.

    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    '''
    seq_len = x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)
