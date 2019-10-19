import torch


# Time Step Dropout
def timestep_dropout(inputs, p=0.5, batch_first=True):
    '''
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    '''
    if not batch_first:
        inputs = inputs.transpose(0, 1)

    batch_size, time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1-p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    # drop_mask = drop_mask.unsqueeze(-1).expand((-1, -1, time_step)).transpose(1, 2)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask


# Independent Dropout
def independent_dropout(x, y, p=0.5, eps=1e-12):
    '''
    :param x: (bz, time_step, feature_size)
    :param y: (bz, time_step, feature_size)
    :param p:
    :param eps:
    :return:
    '''
    x_mask = torch.bernoulli(x.data.new_full(x.shape[:2], 1 - p))
    y_mask = torch.bernoulli(y.data.new_full(y.shape[:2], 1 - p))
    scale = 3. / (2 * x_mask + y_mask + eps)
    x_mask *= scale
    y_mask *= scale
    x *= x_mask.unsqueeze(dim=-1)
    y *= y_mask.unsqueeze(dim=-1)
    return x, y
