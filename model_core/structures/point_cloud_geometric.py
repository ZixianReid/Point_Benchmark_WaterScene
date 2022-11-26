import torch


class PointGeometric(object):
    def __init__(self, batch, x, pos, y, ptr):
        self.batch = batch
        self.x = x
        self.pos = pos
        self.y = y
        self.ptr = ptr

    def to(self, device):
        return PointGeometric(self.batch.to(device), self.x.to(device),
                              self.pos.to(device), self.y.to(device), self.ptr.to(device))


def to_point_geometric(tuple_tensors, targets):
    batch = get_batch(tuple_tensors)
    x = get_x(tuple_tensors)
    pos = get_pos(tuple_tensors)
    y = get_y(targets)
    ptr = get_ptr(tuple_tensors)

    return PointGeometric(batch, x, pos, y, ptr)


def get_ptr(tuple_sensors):
    prt = []
    prt.append(0)
    for ele in tuple_sensors:
        prt.append(ele.shape[0] + prt[-1])
    prt = torch.tensor(prt, dtype=torch.int64)
    return prt


def get_batch(tuple_tensors):
    batch = []
    for idx, _ in enumerate(tuple_tensors):
        batch.append(torch.ones(tuple_tensors[idx].shape[0], dtype=torch.int64) * idx)
    batch = torch.cat(batch, dim=0)
    return batch


def get_x(tuple_tensors):
    return torch.cat(tuple_tensors, dim=0)


def get_pos(tuple_tensors):
    x = get_x(tuple_tensors)
    indices = torch.LongTensor([0, 1, 2])
    pos = torch.index_select(x, 1, indices)
    return pos


def get_y(targets_tuple):
    targets = []
    for idx, ele in enumerate(targets_tuple):
        targets.append(torch.tensor(ele))
    targets = torch.cat(targets, dim=0)
    targets = torch.squeeze(targets).type(torch.int64)
    return targets
