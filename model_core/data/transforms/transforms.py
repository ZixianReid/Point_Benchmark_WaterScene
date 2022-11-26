# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from torchvision.transforms import functional as F
from torch.nn.functional import normalize as normalizePC


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pc, target, pc_target):
        for t in self.transforms:
            pc, target, pc_target = t(pc, target, pc_target)
        return pc, target, pc_target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, pc, target, pc_target):
        return F.to_tensor(pc).squeeze(0), target, pc_target


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, pc, target=None, pc_target=None):
        pc = normalizePC(pc, p=1, dim=0)
        if target is None:
            return pc
        return pc, target, pc_target
