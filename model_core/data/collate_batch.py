# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from model_core.structures.point_cloud_geometric import to_point_geometric


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        pc = to_point_geometric(transposed_batch[0], transposed_batch[2])
        targets = transposed_batch[1]
        img_ids = transposed_batch[3]
        return pc, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
