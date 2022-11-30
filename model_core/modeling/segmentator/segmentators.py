from .pointnet2 import PointNet2
from .pointtransformer import PointTransformer

_SEGMENTATION_META_ARCHITECTURES = {"PointNet2": PointNet2, "PointTransformer": PointTransformer}


def build_segmentation_model(cfg):
    meta_arch = _SEGMENTATION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
