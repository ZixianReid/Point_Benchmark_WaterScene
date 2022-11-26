from .pointnet2 import PointNet2

_SEGMENTATION_META_ARCHITECTURES = {"PointNet2": PointNet2}


def build_segmentation_model(cfg):
    meta_arch = _SEGMENTATION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
