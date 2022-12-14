# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from model_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from model_core.config import cfg
from model_core.data import make_data_loader
from model_core.engine.inference import inference
from model_core.modeling.segmentator import build_segmentation_model
from model_core.utils.checkpoint import DetectronCheckpointer
from model_core.utils.collect_env import collect_env_info
from model_core.utils.comm import synchronize, get_rank
from model_core.utils.logger import setup_logger
from model_core.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/mnt/develop/PycharmProjects/Point_Benchmark_WaterScene/configs/waterScene/pointnet2_waterscene_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("fcos_core", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    output_dir = cfg.OUTPUT_DIR
    model = build_segmentation_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    # # model_name = args.checkpointer_file.split('/')[-1]
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    #
    # iou_types = ("bbox",)
    # if cfg.MODEL.MASK_ON:
    #     iou_types = iou_types + ("segm",)
    # if cfg.MODEL.KEYPOINT_ON:
    #     iou_types = iou_types + ("keypoints",)
    # output_folders = [None] * len(cfg.DATASETS.TEST)
    # dataset_names = cfg.DATASETS.TEST
    # if cfg.OUTPUT_DIR:
    #     for idx, dataset_name in enumerate(dataset_names):
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    #         mkdir(output_folder)
    #         output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for data_loader_val in data_loaders_val:
        inference(
            model,
            data_loader_val,
            device=cfg.MODEL.DEVICE
        )
        synchronize()


if __name__ == "__main__":
    main()
