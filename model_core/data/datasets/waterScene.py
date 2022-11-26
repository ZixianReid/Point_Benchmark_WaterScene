from . import coco
import torch
from PIL import Image
import os
import numpy as np
import pandas as pd

from model_core.structures.bounding_box import BoxList
from model_core.structures.keypoint import PersonKeypoints
from model_core.structures.segmentation_mask import SegmentationMask


class WaterSceneDataset(coco.COCODataset):
    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None):
        super(WaterSceneDataset, self).__init__(ann_file, root, remove_images_without_annotations, transforms)

    def __getitem__(self, idx):
        coco = self.coco
        image_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anno = coco.loadAnns(ann_ids)

        im_path = coco.loadImgs(image_id)[0]['file_name']

        # extracte points cloud info
        pc_path = coco.loadImgs(image_id)[0]['pc_file_name']
        pc = pd.read_csv(os.path.join(self.root, pc_path), dtype=np.float32)

        pc = pc.drop(pc[pc['u'] <= 0].index)
        pc = pc.drop(pc[pc['v'] <= 0].index)
        pc = pc.drop(pc[pc['u'] >= 1920].index)
        pc = pc.drop(pc[pc['v'] >= 1080].index)

        features = pc[['x', 'y', 'z', 'rcs', 'doppler']].to_numpy()
        pc_target = pc[['label']].to_numpy()
        pc = features

        img = Image.open(os.path.join(self.root, im_path)).convert('RGB')
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='poly')
        target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            pc, target, pc_target = self._transforms(pc, target, pc_target)

        return pc, target, pc_target, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
