from tqdm import tqdm
from torchmetrics.functional import jaccard_index
import torch

CODE_LABEL_STR = {1: "pier",
                  2: "buoy",
                  3: "ship",
                  4: "boat",
                  5: "vessel"}


def inference(model, data_loader, device):
    ious = []

    ious_dict = {"pier": [], "ship": [], "boat": [], "vessel": []}

    for pc, targets, img_ids in tqdm(data_loader):
        pc = pc.to(device)
        outs = model(pc)
        sizes = (pc.ptr[1:] - pc.ptr[:-1]).tolist()
        for out, y, target in zip(outs.split(sizes), pc.y.split(sizes), targets):
            label = CODE_LABEL_STR[target.extra_fields['labels'].numpy()[0]]
            iou = jaccard_index(out.argmax(dim=-1), y, num_classes=6, absent_score=1.0)
            ious.append(iou)
            ious_dict[label].append(iou)

    iou = torch.tensor(ious, device=device)
    mean_iou = iou.mean()
    print(f"Test IOU: {mean_iou}")

    for key in ious_dict.keys():
        iou_per_class = torch.tensor(ious_dict[key], device=device)
        mean_iou_per_class = iou_per_class.mean()
        print(f"for {key}, the Test IoU is {mean_iou_per_class}")
