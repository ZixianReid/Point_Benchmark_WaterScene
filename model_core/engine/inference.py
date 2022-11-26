from tqdm import tqdm
from torchmetrics.functional import jaccard_index
import torch

def inference(model, data_loader, device):
    ious = []
    for pc, targets, img_ids in tqdm(data_loader):
        pc = pc.to(device)
        outs = model(pc)
        sizes = (pc.ptr[1:] - pc.ptr[:-1]).tolist()
        for out, y, in zip(outs.split(sizes), pc.y.split(sizes)):
            iou = jaccard_index(out.argmax(dim=-1), y, num_classes=6, absent_score=1.0)
            ious.append(iou)

    iou = torch.tensor(ious, device=device)
    mean_iou = iou.mean()
    print(f"Test IOU: {mean_iou}")
