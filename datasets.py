"""
Datasets file. Code adapted from TokenCut: https://github.com/YangtaoWANG95/TokenCut
"""
import os
import torch
import json
import torchvision
import numpy as np
import skimage.io
import math

from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
from torchvision import transforms as pth_transforms


# Image transformation applied to all images
transform = pth_transforms.Compose(
    [
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

class ImageDataset:
    def __init__(self, image_path, resize=None):
        
        self.image_path = image_path
        self.name = image_path.split("/")[-1]

        # Read the image
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        # Build a dataloader
        if resize is not None:
            transform_resize = pth_transforms.Compose(
                [ 
                    pth_transforms.ToTensor(),
                    pth_transforms.Resize(resize),
                    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            img = transform_resize(img)
            self.img_size = list(img.shape[-1:-3:-1])
        else:
            img = transform(img)
            self.img_size = list(img.shape[-1:-3:-1])
        self.dataloader = [[img, image_path]]

    def get_image_name(self, *args, **kwargs):
        return self.image_path.split("/")[-1].split(".")[0]

    def load_image(self, *args, **kwargs):
        return Image.open(self.image_path).convert("RGB").resize(self.img_size)

class Dataset:
    def __init__(self, dataset_name, dataset_set, remove_hards):
        """
        Build the dataloader
        """

        self.dataset_name = dataset_name
        self.set = dataset_set

        if dataset_name == "COCO14":
            self.year = "2014"
            self.root_path = f"datasets/COCO14/images/{dataset_set}{self.year}"
            self.all_annfile = "datasets/COCO14/annotations/instances_train2014.json"
        elif dataset_name == "COCO17":
            self.year = "2017"
            self.root_path = f"datasets/COCO17"
            self.all_annfile = "datasets/COCO17/annotations/instances_train2017.json"
        else:
            raise ValueError("Unknown dataset 1.")
        
        self.name = f"{self.dataset_name}_{self.set}"
        print(self.root_path, self.all_annfile)

        # Build the dataloader
        if "COCO14" == dataset_name:
            self.dataloader = torchvision.datasets.CocoDetection(
                self.root_path, self.all_annfile, transform=transform
            )
        else:
            raise ValueError("Unknown dataset 2.")

        if "COCO17" == dataset_name:
            self.dataloader = torchvision.datasets.CocoDetection(
                self.root_path, self.all_annfile, transform=transform
            )
        else:
            raise ValueError("Unknown dataset 2.")

        # Set hards images that are not included
        self.remove_hards = remove_hards
        self.hards = []
        if remove_hards:
            self.name += f"-nohards"
            self.hards = self.get_hards()
            print(f"Nb images discarded {len(self.hards)}")

    def load_image(self, im_name):
        """
        Load the image corresponding to the im_name
        """
        if "COCO" in self.dataset_name:
            image = skimage.io.imread(f"./datasets/COCO14/images/train2014/{im_name}")
        else:
            raise ValueError("Unkown dataset.")
        return image

    def get_image_name(self, inp):
        """
        Return the image name
        """
        if "COCO" in self.dataset_name:
            im_name = str(inp[0]["image_id"])
            im_name = self.train2014['images']['file_name']
        return im_name

    def extract_gt(self, targets, im_name):
        if "COCO" in self.dataset_name:
            return extract_gt_COCO(targets, remove_iscrowd=True)
        else:
            raise ValueError("Unknown dataset")
        
    #### Could cause an error or many errors with coco17 fix if needed ####
    def extract_classes(self):
        if "COCO" in self.dataset_name:
            cls_path = f"classes_{self.dataset}_{self.set}_{self.year}.txt"
        # Load if exists
        if os.path.exists(cls_path):
            all_classes = []
            with open(cls_path, "r") as f:
                for line in f:
                    all_classes.append(line.strip())
        else:
            print("Extract all classes from the dataset")
            if "COCO" in self.dataset_name:
                all_classes = self.extract_classes_COCO()

            with open(cls_path, "w") as f:
                for s in all_classes:
                    f.write(str(s) + "\n")

        return all_classes

    def extract_classes_COCO(self):
        all_classes = []
        for im_id, inp in enumerate(tqdm(self.dataloader)):
            objects = inp[1]

            for o in range(len(objects)):
                if objects[o]["category_id"] not in all_classes:
                    all_classes.append(objects[o]["category_id"])

        return all_classes

    def get_hards(self):
        hard_path = "datasets/hard_%s_%s_%s.txt" % (self.dataset_name, self.set, self.year)
        if os.path.exists(hard_path):
            hards = []
            with open(hard_path, "r") as f:
                for line in f:
                    hards.append(int(line.strip()))
        else:
            print("Discover hard images that should be discarded")

            with open(hard_path, "w") as f:
                for s in hards:
                    f.write(str(s) + "\n")

        return hards

def extract_gt_COCO(targets, remove_iscrowd=True):
    objects = targets
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for o in range(nb_obj):
        # Remove iscrowd boxes
        if remove_iscrowd and objects[o]["iscrowd"] == 1:
            continue
        gt_cls = objects[o]["category_id"]
        gt_clss.append(gt_cls)
        bbx = objects[o]["bbox"]
        x1y1x2y2 = [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]
        x1y1x2y2 = [int(round(x)) for x in x1y1x2y2]
        gt_bbxs.append(x1y1x2y2)

    return np.asarray(gt_bbxs), gt_clss

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # https://github.com/ultralytics/yolov5/blob/develop/utils/general.py
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def get_train2014(all_annotations_file):
    # load all annotations
    with open(all_annotations_file, "r") as f:
        train2014 = json.load(f)
    return train2014

def get_train2017(all_annotations_file):
    # load all annotations
    with open(all_annotations_file, "r") as f:
        train2017 = json.load(f)
    return train2017