import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import cv2
import numpy as np
import utils.parameters as param


input_size = param.image_size

class ObjectDetectionDataset(Dataset):

    def __init__(self, img_folder, annotation_folder, rgb=False, transform=None):
        self.img_folder = img_folder
        self.annotation_folder = annotation_folder
        self.rgb = rgb
        self.transform = transform
        self.img_paths = [os.path.join(img_folder, img) for img in os.listdir(img_folder)]
        self.annotation_path = [os.path.join(annotation_folder, annot) for annot in os.listdir(annotation_folder)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        annotation_path = self.annotation_path[idx]
        # print(annotation_path)

        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)


        if not self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        with open(annotation_path, 'r') as fp:
            # line = fp.readlines()[0].strip()
            lines = fp.readlines()
            print(lines)

        boxes = []
        labels = []

        if len(lines) == 0:
            lines.append("11 0 0 0 0\n")
        for line in lines:
            print(line)
            values = line.split()
            print(values)
            box = np.array(values[1:], dtype=float)
            label = int(values[0])
            boxes.append(box)
            labels.append(label)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        if not self.rgb:
            img, boxes, labels = self.transform(img, boxes, labels)
        else:
            img, boxes, labels = self.transform(img, boxes, labels, True)

        return img, boxes, labels


def transform(img, boxes, labels, rgb=False):
    if not rgb:
        height, width = img.shape
    else:
        height, width = img.shape[0], img.shape[1]
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    if rgb:
        new_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        new_image[0:new_height, 0:new_width, :] = resized
    else:
        new_image = np.zeros((input_size, input_size), dtype=np.uint8)
        new_image[0:new_height, 0:new_width] = resized

    # x, y, w, h = box[0], box[1], box[2], box[3]
    # new_box = [int((x - 0.5 * w) * width / r), int((y - 0.5 * h) * height / r), int(w * width / r), int(h * height / r)]

    img = ToTensor()(new_image)

    return img, boxes, labels


def collate_fn(batch):
    images = []
    bbox_labels = []
    bbox_masks = []
    max_num_boxes = param.max_num_boxes

    for img, boxes, labels in batch:
        images.append(img)

        num_boxes = len(boxes)
        padded_boxes = torch.zeros((max_num_boxes, 4), dtype=torch.float32) - 1  # Padding value is -1
        padded_labels = torch.zeros(max_num_boxes, dtype=torch.long) - 1  # Padding value is -1

        padded_boxes[:num_boxes, :] = torch.tensor(boxes)
        padded_labels[:num_boxes] = torch.tensor(labels)

        bbox_labels.append(padded_labels)
        bbox_masks.append(torch.tensor([1] * num_boxes + [0] * (max_num_boxes - num_boxes)))

    images = torch.stack(images)
    bbox_labels = torch.stack(bbox_labels)
    bbox_masks = torch.stack(bbox_masks)

    return images, bbox_labels, bbox_masks


