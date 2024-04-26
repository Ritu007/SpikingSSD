
from __future__ import print_function
from data.dataloader import *
import os
import time
import random
# from model import *
import torch
from models.spiking_ssd300 import *
# from models.ssd300 import *
import utils.parameters as param
from data.encoding import *
from loss_function.multibox_loss import MultiBoxLoss
import torchvision
import torchvision.transforms as transforms
from utils.prior_boxes import *
# from encoding import *

import torch.nn.functional as F


names = 'spiking_model_custom_data_rgb'
count = 0
data_path = './raw1/'  # ta" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_image_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/valid/images"
# val_image_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/valid/images"
# val_image_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/valid/images'
# val_annotations_folder = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/valid/labels/"
val_image_folder = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_train_val/VOC2012_train_val/test/images'
val_annotations_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/valid/labels"
# val_annotations_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/valid/labels"
# val_annotations_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/valid/labels'
val_annotations_folder = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_train_val/VOC2012_train_val/test/labels'


validation_dataset = ObjectDetectionDataset(val_image_folder, val_annotations_folder,rgb=False,  transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=param.batch_size,collate_fn=collate_fn, shuffle=True, num_workers=0)


snn = SSD300(n_classes=param.num_classes, device=device)
snn.to(device)

snn.load_state_dict(torch.load('E:/Python/SpikingSSD/training/trained_models/pascal_object_detection_model.pth'))
snn.eval()

# ================================== Test ==============================
correct = 0
total = 0
# optimizer = lr_scheduler(optimizer, real_epoch, param.learning_rate, 10)
# cm = np.zeros((10, 10), dtype=np.int32)

with torch.no_grad():
    for i, (images, labels, masks, boxes) in enumerate(validation_dataloader):

        images2 = torch.empty((images.shape[0], param.time_window, images.shape[2], images.shape[3]))
        labels2 = torch.empty((images.shape[0], param.max_num_boxes), dtype=torch.int64)
        boxes2 = torch.empty((images.shape[0], param.max_num_boxes, 4), dtype=torch.float32)

        for j in range(images.shape[0]):
            img0 = frequency_coding(images[j, 0, :, :])
            # print(img0)
            images2[j, :, :, :] = (img0)
            labels2[j] = labels[j]
            boxes2[j] = boxes[j]

        labels_new = labels2.unsqueeze(-1)
        print("labels", labels_new)
        # Concatenate 'Boxes' and 'Labels' tensors along the last dimension
        boxes_with_labels = torch.cat((boxes2, labels_new), dim=-1)

        boxes_with_labels = boxes_with_labels.to(device)
        images2 = images2.float().to(device)

        locs, class_scores = snn(images2)

        print(locs)
        print("class",class_scores)

        logits = class_scores

        class_probabilities = F.softmax(logits, dim=2)
        # print("class prob", class_probabilities)

        predicted_classes = torch.argmax(class_probabilities, dim=2)
        print("pred_class", predicted_classes)

        print(torch.max(predicted_classes))
        predicted_confidences = torch.max(class_probabilities, dim=2).values
        # print("pred_conf",predicted_confidences)

        prior_boxes, info = create_prior_boxes(device=device)

        outputs = (locs, class_scores, prior_boxes)
        targets = (boxes_with_labels, masks)
