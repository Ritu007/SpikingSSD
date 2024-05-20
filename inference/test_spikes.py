
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
from inference.evaluation import *

import torch.nn.functional as F


names = 'spiking_model_custom_data_rgb'
count = 0
data_path = './raw1/'  # ta" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

val_image_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/valid/images"
# val_image_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/valid/images"
# val_image_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/valid/images'
# val_annotations_folder = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/valid/labels/"
val_image_folder = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_train_val/VOC2012_train_val/test/images'
val_annotations_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/valid/labels"
# val_annotations_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/valid/labels"
# val_annotations_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/valid/labels'
val_annotations_folder = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_train_val/VOC2012_train_val/test/labels'


validation_dataset = ObjectDetectionDataset(val_image_folder, val_annotations_folder, rgb=False,  transform=transform)

if param.batch_size > 1:
    validation_dataloader = DataLoader(validation_dataset, batch_size=param.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=0)
else:
    validation_dataloader = DataLoader(validation_dataset, batch_size=param.batch_size, shuffle=True, num_workers=0)


snn = SSD300(n_classes=param.num_classes, device=device)
snn.to(device)

snn.load_state_dict(torch.load('E:/Python/SpikingSSD/training/trained_models/alt_pascal_object_detection_model.pth'))
snn.eval()

# ================================== Test ==============================
correct = 0
total = 0
# optimizer = lr_scheduler(optimizer, real_epoch, param.learning_rate, 10)
# cm = np.zeros((10, 10), dtype=np.int32)

with torch.no_grad():
    for i, (images, boxes, labels, image_path) in enumerate(validation_dataloader):

        images2 = torch.empty((images.shape[0], param.time_window, images.shape[2], images.shape[3]))
        labels2 = torch.empty((images.shape[0], param.max_num_boxes), dtype=torch.int64)
        boxes2 = torch.empty((images.shape[0], param.max_num_boxes, 4), dtype=torch.float32)
        decoded_boxes = torch.empty((images.shape[0], 8732, 4), dtype=torch.float32, device=device)
        print("images",images.shape)
        # for j in range(images.shape[0]):
        img0 = images[0, 0, :, :]

        print(img0.shape)
        image_np = np.array(img0)
        cv2.imshow("Image", image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        img0 = frequency_coding(images[0, 0, :, :])
            # print(img0)
        images2[:, :, :] = (img0)
        labels2 = labels
        boxes2 = boxes

        print(labels.shape)
        print(boxes.shape)

        labels_new = labels2.unsqueeze(-1)
        # print("labels", labels_new)
        # Concatenate 'Boxes' and 'Labels' tensors along the last dimension
        boxes_with_labels = torch.cat((boxes2, labels_new), dim=-1)

        boxes_with_labels = boxes_with_labels.to(device)
        images2 = images2.float().to(device)

        locs, class_scores, c1, c4, c7, c8, c9, c10, c11 = snn(images2)

        print("c1",c1.shape)

        input_tensor_reshaped = c1.unsqueeze(2)

        resized_tensor = F.interpolate(c1, size=(512, 38, 38), mode='bilinear', align_corners=False)

        print("c4", resized_tensor.shape)

        selected_feature_maps = resized_tensor[0, :32, :, :]  # Assuming batch size is 1
        num_rows = 4
        num_cols = 8
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
        axes = axes.flatten()
        # Plot each feature map
        for i in range(len(axes)):
            ax = axes[i]
            ax.imshow(selected_feature_maps[i].detach().cpu().numpy(), cmap='gray')
            ax.axis('off')

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

        # print("locs", locs.shape)
        class_probabilities = F.softmax(class_scores, dim=2)
        # print("class prob", class_probabilities)

        predicted_classes = torch.argmax(class_probabilities, dim=2)
        print("pred_class", predicted_classes)


        prior_box, info = create_prior_boxes(device)

        for j in range(locs.shape[0]):

            decoded_boxes[j,:,:] = decoding_boxes(locs[j,:,:], prior_box)

            keep, count = nms(decoded_boxes[j,:,:], predicted_classes[j,:])
            final_boxes = decoded_boxes[j,:,:][keep[:count]]
            final_scores = class_scores[j,:,:][keep[:count]]
        print("keep", keep)
        print("count", count)

        # final_boxes = decoded_boxes[keep[:count]]

        # print("final", final_boxes.shape)
        # print("final", final_scores)



        print("decoded boxes", decoded_boxes.shape)
        # decoded_boxes.to(device)


        # print(decoded_boxes.device)

        #Filtering process
        filtered_boxes, filtered_scores= filter_by_confidence(final_scores, final_boxes, param.num_classes)
        print("filtered_boxes", filtered_boxes)
        print("filtered_scores", filtered_scores)
        # print("filtered_classes", filtered_classes.shape)

        # print(locs)
        # print("class",class_scores)
        #
        # logits = class_scores
        #

        #
        # print(torch.max(predicted_classes))
        # predicted_confidences = torch.max(class_probabilities, dim=2).values
        # # print("pred_conf",predicted_confidences)
        #
        # prior_boxes, info = create_prior_boxes(device=device)
        #
        # outputs = (locs, class_scores, prior_boxes)
        # targets = (boxes_with_labels, masks)
