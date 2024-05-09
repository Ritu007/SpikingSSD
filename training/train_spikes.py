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

names = 'spiking_model_custom_data_rgb'
count = 0
data_path = './raw1/'  # ta" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# image_folder = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train/images/"
image_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/train/images"
image_folder = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_train_val/VOC2012_train_val/train_val/images'
# image_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/train/images"
# image_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/train/images'
# annotations_folder = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train/labels/"
annotations_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/train/labels"
# annotations_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/train/labels"
annotations_folder = 'E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_train_val/VOC2012_train_val/train_val/labels'
# annotations_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/train/labels'

custom_dataset = ObjectDetectionDataset(image_folder, annotations_folder, rgb=False, transform=transform)
custom_dataloader = DataLoader(custom_dataset, batch_size=param.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=0)


print("Total Number of Samples:", len(custom_dataset))
# print("Total Number of Samples:", len(validation_dataset))

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

# snn = SCNN()
# snn = CONVLSTM()
snn = SSD300(n_classes=param.num_classes, device=device)

snn.to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(snn.parameters(), lr=param.learning_rate)

optimizer = torch.optim.SGD(snn.parameters(), lr=param.learning_rate)
criterion = MultiBoxLoss(param.num_classes, 0.3, True, 0, True, 3, 0.5,
                         False, True)

# ================================== Train ==============================
for real_epoch in range(param.num_epoch):
    print("Epoch: ", real_epoch)
    running_loss = 0
    start_time = time.time()
    for epoch in range(param.sub_epoch):
        print("Sub-epoch: ", epoch)
        for i, (images, labels, masks, boxes) in enumerate(custom_dataloader):

            # print("Labels", labels.shape)
            # print("Boxes", boxes[0][0])

            images2 = torch.empty((images.shape[0], param.time_window, images.shape[2], images.shape[3]))
            labels2 = torch.empty((images.shape[0], param.max_num_boxes), dtype=torch.int64)
            boxes2 = torch.empty((images.shape[0], param.max_num_boxes, 4), dtype=torch.float32)


            for j in range(images.shape[0]):
                # print("image", images[j, 0, :, :])
                img0 = population_encoding(images[j, 0, :, :])
                print("image after coding",img0)
                images2[j, :, :, :] = (img0)
                labels2[j] = labels[j]
                boxes2[j] = boxes[j]
                # print(boxes2[j])
                # print(labels2[j])
            # print(boxes2.shape)
            labels_new = labels2.unsqueeze(-1)

            # Concatenate 'Boxes' and 'Labels' tensors along the last dimension
            boxes_with_labels = torch.cat((boxes2, labels_new), dim=-1)

            # print("filtered", filtered_bbox_labels_tensor)

            boxes_with_labels = boxes_with_labels.to(device)

            # print(boxes_with_labels.shape)
            snn.zero_grad()
            optimizer.zero_grad()

            images2 = images2.float().to(device)
            # print("labels:", labels2.shape)

            # labels_ = torch.zeros(param.batch_size, param.num_classes).scatter_(1, labels2.view(-1, 1), 1)
            # print("Labels: ", labels_.shape)
            locs, class_scores = snn(images2)

            # print("locaton shape", locs.shape)
            # print("class score shape", class_scores.shape)

            prior_boxes, info = create_prior_boxes(device=device)

            outputs = (locs, class_scores, prior_boxes)
            targets = (boxes_with_labels, masks)

            # print("labels2:", labels_)
            loss_l, loss_c = criterion(outputs, boxes_with_labels)

            # print("loss_l", loss_l.item())
            loss = loss_c + loss_l
            running_loss += loss.item()
            #
            print("Loss: ", loss, "conf_loss", loss_c, "loc_loss", loss_l)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print('Real_Epoch [%d/%d], Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %( real_epoch, param.num_epoch, epoch, param.sub_epoch, i+1, len(custom_dataset)//param.batch_size, running_loss))
                running_loss = 0
                print('Time elasped:', time.time() - start_time)

if not os.path.isdir('trained_models'):
    os.mkdir('trained_models')

torch.save(snn.state_dict(), './trained_models/pascal_object_detection_model.pth')

    # # ================================== Test ==============================
    # correct = 0
    # total = 0
    # optimizer = lr_scheduler(optimizer, real_epoch, param.learning_rate, 10)
    # # cm = np.zeros((10, 10), dtype=np.int32)
    #
    # with torch.no_grad():
    #     for batch_idx, (images, labels) in enumerate(validation_dataloader):
    #         images2 = torch.empty((images.shape[0], 10, images.shape[2], images.shape[3]))
    #         labels2 = torch.empty((images.shape[0]), dtype=torch.int64)
    #         for j in range(images.shape[0]):
    #             theta1 = 0
    #             theta2 = 360
    #             img0 = frequency_coding(images[j, 0, :, :])
    #             # print(img0.shape)
    #             images2[j, :, :, :] = img0
    #             labels2[j] = labels[j]
    #         inputs = images2.to(device)
    #         optimizer.zero_grad()
    #         outputs = snn(inputs)
    #         labels_ = torch.zeros(param.batch_size, param.num_classes).scatter_(1, labels2.view(-1, 1), 1)
    #         loss = criterion(outputs.cpu(), labels_)
    #         _, predicted = outputs.cpu().max(1)
    #         # print(predicted.shape)
    #         # ----- showing confussion matrix -----
    #
    #         # cm += confusion_matrix(labels2, predicted)
    #         # ------ showing some of the predictions -----
    #         # for image, label in zip(inputs, predicted):
    #         #     for img0 in image.cpu().numpy():
    #         #         cv2.imshow('image', img0)
    #         #         cv2.waitKey(100)
    #         #     print(label.cpu().numpy())
    #
    #         total += float(labels2.size(0))
    #         correct += float(predicted.eq(labels2).sum().item())
    #         if batch_idx % 100 == 0:
    #             acc = 100. * float(correct) / float(total)
    #             print(batch_idx, len(validation_dataset), ' Acc: %.5f' % acc)
    # class_names = ['0', '1',]
    # # plot_confusion_matrix(cm, class_names)
    # print('Iters:', epoch, '\n\n\n')
    # print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    # acc = 100. * float(correct) / float(total)
    # acc_record.append(acc)
    # if real_epoch % 2 == 0:
    #     count+=1
    #     print(acc)
    #     print('Saving..')
    #     state = {
    #         'net': snn.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #         'acc_record': acc_record,
    #     }
    #     if not os.path.isdir('new_checkpoint'):
    #         os.mkdir('new_checkpoint')
    #     torch.save(state, './new_checkpoint/ckpt' + names + str(count) + '.t7')
    #     best_acc = acc