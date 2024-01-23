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
import torchvision
import torchvision.transforms as transforms
# from encoding import *

names = 'spiking_model_custom_data_rgb'
count = 0
data_path = './raw1/'  # ta" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image_folder = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train/images/"
# image_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/train/images"
image_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/train/images"
# image_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/train/images'
# annotations_folder = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/train/labels/"
# annotations_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/train/labels"
annotations_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/train/labels"

# annotations_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/train/labels'
#
# val_image_folder = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/valid/images/"
# val_image_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/valid/images"
val_image_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/valid/images"
# val_image_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/valid/images'
# val_annotations_folder = "E:/Project Work/Datasets/Oxford Pets.v2-by-species.yolov8/valid/labels/"

# val_annotations_folder = "E:/Project Work/Datasets/Self Driving Car.v3-fixed-small.yolov8/valid/labels"
val_annotations_folder = "D:/RiturajMtechProject/Datasets/Self Driving Car.v2-fixed-large.yolov8/export/valid/labels"
# val_annotations_folder = 'D:/RiturajMtechProject/Datasets/Oxford Pets.v2-by-species.yolov8/valid/labels'

custom_dataset = ObjectDetectionDataset(image_folder, annotations_folder, rgb=False, transform=transform)
custom_dataloader = DataLoader(custom_dataset, batch_size=param.batch_size, shuffle=True, num_workers=0)


validation_dataset = ObjectDetectionDataset(val_image_folder, val_annotations_folder,rgb=False,  transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=param.batch_size, shuffle=True, num_workers=0)


print("Total Number of Samples:", len(custom_dataset))
print("Total Number of Samples:", len(validation_dataset))

# for images, targets in custom_dataloader:
#     print(f"Batch Size: {images.size(0)}")
    # print(images)
    # print(targets)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

# snn = SCNN()
# snn = CONVLSTM()
snn = SSD300(n_classes=param.num_classes, device=device)

snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=param.learning_rate)

# ================================== Train ==============================
for real_epoch in range(param.num_epoch):
    print("Epoch: ", real_epoch)
    running_loss = 0
    start_time = time.time()
    for epoch in range(param.sub_epoch):
        print("Sub-epoch: ", epoch)
        for i, (images, labels) in enumerate(custom_dataloader):

            images2 = torch.empty((images.shape[0], param.time_window, images.shape[2], images.shape[3]))
            labels2 = torch.empty((images.shape[0]), dtype=torch.int64)

            for j in range(images.shape[0]):
                img0 = frequency_coding(images[j, 0, :, :])
                print(img0)
                images2[j, :, :, :] = (img0)
                labels2[j] = labels[j]


            snn.zero_grad()
            optimizer.zero_grad()

            images2 = images.float().to(device)
            print("image:", images2)
            conv7, conv8, conv9, conv10, conv11= snn(images2)
            # print("conv 7", conv7)
            # print("Conv 8", conv8)
            # print("Conv 9", conv9)
            # print("Conv 10", conv10)
            # print("Conv 11", conv11)

            # locs, class_score = snn(images2)
            # print("locs: ", locs.shape)
            # print("class_score: ", class_score.shape)

            # print("Output: ",outputs)
            # print("Labels:", labels2)
            # labels_ = torch.zeros(param.batch_size, param.num_classes).scatter_(1, labels2.view(-1, 1), 1)
            # print("Labels: ", labels_)
            # print("labels2:", labels_)
            # loss = criterion(outputs.cpu(), labels_)
            # running_loss += loss.item()

            # print("Loss: ", loss.item())
            # loss.backward()
            # optimizer.step()
            # if (i+1) % 100 == 0:
            #     print('Real_Epoch [%d/%d], Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
            #             %( real_epoch, param.num_epoch, epoch, param.sub_epoch, i+1, len(custom_dataset)//param.batch_size, running_loss))
            #     running_loss = 0
            #     print('Time elasped:', time.time() - start_time)

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