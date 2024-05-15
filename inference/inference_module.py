
from __future__ import print_function

import cv2

from data.dataloader import *
import os
import time
import random
import torch
from models.spiking_ssd300 import *
import utils.parameters as param
from data.encoding import *
from inference.evaluation import *
import torchvision.transforms as transforms
from data.config import *

names = 'spiking_model_custom_data_rgb'
count = 0
data_path = './raw1/'  # ta" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

snn = SSD300(n_classes=param.num_classes, device=device)
snn.to(device)

snn.load_state_dict(torch.load('E:/Python/SpikingSSD/training/trained_models/alt_pascal_object_detection_model.pth'))
snn.eval()

image_path = "E:/Project Work/Datasets/pascalvoc2012/archive/VOC2012_test/VOC2012_test/images"

def load_images_from_folder(folder):
    images = []
    image_path = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            image_path.append(folder + "/" + filename)
    return images, image_path



images, image_path = load_images_from_folder(image_path)
print(image_path)
to_tensor = transforms.ToTensor()  # Scales pixel values to [0, 1]

def get_encoded_image(image):
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(grayscale_img, (param.image_size, param.image_size), interpolation=cv2.INTER_LINEAR)
    # Convert the image to a PyTorch tensor
    image_tensor = to_tensor(resized_img)
    # image_tensor = image_tensor.unsqueeze(0)
    print(image_tensor.shape)
    encoded_img = frequency_coding(image_tensor[0])
    print(encoded_img.shape)
    return encoded_img


def get_model_outputs_new(model, input):
    input = input.to(device)
    locs, scores, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = model(input)
    prior_box, info = create_prior_boxes(device)
    decoded_boxes = torch.empty((param.batch_size, 8732, 4), dtype=torch.float32, device=device)
    # print("locs", locs.shape)
    # class_probabilities = F.softmax(scores, dim=2)
    # print("class prob", class_probabilities)




    # prior_box, info = create_prior_boxes(device)

    for j in range(locs.shape[0]):
        # decoded_boxes[j, :, :] = decoding_boxes(locs[j, :, :], prior_box)
        print("locs", locs[j, :, :])
        decoded_boxes[j, :, :] = decode(locs[j, :, :], prior_box, variances=voc['variance'])
        print("decode", decoded_boxes)

        filtered_boxes, filtered_scores = filter_by_confidence(scores[j, :, :], decoded_boxes[j, :, :], param.num_classes, confidence_threshold=0.3)
        print("filtered_boxes", filtered_boxes)
        print("filtered_scores", filtered_scores)

        predicted_classes = torch.argmax(filtered_scores, dim=1).unsqueeze(0)
        print("pred_class", predicted_classes)
        keep, count = nms(filtered_boxes, predicted_classes[j, :], 0.3)
        final_boxes = filtered_boxes[keep[:count]]
        final_scores = filtered_scores[keep[:count]]
        print("keep", keep)
        print("count", count)
    # filtered_boxes, filtered_scores = filter_by_confidence(final_scores, final_boxes, param.num_classes)

    return final_boxes, final_scores


def get_model_outputs(model, input):
    input = input.to(device)
    locs, scores, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = model(input)
    prior_box, info = create_prior_boxes(device)
    decoded_boxes = torch.empty((param.batch_size, 8732, 4), dtype=torch.float32, device=device)
    # print("locs", locs.shape)
    class_probabilities = F.softmax(scores, dim=2)
    # print("class prob", class_probabilities)

    predicted_classes = torch.argmax(class_probabilities[:, : , 1:], dim=2)
    print("pred_class", predicted_classes)

    # prior_box, info = create_prior_boxes(device)

    for j in range(locs.shape[0]):
        decoded_boxes[j, :, :] = decode(locs[j, :, :], prior_box, variances=voc['variance'])

        keep, count = nms(decoded_boxes[j, :, :], predicted_classes[j, :], 0.1)
        final_boxes = decoded_boxes[j, :, :][keep[:count]]
        final_scores = scores[j, :, :][keep[:count]]
        print("keep", keep)
        print("count", count)
        filtered_boxes, filtered_scores = filter_by_confidence(final_scores, final_boxes, param.num_classes, confidence_threshold=0.3)
        print("filtered_boxes", filtered_boxes)
        print("filtered_scores", filtered_scores.shape)
    return final_boxes, final_scores


with torch.no_grad():
    for image in images:
        encoded_image = get_encoded_image(image)
        batched_image = encoded_image.unsqueeze(0)
        print("bat",batched_image.shape)
        locs, scores = get_model_outputs(snn, batched_image)

        print("locs", locs)
        print("scores", scores)
        get_final_predictions(image, locs, scores)
        time.sleep(2)