import numpy as np
from numpy import interp
from matplotlib import pyplot as plt
import imageio
import math
# from parameters import param as par
# from recep_field import rf
from utils.parameters import *
import torch

# timeinterval = 10

def encode2(pixels):

    #initializing spike train
    train = []

    for l in range(pixels.shape[0]):
        for m in range(pixels.shape[1]):

            temp = np.zeros([(time_window + 1),])

            #calculating firing rate proportional to the membrane potential
            freq = interp(pixels[l][m], [0, 255], [1,20])
            #print(pot[l][m], freq)
            # print freq

            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            if(pixels[l][m]>0):
                while k<(time_window+1):
                    temp[k] = 1
                    k = k + freq1
            train.append(temp)
            # print sum(temp)
    return train

def encode(pot):

    #initializing spike train
    train = []

    for l in range(pot.shape[0]):
        for m in range(pot.shape[1]):

            temp = np.zeros([(time_window+1),])

            #calculating firing rate proportional to the membrane potential
            freq = interp(pot[l][m], [-1.069,2.781], [1,20])
            #print(pot[l][m], freq)
            # print freq
            # print(freq)
            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            # print(freq1)
            if(pot[l][m]>0):
                while k<(time_window+1):
                    temp[int(k)] = 1
                    k = k + freq1
            train.append(temp)
            # print(temp)
    return train


def frequency_coding(images):
    rate = 1 - images
    images2 = torch.empty((time_window, images.shape[0], images.shape[1]))
    for k in range(time_window):
        images2[k, :, :] = (rate > ((k+1) / time_window)).float()
    return images2

    # for row in range(rate.shape[0]):
    #     for col in range(rate.shape[1]):
    #         spikes = np.zeros((time_window))
    #         time = rate[row][col] * (time_window)
    #         time = int(time)
    #         counter = 1
    #         new_time = time * counter
    #         while new_time < time_window:
    #
    #             spikes[new_time] = 1
    #             counter += 1
    #             new_time = new_time * counter
    #         spike_train.append(spikes.tolist())
    #
    # return spike_train


def time_to_first_coding(images):
    rate = 1 - images
    images2 = torch.empty((time_window, images.shape[0], images.shape[1]))
    for k in range(time_window):
        images2[k, :, :] = ((rate * time_window).round().to(torch.int) == k).float()
    return images2
