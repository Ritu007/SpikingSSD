import numpy as np
from numpy import interp
from matplotlib import pyplot as plt
# import imageio
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


def time_to_first_coding(images):
    rate = 1 - images
    images2 = torch.empty((time_window, images.shape[0], images.shape[1]))
    for k in range(time_window):
        images2[k, :, :] = ((rate * time_window).round().to(torch.int) == k).float()
    return images2

# Define Gaussian function
def gaussian(x, mu, sigma):
    # print("mu", mu)
    # print("image", x)
    # diff = x-mu
    # numerator = diff ** 2
    # denominator = (2 * sigma ** 2)
    # out = (numerator / denominator)
    # print("diff", diff)
    # print("numer", numerator)
    # print("denom",denominator)
    # print("out", out)
    # print("ret", torch.exp(-out))
    return torch.exp(-((x-mu)**2)/(2*sigma**2))



def population_encoding(image):
    # I_min = np.min(image)
    # I_max = np.max(image)

    I_min = image.min().item()
    I_max = image.max().item()

    print("min", I_min)
    print("max", I_max)

    beta = 1.5  # Determines the spread of the Gaussians
    M = time_window  # Number of Gaussians for population encoding
    T_ref = 1  # Refractory period
    dt = 1  # Time step

    # Standard deviation for Gaussian functions
    sigma = (1 / beta) * ((I_max - I_min) / (M - 2))

    print("sigma", sigma)

    mu = torch.tensor([I_min + ((2 * i - 3) / 2) * ((I_max - I_min) / (M - 2)) for i in range(M)])

    # print("mu", mu)
    #
    # print("image", image)

    # Compute spike times using vectorized operations
    # Expand dimensions to allow for broadcasting
    image_exp = image.unsqueeze(-1)  # Add a dimension for broadcasting with `mu`
    mu_exp = mu.unsqueeze(0).unsqueeze(0)  # Add dimensions for broadcasting with `image_exp`

    # print("diff",image_exp-mu_exp)

    # print("imageshape", image_exp.shape)
    # print("mushape", mu_exp.shape)
    # Compute Gaussian responses for each pixel and each Gaussian
    responses = gaussian(image_exp, mu_exp, sigma)

    print("response gauss", responses.permute(2, 0, 1))

    # Compute spike times using the refractory period and Gaussian responses
    spike_trains = T_ref * (1 - responses)  # This is now fully vectorized

    # Validate the shape of spike trains (should be 300x300x10)
    # print("Spike Trains Shape:", spike_trains.shape)

    spike_trains_reshaped = spike_trains.permute(2, 0, 1)


    # spike_trains = torch.zeros((image.size(0), image.size(1), M))
    #
    # # Loop through the image and compute spike times for each pixel
    # for row in range(image.shape[0]):
    #     for col in range(image.shape[1]):
    #         pixel_value = image[row, col]
    #         spike_responses = torch.zeros(M)
    #
    #         # Compute spike times for each Gaussian
    #         for k in range(M):
    #             response = gaussian(pixel_value, mu[k], sigma)
    #             adjusted_response = T_ref * (1 - response)  # Adjust with refractory time
    #             spike_responses[k] = adjusted_response
    #
    #         # print("response", response, row, col)
    #
    #         spike_trains[row, col] = spike_responses

    return spike_trains_reshaped

