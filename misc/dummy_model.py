import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.surrogates import *

class Dummy(nn.Module):

    def __init__(self, device):
        super(Dummy, self).__init__()
        self.device = device
        self.conv1_1 = nn.Conv2d(1, 3, kernel_size=2, padding=0)  # stride = 1, by default
        self.conv2_1 = nn.Conv2d(3, 4, kernel_size=2, padding=0)

    def forward(self, image):
        c1_mem = c1_spike = torch.zeros(3, 3, 3, device=self.device)
        c2_mem = c2_spike = torch.zeros(4, 2, 2, device=self.device)

        # out = self.conv1_1(image)
        # print("out1", out.shape)
        # out = self.conv2_1(out)
        # print("out2", out.shape)

        for step in range(param.time_window):
            print("for time: ", step)
            c1_mem, c1_spike = mem_update(self.conv1_1, image, c1_mem, c1_spike)
            c2_mem, c2_spike = mem_update(self.conv2_1, c1_spike, c2_mem, c2_spike)

            print("c1 mem", c1_mem)
            print("c1", c1_spike)
            print("c2 mem", c2_mem)
            print("c2", c2_spike)


