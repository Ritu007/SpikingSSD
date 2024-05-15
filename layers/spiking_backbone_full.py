import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.surrogates import *


class VGGBackbone(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self, device):
        super(VGGBackbone, self).__init__()
        self.device = device
        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.inorm_1 = nn.InstanceNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.inorm_2 = nn.InstanceNorm2d(128)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.inorm_3 = nn.InstanceNorm2d(256)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.inorm_4 = nn.InstanceNorm2d(512)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.inorm_5 = nn.InstanceNorm2d(512)
        self.pool5 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution
        self.inorm_6 = nn.InstanceNorm2d(1024)


        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.inorm_7 = nn.InstanceNorm2d(1024)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
        self.inorm_8 = nn.InstanceNorm2d(512)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
        self.inorm_9 = nn.InstanceNorm2d(256)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0
        self.inorm_10 = nn.InstanceNorm2d(256)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0
        # self.inorm_11 = nn.InstanceNorm2d(256)

        self.init_conv2d()
        # Load pretrained layers
        # self.load_pretrained_layers()

    def init_conv2d(self):
        """
        Initialize convolution parameters using xavier initialization.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """

        # print("For image", image)
        c1_mem = c1_spike = torch.zeros(param.batch_size, 64, 300, 300, device=self.device)
        c2_mem = c2_spike = torch.zeros(param.batch_size, 128, 150, 150, device=self.device)
        c3_mem = c3_spike = torch.zeros(param.batch_size, 256, 75, 75, device=self.device)
        c4_mem = c4_spike = c4_sumspike = torch.zeros(param.batch_size, 512, 38, 38, device=self.device)
        c5_mem = c5_spike = torch.zeros(param.batch_size, 512, 19, 19, device=self.device)
        c6_mem = c6_spike = torch.zeros(param.batch_size, 1024, 19, 19, device=self.device)
        c7_mem = c7_spike = c7_sumspike = torch.zeros(param.batch_size, 1024, 19, 19, device=self.device)

        a1_mem = a1_spike = a1_sumspike = torch.zeros(param.batch_size, 512, 10, 10, device=self.device)
        a2_mem = a2_spike = a2_sumspike = torch.zeros(param.batch_size, 256, 5, 5, device=self.device)
        a3_mem = a3_spike = a3_sumspike = torch.zeros(param.batch_size, 256, 3, 3, device=self.device)
        a4_mem = a4_spike = a4_sumspike = torch.zeros(param.batch_size, 256, 1, 1, device=self.device)

        # print("c1_mem_init",c1_mem)
        # c4_spike_rec = []
        # c7_spike_rec = []

        for step in range(param.time_window):
            # print("time step: ", step)
            new_image = image[:, step:step+1, :, :]

            # print("image", new_image)
            out = F.relu(self.conv1_1(new_image))  # (N, 64, 300, 300)
            # print("conv 1 1 before norm", out)
            out = self.inorm_1(out)
            # out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
            # print("conv 1 1 after norm", out)
            c1_mem, c1_spike = mem_update(self.conv1_2, out, c1_mem, c1_spike)
            # print(c1_spike.shape)
            # print("c1 mem", c1_mem)
            # print("c1 spike before norm", c1_spike)
            #
            # print("c1_spike_after norm", c1_spike)
            out = self.pool1(c1_spike)  # (N, 64, 150, 150)

            spikes_1 = out
            # print("Spikes 1:", spikes_1)

            out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
            # out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
            c2_mem, c2_spike = mem_update(self.conv2_2, out, c2_mem, c2_spike)
            c2_spike = self.inorm_2(c2_spike)
            out = self.pool2(c2_spike)  # (N, 128, 75, 75)

            spikes_2 = out
            # print("Spikes 2:", spikes_2)

            out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
            out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
            # out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
            c3_mem, c3_spike = mem_update(self.conv3_3, out, c3_mem, c3_spike)
            c3_spike = self.inorm_3(c3_spike)
            out = self.pool3(c3_spike)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

            spikes_3 = out
            # print("Spikes 3:", spikes_3)

            out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
            out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
            # out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)

            c4_mem, c4_spike = mem_update(self.conv4_3, out, c4_mem, c4_spike)
            c4_spike = self.inorm_4(c4_spike)
            conv4_3_feats = c4_spike  # (N, 512, 38, 38)
            c4_sumspike += conv4_3_feats
            print("c4sumspike", c4_sumspike)
            # c4_spike_rec.append(conv4_3_feats)
            out = self.pool4(c4_spike)  # (N, 512, 19, 19)

            spikes_4 = out
            # print("Spikes 4:", spikes_4)

            out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
            out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
            # out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
            c5_mem, c5_spike = mem_update(self.conv5_3, out, c5_mem, c5_spike)
            c5_spike = self.inorm_5(c5_spike)
            out = self.pool5(c5_spike)  # (N, 512, 19, 19), pool5 does not reduce dimensions

            spikes_5 = out
            # print("Spikes 5:", spikes_5)

            # out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)
            c6_mem, c6_spike = mem_update(self.conv6, out, c6_mem, c6_spike)
            c6_spike = self.inorm_6(c6_spike)

            spikes_6 = c6_spike
            # print("Spikes 6:", spikes_6)

            # conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)
            c7_mem, c7_spike = mem_update(self.conv7, spikes_6, c7_mem, c7_spike)
            c7_spike = self.inorm_7(c7_spike)
            spikes_7 = c7_spike
            conv7_feats = c7_spike
            c7_sumspike += conv7_feats
            # c7_spike_rec.append(conv7_feats)

            out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
            # out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
            a1_mem, a1_spike = mem_update(self.conv8_2, out, a1_mem, a1_spike)
            a1_spike = self.inorm_8(a1_spike)
            conv8_2_feats = a1_spike  # (N, 512, 10, 10)
            a1_sumspike += conv8_2_feats

            out = F.relu(self.conv9_1(conv8_2_feats))  # (N, 128, 10, 10)
            # out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
            a2_mem, a2_spike = mem_update(self.conv9_2, out, a2_mem, a2_spike)
            a2_spike = self.inorm_9(a2_spike)
            conv9_2_feats = a2_spike  # (N, 256, 5, 5)
            a2_sumspike += conv9_2_feats

            out = F.relu(self.conv10_1(conv9_2_feats))  # (N, 128, 5, 5)
            # out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
            a3_mem, a3_spike = mem_update(self.conv10_2, out, a3_mem, a3_spike)
            a3_spike = self.inorm_10(a3_spike)
            conv10_2_feats = a3_spike  # (N, 256, 3, 3)
            a3_sumspike += conv10_2_feats

            out = F.relu(self.conv11_1(conv10_2_feats))  # (N, 128, 3, 3)
            a4_mem, a4_spike = mem_update(self.conv11_2, out, a4_mem, a4_spike)
            # a4_spike = self.inorm_11(a4_spike)
            conv11_2_feats = a4_spike  # (N, 256, 1, 1)
            a4_sumspike += conv11_2_feats

            # print("conv 11 features:", conv11_2_feats)
            # print("Spikes 7:", spikes_7)
        # Lower-level feature maps
        conv4_3_feats = c4_sumspike/ param.time_window
        conv7_feats = c7_sumspike / param.time_window
        conv8_2_feats = a1_sumspike / param.time_window
        conv9_2_feats = a2_sumspike / param.time_window
        conv10_2_feats = a3_sumspike / param.time_window
        conv11_2_feats = a4_sumspike / param.time_window

        # print("Conv 4", conv4_3_feats)
        # print("Conv 11", conv11_2_feats)

        return conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats,conv10_2_feats,conv11_2_feats

# class AuxiliaryConvolutions(nn.Module):
#     def __init__(self, device):
#         super(AuxiliaryConvolutions, self).__init__()
#         #input (N, 1024, 19, 19) that is conv7_feats
#         # Auxiliary/additional convolutions on top of the VGG base
#         self.device = device
#         self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
#         self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
#
#         self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
#         self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1
#
#         self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
#         self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0
#
#         self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
#         self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0
#
#         # Initialize convolutions' parameters
#         self.init_conv2d()
#
#     def init_conv2d(self):
#         """
#         Initialize convolution parameters using xavier initialization.
#         """
#         for c in self.children():
#             if isinstance(c, nn.Conv2d):
#                 nn.init.xavier_uniform_(c.weight)
#                 nn.init.constant_(c.bias, 0.)
#     def forward(self, conv7_feats):
#         """
#         Forward propagation.
#
#         :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
#         :return: higher-level feature maps (N, 512, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3) and (N, 256, 1, 1)
#         """
#         a1_mem = a1_spike = torch.zeros(param.batch_size, 512, 10, 10, device=self.device)
#         a2_mem = a2_spike = torch.zeros(param.batch_size, 256, 5, 5, device=self.device)
#         a3_mem = a3_spike = torch.zeros(param.batch_size, 256, 3, 3, device=self.device)
#         a4_mem = a4_spike = torch.zeros(param.batch_size, 256, 1, 1, device=self.device)
#
#         for step in range(param.time_window):
#             conv7_feats = conv7_feats[:, step:step+1, :, :]
#             conv7_feats = conv7_feats.squeeze(dim=1)
#             out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
#             # out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
#             a1_mem, a1_spike = mem_update(self.conv8_2, out, a1_mem, a1_spike)
#             conv8_2_feats = a1_spike  # (N, 512, 10, 10)
#
#             out = F.relu(self.conv9_1(conv8_2_feats))  # (N, 128, 10, 10)
#             # out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
#             a2_mem, a2_spike = mem_update(self.conv9_2, out, a2_mem, a2_spike)
#             conv9_2_feats = a2_spike  # (N, 256, 5, 5)
#
#             out = F.relu(self.conv10_1(conv9_2_feats))  # (N, 128, 5, 5)
#             # out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
#             a3_mem, a3_spike = mem_update(self.conv10_2, out, a3_mem, a3_spike)
#             conv10_2_feats = a3_spike  # (N, 256, 3, 3)
#
#             out = F.relu(self.conv11_1(conv10_2_feats))  # (N, 128, 3, 3)
#             a4_mem, a4_spike = mem_update(self.conv11_2, out, a4_mem, a4_spike)
#             conv11_2_feats = a4_spike  # (N, 256, 1, 1)
#
#             print("conv 11 features:", conv11_2_feats)
#
#         # Higher-level feature maps
#         return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats