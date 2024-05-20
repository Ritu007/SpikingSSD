import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.surrogates import *
import torchvision
from utils.box_utils import decimate


class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """

    def __init__(self, device):
        super(VGGBase, self).__init__()
        # self.device = device
        # # Standard convolutional layers in VGG16
        # self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        # self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        #
        #
        # self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        #
        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims
        #
        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        #
        # self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.pool5 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)
        #
        # # Replacements for FC6 and FC7 in VGG16
        # self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution
        #
        # self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.device = device
        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
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


        # Load pretrained layers
        self.load_pretrained()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """

        # print("For image", image)
        c1_1mem = c1_1spike = torch.zeros(param.batch_size, 64, 300, 300, device=self.device)
        c1_2mem = c1_2spike = c1sumspike =  torch.zeros(param.batch_size, 64, 300, 300, device=self.device)
        c2_1mem = c2_1spike = torch.zeros(param.batch_size, 128, 150, 150, device=self.device)
        c2_2mem = c2_2spike = torch.zeros(param.batch_size, 128, 150, 150, device=self.device)
        c3_1mem = c3_1spike = torch.zeros(param.batch_size, 256, 75, 75, device=self.device)
        c3_2mem = c3_2spike = torch.zeros(param.batch_size, 256, 75, 75, device=self.device)
        c3_3mem = c3_3spike = torch.zeros(param.batch_size, 256, 75, 75, device=self.device)
        c4_1mem = c4_1spike = torch.zeros(param.batch_size, 512, 38, 38, device=self.device)
        c4_2mem = c4_2spike = torch.zeros(param.batch_size, 512, 38, 38, device=self.device)
        c4_3mem = c4_3spike = c4_sumspike = torch.zeros(param.batch_size, 512, 38, 38, device=self.device)
        c5_1mem = c5_1spike = torch.zeros(param.batch_size, 512, 19, 19, device=self.device)
        c5_2mem = c5_2spike = torch.zeros(param.batch_size, 512, 19, 19, device=self.device)
        c5_3mem = c5_3spike = torch.zeros(param.batch_size, 512, 19, 19, device=self.device)
        c6_mem = c6_spike = torch.zeros(param.batch_size, 1024, 19, 19, device=self.device)
        c7_mem = c7_spike = c7_sumspike = torch.zeros(param.batch_size, 1024, 19, 19, device=self.device)

        # print("c1_mem_init",c1_mem)
        # c4_spike_rec = []
        # c7_spike_rec = []

        for step in range(param.time_window-2):
            # print("time step: ", step)
            # print("time step: ", step)
            new_image = image[:, step:step + 3, :, :]
            # print("new_image", new_image.shape)

            c1_1mem, c1_1spike = mem_update(self.conv1_1, new_image, c1_1mem, c1_1spike, self.inorm_1, True)   # (N, 64, 300, 300)
            c1_2mem, c1_2spike = mem_update(self.conv1_2, c1_1spike, c1_2mem, c1_2spike, self.inorm_1, True)   # (N, 64, 300, 300)
            conv1feats = c1_2spike
            c1sumspike += conv1feats
            out = self.pool1(c1_2spike)  # (N, 64, 150, 150)

            c2_1mem, c2_1spike = mem_update(self.conv2_1, out, c2_1mem, c2_1spike, self.inorm_2, True)  # (N, 128, 150, 150)
            c2_2mem, c2_2spike = mem_update(self.conv2_2, c2_1spike, c2_2mem, c2_2spike, self.inorm_2, True)  # (N, 128, 150, 150)
            out = self.pool2(c2_2spike)  # (N, 128, 75, 75)

            c3_1mem, c3_1spike = mem_update(self.conv3_1, out, c3_1mem, c3_1spike, self.inorm_3, True)  # (N, 256, 75, 75)
            c3_2mem, c3_2spike = mem_update(self.conv3_2, c3_1spike, c3_2mem, c3_2spike, self.inorm_3, True)  # (N, 256, 75, 75)
            c3_3mem, c3_3spike = mem_update(self.conv3_3, c3_2spike, c3_3mem, c3_3spike, self.inorm_3, True)  # (N, 256, 75, 75)
            out = self.pool3(c3_3spike)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

            c4_1mem, c4_1spike = mem_update(self.conv4_1, out, c4_1mem, c4_1spike, self.inorm_4, True)  # (N, 512, 38, 38)
            c4_2mem, c4_2spike = mem_update(self.conv4_2, c4_1spike, c4_2mem, c4_2spike, self.inorm_4, True)  # (N, 512, 38, 38)
            c4_3mem, c4_3spike = mem_update(self.conv4_3, c4_2spike, c4_3mem, c4_3spike, self.inorm_4, True)  # (N, 512, 38, 38)

            conv4_3_feats = c4_3spike  # (N, 512, 38, 38)
            c4_sumspike += conv4_3_feats
            # print("c4sumspike", c4_sumspike)
            out = self.pool4(c4_3spike)  # (N, 512, 19, 19)

            c5_1mem, c5_1spike = mem_update(self.conv5_1, out, c5_1mem, c5_1spike, self.inorm_5, True)  # (N, 512, 19, 19)
            c5_2mem, c5_2spike = mem_update(self.conv5_2, c5_1spike, c5_2mem, c5_2spike, self.inorm_5, True)  # (N, 512, 19, 19)
            c5_3mem, c5_3spike = mem_update(self.conv5_3, c5_2spike, c5_3mem, c5_3spike, self.inorm_5, True)  # (N, 512, 19, 19)
            out = self.pool5(c5_3spike)  # (N, 512, 19, 19), pool5 does not reduce dimensions

            c6_mem, c6_spike = mem_update(self.conv6, out, c6_mem, c6_spike, self.inorm_6, True) # (N, 1024, 19, 19)

            c7_mem, c7_spike = mem_update(self.conv7, c6_spike, c7_mem, c7_spike, self.inorm_7, True) # (N, 1024, 19, 19)
            conv7_feats = c7_spike
            c7_sumspike += conv7_feats

        # Lower-level feature maps
        conv1feats = c1sumspike/ param.time_window
        # resized_output = torch.nn.functional.interpolate(conv1feats, size=(38, 38), mode='bilinear',
        #                                                  align_corners=False)
        # print("res", resized_output.shape)
        # conv1feats = self.feat_conv_layer(resized_output)
        # print("con1", conv1feats.shape)
        conv4_3_feats = c4_sumspike / param.time_window
        conv7_feats = c7_sumspike / param.time_window
        return conv1feats, conv4_3_feats, conv7_feats

    def load_pretrained(self):
        '''
            Use a VGG-16 pretrained on the ImageNet task for conv1-->conv5
            Convert conv6, conv7 to pretrained
        '''
        print("Loading pretrained base model...")
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, parameters in enumerate(param_names[:26]):
            state_dict[parameters] = pretrained_state_dict[pretrained_param_names[i]]

        # convert fc6, fc7 in pretrained to conv6, conv7 in model
        fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = decimate(fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(fc6_bias, m=[4])  # (1024)

        fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = decimate(fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(fc7_bias, m=[4])

        self.load_state_dict(state_dict)
        print("Loaded base model")

class spikingAuxiliaryConvolutions(nn.Module):
    def __init__(self, device):
        super(AuxiliaryConvolutions, self).__init__()
        #input (N, 1024, 19, 19) that is conv7_feats
        # Auxiliary/additional convolutions on top of the VGG base
        self.device = device
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters using xavier initialization.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps (N, 512, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3) and (N, 256, 1, 1)
        """
        a1_mem = a1_spike = a1_sumspike = torch.zeros(param.batch_size, 512, 10, 10, device=self.device)
        a2_mem = a2_spike = a2_sumspike = torch.zeros(param.batch_size, 256, 5, 5, device=self.device)
        a3_mem = a3_spike = a3_sumspike = torch.zeros(param.batch_size, 256, 3, 3, device=self.device)
        a4_mem = a4_spike = a4_sumspike = torch.zeros(param.batch_size, 256, 1, 1, device=self.device)

        for step in range(param.time_window):
            conv7_feats = conv7_feats[:, step:step+1, :, :]

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

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

class AuxiliaryConvolutions(nn.Module):
    def __init__(self, device):
        super(AuxiliaryConvolutions, self).__init__()
        #input (N, 1024, 19, 19) that is conv7_feats
        # Auxiliary/additional convolutions on top of the VGG base
        self.feat_conv_layer = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters using xavier initialization.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
    def forward(self, conv1, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps (N, 512, 10, 10), (N, 256, 5, 5), (N, 256, 3, 3) and (N, 256, 1, 1)
        """
        resized_output = torch.nn.functional.interpolate(conv1, size=(38, 38), mode='bilinear',
                                                         align_corners=False)
        # print("res", resized_output.shape)
        conv1feats = self.feat_conv_layer(resized_output)
        # print("con1", conv1feats.shape)

        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        # Higher-level feature maps
        return conv1feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats