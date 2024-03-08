
import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.spiking_backbone import *
from layers.spiking_backbone_full import *
from layers.spiking_head import *
from utils.prior_boxes import *



class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes, device):
        super(SSD300, self).__init__()

        self.n_classes = n_classes
        self.device = device
        # self.base = VGGBase(self.device)
        self.backbone = VGGBackbone(self.device)
        # self.aux_convs = AuxiliaryConvolutions(self.device)
        self.pred_convs = PredictionConvolutions(n_classes)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        prior_box, prior_box_info = create_prior_boxes(self.device)
        self.priors_cxcy = prior_box
        self.prior_box_info = prior_box_info
        self.to(device)

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions
        # spikes, conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.backbone(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        #
        # print("C4 shape", conv4_3_feats.shape)
        # print("C7 shape", conv7_feats.shape)
        # print("C8 shape", conv8_2_feats.shape)
        # print("C9 shape", conv9_2_feats.shape)
        # print("C10 shape", conv10_2_feats.shape)
        # print("C11 shape", conv11_2_feats.shape)

        # Rescale conv4_3 after L2 norm
        epsilon = 1e-6
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt() + epsilon  # (N, 1, 38, 38)
        # print("norm", norm)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)

        # Run auxiliary convolutions
        # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)
        # conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)

        # Run prediction convolutions
        # (N, 8732, 4), (N, 8732, n_classes)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)

        # print("Location", locs)
        # print("Class Scores", classes_scores)

        return locs, classes_scores

        # return spikes, conv4_3_feats, conv7_feats
        # return conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

        # return conv7_feats, conv8_feats, conv9_feats, conv10_feats, conv11_feats

