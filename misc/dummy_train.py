import torch
from dummy_model import *
from utils.parameters import *
from data.encoding import *
from utils.prior_boxes import *
from inference.evaluation import *
from utils.box_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image = torch.rand(1,4,4).to(device)
# print("image",image)
#
# encoded_image = frequency_coding(image[0, :, :])
#
# print("encoded image", encoded_image)
#
# model = Dummy(device).to(device)
#
# model(image)
#

# prior_box, info = create_prior_boxes(device)
threshold = 0.5
variances = [0.1, 0.2]
truths = torch.tensor([[0.5, 0.5, 0.7, 0.7], [0.3, 0.3, 0.7, 0.6]])
priors = torch.tensor([[0.1, 0.2, 0.7, 0.7], [0.8, 0.8, 0.9, 0.9], [0.4, 0.1, 0.9, 0.4]])
labels = torch.tensor([1, 2])
# inter = intersect(box_a, box_b)
pred = torch.tensor([[0.2, 0.2, 0.4, 0.1], [0.6, 0.2, 0.1, 0.2], [0.2, 0.1, 0.05, 0.4]])

truths = center_size(truths)
priors = center_size(priors)

print("truth", truths)
print("priors", priors)

# print("inter", inter)
# print("jacc", jaccard(box_a, box_b))

overlaps = jaccard(
    point_form(truths),
    point_form(priors)
)

# print("overlap", overlaps.shape)
# print("overlap", overlaps)
# (Bipartite Matching)
# [1,num_objects] best prior for each ground truth
best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
# [1,num_priors] best ground truth for each prior
best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
best_truth_idx.squeeze_(0)
best_truth_overlap.squeeze_(0)
best_prior_idx.squeeze_(1)
best_prior_overlap.squeeze_(1)
best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
#
# print("best prior", best_prior_idx)
# print("best truth", best_truth_idx)
# TODO refactor: index  best_prior_idx with long tensor
# ensure every gt matches with its prior of max overlap
for j in range(best_prior_idx.size(0)):
    best_truth_idx[best_prior_idx[j]] = j
matches = truths[best_truth_idx]  # Shape: [num_priors,4]
print("matches", matches)
conf = labels[best_truth_idx]  # Shape: [num_priors]
print("Conf", conf)
# v,idx = best_truth_overlap.sort(0)
# print("best truth overlap", v)
conf[best_truth_overlap < threshold] = 0  # label as background
print("conf1", conf)
print("variances", variances)
loc = encode(matches, priors, variances)
print("loc", loc)
# loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
# print("loc_t_box_utils", loc_t)
# conf_t[idx] = conf  # [num_priors] top class label for each prior
# print("conf_t_box_utils", conf_t)

decoded = decode(loc, priors, variances)

print("dcode", decoded)

loss = F.smooth_l1_loss(pred, loc, reduction='none')

print("loss", loss)

