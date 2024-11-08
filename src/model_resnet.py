import importlib, pdb
from math import sqrt
from itertools import product as product
from copy import deepcopy

from torch import nn
import torch.nn.functional as F
import torchvision

from utils.utils import *

args = importlib.import_module('config').args
device = args.device


class ResNetBase(nn.Module):
    """
    ResNet base convolutions to produce lower-level feature maps.
    """
    def __init__(self, three_way=False):
        super(ResNetBase, self).__init__()
        self.three_way = three_way

        # Load the pretrained ResNet models
        resnet_vis = torchvision.models.resnet50(pretrained=True)
        resnet_lwir = torchvision.models.resnet50(pretrained=True)
        
        # RGB path
        self.rgb_conv1 = nn.Sequential(
            resnet_vis.conv1,
            resnet_vis.bn1,
            resnet_vis.relu,
            resnet_vis.maxpool
        )
        self.rgb_layer1 = resnet_vis.layer1
        self.rgb_layer2 = resnet_vis.layer2
        self.rgb_layer3 = resnet_vis.layer3
        self.rgb_layer4 = resnet_vis.layer4
        
        # Thermal path
        self.lwir_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet_lwir.bn1,
            resnet_lwir.relu,
            resnet_lwir.maxpool
        )
        self.lwir_layer1 = resnet_lwir.layer1
        self.lwir_layer2 = resnet_lwir.layer2
        self.lwir_layer3 = resnet_lwir.layer3
        self.lwir_layer4 = resnet_lwir.layer4

        # Convolution layers to reduce channels to 512
        self.conv0 = nn.Conv2d(64, 512, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(512, affine=True)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv1 = nn.Conv2d(256, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512, affine=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(512, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(512, affine=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(2048, 512, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(512, affine=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)


        # Feature fusion layers
        self.feat_1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_1_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_2_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_3 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_3_bn = nn.BatchNorm2d(512, momentum=0.01)
        self.feat_4 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_4_bn = nn.BatchNorm2d(512, momentum=0.01)

        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

    def forward(self, image_vis, image_lwir):
        # RGB path
        out_vis0 = self.rgb_conv1(image_vis)
        out_vis1 = self.rgb_layer1(out_vis0)
        out_vis2 = self.rgb_layer2(out_vis1)
        out_vis3 = self.rgb_layer3(out_vis2)
        out_vis4 = self.rgb_layer4(out_vis3)

        out_vis0 = self.pool0(self.bn0(self.conv0(out_vis0)))
        out_vis1 = self.pool1(self.bn1(self.conv1(out_vis1)))
        out_vis2 = self.pool2(self.bn2(self.conv2(out_vis2)))
        out_vis3 = self.pool3(self.bn3(self.conv3(out_vis3)))
        out_vis4 = self.pool4(self.bn4(self.conv4(out_vis4)))
        
        # Thermal path
        out_lwir0 = self.lwir_conv1(image_lwir)
        out_lwir1 = self.lwir_layer1(out_lwir0)
        out_lwir2 = self.lwir_layer2(out_lwir1)
        out_lwir3 = self.lwir_layer3(out_lwir2)
        out_lwir4 = self.lwir_layer4(out_lwir3)

        out_lwir0 = self.pool0(self.bn0(self.conv0(out_lwir0)))
        out_lwir1 = self.pool1(self.bn1(self.conv1(out_lwir1)))
        out_lwir2 = self.pool2(self.bn2(self.conv2(out_lwir2)))
        out_lwir3 = self.pool3(self.bn3(self.conv3(out_lwir3)))
        out_lwir4 = self.pool4(self.bn4(self.conv4(out_lwir4)))

        # Feature fusion at different stages
        conv1_feats = torch.cat([out_vis1, out_lwir1], dim=1)
        conv1_feats = F.relu(self.feat_1_bn(self.feat_1(conv1_feats)))

        conv2_feats = torch.cat([out_vis2, out_lwir2], dim=1)
        conv2_feats = F.relu(self.feat_2_bn(self.feat_2(conv2_feats)))

        conv3_feats = torch.cat([out_vis3, out_lwir3], dim=1)
        conv3_feats = F.relu(self.feat_3_bn(self.feat_3(conv3_feats)))

        conv4_feats = torch.cat([out_vis4, out_lwir4], dim=1)
        conv4_feats = F.relu(self.feat_4_bn(self.feat_4(conv4_feats)))

        if self.three_way:
            # Return all features
            return conv1_feats, conv2_feats, conv3_feats, conv4_feats, \
                   out_vis1,    out_vis2,    out_vis3,    out_vis4,    \
                   out_lwir1,   out_lwir2,   out_lwir3,   out_lwir4        
        else:
            # Return only fused features
            return conv1_feats, conv2_feats, conv3_feats, conv4_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        n_boxes = {'conv4_3': 6,
                    'conv6': 6,
                    'conv7': 6,
                    'conv8': 6,}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv6 = nn.Conv2d(512, n_boxes['conv6'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(512, n_boxes['conv6'] * 4, kernel_size=3, padding=1)
        self.loc_conv8 = nn.Conv2d(512, n_boxes['conv7'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv6 = nn.Conv2d(512, n_boxes['conv6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(512, n_boxes['conv6'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8 = nn.Conv2d(512, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)


    def forward(self, conv4_3_feats, conv6_feats, conv7_feats, conv8_feats):

        batch_size = conv4_3_feats.size(0)

        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)

        l_conv6 = self.loc_conv6(conv6_feats)
        l_conv6 = l_conv6.permute(0, 2, 3, 1).contiguous()
        l_conv6 = l_conv6.view(batch_size, -1, 4)

        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()
        l_conv7 = l_conv7.view(batch_size, -1, 4)

        l_conv8 = self.loc_conv8(conv8_feats)
        l_conv8 = l_conv8.permute(0, 2, 3, 1).contiguous()
        l_conv8 = l_conv8.view(batch_size, -1, 4)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats) 
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)


        c_conv6 = self.cl_conv6(conv6_feats)
        c_conv6 = c_conv6.permute(0, 2, 3, 1).contiguous()
        c_conv6 = c_conv6.view(batch_size, -1, self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        c_conv8 = self.cl_conv8(conv8_feats)
        c_conv8 = c_conv8.permute(0, 2, 3, 1).contiguous()
        c_conv8 = c_conv8.view(batch_size, -1, self.n_classes)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)

        locs = torch.cat([l_conv4_3, l_conv6, l_conv7, l_conv8], dim=1)
        classes_scores = torch.cat([c_conv4_3, c_conv6, c_conv7, c_conv8],
                                   dim=1)

        return locs, classes_scores


class SSDResNet(nn.Module):
    """
    The SSDResNet network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, n_classes):
        super(SSDResNet, self).__init__()

        self.n_classes = n_classes

        self.base = ResNetBase()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image_vis, image_lwir):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv6_feats , conv7_feats, conv8_feats = self.base(image_vis, image_lwir)
        
        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv6_feats, conv7_feats, conv8_feats)  # (N, 8732, 4), (N, 8732, n_classes)
        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSDResNet, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """

        fmap_dims = {'conv4_3': [80,64],
                     'conv6': [40,32],
                     'conv7': [20,16],
                     'conv8': [10,8]}

        scale_ratios = {'conv4_3': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv6': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv7': [1., pow(2,1/3.), pow(2,2/3.)],
                      'conv8': [1., pow(2,1/3.), pow(2,2/3.)]}


        aspect_ratios = {'conv4_3': [1/2., 1/1.],
                         'conv6': [1/2., 1/1.],
                         'conv7': [1/2., 1/1.],
                         'conv8': [1/2., 1/1.]}


        anchor_areas = {'conv4_3': [40*40.],
                         'conv6': [80*80.],
                         'conv7': [160*160.],
                         'conv8': [200*200.]} 

        fmaps = ['conv4_3', 'conv6', 'conv7', 'conv8']

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap][1]):
                for j in range(fmap_dims[fmap][0]):
                    cx = (j + 0.5) / fmap_dims[fmap][0]
                    cy = (i + 0.5) / fmap_dims[fmap][1]
                    for s in anchor_areas[fmap]:
                        for ar in aspect_ratios[fmap]: 
                            h = sqrt(s/ar)                
                            w = ar * h
                            for sr in scale_ratios[fmap]: # scale
                                anchor_h = h*sr/512.
                                anchor_w = w*sr/640.
                                prior_boxes.append([cx, cy, anchor_w, anchor_h])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSDResNet) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = torch.sigmoid(predicted_scores)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        all_images_bg_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to

            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)) 

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()
            image_bf_scores = list()
            
            predicted_fg_scores = predicted_scores[i][:,1:].mean(dim=1)
            score_above_min_score = predicted_fg_scores > min_score
            n_above_min_score = score_above_min_score.sum().item()

            class_scores = predicted_scores[i][:,1:][score_above_min_score]
            bg_scores = predicted_scores[i][:,0][score_above_min_score]
            class_decoded_locs = decoded_locs[score_above_min_score]
            
            # Sort predicted boxes and scores by scores\
            _, sort_ind = class_scores.mean(dim=1).sort(dim=0, descending=True)
            class_scores = class_scores[sort_ind]
            bg_scores = bg_scores[sort_ind]
            class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

            # Find the overlap between predicted boxes
            overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

            # Non-Maximum Suppression (NMS)
            suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device)  # (n_qualified)

            # Consider each box in order of decreasing scores
            for box in range(class_decoded_locs.size(0)):
                # If this box is already marked for suppression
                if suppress[box] == 1: 
                    continue
                suppress = torch.max(suppress, overlap[box] > max_overlap)
                suppress[box] = 0

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[~suppress])
            image_labels.append(torch.ones((~suppress).sum().item()).to(device))
            image_scores.append(class_scores[~suppress])
            image_bf_scores.append(bg_scores[~suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))
                image_bf_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_bg_scores = torch.cat(image_bf_scores, dim=0)
            n_objects = image_scores.size(0)

            if n_objects > top_k:
                _, sort_ind = image_scores.mean(dim=1).sort(dim=0, descending=True)
                image_scores = image_scores[sort_ind][:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)
                image_bg_scores = image_bg_scores[sort_ind][:top_k]

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)
            all_images_bg_scores.append(image_bg_scores)

        return all_images_boxes, all_images_labels, all_images_scores, all_images_bg_scores  # lists of length batch_size

    def detect_objects_cuda(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k, mode=""):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = torch.sigmoid(predicted_scores)

        all_images_boxes = []
        all_images_labels = []
        all_images_scores = []

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)) 

            predicted_fg_scores = predicted_scores[i][:,1:].mean(dim=1)
            score_above_min_score = predicted_fg_scores > min_score

            class_scores = predicted_scores[i][:,1:][score_above_min_score]
            class_decoded_locs = decoded_locs[score_above_min_score]

            _, sort_ind = class_scores.mean(dim=1).sort(dim=0, descending=True)
            class_scores = class_scores[sort_ind]
            class_decoded_locs = class_decoded_locs[sort_ind]

            keep = torchvision.ops.nms(class_decoded_locs, class_scores.mean(dim=1), max_overlap)

            image_boxes = class_decoded_locs[keep]
            image_labels = torch.ones(keep.size(0), device=class_decoded_locs.device)
            image_scores = class_scores[keep]
            if image_scores.size(0) > top_k:
                _, indices = image_scores.mean(dim=1).sort(descending=True)
                indices = indices[:top_k]
                image_boxes = image_boxes[indices]
                image_labels = image_labels[indices]
                image_scores = image_scores[indices]

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores


class SSDResNet_3Way(SSDResNet):
    def __init__(self, n_classes):
        super().__init__(n_classes)

        self.n_classes = n_classes

        self.base = ResNetBase(True)
        self.pred_convs = PredictionConvolutions(n_classes) # for fusion
        self.pred_convs_vis = PredictionConvolutions(n_classes)
        self.pred_convs_lwir = PredictionConvolutions(n_classes)    

    def forward(self, image_vis, image_lwir):
        conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, \
        out_vis_1,     out_vis_2,   out_vis_3,   out_vis_4,   \
        out_lwir_1,    out_lwir_2,  out_lwir_3,  out_lwir_4 = self.base(image_vis, image_lwir)
        locs_fusion, classes_scores_fusion = self.pred_convs(conv4_3_feats, conv6_feats, conv7_feats, conv8_feats)  # (N, 8732, 4), (N, 8732, n_classes)
        locs_vis, classes_scores_vis = self.pred_convs_vis(out_vis_1, out_vis_2, out_vis_3, out_vis_4)  # (N, 8732, 4), (N, 8732, n_classes)
        locs_lwir, classes_scores_lwir = self.pred_convs_lwir(out_lwir_1, out_lwir_2, out_lwir_3, out_lwir_4)  # (N, 8732, 4), (N, 8732, n_classes)
        return locs_fusion, classes_scores_fusion, locs_vis, classes_scores_vis, locs_lwir, classes_scores_lwir, \
               [conv4_3_feats, conv6_feats , conv7_feats, conv8_feats], \
               [out_vis_1,     out_vis_2,    out_vis_3,   out_vis_4  ], \
               [out_lwir_1,    out_lwir_2,   out_lwir_3,  out_lwir_4 ]
    

class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False, ignore_index=-1)
        self.loss_fn = nn.BCEWithLogitsLoss(reduce=False, reduction ='none')

    def forward(self, predicted_locs, predicted_scores, boxes, labels, cos_weights=None):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float, device=device)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long, device=device)

        # For each image
        for i in range(batch_size):

            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],self.priors_xy)

            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)


            _, prior_for_each_object = overlap.max(dim=1)

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            overlap_for_each_prior[prior_for_each_object] = 1.

            label_for_each_prior = labels[i][object_for_each_prior]

            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0


            true_classes[i] = label_for_each_prior
            

            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        positive_priors = true_classes > 0  # (N, 8732)

        if true_locs[positive_priors].shape[0] == 0:
            loc_loss = torch.tensor([0], device=device)
        else:
            loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors]) 

        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        ignore_index = torch.nonzero((true_classes.view(-1) == -1), as_tuple=False)
        pair_index = torch.nonzero((true_classes.view(-1) == 3), as_tuple=False)
        true_classes = (true_classes.view(-1)+1)
        true_classes = _to_one_hot(true_classes,n_dims=5)[:,1:4]

        if len(pair_index) != 0: 
            true_classes[pair_index,1] = 1
            true_classes[pair_index,2] = 1

        conf_loss_all = self.loss_fn(predicted_scores.view(-1, n_classes), true_classes).sum(dim=1)
        conf_loss_all[ignore_index] = 0
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  

        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / ( 1e-10 + n_positives.sum().float() )  # (), scalar

        return conf_loss + self.alpha * loc_loss , conf_loss , loc_loss, n_positives


def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).type(dtype)        
    return zeros.scatter(scatter_dim, y_tensor, 1)
