#Code from "Perceptual Adversarial Robustness: Defense Against Unseen Threat Models" (Laidlaw et al., 2020)

import torch
import torch.nn.functional as F
import numpy as np
import torchvision.models as torchvision_models
from torch.autograd import Variable
from math import exp
from torch import nn

from .models import AlexNetFeatureModel


class LPIPSDistance(nn.Module):
    """
    Calculates the square root of the Learned Perceptual Image Patch Similarity
    (LPIPS) between two images, using a given neural network.
    """

    def __init__(self, model=None, activation_distance='l2'):
        """
        Constructs an LPIPS distance metric. The given network should return a
        tuple of (activations, logits). If a network is not specified, AlexNet
        will be used. activation_distance can be 'l2' or 'cw_ssim'.
        """

        super().__init__()

        if model is None:
            alexnet_model = torchvision_models.alexnet(pretrained=True)
            self.model = AlexNetFeatureModel(alexnet_model)
        else:
            if isinstance(model, nn.DataParallel):
                self.model = model.module
            else:
                self.model = model

        self.activation_distance = activation_distance

        self.eval()

    def forward(self, image1, image2):
        features1 = self.model.features(image1)
        features2 = self.model.features(image2)

        if self.activation_distance == 'l2':
            return (
                normalize_flatten_features(features1) -
                normalize_flatten_features(features2)
            ).norm(dim=1)
        elif self.activation_distance == 'cw_ssim':
            distance = torch.zeros_like(image1[:, 0, 0, 0])
            for layer1, layer2 in zip(features1, features2):
                size = min(layer1.size()[2:3])
                cw_ssim_level = min(1, np.floor(np.log2(size)))
                distance += CosineWaveletSSIM(
                    level=cw_ssim_level,
                    window_size=1,
                    dissimilarity=True,
                )(layer1, layer2)
            distance /= len(features1)
            return distance


def normalize_flatten_features(features, eps=1e-10):
    """
    Given a tuple of features (layer1, layer2, layer3, ...) from a network,
    flattens those features into a single vector per batch input. The
    features are also scaled such that the L2 distance between features
    for two different inputs is the LPIPS distance between those inputs.
    """

    normalized_features = []

    for feature_layer in features:
        #print(feature_layer.size())
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
            .view(feature_layer.size()[0], -1)
        )
    return torch.cat(normalized_features, dim=1)

def normalize_features(features, eps=1e-10):
    """
    Given a tuple of features (layer1, layer2, layer3, ...) from a network,
    flattens those features into a single vector per batch input. The
    features are also scaled such that the L2 distance between features
    for two different inputs is the LPIPS distance between those inputs.
    """

    normalized_features = []
    for feature_layer in features:
        norm_factor = torch.sqrt(
            torch.sum(feature_layer ** 2, dim=1, keepdim=True)) + eps
        normalized_features.append(
            (feature_layer / (norm_factor *
                              np.sqrt(feature_layer.size()[2] *
                                      feature_layer.size()[3])))
        )
    return normalized_features


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    """
    Copied from https://github.com/Po-Hsun-Su/pytorch-ssim
    """

    def __init__(self, window_size=11, size_average=True, dissimilarity=False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.dissimilarity = dissimilarity

    def forward(self, imgs1, imgs2):
        (_, channel, _, _) = imgs1.size()

        if channel == self.channel and self.window.data.type() == imgs1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if imgs1.is_cuda:
                window = window.cuda(imgs1.get_device())
            window = window.type_as(imgs1)
            
            self.window = window
            self.channel = channel

        sim = torch.tensor([
            _ssim(img1[None], img2[None], window, self.window_size, channel, self.size_average)
            for img1, img2 in zip(imgs1, imgs2)
        ])
        return 1 - sim if self.dissimilarity else sim


class CosineWaveletSSIM(nn.Module):
    """
    Calculates the Cosine Wavelet Structural Similarity Index (CW-SSWIM)
    between two images; see
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5109651
    Code adapted from MatLab implementation at
    https://www.mathworks.com/matlabcentral/fileexchange/43017-complex-wavelet-structural-similarity-index-cw-ssim
    """

    def __init__(self, level=6, window_size=7, K=1e-8, dissimilarity=False):
        self.level = level
        self.window_size = window_size
        self.K = K
        self.cw_transform = DTCWTForward(J=level, biort='near_sym_b',
                                         qshift='qshift_b')
        self.dissimilarity = dissimilarity

    def _construct_gaussian_kernel(self, width, height, sigma):
        x_grid = torch.arange(width).repeat(height) \
            .view(width, height).float()
        y_grid = torch.arange(height).repeat(width).t() \
            .view(width, height).float()

        x_mean = (width - 1) / 2
        y_mean = (height - 1) / 2
        variance = sigma ** 2

        gaussian_kernel = (1 / (2 * np.pi * variance)) * torch.exp(
                              -((x_grid - x_mean) ** 2 +
                                (y_grid - y_mean) ** 2) /
                              (2*variance)
                          )

        return gaussian_kernel / torch.sum(gaussian_kernel)

    def forward(self, image1, image2):
        _, pyr1 = self.cw_transform(image1)
        _, pyr2 = self.cw_transform(image2)

        bands1 = pyr1[-1]
        bands2 = pyr2[-1]
        num_images, num_channels, num_bands, width, height = bands1.size()[:5]
        width_band = width - self.window_size + 1
        height_band = height - self.window_size + 1

        # corr is complex and equals bands1 * conjugate(bands2)
        corr = torch.zeros_like(bands1)
        corr[..., 0] += bands1[..., 0] * bands2[..., 0]
        corr[..., 0] += bands1[..., 1] * bands2[..., 1]
        corr[..., 1] += bands1[..., 1] * bands2[..., 0]
        corr[..., 1] -= bands1[..., 0] * bands2[..., 1]

        # varr is real-valued and equals abs(bands1)^2 + abs(bands2)^2
        varr = (bands1[..., 0] ** 2 + bands1[..., 1] ** 2 +
                bands2[..., 0] ** 2 + bands2[..., 1] ** 2)

        corr = corr.permute(0, 1, 2, 5, 3, 4) \
            .reshape((num_images, -1, width, height))
        corr_band = F.avg_pool2d(corr, self.window_size, stride=1)
        corr_band = corr_band \
            .reshape((num_images, num_channels, num_bands,
                      2, width_band, height_band)) \
            .permute(0, 1, 2, 4, 5, 3)

        varr = varr.reshape((num_images, -1, width, height))
        varr_band = F.avg_pool2d(varr, self.window_size, stride=1)
        varr_band = varr_band \
            .reshape((num_images, num_channels, num_bands,
                      width_band, height_band))

        abs_corr_band = torch.sqrt(corr_band[..., 0] ** 2 +
                                   corr_band[..., 1] ** 2)
        cssim_map = (2 * abs_corr_band + self.K) / (varr_band + self.K)

        weight = self._construct_gaussian_kernel(
            width_band, height_band, (width_band + height_band) / 8)
        cw_ssim = torch.sum(cssim_map.mean(dim=[1, 2]) * weight[None],
                            dim=[1, 2])

        if self.dissimilarity:
            return 1 - cw_ssim
        else:
            return cw_ssim


class L2Distance(nn.Module):
    def forward(self, img1, img2):
        return (img1 - img2).reshape(img1.shape[0], -1).norm(dim=1)
