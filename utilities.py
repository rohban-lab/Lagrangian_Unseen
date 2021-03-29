#Code from "Perceptual Adversarial Robustness: Defense Against Unseen Threat Models" (Laidlaw et al., 2020)

import torch
import os
import torchvision.models as torchvision_models
from torch import nn

class MarginLoss(nn.Module):
    """
    Calculates the margin loss max(kappa, (max z_k (x) k != y) - z_y(x)),
    also known as the f6 loss used by the Carlini & Wagner attack.
    """

    def __init__(self, kappa=float('inf'), targeted=False):
        super().__init__()
        self.kappa = kappa
        self.targeted = targeted

    def forward(self, logits, labels):
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1))

        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max

        if self.targeted:
            return (correct_logits - max_incorrect_logits) \
                .clamp(max=self.kappa).squeeze()
        else:
            return (max_incorrect_logits - correct_logits) \
                .clamp(max=self.kappa).squeeze()