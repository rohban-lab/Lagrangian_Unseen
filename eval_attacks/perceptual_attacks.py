#Code from "Perceptual Adversarial Robustness: Defense Against Unseen Threat Models" (Laidlaw et al., 2020)

import torch
import torchvision.models as torchvision_models
from torch import nn
from torch.nn import functional as F

from .distances import normalize_flatten_features, LPIPSDistance
from .utilities import MarginLoss
from .models import AlexNetFeatureModel, CifarAlexNet
import utilities
import random


def get_lpips_model(lpips_model, model):
    if lpips_model == 'self':
        return model
    elif lpips_model == 'alexnet':
        alexnet_model = torchvision_models.alexnet(pretrained=True)
        lpips_model = AlexNetFeatureModel(alexnet_model)
        if torch.cuda.is_available():
            lpips_model.cuda()
    elif lpips_model == 'alexnet_cifar':
        alexnet_model = CifarAlexNet()
        state = torch.load('./alexnet.pth')
        alexnet_model.load_state_dict(state['model'])
        lpips_model = AlexNetFeatureModel(alexnet_model)
        if torch.cuda.is_available():
            lpips_model.cuda()
    lpips_model.eval()
    return lpips_model

class NoProjection(nn.Module):
    def __init__(self, bound, lpips_model):
        super().__init__()

    def forward(self, inputs, adv_inputs, input_features=None):
        return adv_inputs


class BisectionPerceptualProjection(nn.Module):
    def __init__(self, bound, lpips_model, num_steps=5):
        super().__init__()

        self.bound = bound
        self.lpips_model = lpips_model
        self.num_steps = num_steps

    def forward(self, inputs, adv_inputs, input_features=None):
        batch_size = inputs.shape[0]
        if input_features is None:
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs))

        lam_min = torch.zeros(batch_size, device=inputs.device)
        lam_max = torch.ones(batch_size, device=inputs.device)
        lam = 0.5 * torch.ones(batch_size, device=inputs.device)

        for _ in range(self.num_steps):
            projected_adv_inputs = (
                inputs * (1 - lam[:, None, None, None]) +
                adv_inputs * lam[:, None, None, None]
            )
            adv_features = self.lpips_model.features(projected_adv_inputs)
            adv_features = normalize_flatten_features(adv_features).detach()
            diff_features = adv_features - input_features
            norm_diff_features = torch.norm(diff_features, dim=1)

            lam_max[norm_diff_features > self.bound] = \
                lam[norm_diff_features > self.bound]
            lam_min[norm_diff_features <= self.bound] = \
                lam[norm_diff_features <= self.bound]
            lam = 0.5*(lam_min + lam_max)
        return projected_adv_inputs.detach()


class NewtonsPerceptualProjection(nn.Module):
    def __init__(self, bound, lpips_model, projection_overshoot=1e-2):
        super().__init__()

        self.bound = bound
        self.lpips_model = lpips_model
        self.projection_overshoot = projection_overshoot

    def forward(self, inputs, adv_inputs, input_features=None):
        if input_features is None:
            input_features = normalize_flatten_features(
                self.lpips_model.features(inputs))

        needs_projection = torch.ones_like(adv_inputs[:, 0, 0, 0]) \
            .bool()

        needs_projection.requires_grad = False
        while needs_projection.sum() > 0:
            adv_inputs.requires_grad = True
            adv_features = normalize_flatten_features(
                self.lpips_model.features(adv_inputs[needs_projection]))
            adv_lpips = (input_features[needs_projection] -
                         adv_features).norm(dim=1)
            adv_lpips.sum().backward()

            projection_step_size = (adv_lpips - self.bound) \
                .clamp(min=0)
            projection_step_size[projection_step_size > 0] += \
                self.projection_overshoot

            grad_norm = adv_inputs.grad.data[needs_projection] \
                .view(needs_projection.sum(), -1).norm(dim=1)
            inverse_grad = adv_inputs.grad.data[needs_projection] / \
                grad_norm[:, None, None, None] ** 2

            adv_inputs.data[needs_projection] = (
                adv_inputs.data[needs_projection] -
                projection_step_size[:, None, None, None] *
                (1 + self.projection_overshoot) *
                inverse_grad
            ).clamp(0, 1).detach()

            needs_projection[needs_projection] = \
                projection_step_size > 0

        return adv_inputs


PROJECTIONS = {
    'none': NoProjection,
    'linesearch': BisectionPerceptualProjection,
    'bisection': BisectionPerceptualProjection,
    'gradient': NewtonsPerceptualProjection,
    'newtons': NewtonsPerceptualProjection,
}

class LagrangePerceptualAttack(nn.Module):
    def __init__(self, model, bound=0.5, step=None, num_iterations=20,
                 binary_steps=5, h=0.1, lpips_model='self',
                 projection='bisection', decay_step_size=True,
                 num_classes=None, randomize=False, random_targets=False):
        """
        Perceptual attack using a Lagrangian relaxation of the
        LPIPS-constrainted optimization problem.
        bound is the (soft) bound on the LPIPS distance.
        step is the LPIPS step size.
        num_iterations is the number of steps to take.
        lam is the lambda value multiplied by the regularization term.
        h is the step size to use for finite-difference calculation.
        lpips_model is the model to use to calculate LPIPS or 'self' or
            'alexnet'
        """

        super().__init__()

        assert randomize is False

        self.model = model
        self.bound = bound
        self.decay_step_size = decay_step_size
        self.num_iterations = num_iterations
        if step is None:
            if self.decay_step_size:
                self.step = self.bound
            else:
                self.step = self.bound * 2 / self.num_iterations
        else:
            self.step = step
        self.binary_steps = binary_steps
        self.h = h
        self.random_targets = random_targets
        self.num_classes = num_classes

        self.lpips_model = get_lpips_model(lpips_model, model)
        self.lpips_distance = LPIPSDistance(self.lpips_model)
        self.loss = MarginLoss(kappa=1, targeted=self.random_targets)
        self.projection = PROJECTIONS[projection](self.bound, self.lpips_model)

    def _attack(self, inputs, labels):
        perturbations = torch.zeros_like(inputs)
        perturbations.normal_(0, 0.01)
        perturbations.requires_grad = True

        batch_size = inputs.shape[0]
        step_size = self.step

        lam = 0.01 * torch.ones(batch_size, device=inputs.device)

        input_features = normalize_flatten_features(
            self.lpips_model.features(inputs)).detach()

        live = torch.ones(batch_size, device='cuda', dtype=bool)

        for binary_iter in range(self.binary_steps):
            for attack_iter in range(self.num_iterations):
                if self.decay_step_size:
                    step_size = self.step * \
                        (0.1 ** (attack_iter / self.num_iterations))
                else:
                    step_size = self.step

                if perturbations.grad is not None:
                    perturbations.grad.data.zero_()

                adv_inputs = (inputs + perturbations)[live]

                if self.model == self.lpips_model:
                    adv_features, adv_logits = \
                        self.model.features_logits(adv_inputs)
                else:
                    adv_features = self.lpips_model.features(adv_inputs)
                    adv_logits = self.model(adv_inputs)

                adv_labels = adv_logits.argmax(1)
                adv_loss = self.loss(adv_logits, labels[live])
                adv_features = normalize_flatten_features(adv_features)
                lpips_dists = (adv_features - input_features[live]).norm(dim=1)
                all_lpips_dists = torch.zeros(batch_size, device=inputs.device)
                all_lpips_dists[live] = lpips_dists

                loss = -adv_loss + lam[live] * F.relu(lpips_dists - self.bound)
                loss.sum().backward()

                grad = perturbations.grad.data[live]
                grad_normed = grad / \
                    (grad.reshape(grad.size()[0], -1).norm(dim=1)
                     [:, None, None, None] + 1e-8)

                dist_grads = (
                    adv_features -
                    normalize_flatten_features(self.lpips_model.features(
                        adv_inputs + grad_normed * self.h))
                ).norm(dim=1) / self.h

                updates = -grad_normed * (
                    step_size / (dist_grads + 1e-8)
                )[:, None, None, None]

                perturbations.data[live] = (
                    (inputs[live] + perturbations[live] +
                     updates).clamp(0, 1) -
                    inputs[live]
                ).detach()

                if self.random_targets:
                    live[live] = (adv_labels != labels[live]) | (lpips_dists > self.bound)
                else:
                    live[live.clone()] = (adv_labels == labels[live.clone()]) | (lpips_dists > self.bound)
                if live.sum() == 0:
                    break

            lam[all_lpips_dists >= self.bound] *= 10
            if live.sum() == 0:
                break

        adv_inputs = (inputs + perturbations).detach()
        adv_inputs = self.projection(inputs, adv_inputs, input_features)
        return adv_inputs

    def forward(self, inputs, labels):
        if self.random_targets:
            return utilities.run_attack_with_random_targets(
                self._attack,
                self.model,
                inputs,
                labels,
                self.num_classes,
            )
        else:
            return self._attack(inputs, labels)
