import torch
import torchvision.models as torchvision_models
from torch import nn
from torch.nn import functional as F
from utilities import MarginLoss

class LagrangeAttack(nn.Module):
    def __init__(self, model, bound=0.15, num_iterations=5,
                 lam=1.0, decay_step_size=True, increase_lambda=True):
      

        super().__init__()

        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations
        self.lam = lam
        self.decay_step_size = decay_step_size
        self.increase_lambda = increase_lambda
        self.loss = MarginLoss(kappa=1)

    def forward(self, inputs, labels):
        perturbations = torch.zeros_like(inputs)
        perturbations.normal_(0, 0.01)

        perturbations.requires_grad = True
        step_size = self.bound
        
        for attack_iter in range(self.num_iterations):
            if self.decay_step_size:
                step_size = \
                    self.bound * 0.1 ** (attack_iter / self.num_iterations)
            if self.increase_lambda:
                lam = \
                    self.lam * 0.1 ** (1 - attack_iter / self.num_iterations)

            if perturbations.grad is not None:
                perturbations.grad.data.zero_()

            adv_inputs = (inputs + perturbations)
            adv_logits = self.model(adv_inputs)
            adv_loss = self.loss(adv_logits, labels)

            lpips_dists = (adv_inputs - inputs).reshape(inputs.size()[0], -1).norm(dim=1)

            loss = -adv_loss + lam * F.relu(lpips_dists)
            loss.sum().backward()

            grad = perturbations.grad.data
            grad_normed = grad/(abs(grad.reshape(grad.size()[0], -1)).max(dim=1)[0]
            [:, None, None, None] + 1e-8)
            
            perturbation_updates = -grad_normed * step_size

            perturbations.data = (
                (inputs + perturbations + perturbation_updates).clamp(0, 1) - inputs).detach()

        adv_inputs = (inputs + perturbations).detach()
            
        return adv_inputs