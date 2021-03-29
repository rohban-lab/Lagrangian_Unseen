import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import numpy as np
from resnet import ResNet18

from eval_attacks.perceptual_attacks import LagrangePerceptualAttack
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from recoloradv.utils import get_attack_from_name
import torchgeometry
from eval_attacks.jpeg_attack import JPEGAttack
from eval_attacks.PGD0 import PGDattack

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=50, type=int)
    parser.add_argument('--data-dir', default='data', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--dataset', default='cifar', type=str, choices=['cifar', 'imagenet'])
    parser.add_argument('--load-path', default='checkpoint.pth', type=str)
    parser.add_argument('--normalize', default=False, type=bool)
    parser.add_argument('--attack-type', default='L2', type=str, choices=['LPA', 'L2', 'Linf', 'recolor', 'stadv', 'jpeg', 'noise', 'blur', 'pgd0'])
    parser.add_argument('--LPA-bound', default=0.5, type=float)
    parser.add_argument('--L2-bound', default=1.0, type=float)
    parser.add_argument('--Linf-bound', default=0.0314, type=float)
    parser.add_argument('--Jpeg-bound', default=0.125, type=float)
    return parser.parse_args()

class normalizing():
    def __init__(self, normalize):
        self.normalize = normalize
    def forward(self, examples):
        return self.normalize(examples)

class wraper():
  def __init__(self, model, attack, normalize):

      super().__init__()

      self.model = model
      self.attack = attack
      self.training = False
      self.normalize = normalize
  
  def __call__(self, input):
      
      normalized_in = self.normalize(input)
      out = self.model(normalized_in)
      return out

def main():

    args = get_args()
    device = args.device
    attack_type = args.attack_type

    if args.normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         ])

    if args.dataset == 'cifar':

        test_loader = torch.utils.data.DataLoader(
            CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False)

        test_model = ResNet18()
        image_size = 32
        
        

    elif args.dataset == 'imagenet':
        
        test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

        test_model = torchvision.models.resnet34(pretrained=False, progress=False, num_classes=100)
        image_size = 224


    test_model.load_state_dict(torch.load(args.load_path)['net'])
    test_model.to(device)
    test_model.eval()

    test_model_wrapped = wraper(test_model, attack_type, normalize)

    if attack_type == 'LPA':
        attacker = LagrangePerceptualAttack(test_model_wrapped, bound=args.LPA_bound, num_iterations=200, lpips_model='alexnet_cifar', num_classes=10)
    elif attack_type == 'L2':
        attacker = L2PGDAttack(test_model_wrapped, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=args.L2_bound, nb_iter=40, eps_iter=args.L2_bound/33, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False).perturb
    elif attack_type == 'Linf':
        attacker = LinfPGDAttack(test_model_wrapped, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=args.Linf_bound, nb_iter=40, eps_iter=args.Linf_bound/33, rand_init=False, clip_min=0.0, clip_max=1.0, targeted=False).perturb
    elif attack_type == 'recolor':
        normalizer = normalizing(normalize)
        attacker = get_attack_from_name('recoloradv', test_model, normalizer, verbose=False)
    elif attack_type == 'stadv':
        normalizer = normalizing(normalize)
        attacker = get_attack_from_name('stadv', test_model, normalizer, verbose=False)
    elif attack_type == 'jpeg':
        attacker = JPEGAttack(50, args.Jpeg_bound, args.Jpeg_bound/7.07, image_size)
    elif attack_type == 'pgd0':
        attacker = PGDattack(test_model_wrapped).perturb


    correct = 0
    total = 0
    for j, data in enumerate(test_loader):

        input, target = data
        target = target.to(device)
        input = input.to(device)
        
        if attack_type in ['LPA', 'L2', 'Linf']:
            attacked_input = attacker(input, target)
        elif attack_type in ['recolor', 'stadv']:
            attacked_input = attacker.attack(input, target)[0]
        elif attack_type == 'jpeg':
            attacked_input = attacker(test_model_wrapped, input, target, avoid_target = True)
        elif attack_type == 'noise':
            attacked_input = input + (0.05**0.5)*torch.randn_like(input)
        elif attack_type == 'blur':
            attacked_input = torchgeometry.image.gaussian_blur(input, (5, 5), (1.5, 1.5))
        elif attack_type == 'pgd0':
            attacked_input = attacker(input.cpu().permute(0, 2, 3, 1).numpy(), target.cpu().numpy())

        
        logits = test_model_wrapped(attacked_input)
        _, pred = torch.max(logits, dim=1)
        correct += (pred == target).sum()
        total += target.size(0)

    acc = (float(correct) / total) * 100
    print('Test Accuracy: ', round(acc, 2))


if __name__ == "__main__":
    main()