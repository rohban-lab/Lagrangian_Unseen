import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import time
import numpy as np
from resnet import ResNet18
from Lag_attack import LagrangeAttack

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=50, type=int)
    parser.add_argument('--data-dir', default='data', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr-drop-epochs', default="[70, 90]", type=str)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lam', default=2.0, type=float)
    parser.add_argument('--alpha', default=0.15, type=float)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--save-path', default='checkpoint.pth', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=2e-4, type=float)
    parser.add_argument('--warmup-epochs', default=3, type=int)
    parser.add_argument('--print-every-iterations', default=200, type=int)
    parser.add_argument('--normalize', default=False, type=bool)
    return parser.parse_args()

def main():
    args = get_args()
    device = args.device

    main_model = ResNet18()
    main_model.to(device)
    main_model.train()

    Lag = LagrangeAttack(main_model, bound=args.alpha, lam=args.lam)

    if args.normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1.0, 1.0, 1.0])

    transform_train = transforms.Compose(
            [transforms.Pad(4, padding_mode="reflect"),
             transforms.RandomCrop(32),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize,
             ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         normalize,
         ])

    test_loader = torch.utils.data.DataLoader(
            CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
            CIFAR10(root=args.data_dir, train=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True)
    
    lr_drop_epochs = [float(x) for x in args.lr_drop_epochs.strip('[]').split(',')]
    
    optimizer = optim.SGD(main_model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=lr_drop_epochs, gamma=args.lr_decay)

    for i in range(args.epochs):

        correct = 0
        total = 0
        total_loss = 0

        start_time = time.time()
        for j, data in enumerate(train_loader):

            input, target = data
            target = target.to(device)
            input = input.to(device)

            if i>=args.warmup_epochs:
                attacked_input = Lag(input, target)
            else:
                attacked_input = input
            
            optimizer.zero_grad()
            logits = main_model(attacked_input)

            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(logits, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)
            total_loss += loss

            if j%args.print_every_iterations == 1:
                  acc = (float(correct) / total) * 100
                  message = 'Epoch {}, Loss: {}, Accuracy: {}'.format(i, round(total_loss.item(), 2), round(acc, 2))
                  print(message)
                  correct = 0
                  total = 0
                  total_loss = 0
            
        lr_scheduler.step()
        end_time = time.time()
        batch_time = end_time - start_time

        acc = (float(correct) / total) * 100
        message = 'Epoch {}, Loss: {}, Accuracy: {}'.format(i, round(total_loss.item(), 2), round(acc, 2))
        print(message)
        print("Time: ", batch_time)

        state = {'net': main_model.state_dict(),
                     'epoch': i,
                     'optimizer': optimizer.state_dict(),
                     'scheduler': lr_scheduler.state_dict()}

        torch.save(state, args.save_path)


if __name__ == "__main__":
    main()