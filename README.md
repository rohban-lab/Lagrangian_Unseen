# Lagrangian Objective Function Leads to Improved Unforeseen Attack Generalization in Adversarial Training

This repository contains the code for the paper "Lagrangian Objective Function Leads to Improved Unforeseen Attack Generalization in Adversarial Training". In this paper, we have shown that using the Lagrangian objective function for generating perturbations in adversarial training can improve the robustness against unseen attacks, and we could get better results than previous methods by making some improvements.

## Lagrange Attack
Our proposed attack is available in `Lag_attack.py`.  This attack is specifically designed for adversarial training and we use it for this purpose.

## Adversarial training
The following commands can be used to adversarially train the models with our attack:

CIFAR-10:
```
python train_cifar.py --epochs 100 --warmup-epochs 3
```
ImageNet-100:
```
python train_imagenet.py --data-test 'data/test/' --data-train 'data/train/' --epochs 90 --warmup-epochs 3
```
The ImageNet data should be provided separately in a specific folder.

## Model Evaluation
The trained models can be evaluated against some attacks using:
```
python eval_model.py --dataset 'cifar' --attack-type 'pgd0' --load-path 'checkpoint.pth'  
```
