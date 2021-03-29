#The coed is based on https://github.com/fra31/sparse-imperceivable-attacks (Croce et al.)

import scipy.io
import numpy as np
import torch
import torch.nn.functional as F

device = 'cuda:0'

def project_L0_box(y, k, lb, ub):
  ''' projection of the batch y to a batch x such that:
        - each image of the batch x has at most k pixels with non-zero channels
        - lb <= x <= ub '''
      
  x = np.copy(y)
  p1 = np.sum(x**2, axis=-1)
  p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
  p2 = np.sum(p2**2, axis=-1)
  p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
  x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
  x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
    
  return x
  
def perturb_L0_box(attack, x_nat, y_nat, lb, ub, k):
  ''' PGD attack wrt L0-norm + box constraints
  
      it returns adversarial examples (if found) adv for the images x_nat, with correct labels y_nat,
      such that:
        - each image of the batch adv differs from the corresponding one of
          x_nat in at most k pixels
        - lb <= adv - x_nat <= ub
      
      it returns also a vector of flags where 1 means no adversarial example found
      (in this case the original image is returned in adv) '''
  
  if attack.rs:
    x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
    x2 = np.clip(x2, 0, 1)
  else:
    x2 = np.copy(x_nat)
      
  adv_not_found = np.ones(y_nat.shape)
  adv = np.zeros(x_nat.shape)

  for i in range(attack.num_steps):
    if i > 0:
      inp = torch.FloatTensor(x2).permute(0, 3, 1, 2).to(device)
      inp.requires_grad_(True)
      logits = attack.model(inp)
      _, pred = torch.max(logits, dim=1)
      pred = (pred.cpu().numpy() == y_nat)

      loss = F.cross_entropy(logits, torch.tensor(y_nat).to(device))
      loss.backward()
      grad = inp.grad.permute(0, 2, 3, 1).cpu().numpy()

      adv_not_found = np.minimum(adv_not_found, pred.astype(int))
      adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])
      
      grad /= (1e-10 + np.sum(np.abs(grad), axis=(1,2,3), keepdims=True))
      x2 = np.add(x2, (np.random.random_sample(grad.shape)-0.5)*1e-12 + attack.step_size * grad, casting='unsafe')
      
    x2 = x_nat + project_L0_box(x2 - x_nat, k, lb, ub)
    
  return adv, adv_not_found


class PGDattack():
  def __init__(self, model):
    self.model = model
    self.num_steps = 20                    # number of iterations of gradient descent for each restart
    self.step_size = 30000/255              # step size for gradient descent (\eta in the paper)
    self.n_restarts = 10                     # number of random restarts to perform
    self.rs = True                         # random starting point
    self.k = 10                           # maximum number of pixels that can be modified (k_max in the paper)
    
  def perturb(self, x_nat, y_nat):
    adv = np.copy(x_nat)
  
    
    for k in range(1, self.k):
      #print("k:", k)
      for counter in range(self.n_restarts):

        if counter == 0 and k == 1:


          logits = self.model(torch.tensor(x_nat).permute(0, 3, 1, 2).to(device))
          _, pred = torch.max(logits, dim=1)
          corr_pred = (pred.cpu().numpy() == y_nat)

          pgd_adv_acc = np.copy(corr_pred)
        
      
        x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, -x_nat, 1.0 - x_nat, k)
  
        adv[np.logical_not(curr_pgd_adv_acc)] = x_batch_adv[np.logical_not(curr_pgd_adv_acc)]

        pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)
        #print("Restart {} - Robust accuracy: {}".format(counter + 1, np.sum(pgd_adv_acc)/x_nat.shape[0]))
    
    return torch.tensor(adv).permute(0, 3, 1, 2).to(device)