import pyro
import torch
import matplotlib.pyplot as plt

import time

# Run stochastic variational inference using PyTorch optimizers from torch.optim
def train(loss, train_loader, optimizer, scheduler, use_cuda=False):

    # Initialize loss accumulator
    tot_epoch_loss = 0.0

    # Do a training epoch over each mini-batch x returned
    # by the data loader
    for batch in train_loader:
      
      cod_stations_batch = batch['cod station']
      model_orders_batch = batch['model order']
      x = batch['x']
      y = batch['y']
      date_batch = batch['date']

      # In a single batch are contained measures about single station
      mask = batch['mask'][0]  

      # if on GPU put mini-batch into CUDA memory
      if use_cuda:
          x = x.cuda()
          y = y.cuda()
          mask = mask.cuda()

          date_batch = [date_batch[0].cuda(), date_batch[1].cuda(), date_batch[2].cuda(), date_batch[3].cuda(), date_batch[4].cuda()]
      
      optimizer.zero_grad()

      # ELBO: return the negative of ELBO
      # Objective: Minimizate negative of ELBO --> Maximize ELBO
      loss_batch = loss(x, date_batch, mask, y)
  
      loss_batch.backward()
      optimizer.step()

      tot_epoch_loss += (-loss_batch.item())
  
    if scheduler is not None:
      scheduler.step()

    return tot_epoch_loss

def evaluate(loss, x_test_tensor, y_test_tensor, dates, mask_tensor, use_cuda=False):

    test_loss = 0.0

    with torch.no_grad():

      if use_cuda:
        x_test_tensor = x_test_tensor.cuda()
        y_test_tensor = y_test_tensor.cuda()
        mask_tensor = mask_tensor.cuda()
        dates = [dates[0].cuda(), dates[1].cuda(), dates[2].cuda(), dates[3].cuda(), dates[4].cuda()]

      # ELBO: return the negative of ELBO
      test_loss += - loss(x_test_tensor, dates, mask_tensor, y_test_tensor).item()

    return test_loss

def save_checkpoint(state_dict_model, guide, optimizer, scheduler, epoch, path_best_model, path_best_model_params):
    
    state_dict_scheduler = None

    if scheduler is not None:
      state_dict_scheduler = scheduler.state_dict()

    dict_state = {
                  "model": state_dict_model,
                  "guide": guide,
                  "optimizer": optimizer.state_dict(),
                  "scheduler": state_dict_scheduler,
                  "epoch": epoch + 1
                }


    torch.save(dict_state, path_best_model)
    pyro.get_param_store().save(path_best_model_params)

def load_checkpoint(path_best_model):
    
    saved_model_dict = torch.load(path_best_model)
    guide = saved_model_dict['guide']
    epoch = saved_model_dict['epoch']
    
    return saved_model_dict['model'], guide, saved_model_dict['optimizer'], saved_model_dict['scheduler'], epoch

def total_number_of_params(parameters):
  return sum(param.numel() for param in parameters)

def plot_loss(losses, title, path_save_fig):
   
  plt.figure(figsize=(10, 5))
  plt.plot(losses)
  plt.title(title)
  plt.xlabel("Epoch")
  plt.ylabel("ELBO loss")

  plt.savefig(path_save_fig, dpi=300)

  plt.show()
  plt.close()

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters, last_epoch=-1):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

        self.last_epoch = last_epoch

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor