import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from EWC_model import EWCModel
import numpy as np
from tqdm import tqdm 
import time
import copy
import os


def train(model, train_loader, iters,  val_dataset = None, loss_cbs=list(), eval_cbs=list(), patience=5):
    '''Train a model with a "train_a_batch" method for [iters] iterations on data from [train_loader].
    [model]             model to optimize
    [train_loader]      <dataloader> for training [model] on
    [iters]             <int> (max) number of iterations (i.e., batches) to train for
    [loss_cbs]          <list> of callback-<functions> to keep track of training progress
    [eval_cbs]          <list> of callback-<functions> to evaluate model on separate data-set'''

    device = model._device()
    # Create progress-bar (with manual control)
    iteration = epoch = 0
    acc = []
    loss = []
    prevLoss = 100
    lossCounter = 0      # For Early Stopping
    for it in tqdm(range(iters), desc = 'Iterations'):
        # Loop over all batches of an epoch
        
        for batch_idx, (data, y) in enumerate(train_loader):
            # Perform training-step on this batch
            data, y = data.to(device), y.to(device)
            loss_dict = model.train_a_batch(data, y=y)
            # print('batch training done')
            # print(data)
            y_hat = model(data)
            
        acc.append((y == y_hat.max(1)[1]).sum().item() / data.size(0))
        loss.append(loss_dict['loss_current'])
        if it %10 == 0:
            print('\n accuracy for iteration ',it, ' :', np.mean(acc))
        
        """
        # Early stopping
        currLoss = loss_dict['loss_current']
        if currLoss > prevLoss:
            lossCounter += 1
            if lossCounter >= patience:
                print('Early stopping!')
                return acc, loss
        else:
            lossCounter = 0
        prevLoss = currLoss
        """

    return acc, loss

def train_cl_val(model, train_datasets, test_datasets,  val_datasets = None, iters=2000, batch_size=32, baseline='none',
             loss_cbs=list(), eval_cbs=list(), sample_cbs=list(), context_cbs=list(),
             generator=None, gen_iters=0, gen_loss_cbs=list(), continue_from_context = 1, training_environment = 'none', patience=5, **kwargs):
    '''Train a model (with a "train_a_batch" method) on multiple contexts.
    [model]               <nn.Module> main model to optimize across all contexts
    [train_datasets]      <list> with for each context the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per context
    [batch_size]          <int>, # of samples per mini-batch
    [baseline]            <str>, 'joint': model trained once on data from all contexts
                                 'cummulative': model trained incrementally, always using data all contexts so far
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Loop over all contexts.
    for context, train_dataset, val_dataset in enumerate(tqdm(train_datasets, desc = 'context'), val_datasets, 1):
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

        acc, loss = train(model, dataloader,iters,  val_dataset,  loss_cbs = loss_cbs, eval_cbs=eval_cbs, patience=patience)
        PATH = f'./checkpoint/model_{training_environment}_context{context}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'context': context,
            'acc': acc,
            'loss': loss
            }, PATH)
        ##----------> UPON FINISHING EACH CONTEXT...

        # Parameter regularization: update and compute the parameter importance estimates
        # if context<len(train_datasets) and isinstance(model, EWCModel):
        #     # -find allowed classes
        #     allowed_classes = None
        #     ##--> EWC/NCL: estimate the Fisher Information matrix
        #     if model.importance_weighting=='fisher' and (model.weight_penalty or model.precondition):
        #         if model.fisher_kfac:
        #             model.estimate_kfac_fisher(train_dataset, allowed_classes=allowed_classes)
        #         else:
        #             model.estimate_fisher(train_dataset, allowed_classes=allowed_classes)

        # # Run the callbacks after finishing each context
        # for context_cb in context_cbs:
        #     if context_cb is not None:
        #         context_cb(model, iters, context=context)

def train_cl(model, train_datasets, test_datasets,  iters=2000, batch_size=32, baseline='none',
             loss_cbs=list(), eval_cbs=list(), sample_cbs=list(), context_cbs=list(),
             generator=None, gen_iters=0, gen_loss_cbs=list(), continue_from_context = 1, training_environment = 'none', patience=5, **kwargs):
    '''Train a model (with a "train_a_batch" method) on multiple contexts.
    [model]               <nn.Module> main model to optimize across all contexts
    [train_datasets]      <list> with for each context the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per context
    [batch_size]          <int>, # of samples per mini-batch
    [baseline]            <str>, 'joint': model trained once on data from all contexts
                                 'cummulative': model trained incrementally, always using data all contexts so far
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    load_context = continue_from_context -1
    MODEL_PATH = f'./checkpoint/model_{training_environment}_context{load_context}.pt'
    if os.path.isfile(MODEL_PATH) == True:
      model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
      print("Model Loaded ...")

    # Loop over all contexts.
    for context, train_dataset in enumerate(tqdm(train_datasets, desc = 'context'),1):
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

        print(context, " " , continue_from_context)
        if context < continue_from_context:
          continue

        print('context ',  context, ' has not been skipped')
        acc, loss = train(model, dataloader,iters, loss_cbs = loss_cbs, eval_cbs=eval_cbs, patience=patience)
        PATH = f'./checkpoint/model_{training_environment}_context{context}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'context': context,
            'acc': acc,
            'loss': loss
            }, PATH)
        ##----------> UPON FINISHING EACH CONTEXT...

        # Parameter regularization: update and compute the parameter importance estimates
        # if context<len(train_datasets) and isinstance(model, EWCModel):
        #     # -find allowed classes
        #     allowed_classes = None
        #     ##--> EWC/NCL: estimate the Fisher Information matrix
        #     if model.importance_weighting=='fisher' and (model.weight_penalty or model.precondition):
        #         if model.fisher_kfac:
        #             model.estimate_kfac_fisher(train_dataset, allowed_classes=allowed_classes)
        #         else:
        #             model.estimate_fisher(train_dataset, allowed_classes=allowed_classes)

        # # Run the callbacks after finishing each context
        # for context_cb in context_cbs:
        #     if context_cb is not None:
        #         context_cb(model, iters, context=context)

def train_EWC(model, train_datasets, test_datasets, iters=2000, batch_size=32, baseline='none',
             loss_cbs=list(), eval_cbs=list(), sample_cbs=list(), context_cbs=list(),
             generator=None, gen_iters=0, gen_loss_cbs=list(), continue_from_context = 1, training_environment = 'none', patience=5, **kwargs):
    '''Train a model (with a "train_a_batch" method) on multiple contexts.
    [model]               <nn.Module> main model to optimize across all contexts
    [train_datasets]      <list> with for each context the training <DataSet>
    [iters]               <int>, # of optimization-steps (i.e., # of mini-batches) per context
    [batch_size]          <int>, # of samples per mini-batch
    [baseline]            <str>, 'joint': model trained once on data from all contexts
                                 'cummulative': model trained incrementally, always using data all contexts so far
    [*_cbs]               <list> of call-back functions to evaluate training-progress
    '''

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    load_context = continue_from_context -1
    MODEL_PATH = f'./checkpoint/trial_model_{training_environment}_context{load_context}.pt'
    if os.path.isfile(MODEL_PATH) == True:
      model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
      print("Model Loaded ...")

    # Loop over all contexts.
    for context, train_dataset in enumerate(tqdm(train_datasets, desc = 'context'), 1):
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

        if context < continue_from_context:
          continue

        acc, loss = train(model, dataloader, iters, loss_cbs = loss_cbs, eval_cbs=eval_cbs, patience=patience)
        PATH = f'./checkpoint/trial_model_{training_environment}_context{context}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'context': context,
            'acc': acc,
            'loss': loss
            }, PATH)
        ##----------> UPON FINISHING EACH CONTEXT...

        # Parameter regularization: update and compute the parameter importance estimates
        if context<len(train_datasets):
            # -find allowed classes
            allowed_classes = None
            ##--> EWC/NCL: estimate the Fisher Information matrix
            if model.importance_weighting=='fisher' and (model.weight_penalty or model.precondition):
                if model.fisher_kfac:
                    model.estimate_kfac_fisher(train_dataset, allowed_classes=allowed_classes)
                else:
                    model.estimate_fisher(train_dataset, allowed_classes=allowed_classes)
        # # Run the callbacks after finishing each context
        # for context_cb in context_cbs:
        #     if context_cb is not None:
        #         context_cb(model, iters, context=context)