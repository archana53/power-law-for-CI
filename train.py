import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from EWC_model import EWCModel
import numpy as np
from tqdm import tqdm 
import time
import copy


def train(model, train_loader, iters, loss_cbs=list(), eval_cbs=list()):
    '''Train a model with a "train_a_batch" method for [iters] iterations on data from [train_loader].
    [model]             model to optimize
    [train_loader]      <dataloader> for training [model] on
    [iters]             <int> (max) number of iterations (i.e., batches) to train for
    [loss_cbs]          <list> of callback-<functions> to keep track of training progress
    [eval_cbs]          <list> of callback-<functions> to evaluate model on separate data-set'''

    device = model._device()
    # Create progress-bar (with manual control)
    iteration = epoch = 0
    for it in tqdm(range(iters), desc = 'Iterations'):
        # Loop over all batches of an epoch
        acc = []
        for batch_idx, (data, y) in enumerate(train_loader):
            # Perform training-step on this batch
            data, y = data.to(device), y.to(device)
            loss_dict = model.train_a_batch(data, y=y)
            y_hat = model(data)
            acc.append((y == y_hat.max(1)[1]).sum().item() / data.size(0))
        if it %100 == 0:
            print('accuracy for iteration ',it, ' :', np.mean(acc))
def train_cl(model, train_datasets, iters=2000, batch_size=32, baseline='none',
             loss_cbs=list(), eval_cbs=list(), sample_cbs=list(), context_cbs=list(),
             generator=None, gen_iters=0, gen_loss_cbs=list(), **kwargs):
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
    for context, train_dataset in enumerate(tqdm(train_datasets, desc = 'context'), 1):
        # If using the "joint" baseline, skip to last context, as model is only be trained once on data of all contexts
        print(context)
        if baseline=='joint':
            if context<len(train_datasets):
                continue
            else:
                baseline = "cummulative"

        # If using the "cummulative" baseline, create a large training dataset of all contexts so far
        if baseline=="cummulative":
            train_dataset = ConcatDataset(train_datasets[:context])

        dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
        
        train(model, dataloader, iters, loss_cbs = loss_cbs, eval_cbs=eval_cbs)
        ##----------> UPON FINISHING EACH CONTEXT...

        # Parameter regularization: update and compute the parameter importance estimates
        if context<len(train_datasets) and isinstance(model, EWCModel):
            # -find allowed classes
            allowed_classes = None
            ##--> EWC/NCL: estimate the Fisher Information matrix
            if model.importance_weighting=='fisher' and (model.weight_penalty or model.precondition):
                if model.fisher_kfac:
                    model.estimate_kfac_fisher(train_dataset, allowed_classes=allowed_classes)
                else:
                    model.estimate_fisher(train_dataset, allowed_classes=allowed_classes)

        # Run the callbacks after finishing each context
        for context_cb in context_cbs:
            if context_cb is not None:
                context_cb(model, iters, context=context)

