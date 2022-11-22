import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
from EWC_model import EWCModel
import numpy as np
import tqdm
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
    bar = tqdm.tqdm(total=iters)

    iteration = epoch = 0
    while iteration < iters:
        epoch += 1

        # Loop over all batches of an epoch
        for batch_idx, (data, y) in enumerate(train_loader):
            iteration += 1

            # Perform training-step on this batch
            data, y = data.to(device), y.to(device)
            loss_dict = model.train_a_batch(data, y=y)

            # Fire training-callbacks (for visualization of training-progress)
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(bar, iteration, loss_dict)

            # Fire evaluation-callbacks (to be executed every [eval_log] iterations, as specified within the functions)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, iteration)

            # Break if max-number of iterations is reached
            if iteration == iters:
                bar.close()
                break

#------------------------------------------------------------------------------------------------------------#

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
    for context, train_dataset in enumerate(train_datasets, 1):

        # If using the "joint" baseline, skip to last context, as model is only be trained once on data of all contexts
        if baseline=='joint':
            if context<len(train_datasets):
                continue
            else:
                baseline = "cummulative"

        # If using the "cummulative" baseline, create a large training dataset of all contexts so far
        if baseline=="cummulative":
            train_dataset = ConcatDataset(train_datasets[:context])

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1

        # Define tqdm progress bar(s)
        progress = tqdm.tqdm(range(1, iters+1))

        for batch_index in range(1, iters):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(DataLoader(train_dataset, batch_size, Shuffle = True, drop_last=True,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {}))())
                # NOTE:  [train_dataset]  is training-set of current context
                #      [training_dataset] is training-set of current context with stored samples added (if requested)
                iters_left = len(data_loader)

            # -----------------Collect data------------------#
                x, y = next(data_loader)                  
                x, y = x.to(device), y.to(device)             

            x_ = y_ = scores_ = context_used = None   #-> if no replay

            #---> Train MAIN MODEL
            if batch_index <= iters:

                #Setup active classes
                active_classes = None

                # Train the main model with this batch
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=None, scores_=scores_, rnt = 1./context,
                                                contexts_=context_used, active_classes=active_classes, context=context)

                # Fire callbacks (for visualization of training-progress / evaluating performance after each context)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, context=context)
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, context=context)
                if model.label == "VAE":
                    for sample_cb in sample_cbs:
                        if sample_cb is not None:
                            sample_cb(model, batch_index, context=context)


        ##----------> UPON FINISHING EACH CONTEXT...

        # Close progres-bar(s)
        progress.close()

        # Parameter regularization: update and compute the parameter importance estimates
        if context<len(train_datasets) and isinstance(model, EWCModel):
            # -find allowed classes
            allowed_classes = active_classes
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

