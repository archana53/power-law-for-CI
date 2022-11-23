import abc
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from models.fc.layers import fc_layer
from models.fc.nets import MLP
from models.conv.nets import ConvLayers
from models.utils import loss_functions as lf, modules
from models.utils.ncl import additive_nearest_kf
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm


class EWCModel(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        super().__init__()
        # List with the methods to create generators that return the parameters on which to apply param regularization
        self.param_list = [self.named_parameters]  #-> lists the parameters to regularize with SI or diagonal Fisher
                                                   #   (default is to apply it to all parameters of the network)
        # Optimizer (and whether it needs to be reset)
        
        self.optim_type = "adam"
        #--> self.[optim_type]   <str> name of optimizer, relevant if optimizer should be reset for every context
        self.optim_list = []
        #--> self.[optim_list]   <list>, if optimizer should be reset after each context, provide list of required <dicts>


        # Parameter-regularization
        self.weight_penalty = True
        self.reg_strength = 0       #-> hyperparam: how strong to weigh the weight penalty ("regularisation strength")
        self.precondition = False
        self.alpha = 1e-10          #-> small constant to stabilize inversion of the Fisher Information Matrix
                                    #   (this is used as hyperparameter in OWM)
        self.importance_weighting = 'fisher'  #-> Options for estimation of parameter importance:
                                              #   - 'fisher':   Fisher Information matrix (e.g., as in EWC, NCL)
                                              #   - 'si':       ... diagonal, online importance estimation ...
                                              #   - 'owm':      ...
        self.fisher_kfac = False    #-> whether to use a block-diagonal KFAC approximation to the Fisher Information
                                    #   (alternative is a diagonal approximation)
        self.fisher_n = None        #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.fisher_labels = "all"  #-> what label(s) to use for any given sample when calculating the FI matrix?
                                    #   - 'all':    use all labels, weighted according to their predicted probabilities
                                    #   - 'sample': sample one label to use, using predicted probabilities for sampling
                                    #   - 'pred':   use the predicted label (i.e., the one with highest predicted prob)
                                    #   - 'true':   use the true label (NOTE: this is also called "empirical FI")
        self.fisher_batch = 1       #-> batch size for estimating FI-matrix (should be 1, for best results)
                                    #   (different from 1 only works if [fisher_labels]='pred' or 'true')
        self.context_count = 0      #-> counts 'contexts' (if a prior is used, this is counted as the first context)
        self.data_size = None       #-> inverse prior (can be set to # samples per context, or used as hyperparameter)



        self.offline = False        #-> use separate penalty term per context (as in original EWC paper)
        self.gamma = 1.             #-> decay-term for old contexts' contribution to cummulative FI (as in 'Online EWC')


    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


    def initialize_fisher(self):
        '''Initialize diagonal fisher matrix with the prior precision (as in NCL).'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__') 
                    # -take initial parameters as zero for regularization purposes
                    self.register_buffer('{}_EWC_prev_context'.format(n), p.detach().clone()*0)
                    # -precision (approximated by diagonal Fisher Information matrix)
                    self.register_buffer( '{}_EWC_estimated_fisher'.format(n), torch.ones(p.shape) / self.data_size)

    def estimate_fisher(self, dataset, allowed_classes=None):
        '''After completing training on a context, estimate diagonal of Fisher Information matrix.
        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        print("fisher estimation start")
        # Create data-loader to give batches of size 1 (unless specifically asked to do otherwise)

        data_loader = DataLoader(dataset, batch_size=1 if self.fisher_batch==1 else self.fisher_batch)

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index,(x,y) in enumerate(tqdm(data_loader, desc = 'Fisher Estimation')):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            # calculate FI-matrix (according to one of the four options)
            if self.fisher_labels=='all':
                # -use a weighted combination of all labels
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)  # --> get weights, which shouldn't have gradient tracked
                for label_index in range(output.shape[1]):
                    label = torch.LongTensor([label_index]).to(self._device())
                    negloglikelihood = F.cross_entropy(output, label)  #--> get neg log-likelihoods for this class
                    # Calculate gradient of negative loglikelihood
                    self.zero_grad()
                    negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
                    # Square gradients and keep running sum (using the weights)
                    for gen_params in self.param_list:
                        for n, p in gen_params():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
            else:
                # -only use one particular label for each datapoint
                if self.fisher_labels=='true':
                    # --> use provided true label to calculate loglikelihood --> "empirical Fisher":
                    label = torch.LongTensor([y]) if type(y)==int else y  #-> shape: [self.fisher_batch]
                    if allowed_classes is not None:
                        label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                        label = torch.LongTensor(label)
                    label = label.to(self._device())
                elif self.fisher_labels=='pred':
                    # --> use predicted label to calculate loglikelihood:
                    label = output.max(1)[1]
                elif self.fisher_labels=='sample':
                    # --> sample one label from predicted probabilities
                    with torch.no_grad():
                        label_weights = F.softmax(output, dim=1)       #--> get predicted probabilities
                    weights_array = np.array(label_weights[0].cpu())   #--> change to np-array, avoiding rounding errors
                    label = np.random.choice(len(weights_array), 1, p=weights_array/weights_array.sum())
                    label = torch.LongTensor(label).to(self._device()) #--> change label to tensor on correct device
                # calculate negative log-likelihood
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
                # calculate gradient of negative loglikelihood
                self.zero_grad()
                negloglikelihood.backward()
                # square gradients and keep running sum
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    # -mode (=MAP parameter estimate)
                    self.register_buffer('{}_EWC_prev_context{}'.format(n, self.context_count+1 if self.offline else ""),
                                         p.detach().clone())
                    # -precision (approximated by diagonal Fisher Information matrix)
                    if (not self.offline) and hasattr(self, '{}_EWC_estimated_fisher'.format(n)):
                        existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                        est_fisher_info[n] += self.gamma * existing_values
                    self.register_buffer(
                        '{}_EWC_estimated_fisher{}'.format(n, self.context_count+1 if self.offline else ""), est_fisher_info[n]
                    )

        # Increase context-count
        self.context_count += 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        try:
            losses = []
            # If "offline EWC", loop over all previous contexts as each context has separate penalty term
            num_penalty_terms = self.context_count if (self.offline and self.context_count>0) else 1
            for context in range(1, num_penalty_terms+1):
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                            n = n.replace('.', '__')
                            mean = getattr(self, '{}_EWC_prev_context{}'.format(n, context if self.offline else ""))
                            fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, context if self.offline else ""))
                            # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                            fisher = fisher if self.offline else self.gamma*fisher
                            # Calculate weight regularization loss
                            losses.append((fisher * (p-mean)**2).sum())
            # Sum the regularization loss from all parameters (and from all contexts, if "offline EWC")
            return (1./2)*sum(losses)
        except AttributeError:
            # Regularization loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())

class Classifier(EWCModel):
    '''Model for classifying images, using the EWC Continual Learning Module'''

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False, phantom=False):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.depth = depth
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")


        ######------SPECIFY MODEL------######
        #--> convolutional layers
        self.convE = ConvLayers(
            conv_type=conv_type, block_type="basic", num_blocks=num_blocks, image_channels=image_channels,
            depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
            global_pooling=global_pooling, gated=conv_gated, output="none" if no_fnl else "normal",
        )
        self.flatten = modules.Flatten()  # flatten image to 2D-tensor
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        #------------------------------------------------------------------------------------------#
        #--> fully connected hidden layers
        self.fcE = MLP(input_size=image_size ,output_size=fc_units, layers=fc_layers-1,
                       hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                       excitability=excitability, excit_buffer=excit_buffer, gated=fc_gated, phantom=phantom)
        mlp_output_size = fc_units if fc_layers>1 else self.conv_out_units
        #--> classifier
        self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none', drop=fc_drop)

        # Flags whether parts of the network are frozen (so they can be set to evaluation mode during training)
        self.convE.frozen = False
        self.fcE.frozen = False

        optim_list = [{'params': list(filter(lambda p: p.requires_grad, self.parameters())),
                                        'lr': 1e-3}]
        self.optimizer = optim.Adam(optim_list, betas=(0.9, 0.999))


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.depth>0 and self.fc_layers>1:
            return "{}_{}_c{}".format(self.convE.name, self.fcE.name, self.classes)
        elif self.depth>0:
            return "{}_{}c{}".format(self.convE.name, "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                     self.classes)
        elif self.fc_layers>1:
            return "{}_c{}".format(self.fcE.name, self.classes)
        else:
            return "i{}_{}c{}".format(self.conv_out_units, "drop{}-".format(self.fc_drop) if self.fc_drop>0 else "",
                                      self.classes)


    def forward(self, x, return_intermediate=False):
        hidden = self.convE(x)
        flatten_x = self.flatten(hidden)
        if not return_intermediate:
            final_features = self.fcE(flatten_x)
        else:
            final_features, intermediate = self.fcE(flatten_x, return_intermediate=True)
            intermediate["classifier"] = final_features
        out = self.classifier(final_features)
        return (out, intermediate) if return_intermediate else out


    def feature_extractor(self, images):
        return self.fcE(self.flatten(self.convE(images)))

    def classify(self, x, allowed_classes=None, no_prototypes=False):
        '''For input [x] (image/"intermediate" features), return predicted "scores"/"logits" for [allowed_classes].'''
        if self.prototypes and not no_prototypes:
            return self.classify_with_prototypes(x, allowed_classes=allowed_classes)
        else:
            image_features = self.flatten(self.convE(x))
            hE = self.fcE(image_features)
            scores = self.classifier(hE)
            return scores if (allowed_classes is None) else scores[:, allowed_classes]


    def train_a_batch(self, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, context=1,
                      **kwargs):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).
        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]
                            NOTE: only to be used for "BCE with distill" (only when scenario=="class")
        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new context
        [active_classes]  None or (<list> of) <list> with "active" classes
        [context]         <int> context-ID, with first context labelled as '1' 
        '''

        # Set model to training-mode
        self.train()
        # -however, if some layers are frozen, they should be set to eval() to prevent batch-norm layers from changing
        if self.convE.frozen:
            self.convE.eval()
        if self.fcE.frozen:
            self.fcE.eval()

        # Reset optimizer
        ##--(2)-- CURRENT DATA --##
        loss_total  = 0

        if x is not None:
            # Run model
            y_hat = self(x)
            predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

            # Weigh losses
            loss_cur = predL
            loss_total += loss_cur
            # Calculate training-accuracy
            accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
        ##--(3)-- PARAMETER REGULARIZATION LOSSES --##

        # Add a parameter regularization penalty to the loss function
        weight_penalty_loss = None
        if self.weight_penalty:
            if self.importance_weighting=='si':
                weight_penalty_loss = self.surrogate_loss()
            elif self.importance_weighting=='fisher':
                if self.fisher_kfac:
                    weight_penalty_loss = self.ewc_kfac_loss()
                else:
                    weight_penalty_loss = self.ewc_loss()
            loss_total += self.reg_strength * weight_penalty_loss


        ##--(4)-- COMPUTE (AND MANIPULATE) GRADIENTS --##

        # Backpropagate errors (for the part of the loss that has not yet been backpropagated)
        loss_total.backward()

        # Precondition gradient of current data using projection matrix constructed from parameter importance estimates
        if self.precondition:

            if self.importance_weighting=='fisher' and not self.fisher_kfac:
                #--> scale gradients by inverse diagonal Fisher
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            # Retrieve prior fisher matrix
                            n = n.replace(".", "__")
                            fisher = getattr(self, "{}_EWC_estimated_fisher{}".format(n, "" if self.online else context))
                            # Scale loss landscape by inverse prior fisher and divide learning rate by data size
                            scale = (fisher + self.alpha**2) ** (-1)
                            p.grad *= scale  # scale lr by inverse prior information
                            p.grad /= self.data_size  # scale lr by prior (necessary for stability in 1st context)

            elif self.importance_weighting=='fisher' and self.fisher_kfac:
                #--> scale gradients by inverse Fisher kronecker factors
                def scale_grad(label, layer):
                    assert isinstance(layer, fc_layer)
                    info = self.KFAC_FISHER_INFO[label]  # get previous KFAC fisher
                    A = info["A"].to(self._device())
                    G = info["G"].to(self._device())
                    linear = layer.linear
                    if linear.bias is not None:
                        g = torch.cat( (linear.weight.grad, linear.bias.grad[..., None]), -1).clone()
                    else:
                        g = layer.linear.weight.grad.clone()

                    assert g.shape[-1] == A.shape[-1]
                    assert g.shape[-2] == G.shape[-2]
                    iA = torch.eye(A.shape[0]).to(self._device()) * (self.alpha)
                    iG = torch.eye(G.shape[0]).to(self._device()) * (self.alpha)

                    As, Gs = additive_nearest_kf({"A": A, "G": G}, {"A": iA, "G": iG})  # kronecker sums
                    Ainv = torch.inverse(As)
                    Ginv = torch.inverse(Gs)

                    scaled_g = Ginv @ g @ Ainv
                    if linear.bias is not None:
                        linear.weight.grad = scaled_g[..., 0:-1].detach() / self.data_size
                        linear.bias.grad = scaled_g[..., -1].detach() / self.data_size
                    else:
                        linear.weight.grad = scaled_g[..., 0:-1, :] / self.data_size

                    # make sure to reset all phantom to have no zeros
                    if not hasattr(layer, 'phantom'):
                        raise ValueError(f"Layer {label} does not have phantom parameters")
                    # make sure phantom stays zero
                    layer.phantom.grad.zero_()
                    layer.phantom.data.zero_()

                scale_grad("classifier", self.classifier)
                for i in range(1, self.fcE.layers + 1):
                    label = f"fcLayer{i}"
                    scale_grad(label, getattr(self.fcE, label))


        ##--(5)-- TAKE THE OPTIMIZATION STEP --##
        self.optimizer.step()


        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'param_reg': weight_penalty_loss.item() if weight_penalty_loss is not None else 0,
            'accuracy': accuracy if accuracy is not None else 0
        }