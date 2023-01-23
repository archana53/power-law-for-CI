from environments import get_training_environment
from EWC_model import *
from train import *
import sys

def get_model(device):
    model = Classifier(
        image_size=32*32,
        image_channels=1,
        classes=10,
        # -fc-layers
        fc_units=1000,
        fc_drop=0.5,
        fc_layers=4,
        fc_bn=True

    )
    model.to(device)
    model.importance_weighting == 'fisher'
    model.precondition = False
    model.fisher_n = None
    model.fisher_labels = 'all'
    model.fisher_batch = 1
    # -options relating to 'Offline EWC' (Kirkpatrick et al., 2017) and 'Online EWC' (Schwarz et al., 2018)
    model.offline = True
    model.weight_penalty = False
    model.reg_strength = 500
    model.to(device)
    return model

BASELINE_DISTRIBUTIONS = ['upper_baseline','lower_baseline','EWC','powerlaw_unrestricted']
MAX_TASK_EXAMPLES = 57927
MIN_TASK_EXAMPLES = 1195
NUM_TASKS = 10
replay_values = [1,0.75,0.5,0.25,0.1]

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

distribution = sys.argv[1]
seed = int(sys.argv[2])
model = get_model(device)



if distribution in BASELINE_DISTRIBUTIONS:
    train_env, test = get_training_environment(replay_fraction = None, replay_distribution = distribution, MAX_EPISODE_SAMPLES=MAX_TASK_EXAMPLES, MIN_EPISODE_SAMPLES=MIN_TASK_EXAMPLES,NUM_TASKS=NUM_TASKS,seed=seed, hyperparameters={"alpha": 5, "beta": 1})
    if distribution == 'EWC' :
        model.weight_penalty = True
        train_EWC(model, train_datasets = train_env, test_datasets = test, iters=100, batch_size=2048, continue_from_context = 1, label=distribution+ '_seed = ' +seed)
    else :
        train_cl(model, train_datasets = train_env, test_datasets = test, iters=100, batch_size=2048, continue_from_context = 1, label=distribution+ '_seed = ' +seed)

else:
    for r in replay_values:
        train_env , test = get_training_environment(replay_fraction = r, replay_distribution = distribution, MAX_EPISODE_SAMPLES = MAX_TASK_EXAMPLES, MIN_EPISODE_SAMPLES = MIN_TASK_EXAMPLES, NUM_TASKS =  NUM_TASKS, seed= seed, hyperparameters={"alpha": 5, "beta": 1})
        train_cl(model, train_datasets = train_env, test_datasets = test, iters=100, batch_size=2048, continue_from_context = 1, label=distribution+"_seed = " + seed + "_r = "+str(r))   

     
   