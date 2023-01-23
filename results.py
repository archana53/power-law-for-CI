import numpy as np
import torch
from run import get_model
from environments import get_test_permutations


if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

BASELINE_DISTRIBUTIONS = ['upper_baseline','lower_baseline','EWC','powerlaw_unrestricted']
SEEDS = [65,49]
ACCURACIES = {}
for seed in SEEDS:
    ACCURACIES[seed] = {} 
    for dist in BASELINE_DISTRIBUTIONS:
        ACCURACIES[seed][dist] = {}
        test_data = get_test_permutations(seed)
        for context in range(1,11):
            model = get_model()
            MODEL_PATH = f'./checkpoint/model_{env}_context_{context}.pt'
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))['model_state_dict'], strict = False)
            model.to(torch.device(device))
            _, test =  get_test_permutations(seed)
            correct = 0
            total  = 0
            for i, test_set in enumerate(tqdm(test),1):
                if(i <= ii):
                    test_loader = DataLoader(test_set,batch_size=1024, shuffle=True)
                    for batch_idx, (data, y) in enumerate(tqdm(test_loader)):
                            data, y = data.to(device), y.to(device)
                            y_hat = model(data)
                            correct += (y == y_hat.max(1)[1]).sum().item()
                            total += data.size(0)
            ACCURACIES[seed][dist][context] = correct/total

#Add code to save ACCURACIES dictionary