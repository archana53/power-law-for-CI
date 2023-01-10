import math
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
from torchvision import datasets,transforms
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset



def permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].
    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation[1]]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image

class TransformedDataset(Dataset):
    '''To modify an existing dataset with a transform.
    This is useful for creating different permutations of MNIST without loading the data multiple times.'''

    def __init__(self, original_dataset, transform=None, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        (input, target) = self.dataset[index]
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return (input, target)

def permute_train_test_data(mnist_trainset, mnist_testset, seed) :
        # get train and test datasets
        # generate pixel-permutations
        np.random.seed(seed)
        permutations = [np.random.permutation(32**2) for _ in range(10)]
        # specify transformed datasets per context
        train_datasets = []
        test_datasets = []
        for perm in enumerate(permutations):
            target_transform = None
            train_datasets.append(TransformedDataset(
                mnist_trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                mnist_testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))

        return train_datasets, test_datasets

def get_phase_task_frequency_matrix(
                                    replay_fraction,
                                    replay_distribution,
                                    MAX_EPISODE_SAMPLES,
                                    MIN_EPISODE_SAMPLES,
                                    NUM_TASKS,
                                    hyperparameters):
  if replay_fraction <= 0 or replay_fraction > 1:
    print("Invalid replay fraction: "+str(replay_fraction))
    return None
  replay_cnt = math.floor(MAX_EPISODE_SAMPLES*replay_fraction) 
  phase_task_frequency_matrix = np.diag(np.repeat(MAX_EPISODE_SAMPLES, NUM_TASKS))
  if replay_distribution == "uniform":
    for i in range(1, NUM_TASKS):
      task_freq = replay_cnt//i
      for j in range(i):
        phase_task_frequency_matrix[j,i]+=task_freq
    return phase_task_frequency_matrix
  elif replay_distribution == "powerlaw":
    if "alpha" not in hyperparameters.keys():
      print("Missing Hyperparameter: alpha")
      return None
    alpha = hyperparameters["alpha"]
    if alpha <= 2:
      print("Invalid Hyperparameter: alpha")
      return None
    not_enough_replay_count = False
    for i in range(1, NUM_TASKS):
      sample_positions = np.arange(10, 10 + (i)) # Hyperparameterizable for sampling?
      sample_positions=0.01*sample_positions
      sample_points = alpha*np.power(sample_positions, alpha - 1)
      sample_sum = np.sum(sample_points)
      rescaled_samples = (replay_cnt//sample_sum)*sample_points
      # print("column: "+str(i))
      # print(np.sum(rescaled_samples))
      # print((sample_points))
      # print((rescaled_samples))
      for j in range(i):
        phase_task_frequency_matrix[j,i]+=rescaled_samples[j]
        if rescaled_samples[i-j-1] < MIN_EPISODE_SAMPLES:
          not_enough_replay_count = True
    if not_enough_replay_count:
      print("WARNING: Replay data set size not enough to ensure existence of all previous tasks.")
    return phase_task_frequency_matrix
  elif replay_distribution == "exponential":
    if "beta" not in hyperparameters.keys():
      print("Missing Hyperparameter: beta")
      return None
    beta = hyperparameters["beta"]
    if beta <= 0:
      print("Invalid Hyperparameter: beta")
      return None
    not_enough_replay_count = False
    for i in range(1, NUM_TASKS):
      sample_positions = np.arange(1,i+1) # Hyperparameterizable for sampling?
      sample_points = np.exp(-sample_positions/beta)/beta
      sample_sum = np.sum(sample_points)
      rescaled_samples = (replay_cnt//sample_sum)*sample_points
      for j in range(i):
        phase_task_frequency_matrix[j,i]+=rescaled_samples[i-j-1]
        if rescaled_samples[i-j-1] < MIN_EPISODE_SAMPLES:
          not_enough_replay_count = True
    if not_enough_replay_count:
      print("WARNING: Replay data set size not enough to ensure existence of all previous tasks.")
    return phase_task_frequency_matrix
  elif replay_distribution == "lower_baseline" or replay_distribution =="EWC" :
    return phase_task_frequency_matrix
  elif replay_distribution == "upper_baseline" :
        for i in range(1, NUM_TASKS):
            for j in range(i):
                phase_task_frequency_matrix[j,i] = MAX_EPISODE_SAMPLES
  else:
    print("Invalid Distribution")
    return None


def get_training_environment(  replay_fraction,
                                    replay_distribution,
                                    MAX_EPISODE_SAMPLES, 
                                    MIN_EPISODE_SAMPLES,
                                    NUM_TASKS,
                                    seed,
                                    hyperparameters):
  dataset_tranforms = transforms.Compose([
          transforms.Pad(2),
          transforms.ToTensor(),
      ]
  )

  mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=dataset_tranforms)
  mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=dataset_tranforms)
  
  training_permutations , test_permutations = permute_train_test_data((mnist_trainset),
                                                                      (mnist_testset), seed)

  # phase_task_freq_count = get_phase_task_frequency_matrix(REPLAY_FRACTION, "powerlaw",MAX_TASK_EXAMPLES, MIN_TASK_EXAMPLES, NUM_TASKS, {"alpha": 5, "beta": 1})
  phase_task_freq_count = get_phase_task_frequency_matrix(replay_fraction, replay_distribution, MAX_EPISODE_SAMPLES,  MIN_EPISODE_SAMPLES, NUM_TASKS, hyperparameters)

  training_environments = []
  np.random.seed(seed)
  for phase in range(NUM_TASKS): 
      phase_training = []
      for task in range(NUM_TASKS):
          num_ele_to_pick = phase_task_freq_count[task,phase]
          selected_indices = np.random.choice(np.arange(MAX_EPISODE_SAMPLES), int(num_ele_to_pick), replace=False)
          task_subset = torch.utils.data.Subset(training_permutations[task], selected_indices)
          phase_training.append(task_subset)
      final_phase_dataset = ConcatDataset(phase_training)
      training_environments.append(final_phase_dataset)

  return training_environments, test_permutations
