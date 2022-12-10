#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import math
# get_ipython().run_line_magic('matplotlib', 'inline')


from torchvision import datasets,transforms


# ## Data_loading module

# In[5]:


import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import Dataset


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

def permute_train_test_data(mnist_trainset, mnist_testset) :
        # get train and test datasets
        # generate pixel-permutations
        np.random.seed(10)
        permutations = [np.random.permutation(32**2) for _ in range(10)]
        print(permutations)
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

def get_phase_task_frequency_matrix(new_task_cnt, 
                                    replay_fraction,
                                    replay_distribution,
                                    min_example,
                                    phase_cnt,
                                    hyperparameters):
  if replay_fraction <= 0 or replay_fraction > 1:
    print("Invalid replay fraction: "+str(replay_fraction))
    return None
  replay_cnt = math.floor(new_task_cnt*replay_fraction) 
  phase_task_frequency_matrix = np.diag(np.repeat(new_task_cnt, phase_cnt))
  if replay_distribution == "uniform":
    for i in range(1, phase_cnt):
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
    for i in range(1, phase_cnt):
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
        if rescaled_samples[i-j-1] < min_example:
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
    for i in range(1, phase_cnt):
      sample_positions = np.arange(1,i+1) # Hyperparameterizable for sampling?
      sample_points = np.exp(-sample_positions/beta)/beta
      sample_sum = np.sum(sample_points)
      rescaled_samples = (replay_cnt//sample_sum)*sample_points
      for j in range(i):
        phase_task_frequency_matrix[j,i]+=rescaled_samples[i-j-1]
        if rescaled_samples[i-j-1] < min_example:
          not_enough_replay_count = True
    if not_enough_replay_count:
      print("WARNING: Replay data set size not enough to ensure existence of all previous tasks.")
    return phase_task_frequency_matrix
  else:
    print("Invalid Distribution")
    return None
  
  return phase_task_frequency_matrix


from torch.utils.data import ConcatDataset
import torch

def get_all_experimental_training_environments(new_task_cnt, 
                                    replay_fraction,
                                    replay_distribution,
                                    min_example,
                                    phase_cnt,
                                    hyperparameters):
  dataset_tranforms = transforms.Compose([
          transforms.Pad(2),
          transforms.ToTensor(),
      ]
  )


  # In[7]:


  mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=dataset_tranforms)
  mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=dataset_tranforms)


  # In[8]:


  training_permutations , test_permutations = permute_train_test_data((mnist_trainset),
                                                                      (mnist_testset))


  # ## Power Law Sampling 

  # In[9]:


  power_law_sampling = [57927,
  44078,
  33969,
  26483,
  20867,
  16603,
  13329,
  10790,
  8802,
  7232,
  5981,
  4978,
  4167,
  3508,
  2968,
  2523,
  2155,
  1849,
  1592,
  1377,
  1195]
  MAX_TASK_EXAMPLES = 57927
  MIN_TASK_EXAMPLES = 1195
  NUM_TASKS = 10

  DOWNSAMPLE = 1
  power_law_sampling[:] = [x//DOWNSAMPLE for x in power_law_sampling]
  MAX_TASK_EXAMPLES//=DOWNSAMPLE
  MIN_TASK_EXAMPLES//=DOWNSAMPLE


  # ## Skip Alternate Policy

  # In[12]:


  # freq_vector = []
  # for idx in range(0,20,2):
  #     freq_vector.append(power_law_sampling[idx])
  # freq_vector


  # In[13]:


  # phase_task_freq_count = np.zeros((NUM_TASKS,NUM_TASKS)) 
  # for row in range(NUM_TASKS):
  #     phase_task_freq_count[row,:]+=freq_vector
  #     freq_vector.insert(0, 0)
  #     freq_vector.pop()


  # In[14]:

  REPLAY_FRACTION = 0.5
  # phase_task_freq_count = get_phase_task_frequency_matrix(MAX_TASK_EXAMPLES, REPLAY_FRACTION, "powerlaw", MIN_TASK_EXAMPLES, NUM_TASKS, {"alpha": 5, "beta": 1})
  phase_task_freq_count = get_phase_task_frequency_matrix(new_task_cnt, replay_fraction, replay_distribution, min_example, phase_cnt, hyperparameters)

  # ### Creating Unified Testing Dataset for each phase

  # In[16]:



  # ### Power Law Distirbuted Training Environment

  # In[17]:


  training_environments = []
  np.random.seed(10)
  MAX_TASK_EXAMPLES = 60000
  # NUM_PHASES = NUM_TASKS
  for phase in range(NUM_TASKS): 
      phase_training = []
      for task in range(NUM_TASKS):
          num_ele_to_pick = phase_task_freq_count[task,phase]
          # print("phase: "+str(phase)+" task: "+str(task)+" #ele: "+str(num_ele_to_pick))
          selected_indices = np.random.choice(np.arange(MAX_TASK_EXAMPLES), int(num_ele_to_pick), replace=False)
          task_subset = torch.utils.data.Subset(training_permutations[task], selected_indices)
          phase_training.append(task_subset)
      final_phase_dataset = ConcatDataset(phase_training)
      training_environments.append(final_phase_dataset)


  # In[18]:


  print(len(training_environments))
  for i in range(1,NUM_TASKS+1):
    print("Training environment size for task ", i , "is :", len(training_environments[i-1]))


  # ### SGD (Lower Baseline) Training Environment

  # In[25]:
  SGD_training_environments = []
  num_ele_to_pick = phase_task_freq_count[0,0]
  for i in range(NUM_TASKS):
    selected_indices = np.random.choice(np.arange(MAX_TASK_EXAMPLES), int(num_ele_to_pick), replace=False)
    task_subset = torch.utils.data.Subset(training_permutations[i], selected_indices)
    SGD_training_environments.append(task_subset)


  # In[26]:


  print(len(SGD_training_environments))
  for i in range(1,NUM_TASKS+1):
    print("Training environment size for task ", i , "is :", len(SGD_training_environments[i-1]))


  # ### Upper Baseline Training Environment

  # In[27]:


  UBL_training_environments = []
  np.random.seed(10)
  MAX_TASK_EXAMPLES = 60000
  num_ele_to_pick = phase_task_freq_count[0,0]
  # NUM_PHASES = NUM_TASKS
  phase_training = []
  for phase in range(NUM_TASKS): # only 1 phase is enough for upper baseline
        selected_indices = np.random.choice(np.arange(MAX_TASK_EXAMPLES), int(num_ele_to_pick), replace=False)
        task_subset = torch.utils.data.Subset(training_permutations[phase], selected_indices)
        phase_training.append(task_subset)
        final_phase_dataset = ConcatDataset(phase_training)
        UBL_training_environments.append(final_phase_dataset)



  print(len(UBL_training_environments))
  for i in range(1,len(UBL_training_environments)+1):
    print("Training environment size for task ", i , "is :", len(UBL_training_environments[i-1]))

  return training_environments, SGD_training_environments, UBL_training_environments, test_permutations

# get_all_training_environments()