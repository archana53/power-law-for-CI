#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get Dataset

# In[2]:


import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist


# # Training Environment

# In[3]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0


# In[4]:


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


from torch.utils.data import ConcatDataset
import torch

def get_all_training_environments():
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

  DOWNSAMPLE = 6
  power_law_sampling[:] = [x//DOWNSAMPLE for x in power_law_sampling]
  MAX_TASK_EXAMPLES//=DOWNSAMPLE
  MIN_TASK_EXAMPLES//=DOWNSAMPLE


  # ## Skip Alternate Policy

  # In[12]:


  freq_vector = []
  for idx in range(0,20,2):
      freq_vector.append(power_law_sampling[idx])
  freq_vector


  # In[13]:


  phase_task_freq_count = np.zeros((NUM_TASKS,NUM_TASKS)) 
  for row in range(NUM_TASKS):
      phase_task_freq_count[row,:]+=freq_vector
      freq_vector.insert(0, 0)
      freq_vector.pop()


  # In[14]:


  phase_task_freq_count


  # In[15]:


  phase_task_freq_frac=phase_task_freq_count/np.sum(phase_task_freq_count, axis=0)
  phase_task_freq_frac


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
          selected_indices = np.random.choice([*range(int(MAX_TASK_EXAMPLES)+1)], int(num_ele_to_pick), replace=False)
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
  np.random.seed(10)
  MAX_TASK_EXAMPLES = 60000
  # NUM_PHASES = NUM_TASKS
  num_ele_to_pick = phase_task_freq_count[0,0] # to select only new task uniformly 
  for phase in range(NUM_TASKS): 
      phase_training = []
      selected_indices = np.random.choice([*range(int(MAX_TASK_EXAMPLES)+1)], int(num_ele_to_pick), replace=False)
      new_task = phase # only new task for each phase
      task_subset = torch.utils.data.Subset(training_permutations[new_task], selected_indices)
      phase_training.append(task_subset)
      final_phase_dataset = ConcatDataset(phase_training)
      SGD_training_environments.append(final_phase_dataset)


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
  for phase in range(1): # only 1 phase is enough for upper baseline
      phase_training = []
      for task in range(NUM_TASKS):
          selected_indices = np.random.choice([*range(int(MAX_TASK_EXAMPLES)+1)], int(num_ele_to_pick), replace=False)
          task_subset = torch.utils.data.Subset(training_permutations[task], selected_indices)
          phase_training.append(task_subset)
      final_phase_dataset = ConcatDataset(phase_training)
      UBL_training_environments.append(final_phase_dataset)


  # In[32]:


  print(len(UBL_training_environments))
  for i in range(len(UBL_training_environments)):
    print("Training environment size for task ", i , "is :", len(UBL_training_environments[i-1]))

  return training_environments, SGD_training_environments, UBL_training_environments