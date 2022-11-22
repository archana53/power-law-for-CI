import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import Dataset
from config import PERMUTATIONS,TASK_CONFIGURATIONS


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

def no_permute_train_test_data(mnist_trainset, mnist_testset) :
        # get train and test datasets
        # generate pixel-permutations
        permutations = [np.arange(32*32)]
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

