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
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image

#----------------------------------------------------------------------------------------------------------#

class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].
    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class MemorySetDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).
    The images at the i-th entry of [memory_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, memory_sets, target_transform=None):
        super().__init__()
        self.memory_sets = memory_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.memory_sets)):
            total += len(self.memory_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.memory_sets)):
            examples_in_this_class = len(self.memory_sets[class_id])
            if index < (total + examples_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                example_id = index - total
                break
            else:
                total += examples_in_this_class
        image = torch.from_numpy(self.memory_sets[class_id][example_id])
        return (image, class_id_to_return)




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

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Denormalize image, either single image (C,H,W) or image batch (N,C,H,W)"""
        batch = (len(tensor.size()) == 4)
        for t, m, s in zip(tensor.permute(1, 0, 2, 3) if batch else tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return 
        


if name == 'permMNIST':
        # get train and test datasets
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=None, verbose=verbose)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=None, verbose=verbose)
        # generate pixel-permutations
        if exception:
            permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(contexts-1)]
        else:
            permutations = [np.random.permutation(config['size']**2) for _ in range(contexts)]
        # specify transformed datasets per context
        train_datasets = []
        test_datasets = []
        for context_id, perm in enumerate(permutations):
            target_transform = transforms.Lambda(
                lambda y, x=context_id: y + x*classes_per_context
            ) if scenario in ('task', 'class') and not (scenario=='task' and singlehead) else None
            train_datasets.append(TransformedDataset(
                trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
    else:
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))
        # prepare train and test datasets with all classes
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=target_transform,
                               verbose=verbose, augment=augment, normalize=normalize)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=target_transform, verbose=verbose,
                              augment=augment, normalize=normalize)
        # generate labels-per-dataset (if requested, training data is split up per class rather than per context)
        labels_per_dataset_train = [[label] for label in range(classes)] if train_set_per_class else [
            list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
        ]
        labels_per_dataset_test = [
            list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
        ]
        # split the train and test datasets up into sub-datasets
        train_datasets = []
        for labels in labels_per_dataset_train:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
                    scenario=='domain' or (scenario=='task' and singlehead)
            ) else None
            train_datasets.append(SubDataset(trainset, labels, target_transform=target_transform))
        test_datasets = []
        for labels in labels_per_dataset_test:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
                    scenario=='domain' or (scenario=='task' and singlehead)
            ) else None
            test_datasets.append(SubDataset(testset, labels, target_transform=target_transform))


def getData():
    mnist_trainset = datasets.MNIST32(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST32(root='./data', train=False, download=True, transform=None)
