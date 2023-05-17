import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels




class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def all_coords(size = 15):
    x, y = np.meshgrid(np.arange(-size+1, size), np.arange(-size+1, size))
    all_coords = np.column_stack([x.ravel(), y.ravel()])
    np.random.shuffle(all_coords)
    return all_coords


def get_coord_hd(set_of_points):
    head_directions = np.arange(4)

    coord_hd = []
    for coord in set_of_points:
        for hd in head_directions:
            coord_hd.append(np.append(coord, hd))
    return np.array(coord_hd)

def split_square_coords(size, split_proportion=0.8):
    """
    This function splits square coordinates into a training and test set based on a specified split proportion.
    
    Parameters:
    size (int): The size of the square with x,y coordinates ranging from -size to size.
    split_proportion (float, optional): The proportion of coordinates to be included in the training set. The default is 0.8.
    
    Returns:
    tuple: A tuple of two arrays, the first array is the training set and the second is the test set.
    
    """
    # Create the array of square coordinates
    coords = np.array([(x, y) for x in range(-size+1, size) for y in range(-size+1, size)])
    
    # Calculate the number of training and test samples
    n_samples = len(coords)
    n_train = int(n_samples * split_proportion)
    n_test = n_samples - n_train
    
    # Randomly choose training samples from the square coordinates
    train_indices = np.random.choice(n_samples, size=n_train, replace=False)
    train_coords = coords[train_indices]
    
    # The test set is the remaining samples
    test_indices = np.array([i for i in range(n_samples) if i not in train_indices])
    test_coords = coords[test_indices]
    
    return (train_coords, test_coords)

class iWaterMaze(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = []

    class_order = np.arange(5).tolist()

    def download_data(self):
        

        all_train_dataset = None #np.array([])
        all_test_dataset = None #np.array([])

        all_train_labels = np.array([])
        all_test_labels = np.array([])

        for i in np.arange(5): #5 different environments #HYPERPARAM
            train_dataset = all_coords(size = 15)
            train_dataset, test_dataset = split_square_coords(size = 15)
            train_dataset = get_coord_hd(train_dataset)[:1000]
            test_dataset = get_coord_hd(test_dataset)[:100]
            

            if all_train_dataset is None:
                all_train_dataset = train_dataset
            else:
                all_train_dataset = np.vstack((all_train_dataset, train_dataset))

            if all_test_dataset is None:
                all_test_dataset = test_dataset
            else:
                all_test_dataset = np.vstack((all_test_dataset, test_dataset))


            train_labels = [i for a in np.arange(len(train_dataset))]
            test_labels = [i for a in np.arange(len(test_dataset))]

            all_test_labels = np.append(all_test_labels, test_labels)
            all_train_labels = np.append(all_train_labels , train_labels)
        self.train_data = all_train_dataset
        self.train_targets = all_train_labels

        self.test_data = all_test_dataset
        self.test_targets = all_test_labels


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
