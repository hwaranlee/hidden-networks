import os
import torch
import torchvision
from torchvision import transforms
import random
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from args import args

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

class Cub200_dataset(Dataset):
    # Modified from https://github.com/TDeVries/cub2011_dataset
    base_folder = 'images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train, transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(os.path.join(root, "CUB_200_2011"))
        self.transform = transform
        self.loader = loader
        self.is_train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        print(self.root)
        images = pd.read_csv(os.path.join(self.root, 'images.txt'),
                             sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.is_train:
            self.data = self.data[self.data.is_training_img == 1]
            self.target = self.data[self.data.is_training_img == 1].target - 1
        else:
            self.data = self.data[self.data.is_training_img == 0]
            self.target = self.data[self.data.is_training_img == 0].target - 1

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, os.path.join(self.root, ".."), self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, '..', self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CUB200:
    def __init__(self, args):
        super(CUB200, self).__init__()

        data_root = os.path.join(args.data, "cub200")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        # Train data
        transform=transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize
                               ])

        train_dataset = Cub200_dataset(data_root, train=True, download=True, transform = transform)
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        # Test data
        transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
                               ])
        test_dataset = Cub200_dataset(data_root, train=False, download=False, transform = transform)

        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
