import argparse
import os
import sys
import math
import copy
import shutil
import pathlib
import torch
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from model import vgg19_bn, resnet50


def load_model(architecture, num_classes, prune_rate=0.0):
    if architecture == "vgg19":
        return vgg19_bn(num_classes=num_classes, prune_rate=prune_rate)
    elif architecture == "resnet50":
        return resnet50(num_classes=num_classes, prune_rate=prune_rate)
    else:
        raise ValueError(architecture + " architecture not supported.")

def load_state_dict(model, path):
	state_dict = torch.load(path)['state_dict']
	if hasattr(model, "module"):
		model.module.load_state_dict(state_dict)
	else:
		model.load_state_dict(state_dict)
	print("loaded successfully from [%s]" % path)


def load_dataset(dataset, batch_size = 512, is_train_split=True):
	"""
	Loads the dataset loader object

	Arguments
	---------
	dataset : Name of dataset which has to be loaded
	batch_size : Batch size to be used 
	is_train_split : Boolean which when true, indicates that training dataset will be loaded

	Returns
	-------
	Pytorch Dataloader object
	"""
	if is_train_split:
		if dataset == 'mnist':
			transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.MNIST(root='~/data/mnist', train=is_train_split, download=True, 
                                                  transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'cifar10':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=is_train_split, download=True, 
                                                    transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'cifar100':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR100(root='~/data/cifar100', train=is_train_split, download=True, 
                                                     transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'svhn':
			if not is_train_split:
				svhn_split = 'test'
			else:
				svhn_split = 'train'
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.SVHN(root='~/data/svhn', split=svhn_split , download=True, 
                                                 transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'fashionmnist':
			transform = transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.FashionMNIST(root='~/data/fmnist', train=is_train_split, download=True, 
                                                         transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=8)
		elif dataset == 'imagenet':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
			transform_augment = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
			data_set = torchvision.datasets.ImageFolder(root='~/data/imagenet/train', transform=transforms.Compose([transform_augment, transform]))
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=8)
		elif dataset == 'cifar10a' or dataset == 'cifar10b':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
			cifarset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=is_train_split, download=True, 
                                                    transform=transforms.Compose([transform_augment, transform]))
			label_flag = {x:True for x in range(10)}
			cifarA = []
			cifarB = []
			for sample in cifarset:
				if label_flag[sample[-1]]:
					cifarA.append(sample)
					label_flag[sample[-1]] = False
				else:
					cifarB.append(sample)
					label_flag[sample[-1]] = True
			class DividedCifar10A(torch.utils.data.dataset.Dataset):
				def __init__(self):
					self.samples = cifarA
				def __len__(self):
					return len(self.samples)
				def __getitem__(self, index):
					return self.samples[index]
			class DividedCifar10B(torch.utils.data.dataset.Dataset):
				def __init__(self):
					self.samples = cifarB
				def __len__(self):
					return len(self.samples)
				def __getitem__(self, index):
					return self.samples[index]
			if dataset == 'cifar10a' :
				data_set = DividedCifar10A()
			else:
				data_set == DividedCifar10B()
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		else:
			raise ValueError("Dataset not supported.")
	else:
		if dataset == 'mnist':
			transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
			data_set = torchvision.datasets.MNIST('~/data/mnist', train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'cifar10':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'cifar100':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.CIFAR100(root='~/data/cifar100', train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'svhn':
			if not is_train_split:
				svhn_split = 'test'
			else:
				svhn_split = 'train'
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.SVHN(root='~/data/svhn', split=svhn_split , download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		elif dataset == 'fashionmnist':
			transform = transforms.Compose([transforms.Grayscale(3),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.FashionMNIST(root='~/data/fmnist', train=is_train_split, download=True, transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=8)
		elif dataset == 'imagenet':
			transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), 
            transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
			# transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			data_set = torchvision.datasets.ImageFolder(root='~/data/imagenet/val', transform=transform)
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=8)
		elif dataset == 'cifar10a' or dataset == 'cifar10b':
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
			transform_augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms. RandomCrop(32, padding=4)])
			cifarset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=is_train_split, download=True, 
            transform=transforms.Compose([transform_augment, transform]))
			label_flag = {x:True for x in range(10)}
			cifarA = []
			cifarB = []
			for sample in cifarset:
				if label_flag[sample[-1]]:
					cifarA.append(sample)
					label_flag[sample[-1]] = False
				else:
					cifarB.append(sample)
					label_flag[sample[-1]] = True
			class DividedCifar10A(torch.utils.data.dataset.Dataset):
				def __init__(self):
					self.samples = cifarA
				def __len__(self):
					return len(self.samples)
				def __getitem__(self, index):
					return self.samples[index]
			class DividedCifar10B(torch.utils.data.dataset.Dataset):
				def __init__(self):
					self.samples = cifarB
				def __len__(self):
					return len(self.samples)
				def __getitem__(self, index):
					return self.samples[index]
			if dataset == 'cifar10a' :
				data_set = DividedCifar10A()
			else:
				data_set == DividedCifar10B()
			data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
		else:
			raise ValueError(dataset + " dataset not supported.")
	return data_loader

def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def set_prune_rate(model, prune_rate):
    if hasattr(model, "module"):
        model.module.set_prune_rate(prune_rate)
    else:
        model.set_prune_rate(prune_rate)


def init_score(model, mute=True):
	print("init score by weight")
	if hasattr(model, "module"):
		model.module.init_score()
	else:
		model.init_score()

	if not mute:
		check_prune_rate(model)


def check_prune_rate(model):
	if hasattr(model, "module"):
		check = copy.deepcopy(model.module)
	else:
		check = copy.deepcopy(model)

	modules = {name: module for name, module in check.named_modules()}

	for name, params in check.named_parameters():
		if "score" in name:
			module = modules[name.replace('.score', '')]
			mask = module.get_mask()
			zero, total = (mask == 0).sum().float().item(), mask.numel()
			print("%s's prune ratio: %.2f%% (%d/%d)" % (name, 100 * zero / total, zero, total))
	del check


def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            # print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                # print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                # print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    # print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            # print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                # print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def freeze_model_scores(model):
    print("=> Freezing model scores")
    if hasattr(model, "module"):
        model.module.freeze_score()
    else:
        model.freeze_score()


def unfreeze_model_scores(model):
    print("=> Unfreezing model scores")
    if hasattr(model, "module"):
        model.module.unfreeze_score()
    else:
        model.unfreeze_score()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res