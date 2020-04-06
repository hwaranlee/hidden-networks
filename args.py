import argparse

def parse_arguments():
    """
    returns argument parser object used while training a model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', required=True, type=str, help='neural network architecture [vgg19, resnet50]')
    parser.add_argument('--dataset',type=str, required=True, help='dataset [cifar10, cifar100, svhn, fashionmnist]')
    parser.add_argument('--batch-size', type=int, default=512, help='input batch size for training (default: 512)')
    parser.add_argument('--num-epochs', type=int, default=None, help='training epochs (default: None)')
    parser.add_argument('--lr', type=float, default=None, help='initial learning rate (default: None)')
    parser.add_argument('--lr-anneal-epochs', default=None, type=lambda x: [int(a) for a in x.split(",")], help='anneal learning rate by ten on')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='warm up epochs before pruning (default: 0)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer [sgd, adam]')
    parser.add_argument('--prune-rate', type=float, default=0.0, help='prune rate (default: 0.0)')
    parser.add_argument('--init-path',type=str, default="None", help='path to initialization model (default = None)')
    parser.add_argument('--init-score', action="store_true", default=False, help='initiate scores by weights')
    parser.add_argument('--evaluate', action="store_true", default=False, help='evaluate only')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--model-saving-path', type=str, default = ".",help='path to directory where you want to save trained models (default = .)')
    parser.add_argument("--multigpu", default="0,1", type=lambda x: [int(a) for a in x.split(",")], help="Which GPUs to use for multigpu training")
    parser.add_argument("--save-every", default=-1, type=int, help="Save every ___ epochs")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency (default: 10)")
    parser.add_argument("--train-type", default="None", type=str, help="train type [freeze-weight, freeze-score, alternative (default: None)]")
    return parser
