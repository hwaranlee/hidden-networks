import os
import time
import shutil
import random
import pathlib
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from args import parse_arguments
from utils import *
from logger import AverageMeter, ProgressMeter
from trainer import warm, train, validate
from torch.utils.tensorboard import SummaryWriter


def main():
	#Parsers the command line arguments
	parser = parse_arguments()
	args = parser.parse_args()

	#Sets random seed
	random.seed(args.seed)

	#Uses GPU is available
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	#Checks number of classes to aa appropriate linear layer at end of model
	if args.dataset in ['cifar10', 'fashionmnist', 'svhn']:
		num_classes = 10
	elif args.dataset in ['cifar100']:
		num_classes = 100
	elif args.dataset in ['imagenet']:
		num_classes = 1000
	else:
		raise ValueError(args.dataset + " dataset not supported")

	#Loads model
	model = load_model(args.architecture, num_classes)

	if args.architecture == "vgg19":
		num_epochs = 160 if not args.num_epochs else args.num_epochs
		lr_anneal_epochs = [80, 120] if not args.lr_anneal_epochs else args.lr_anneal_epochs
	elif args.architecture == "resnet50":
		num_epochs = 90 if not args.num_epochs else args.num_epochs
		lr_anneal_epochs = [50, 65, 80] if not args.lr_anneal_epochs else args.lr_anneal_epochs
	else:
		raise ValueError(args.architecture + " architecture not supported")

	criterion = nn.CrossEntropyLoss().to(device)

	if args.optimizer == 'sgd':
		lr = 0.1 if not args.lr else args.lr
		optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
	elif args.optimizer == 'adam':
		lr = 0.0003 if not args.lr else args.lr
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
	else:
		raise ValueError(args.optimizer + " optimizer not supported")

	model = set_gpu(args, model)

	if not args.init_path == "None":
		load_state_dict(model, args.init_path)

	best_acc1 = 0.0
	best_acc5 = 0.0
	best_train_acc1 = 0.0
	best_train_acc5 = 0.0

	# Data loading code
	if args.evaluate:
		set_prune_rate(model, args.prune_rate)
		val_loader = load_dataset(args.dataset, args.batch_size, False)
		acc1, acc5 = validate(
			val_loader, model, criterion, args, writer=None, epoch=None
		)
		return
	else:
		train_loader = load_dataset(args.dataset, args.batch_size, True)
		val_loader = load_dataset(args.dataset, args.batch_size, False)

	# Set up directories
	run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
	args.ckpt_base_dir = ckpt_base_dir

	writer = SummaryWriter(log_dir=log_base_dir)
	epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
	validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
	train_time = AverageMeter("train_time", ":.4f", write_avg=False)
	progress_overall = ProgressMeter(
		1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
	)

	end_epoch = time.time()
	acc1 = None

	# Warmup model
	if not args.warmup_epochs == 0:
		print("[Warm model with freezed scores by %d epochs]" % args.warmup_epochs)
		freeze_model_scores(model)
		for epoch in range(1, args.warmup_epochs + 1):
			set_prune_rate(model, prune_rate=0.0)
			warm(train_loader, model, criterion, optimizer, epoch, args)
			validate(val_loader, model, criterion, args, None, epoch)
		unfreeze_model_scores(model)

	if args.train_type == "freeze-weight":
		freeze_model_weights(model)
	
	if args.train_type == "freeze-score":
		freeze_model_scores(model)

	set_prune_rate(model, prune_rate=args.prune_rate)

	if not args.warmup_epochs == 0 or args.init_score:
		init_score(model)

	# Save the initial state
	save_checkpoint(
		{
			"warmup_epochs": args.warmup_epochs,
			"epoch": 0,
			"arch": args.architecture,
			"state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
			"best_acc1": best_acc1,
			"best_acc5": best_acc5,
			"best_train_acc1": best_train_acc1,
			"best_train_acc5": best_train_acc5,
			"optimizer": optimizer.state_dict(),
			"curr_acc1": acc1 if acc1 else "Not evaluated",
		},
		False,
		filename=ckpt_base_dir / f"initial.state",
		save=False,
	)

	# Start training
	print("[Start training model]")
	for epoch in range(1, num_epochs + 1):
		if epoch in lr_anneal_epochs:
			optimizer.param_groups[0]['lr'] /= 10
			print("learning rate is annealed to %f" % get_lr(optimizer))
			
		if args.train_type == "alternative":
			if epoch % 2 == 1:
				freeze_model_scores(model)
				unfreeze_model_weights(model)
			else:
				unfreeze_model_scores(model)
				freeze_model_weights(model)

		# train for one epoch
		start_train = time.time()
		train_acc1, train_acc5 = train(
			train_loader, model, criterion, optimizer, epoch, args, writer=writer
		)
		train_time.update((time.time() - start_train) / 60)

		# evaluate on validation set
		start_validation = time.time()
		acc1, acc5 = validate(val_loader, model, criterion, args, writer, epoch)
		validation_time.update((time.time() - start_validation) / 60)

		# remember best acc@1 and save checkpoint
		is_best = acc1 > best_acc1
		best_acc1 = max(acc1, best_acc1)
		best_acc5 = max(acc5, best_acc5)
		best_train_acc1 = max(train_acc1, best_train_acc1)
		best_train_acc5 = max(train_acc5, best_train_acc5)

		save = ((epoch % args.save_every) == 0) and args.save_every > 0
		if is_best or save or epoch == num_epochs:
			if is_best:
				print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

			save_checkpoint(
				{
					"warmup_epochs": args.warmup_epochs,
					"epoch": epoch,
					"arch": args.architecture,
					"state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
					"best_acc1": best_acc1,
					"best_acc5": best_acc5,
					"best_train_acc1": best_train_acc1,
					"best_train_acc5": best_train_acc5,
					"optimizer": optimizer.state_dict(),
					"curr_acc1": acc1,
					"curr_acc5": acc5,
				},
				is_best,
				filename=ckpt_base_dir / f"epoch_{epoch}.state",
				save=save,
			)

		epoch_time.update((time.time() - end_epoch) / 60)
		progress_overall.display(epoch)
		progress_overall.write_to_tensorboard(
			writer, prefix="diagnostics", global_step=epoch
		)
		cur_lr = get_lr(optimizer)
		writer.add_scalar("test/lr", cur_lr, epoch)
		end_epoch = time.time()

	write_result_to_csv(
		dataset=args.dataset,
		prune_rate=args.prune_rate,
		warmup_epochs=args.warmup_epochs,
		num_epochs=num_epochs,
		init_path=args.init_path,
		train_type=args.train_type,
		optimizer=args.optimizer,
		lr=lr,
		best_acc1=best_acc1,
		best_acc5=best_acc5,
		curr_acc1=acc1,
		curr_acc5=acc5,
		best_train_acc1=best_train_acc1,
		best_train_acc5=best_train_acc5,
		ckpt_base_dir=ckpt_base_dir,
	)
	print('-' * 40)
	print("best_acc1: %.4f best_acc5: %.4f" % (best_acc1, best_acc5))

def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date-Finished, "
			"Dataset, "
            "Prune-Rate, "
			"Warmup-Epochs, "
			"Train-Epochs, "
			"Init-Path, "
			"Train-Type, "
			"Optimizer, "
			"Learning-Rate, "
            "Best-Val-Top-1, "
            "Best-Val-Top-5, "
            "Current-Val-Top-1, "
            "Current-Val-Top-5, "
            "Best-Train-Top-1, "
            "Best-Train-Top-5, "
			"Save-Path\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
				"{dataset}, "
                "{prune_rate}, "
				"{warmup_epochs}, "
				"{num_epochs}, "
				"{init_path}, "
				"{train_type}, "
				"{optimizer}, "
				"{lr}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}, "
				"{ckpt_base_dir}\n"
            ).format(now=now, **kwargs)
        )


def get_directories(args):
	run_base_dir = pathlib.Path(
		f"runs/{args.architecture}/{args.dataset}/prune_rate={args.prune_rate}"
	)

	if _run_dir_exists(run_base_dir):
		rep_count = 0
		while _run_dir_exists(run_base_dir / str(rep_count)):
			rep_count += 1

		run_base_dir = run_base_dir / str(rep_count)

	log_base_dir = run_base_dir / "logs"
	ckpt_base_dir = run_base_dir / "checkpoints"

	if not run_base_dir.exists():
		os.makedirs(run_base_dir)

	(run_base_dir / "settings.txt").write_text(str(args))

	return run_base_dir, ckpt_base_dir, log_base_dir

def set_gpu(args, model):
	assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
	# DataParallel will divide and allocate batch_size to all available GPUs
	print(f"=> Parallelizing on {args.multigpu} gpus")
	torch.cuda.set_device(args.multigpu[0])
	args.gpu = args.multigpu[0]
	model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
		args.multigpu[0]
	)

	cudnn.benchmark = True

	return model

if __name__ == '__main__':
    main()