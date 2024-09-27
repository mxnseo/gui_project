import datetime
import os
import time
import warnings

import segmentation_presets
import torch
import torch.utils.data
import torchvision
import segmention__utils
from segmention_coco_utils import get_coco
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.transforms import functional as F, InterpolationMode
import numpy as np

def get_dataset(args, is_train):
    def sbd(*args, **kwargs):
        kwargs.pop("use_v2")
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    def voc(*args, **kwargs):
        kwargs.pop("use_v2")
        return torchvision.datasets.VOCSegmentation(*args, **kwargs)

    paths = {
        "voc": (args.data_path, voc, 21),
        "voc_aug": (args.data_path, sbd, 21),
        "coco": (args.data_path, get_coco, 9),
    }
    p, ds_fn, num_classes = paths[args.dataset]

    image_set = "train" if is_train else "val"
    ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train, args), use_v2=args.use_v2)
    return ds, num_classes


def get_transform(is_train, args):
    if is_train:
        return segmentation_presets.SegmentationPresetTrain(base_size=520, crop_size=480, backend=args.backend, use_v2=args.use_v2)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return segmentation_presets.SegmentationPresetEval(base_size=520, backend=args.backend, use_v2=args.use_v2)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes, epoch, cut_train, metric_logger, args):
    model.eval()
    confmat = segmention__utils.ConfusionMatrix(num_classes)
    header = "Test:"
    header = f"Test: [{epoch}] cut : " + cut_train
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 30, header):
            image, target = image.to(device), target.to(device)
            #if문으로 Data_loader가 Val일 때에만 진행해야 함.
            scaler = torch.cuda.amp.GradScaler() if args.amp else None
            #scaler = torch.cpu.amp.GradScaler() if args.amp else None
            model_without_ddp = model
            params_to_optimize = [
                {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
                {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
            ]
            if args.aux_loss:
                params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params, "lr": args.lr * 10})
 
            optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, 
                weight_decay=args.weight_decay)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
            #with torch.cpu.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss = criterion(output, target)
            #metric_logger.add_meter('valid_loss', loss.item())
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            output = model(image)
            output = output["out"]
 
            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]
 
        confmat.reduce_from_all_processes()
 
    num_processed_samples = segmention__utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )
 
    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, metric_logger, scaler=None):
    model.train()
    metric_logger.add_meter("lr", segmention__utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
        #with torch.cpu.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        #metric_logger.add_meter('train_loss', loss.item()) 
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def main(myWindow, args):
    train_list = [[],[]]
    valid_list = [[],[]]
    if args.backend.lower() != "pil" and not args.use_v2:
        # TODO: Support tensor backend in V1?
        raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
    if args.use_v2 and args.dataset != "coco":
        raise ValueError("v2 is only support supported for coco dataset for now.")

    if args.output_dir:
        segmention__utils.mkdir(args.output_dir)

    segmention__utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    dataset, num_classes = get_dataset(args, is_train=True)
    dataset_test, _ = get_dataset(args, is_train=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=segmention__utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=segmention__utils.collate_fn
    )

    model = torchvision.models.get_model(
        args.model,
        weights=args.weights,
        weights_backbone=args.weights_backbone,
        num_classes=num_classes,
        aux_loss=args.aux_loss,
    )
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    #scaler = torch.cpu.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, epoch=epoch, cut_train='V',metric_logger=metric_logger, args=args)
        print(confmat)
        return

    start_time = time.time()
    metric_logger = segmention__utils.MetricLogger("  ")
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        key=myWindow.stop_training()
        if(key==False):
            break
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, metric_logger, scaler=None)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, epoch=epoch, cut_train='V',metric_logger=metric_logger, args=args)
        print(confmat)
        train_confmat = evaluate(model, data_loader, device=device, num_classes=num_classes, epoch=epoch, cut_train='T',metric_logger=metric_logger, args=args)
        print(train_confmat)
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        t_acc, __, __ = confmat.compute()
        v_acc, __, __ = train_confmat.compute()
        print("--------------------------------------------------")
        print(" 출력 : ")
        print(" T_ACC : ", t_acc.item()) #텐서로 출력 중
        print(" V_ACC : ", v_acc.item()) #텐서로 출력 중
        print("TRAIN_LOSS",metric_logger.meters['train_loss'])
        print("VALID_LOSS",metric_logger.meters['valid_loss'])
        print("--------------------------------------------------")
        train_list[0].append(metric_logger.meters['train_loss'])
        train_list[1].append(t_acc.item())
        valid_list[0].append(metric_logger.meters['valid_loss'])
        valid_list[1].append(v_acc.item())
        print("학습 손실값 기록 : ", train_list[0])
        print("학습 정확도 기록 : ", train_list[1])
        print("검증 손실값 기록 : ", valid_list[0])
        print("검증 정확도 기록 : ", valid_list[1])
        segmention__utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        segmention__utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        
        to_numpy_train = np.array(train_list)
        to_numpy_valid = np.array(valid_list)
        
        #myWindow 클래스 추가해줘야함
        x_arr = np.arange(epoch + 1)
        myWindow.plot(x_arr, to_numpy_valid, to_numpy_train)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    print("학습이 종료되었습니다.")

def get_args_parser(folder_path, epoch, worker ,lr,model ,weight ,device,resultfolder_path, batch_size, dataset, add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliary loss")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    test_args2 = ["--data-path",folder_path,"--epochs",str(epoch),"--workers",str(worker),"--lr",str(lr),"--model", model,"--weights-backbone",weight,"--device",device,"--output-dir",resultfolder_path,
                "--batch-size", str(batch_size), "--dataset", dataset]
    args2 = parser.parse_args(test_args2)
    return args2



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)