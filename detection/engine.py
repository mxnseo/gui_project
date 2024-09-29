import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import detection_utils
from coco_eval import CocoEvaluator
from detection_coco_utils import get_coco_api_from_dataset


# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, metric_logger,scaler=None):
#     model.train()
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     header = f"Epoch: [{epoch}]"

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=warmup_factor, total_iters=warmup_iters
#         )

#     for images, targets in metric_logger.log_every(data_loader, print_freq, header):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

#         loss_value = losses_reduced.item()

#         if not math.isfinite(loss_value):
#             print(f"Loss is {loss_value}, stopping training")
#             print(loss_dict_reduced)
#             sys.exit(1)

#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(losses).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             losses.backward()
#             optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()
#         metric_logger.add_meter('train_loss', loss_value)
#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

#     return metric_logger

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()  # 모델을 학습 모드로 설정
    metric_logger = detection_utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    running_loss = 0.0
    total_samples = 0

    coco_evaluator = CocoEvaluator(get_coco_api_from_dataset(data_loader.dataset), iou_types=["bbox"])

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

        # 모델에 이미지와 타겟을 전달하여 손실 계산 (학습 모드)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)  
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        batch_size = len(images)
        running_loss += losses.item() * batch_size
        total_samples += batch_size

        
        model.eval() 
        outputs = model(images) 
        model.train()  

        
        res = {}
        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"]
            
            # image_id가 문자열이면 정수로 변환
            if isinstance(image_id, torch.Tensor):
                image_id = image_id.item()
            elif isinstance(image_id, str):
                try:
                    image_id = int(image_id)  # 문자열이면 정수로 변환
                except ValueError:
                    raise ValueError(f"Invalid image_id: {image_id}. Expected int or convertible string.")

            res[image_id] = {
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu()
            }

        coco_evaluator.update(res)
        metric_logger.update(loss=losses.item())

    # 에포크별 손실 계산
    epoch_loss = running_loss / total_samples

    # COCO evaluator 처리
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # mAP 계산
    coco_stats = coco_evaluator.coco_eval['bbox'].stats
    train_mAP = coco_stats[0]  # mAP (IoU=0.50:0.95)
    print(f"epcoh_loss: {epoch_loss}")
    print(f"train_mAP: {train_mAP}")
    return epoch_loss, train_mAP  # 훈련 손실 및 mAP 반환



def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
# def evaluate(model, data_loader, device, metric_logger):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     for images, targets in metric_logger.log_every(data_loader, 100, header):
#         images = list(img.to(device) for img in images)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"]: output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator

def evaluate(model, data_loader_test, device):
    model.eval()
    coco = get_coco_api_from_dataset(data_loader_test.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(image.to(device) for image in images)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            model.eval()

            batch_size = len(images)
            running_loss += losses.item() * batch_size
            total_samples += batch_size

            # 모델의 예측을 coco evaluator로 업데이트
            outputs = model(images)
            #res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            
            res = {}
            for i, output in enumerate(outputs):
                image_id = targets[i]["image_id"]
            
                # image_id가 문자열이면 정수로 변환
                if isinstance(image_id, torch.Tensor):
                    image_id = image_id.item()
                elif isinstance(image_id, str):
                    try:
                        image_id = int(image_id)  # 문자열이면 정수로 변환
                    except ValueError:
                        raise ValueError(f"Invalid image_id: {image_id}. Expected int or convertible string.")

                res[image_id] = {
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                    "labels": output["labels"].detach().cpu()
                }
                coco_evaluator.update(res)

    epoch_loss = running_loss / total_samples
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # mAP 값 추출
    coco_stats = coco_evaluator.coco_eval['bbox'].stats
    val_mAP = coco_stats[0]  # mAP (IoU=0.50:0.95)
    print(f"epcoh_loss: {epoch_loss}")
    print(f"val_mAP: {val_mAP}")
    return epoch_loss, val_mAP  # 검증 손실 및 mAP 반환


