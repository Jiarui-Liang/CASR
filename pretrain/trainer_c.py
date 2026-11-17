"""Stage 1: warm-up (CS-SA)."""
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data_loader import CXRDataset
import utils
from models import AnchoringHead
from utils import get_config
import models
import vision_transformer as vits
from trainer_cf import CSSALoss, Transform

def train_c(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing training params ... ============
    epochs = list(map(int, args.epochs.split('-')))
    data_granularity = list(map(int, args.data_granularity.split('-')))
    assert len(epochs) == len(data_granularity)

    datasets_config = get_config(args.datasets_config)
    args.total_epochs=sum(epochs)

    # ============ preparing augmentation ... ============
    transform = Transform(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number
    )

    # ============ building student and teacher networks ... ============
    student = vits.__dict__[args.arch](
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,  # stochastic depth
    )
    teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
    embed_dim = student.embed_dim

    student = models.MultiCropWrapper(student, AnchoringHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = models.MultiCropWrapper(
        teacher,
        AnchoringHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    student, teacher = student.cuda(), teacher.cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    cssa_criterion = CSSALoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.total_epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ training ... ============
    to_restore = {"epoch": 0}
    if args.weights is not None:
        utils.restart_from_checkpoint(
            args.weights,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            cssa_loss=cssa_criterion,
        )

    start_time = time.time()
    print("Starting training....")
    start_epoch=0
    for step in range(len(epochs)):
        concat_datasets = []
        for dataset_name in args.datasets:
            dataset = CXRDataset(images_path=datasets_config[dataset_name]['data_dir'], file_path=datasets_config[dataset_name]['train_list'],
                                     augment=transform, data_granularity=data_granularity[step])

            concat_datasets.append(dataset)
        train_dataset = torch.utils.data.ConcatDataset(concat_datasets)
        print(f"Data loaded: there are {len(train_dataset)} training images.")
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if step==0:
            # ============ init schedulers ... ============
            lr_schedule = utils.cosine_scheduler(
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
                args.min_lr,
                args.total_epochs, len(train_data_loader),
                warmup_epochs=args.warmup_epochs,
            )
            wd_schedule = utils.cosine_scheduler(
                args.weight_decay,
                args.weight_decay_end,
                args.total_epochs, len(train_data_loader),
            )
            momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                                       args.total_epochs, len(train_data_loader))
        for epoch in range(start_epoch, start_epoch+epochs[step]):
            train_data_loader.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(student, teacher, teacher_without_ddp, cssa_criterion,
                train_data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                epoch, fp16_scaler, args)
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'cssa_loss': cssa_criterion.state_dict(),
            }
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        start_epoch=start_epoch+epochs[step]
        print("start_epoch",start_epoch)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student, teacher, teacher_without_ddp, cssa_criterion, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.total_epochs)
    for it, (images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = cssa_criterion(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
