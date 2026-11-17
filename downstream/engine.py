import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import vision_transformer as vits
from trainer import train_one_epoch, evaluate, test_segmentation, test_classification
from unet import UNet
from utils import cosine_anneal_schedule, dice, mean_dice_coef, meanAUC, save_checkpoint


def load_teacher_state_dict(path):
  state_dict = torch.load(path, map_location="cpu")
  if "teacher" in state_dict:
    state_dict = state_dict["teacher"]
  elif "model" in state_dict:
    state_dict = state_dict["model"]
  new_state_dict = {}
  for key, value in state_dict.items():
    new_key = key.replace("module.", "").replace("backbone.", "")
    if new_key.startswith(("head.", "fc")):
      continue
    new_state_dict[new_key] = value
  return new_state_dict


def segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test, criterion, num_classes):
  device = torch.device(args.device)
  os.makedirs(model_path, exist_ok=True)
  log_writter = open(os.path.join(model_path, "log.txt"), 'a')
  checkpoint_path = os.path.join(model_path, "checkpoint.pt")

  data_loader_train = DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                 num_workers=args.train_num_workers)
  data_loader_val = DataLoader(dataset_val, batch_size=args.train_batch_size,
                               shuffle=False, num_workers=args.train_num_workers)

  net = UNet(num_classes=num_classes, encoder_name="vit_base")
  state_dict = {"encoder." + k: v for k, v in load_teacher_state_dict(args.pretrained).items()}
  msg = net.load_state_dict(state_dict, strict=False)
  print("Loaded pretrained encoder, missing keys:", msg.missing_keys)
  for param in net.encoder.parameters():
    param.requires_grad = False

  optimizer = torch.optim.Adam(net.parameters(), args.learning_rate)
  model = torch.nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
  model.to(device)
  best_val_loss = 100000
  patience_counter = 0

  for epoch in range(args.epochs):
    train_one_epoch(data_loader_train, device, model, criterion, optimizer, epoch)
    val_loss = evaluate(data_loader_val, device, model, criterion)
    lr_ = cosine_anneal_schedule(epoch, args.epochs, args.learning_rate)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr_

    if val_loss < best_val_loss:
      print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(
        epoch, best_val_loss, val_loss, checkpoint_path))
      torch.save({
        'epoch': epoch + 1,
        'lossMIN': val_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
      }, checkpoint_path)
      best_val_loss = val_loss
      patience_counter = 0
    else:
      print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss))
      patience_counter += 1

    if patience_counter > args.patience:
      print("Early Stopping")
      break
  torch.cuda.empty_cache()

  data_loader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False,
                                num_workers=args.test_num_workers)
  test_y, test_p = test_segmentation(net, checkpoint_path, data_loader_test, device, log_writter)

  print("[INFO] Dice = {:.2f}%".format(100.0 * dice(test_p, test_y)), file=log_writter)
  print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter)
  log_writter.close()


def classification_engine(args, model_path, diseases, dataset_train, dataset_val, dataset_test, num_classes):
  device = torch.device(args.device)
  cudnn.benchmark = True
  os.makedirs(model_path, exist_ok=True)
  output_file = os.path.join(model_path, "results.txt")

  data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.workers, pin_memory=True)
  data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
  data_loader_test = DataLoader(dataset=dataset_test, batch_size=int(args.batch_size / 4), shuffle=False,
                                num_workers=args.workers, pin_memory=True)
  criterion = torch.nn.BCEWithLogitsLoss()

  mean_auc = []
  for i in range(args.num_trial):
    experiment = "run_" + str(i)
    save_model_path = os.path.join(model_path, experiment)
    print("run:", i + 1)

    net = vits.__dict__[args.model](patch_size=8, num_classes=num_classes, apply_head=True)
    msg = net.load_state_dict(load_teacher_state_dict(args.pretrained), strict=False)
    print("Loaded pretrained encoder, missing keys:", msg.missing_keys)
    if args.freeze_encoder:
      for name, param in net.named_parameters():
        if name not in ['head.weight', 'head.bias']:
          param.requires_grad = False

    optimizer = create_optimizer(args, net)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    model = torch.nn.DataParallel(net) if torch.cuda.device_count() > 1 else net
    model.to(device)

    best_val_loss = 1000000
    patience_counter = 0
    for epoch in range(args.epochs):
      train_one_epoch(data_loader_train, device, model, criterion, optimizer, epoch)
      val_loss = evaluate(data_loader_val, device, model, criterion)
      lr_scheduler.step(val_loss)

      if val_loss < best_val_loss:
        print("Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(
          epoch, best_val_loss, val_loss, save_model_path))
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint({
          'epoch': epoch + 1,
          'lossMIN': best_val_loss,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'scheduler': lr_scheduler.state_dict(),
        }, filename=save_model_path)
      else:
        print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss))
        patience_counter += 1

      if patience_counter >= args.patience:
        print("Early Stopping")
        break

    # Test the best checkpoint of this trial.
    y_test, p_test = test_classification(net, save_model_path + ".pth", data_loader_test, device, args)
    mAUC, auc_scores = meanAUC(y_test.cpu().numpy(), p_test.cpu().numpy())
    print(">> {}: mean AUC = {:.4f}".format(experiment, mAUC))
    with open(output_file, 'a') as writer:
      writer.write("Disease = {}\n".format(diseases))
      writer.write("{}: AUC = {}\n".format(experiment, np.array2string(np.array(auc_scores), precision=4, separator=',')))
      writer.write("{}: mAUC = {:.4f}\n".format(experiment, mAUC))
    mean_auc.append(mAUC)

  mean_auc = np.array(mean_auc)
  print(">> Mean / STD over all trials: AUC = {:.4f} ({:.4f})".format(np.mean(mean_auc), np.std(mean_auc)))
  with open(output_file, 'a') as writer:
    writer.write("Mean / STD over all trials: AUC = {:.4f} ({:.4f})\n".format(np.mean(mean_auc), np.std(mean_auc)))
