import os
import argparse

from dataloader import SegmentationDataset, VinDr_Rib, build_transform_segmentation
from engine import segmentation_engine
from utils import torch_dice_coef_loss


def get_args_parser():
    parser = argparse.ArgumentParser(description='Segmentation with a frozen CASR encoder.')
    parser.add_argument('--data_set', required=True, choices=['JSRT-Heart', 'JSRT-Clavicle', 'Montgomery', 'VinDr-Rib'])
    parser.add_argument('--pretrained', required=True, help='path to the pretrained CASR checkpoint')
    parser.add_argument('--train_data_dir', help='train images (VinDr-Rib: train json)', default=None)
    parser.add_argument('--train_mask_dir', help='train masks', default=None)
    parser.add_argument('--valid_data_dir', help='validation images', default=None)
    parser.add_argument('--valid_mask_dir', help='validation masks', default=None)
    parser.add_argument('--test_data_dir', help='test images (VinDr-Rib: test json)', default=None)
    parser.add_argument('--test_mask_dir', help='test masks', default=None)
    parser.add_argument('--image_root', help='VinDr-Rib only: root that image paths in the json are relative to '
                                             '(default: inferred from the json location)', default=None)
    parser.add_argument('--few_shot', help='number (>=1) or fraction (<1) of training images; -1 uses all',
                        default=-1, type=float)
    parser.add_argument('--output_dir', help='results are written to <output_dir>/<data_set>/<shot>', default='./outputs')
    parser.add_argument('--train_batch_size', default=48, type=int)
    parser.add_argument('--test_batch_size', default=48, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--patience', default=30, type=int, help='early-stopping patience (epochs)')
    parser.add_argument('--train_num_workers', default=0, type=int)
    parser.add_argument('--test_num_workers', default=2, type=int)
    parser.add_argument('--device', default="cuda", help='cuda|cpu')
    return parser.parse_args()


def main(args):
    print(args)
    shot = "full" if args.few_shot <= 0 else ("%g" % args.few_shot)
    model_path = os.path.join(args.output_dir, args.data_set, shot)

    if args.data_set == "VinDr-Rib":
        dataset_train, dataset_val = VinDr_Rib(args.train_data_dir, mode='train-val', few_shot=args.few_shot, image_root=args.image_root)
        dataset_test = VinDr_Rib(args.test_data_dir, mode='test', image_root=args.image_root)
    else:  # JSRT-Heart, JSRT-Clavicle, Montgomery
        dataset_train = SegmentationDataset(args.train_data_dir, args.train_mask_dir, transforms=build_transform_segmentation(), few_shot=args.few_shot)
        dataset_val = SegmentationDataset(args.valid_data_dir, args.valid_mask_dir, transforms=build_transform_segmentation())
        dataset_test = SegmentationDataset(args.test_data_dir, args.test_mask_dir, transforms=None)

    segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test, torch_dice_coef_loss, num_classes=1)


if __name__ == '__main__':
    main(get_args_parser())
