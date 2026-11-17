import argparse
from pathlib import Path
from trainer_c import train_c
from trainer_cf import train_cf
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('CASR', add_help=False)
    parser.add_argument('--arch', default='vit_base', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'],
        help="Backbone architecture")
    parser.add_argument('--patch_size', default=8, type=int, help="For ViT backbone")
    parser.add_argument('--out_dim', default=65536, type=int, help="Output dimensionality of the CS-SA anchoring head.")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="Whether or not to weight normalize the last layer of the CS-SA anchoring head")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="Base EMA parameter for teacher update.")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag, help="use batch normalizations in the CS-SA anchoring head")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="Final value (after linear warmup) of the teacher temperature.")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. """)
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size.')
    parser.add_argument('--epochs', default='200-200-100', type=str,
        help='Number of epochs at each MCC granularity level (paired with --data_granularity). '
             'Stage 1 (--mode C): 200-200-100; stage 2 (--mode CF): 63-5-12.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""optimizer""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.8, 1.))
    parser.add_argument('--local_crops_number', type=int, default=8)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4))
    parser.add_argument('--datasets', type=str, nargs='+', default=[])
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=100, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--weights", default=None, type=str, help="weights")
    parser.add_argument('--data_granularity', default='0-1-2', type=str, help='MCC anchor granularity k per stage, coarse to fine (paired with --epochs)')
    parser.add_argument('--mode', default='CF', type=str, help='C: stage 1, CS-SA warm-up | CF: stage 2, full CASR objective')
    parser.add_argument('--datasets_config', default=None, type=str, help='path to datasets config file')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CASR', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode.lower()=="c":
        train_c(args)
    else:
        train_cf(args)
