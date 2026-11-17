import os
from optparse import OptionParser

from dataloader import ChestXray14, VinDrCXR, build_transform_classification
from engine import classification_engine
from utils import vararg_callback_bool


def get_args_parser():
    parser = OptionParser(description='Linear probing of a frozen CASR encoder.')

    parser.add_option("--data_set", dest="data_set", help="ChestXray14|VinDrCXR", default="ChestXray14", type="string")
    parser.add_option("--pretrained", dest="pretrained", help="path to the pretrained CASR checkpoint", default=None, type="string")
    parser.add_option("--data_dir", dest="data_dir", help="image root", default=None, type="string")
    parser.add_option("--train_list", dest="train_list", help="file for training list", default=None, type="string")
    parser.add_option("--val_list", dest="val_list", help="file for validating list", default=None, type="string")
    parser.add_option("--test_list", dest="test_list", help="file for test list", default=None, type="string")
    parser.add_option('--few_shot', dest="few_shot", help='number (>=1) or fraction (<1) of training images; -1 uses all',
                      default=-1, type=float)
    parser.add_option("--output_dir", dest="output_dir", help="results are written to <output_dir>/<data_set>/<shot>",
                      default="./outputs", type="string")
    parser.add_option("--model", dest="model", help="vit_base", default="vit_base", type="string")
    parser.add_option("--normalization", dest="normalization", help="how to normalize data (imagenet|chestx-ray)", default="imagenet",
                      type="string")
    parser.add_option("--img_size", dest="img_size", help="resize image resolution", default=256, type="int")
    parser.add_option("--input_size", dest="input_size", help="input image resolution", default=224, type="int")
    parser.add_option("--batch_size", dest="batch_size", help="batch size", default=256, type="int")
    parser.add_option("--epochs", dest="epochs", help="num of epoches", default=300, type="int")
    parser.add_option("--freeze_encoder", dest="freeze_encoder", help="whether freeze encoder", default=True, action="callback",
                      callback=vararg_callback_bool)
    # Optimizer parameters
    parser.add_option('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "momentum")')
    parser.add_option('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_option('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_option('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_option('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_option('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    # Learning rate schedule parameters
    parser.add_option('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_option('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_option('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_option('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_option('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_option('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_option('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_option('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_option('--warmup-epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_option('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_option('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.5)')

    parser.add_option("--patience", dest="patience", help="early-stopping patience (epochs)", default=30, type="int")
    parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
    parser.add_option("--workers", dest="workers", help="number of CPU workers", default=2, type="int")
    parser.add_option("--device", dest="device", help="cpu|cuda", default="cuda", type="string")

    (options, args) = parser.parse_args()

    return options


def main(args):
    print(args)
    assert args.data_dir is not None and args.pretrained is not None
    shot = "full" if args.few_shot <= 0 else ("%g" % args.few_shot)
    model_path = os.path.join(args.output_dir, args.data_set, shot)

    def transform(mode):
        return build_transform_classification(normalize=args.normalization, mode=mode, crop_size=args.input_size, resize=args.img_size)

    if args.data_set == "ChestXray14":
        diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        dataset_train = ChestXray14(images_path=args.data_dir, file_path=args.train_list, augment=transform("train"), few_shot=args.few_shot)
        dataset_val = ChestXray14(images_path=args.data_dir, file_path=args.val_list, augment=transform("valid"))
        dataset_test = ChestXray14(images_path=args.data_dir, file_path=args.test_list, augment=transform("test"))
        classification_engine(args, model_path, diseases, dataset_train, dataset_val, dataset_test, num_classes=14)

    elif args.data_set == "VinDrCXR":
        diseases = ['PE', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']
        dataset_train = VinDrCXR(images_path=args.data_dir, file_path=args.train_list, augment=transform("train"), few_shot=args.few_shot)
        dataset_val = VinDrCXR(images_path=args.data_dir, file_path=args.val_list, augment=transform("valid"))
        dataset_test = VinDrCXR(images_path=args.data_dir, file_path=args.test_list, augment=transform("test"))
        classification_engine(args, model_path, diseases, dataset_train, dataset_val, dataset_test, num_classes=6)


if __name__ == '__main__':
    args = get_args_parser()
    main(args)
