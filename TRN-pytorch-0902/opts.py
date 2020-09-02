import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str, choices=['something','jester','moments', 'somethingv2','epic','epic_test'], default='epic')
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'], default= 'RGB')
parser.add_argument('--train_list', type=str,default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--test_pickle', type=str, default="")


parser.add_argument('--root_path', type=str, default="/data/acq18jg/epic/frames_rgb_flow", help='root_path for dataset ONLY')
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--new_length', type=int, default= None)
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.8, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size ')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[30, 60], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False )

# ========================= Monitor Configs ==========================


parser.add_argument('--print_model', '-pm', default=False, 
                    help='print_model (default: 1)')
parser.add_argument('--print_training_in_terminal', '-ifp', default=True, 
                    help='print_training_in_terminal (default: true)')
parser.add_argument('--print_freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency of batch (default: 10)')
parser.add_argument('--eval_freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency of epoch , no use now')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers ')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', #action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--root_output',type=str, default='output')



