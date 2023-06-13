from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.misc import *
from utils.adapt_helpers import *

import torchvision.transforms as T



from utils.test_helpers import test_nm, test
import json 


parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
# parser.add_argument('--dataroot', default='/data/datasets/imagenet/')
parser.add_argument('--shared', default=None)

# TEDDI added... need train_labels and val_labels ?
parser.add_argument('--dataroot', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/celebAdata/')
# parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/setting1_test.csv')
# parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/setting1_val.csv')
parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/setting1_val.csv')
########################################################################
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=4, type=int)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=10, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='clf_results')




args = parser.parse_args()
args.threshold += 0.001		# to correct for numeric errors
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)

def gn_helper(planes):
	return nn.GroupNorm(args.group_norm, planes)
norm_layer = gn_helper
net = models.resnet18(norm_layer=norm_layer).cuda()
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 2).cuda()
net = torch.nn.DataParallel(net)
teset, _ = prepare_test_data(args, use_transforms=True)
# TEDDI TODO make sure prepare_test_data (defined in utils/train_helpers.py) properly loads celebA data
print('Resuming from %s...' %(args.resume))
# ckpt_path_name = 'results/debug_nicole/ckpt_1' 

ckpt_path_name = 'results/resnet18_layer3_gn_seed1_fixed/ckpt_90'

# ckpt_path_name = 'results/resnet18_layer3_gn_wd_0.1_fixed/ckpt_1'
ckpt = torch.load(ckpt_path_name+'.pth')
# ckpt = torch.load('%s/ckpt.pth' %(args.resume))

def get_new_statedict(prev_state_dict):
	new_state_dict = {}
	load_prefix = list(prev_state_dict.keys())[0][:6]
	for key in prev_state_dict:
		value = prev_state_dict[key]
		if load_prefix == 'module':
			new_key = key[7:]
			new_state_dict[new_key] = value
		else:
			new_state_dict[key] = value

	return new_state_dict


if args.online:
	net.load_state_dict(ckpt['net'])
	# net.load_state_dict(get_new_statedict(ckpt['net']))
	head.load_state_dict(ckpt['head'])

net.load_state_dict(ckpt['net'])
head.load_state_dict(ckpt['head'])


criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(ssh.parameters(), lr=args.lr)

## run validation eval to get  11%
_, valloader = prepare_test_data(args)
err_cls, one_hot_cls, losses_cls = test_nm(valloader, net, ckpt, verbose=True)
# err_cls, one_hot_cls, losses_cls = test(valloader, net, verbose=True)
print('err_cls: ', err_cls)
