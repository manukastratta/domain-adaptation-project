from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.misc import *
from utils.adapt_helpers import *

import torchvision.transforms as T



from utils.test_helpers import test
import json 


parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
# parser.add_argument('--dataroot', default='/data/datasets/imagenet/')
parser.add_argument('--shared', default=None)

# TEDDI added... need train_labels and val_labels ?
parser.add_argument('--dataroot', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/celebAdata/')
# parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/setting1_test.csv')
parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/debug30_setting1_val.csv')
# parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/debug_setting1_val.csv')


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
teset, _ = prepare_test_data(args, use_transforms=True)
# TEDDI TODO make sure prepare_test_data (defined in utils/train_helpers.py) properly loads celebA data
print('Resuming from %s...' %(args.resume))
ckpt_path_name = 'results/resnet18_layer3_gn_wd_0.1_FINAL/ckpt_90'
ckpt = torch.load(ckpt_path_name+'.pth')
# ckpt = torch.load('%s/ckpt.pth' %(args.resume))
print('ckpt net!!!!')
print(ckpt['net'].keys())
print('ckpt head!!!')
print(ckpt['head'].keys())

load_state_dict_net = ckpt['net']
new_state_dict = {}
load_prefix = list(load_state_dict_net.keys())[0][:6]
for key in load_state_dict_net:
	value = load_state_dict_net[key]
	# # Multi-GPU state dict has the prefix 'module.' appended in front of each key
	# if torch.cuda.device_count() > 1:
	# 	if load_prefix != 'module':
	# 		new_key = 'module.' + key
	# 		new_state_dict[new_key] = value
	# 	else:
	# 		new_state_dict[key] = value
	# else:
	if load_prefix == 'module':
		new_key = key[7:]
		new_state_dict[new_key] = value
	else:
		new_state_dict[key] = value

if args.online:
	net.load_state_dict(new_state_dict)
	head.load_state_dict(ckpt['head'])

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(ssh.parameters(), lr=args.lr)


## run validation eval to get  11%


print('Running...')
teset.switch_mode(True, False)
if args.dset_size > 0:
	teset, _ = torch.utils.data.random_split(teset, (args.dset_size, len(teset)-args.dset_size))
if args.shuffle:
	teset, _ = torch.utils.data.random_split(teset, (len(teset), 0))
correct = []
sshconf = []
trerror = []
results = {}
result_ls = []
for i in tqdm(range(1, len(teset)+1)):
        if not args.online:
                net.load_state_dict(new_state_dict)
                head.load_state_dict(ckpt['head'])

        image, label, img_filename = teset[i-1]
        transform = T.ToPILImage()
        image = transform(image)
        ssh_correctness, ssh_confidence, ssh_prediction = test_single(ssh, image, 0)
        sshconf.append(ssh_confidence)
        print()
        print('ssh confidence:', ssh_confidence)
        # if the self supervised head has low confidence, train (adapt) model  ## could check here 
        
        # if sshconf[-1] < args.threshold:
                # adapt_single(ssh, image, optimizer, criterion, args.niter, args.batch_size)

        # testing the classification head
        correctness, confidence, prediction = test_single(net, image, label)
        correct.append(correctness)
        trerror.append(trerr_single(ssh, image))
        print('label:', label)
        print('prediction:', prediction)
        prediction = {'img_filename': img_filename, 'prediction': prediction}
        result_ls.append(prediction)
        results[img_filename] = prediction

with open("/data/yusun/manuka_nicole_teddi/test-time-training-project/ttt_imagenet_release/results/ttt_eval.json", 'w') as fp:
    json.dump(results, fp)
with open("/data/yusun/manuka_nicole_teddi/test-time-training-project/ttt_imagenet_release/clf_results/"+ckpt_path_name+".json", 'w') as fp: 
    json.dump(result_ls, fp)

print('Adapted test error cls %.2f' %((1-mean(correct))*100))
rdict = {'cls_correct': np.asarray(correct), 'ssh_confide': np.asarray(sshconf), 'cls_adapted':1-mean(correct), 'trerror': trerror}
torch.save(rdict, args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))