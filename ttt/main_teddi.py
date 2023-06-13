import argparse
import time
import wandb
import json

import torch
import torch.nn as nn
import torch.optim

from utils.misc import *
from utils.test_helpers import test
from utils.train_helpers import *

import difflib

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/celebAdata/')
parser.add_argument('--train_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/debug_setting1_train.csv')
parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/debug30_setting1_val.csv')
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--use_pretrained', default="False", type=str)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')

args = parser.parse_args()
args.use_pretrained= args.use_pretrained=='True'
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)
_, teloader = prepare_test_data(args)
_, trloader = prepare_train_data(args)

parameters = list(net.parameters())+list(head.parameters())
optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(reduction='none').cuda()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="ttt",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
	"weight_decay": args.weight_decay,
    "dataset": "CELEBA",
    "epochs": args.epochs,
    }
)




def train(trloader, epoch):
	net.train()
	ssh.train()
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	progress = ProgressMeter(len(trloader), batch_time, data_time, losses, top1, 
											prefix="Epoch: [{}]".format(epoch))

	end = time.time()
	all_loss_cls, all_loss_ssh, all_loss_combined, all_acc = 0,0,0,0
	for i, dl in enumerate(trloader):
		data_time.update(time.time() - end)
		optimizer.zero_grad()

		inputs_cls, labels_cls = dl[0].cuda(), dl[1].cuda()
		outputs_cls = net(inputs_cls)
		loss_cls = criterion(outputs_cls, labels_cls)
		loss = loss_cls.mean()
		losses.update(loss.item(), len(labels_cls))
		
		_, predicted = outputs_cls.max(1)
		acc1 = predicted.eq(labels_cls).sum().item() / len(labels_cls)
		top1.update(acc1, len(labels_cls))

		if args.shared is not None:
			inputs_ssh, labels_ssh = dl[2].cuda(), dl[3].cuda()
			outputs_ssh = ssh(inputs_ssh)
			loss_ssh = criterion(outputs_ssh, labels_ssh)
			loss += loss_ssh.mean()

		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()
		if i % args.print_freq == 0:
			progress.print(i)

		all_loss_cls+=loss_cls.mean().item()
		all_loss_ssh+=loss_ssh.mean().item()
		all_loss_combined+=loss.item()
		all_acc+=acc1


	N = len(trloader)
	return all_loss_cls/N, all_loss_ssh/N, all_loss_combined/N, all_acc/N

all_err_cls = []
all_err_ssh = []

if args.resume is not None:
	print('Resuming from checkpoint..')
	ckpt = torch.load('%s/ckpt.pth' %(args.resume))
	net.load_state_dict(ckpt['net'])
	head.load_state_dict(ckpt['head'])
	optimizer.load_state_dict(ckpt['optimizer'])
	loss = torch.load('%s/loss.pth' %(args.resume))
	all_err_cls, all_err_ssh = loss

for epoch in range(args.start_epoch, args.epochs+1):
	adjust_learning_rate(optimizer, epoch, args)
	for param_group in optimizer.param_groups:
		param_group['weight_decay'] = args.weight_decay

	train_loss_cls, train_loss_ssh, train_loss_combined, train_acc = train(trloader, epoch)
	teloader.dataset.switch_mode(True, False)
	err_cls, one_hot_cls, losses_cls = test(teloader, net, verbose=True)
	if args.shared is not None:
		teloader.dataset.switch_mode(False, True)
		err_ssh, one_hot_ssh, losses_ssh = test(teloader, ssh, verbose=True)
	else:
		err_ssh = 0

	all_err_cls.append(err_cls)
	all_err_ssh.append(err_ssh)
	torch.save((all_err_cls, all_err_ssh), args.outf + '/loss.pth')
	wandb.log({"test_err_cls": all_err_cls[-1], "test_err_ssh": all_err_ssh[-1], 
		'test_losses_ssh':losses_ssh.mean(), 'test_losses_cls': losses_cls.mean(),
		'train_classificational_loss': train_loss_cls, 'train_loss_ssh': train_loss_ssh, 
		'train_loss_combined': train_loss_combined, 'train_acc': train_acc})
	
	test_results = {'one_hot_ssh': one_hot_ssh.tolist(),'one_hot_cls': one_hot_cls.tolist()}

	# with open(args.outf + 'test_results_{}.json'.format(epoch), 'w') as fp:
                # json.dump(test_results, fp)

	# plot_epochs(all_err_cls, all_err_ssh, args.outf + '/loss.pdf')

	state = {'args': args, 'err_cls': err_cls, 'err_ssh': err_ssh, 
				'optimizer': optimizer.state_dict(), 'net': net.state_dict(), 'head': head.state_dict()}
	torch.save(state, args.outf + '/ckpt_{}.pth'.format(epoch))


# load model from ckpt
net2, ext2, head2, ssh2 = build_model(args)

def gn_helper(planes):
        return nn.GroupNorm(args.group_norm, planes)

norm_layer = gn_helper
net2 = models.resnet18(norm_layer=norm_layer).cuda()
num_features = net2.fc.in_features
net2.fc = nn.Linear(num_features, 2).cuda()
# teset, _ = prepare_test_data(args, use_transforms=True)
# TEDDI TODO make sure prepare_test_data (defined in utils/train_helpers.py) properly loads celebA data
# print('Resuming from %s...' %(args.resume))
ckpt_path_name = args.outf + '/ckpt_1'
# ckpt_path_name = 'results/resnet18_layer3_gn_wd_0.1_fixed/ckpt_1'
ckpt = torch.load(ckpt_path_name+'.pth')
# ckpt = torch.load('%s/ckpt.pth' %(args.resume))
load_state_dict_net = ckpt['net']
new_state_dict = {}
load_prefix = list(load_state_dict_net.keys())[0][:6]
for key in load_state_dict_net:
        value = load_state_dict_net[key]
        if load_prefix == 'module':
                new_key = key[7:]
                new_state_dict[new_key] = value
        else:
                new_state_dict[key] = value

net2 = torch.nn.DataParallel(net2) 
net2.load_state_dict(ckpt['net'])
head2.load_state_dict(ckpt['head'])

# check models in terms of parameter equality
state_net = net.state_dict().__str__()
state_net2 = net2.state_dict().__str__()
state_head = head.state_dict().__str__()
state_head2 = head2.state_dict().__str__()
if state_net == state_net2:
    print("Nets loaded from ckpt correctly")
else:
    print("Nets loaded from ckpt incorrectly")
    #diff = difflib.ndiff(state_net, state_net2)
    #print(list(diff))
    with open("state_net.txt", "w") as text_file:
        text_file.write(state_net)
    with open("state_net2.txt", "w") as text_file:
                text_file.write(state_net2)
if state_head == state_head2:
    print("Heads loaded from ckpt correctly")
else:
    print("Heads loaded from ckpt incorrectly")
    diff = difflib.ndiff(state_head, state_head2)
    print(list(diff))
