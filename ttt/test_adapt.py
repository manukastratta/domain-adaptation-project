from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.misc import *
from utils.adapt_helpers import *

import torchvision.transforms as T

import random

from utils.test_helpers import test
import json 

# Set seed
SEED = 13
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED) # Set a fixed value for the hash seed 
print(f"Random seed set as {SEED}")

parser = argparse.ArgumentParser()
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
# parser.add_argument('--dataroot', default='/data/datasets/imagenet/')
parser.add_argument('--shared', default=None)

# TEDDI added... need train_labels and val_labels ?
parser.add_argument('--dataroot', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/celebAdata/')
parser.add_argument('--debug', default=False)
parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/setting1_test.csv')
# parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/debug30_setting1_test.csv')
# parser.add_argument('--val_labels', default='/data/yusun/manuka_nicole_teddi/test-time-training-project/data/CelebA/splits/setting1_val.csv')
parser.add_argument('--ckpt', type=str)
########################################################################
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--batch_examples', default=1, type=int)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--niter', default=10, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--nonstationary_distrib', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
parser.add_argument('--dset_size', default=0, type=int)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='clf_results')
########################################################################
parser.add_argument('--visualize_pred', default=0, type=int)


args = parser.parse_args()
args.threshold += 0.001		# to correct for numeric errors
my_makedir(args.outf)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
net, ext, head, ssh = build_model(args)


def gn_helper(planes):
	return nn.GroupNorm(args.group_norm, planes)
norm_layer = gn_helper
# net = models.resnet18(norm_layer=norm_layer).cuda()
#num_features = net.fc.in_features
#net.fc = nn.Linear(num_features, 2).cuda()
teset, teloader = prepare_test_data(args, use_transforms=True, return_imgnames=True)
print('Resuming from %s...' %(args.resume))
ckpt_path_name = args.ckpt

ckpt = torch.load(ckpt_path_name+'.pth')
# ckpt = torch.load('%s/ckpt.pth' %(args.resume))

load_state_dict_net = ckpt['net']
# new_state_dict = {}
# load_prefix = list(load_state_dict_net.keys())[0][:6]
# for key in load_state_dict_net:
	# value = load_state_dict_net[key]
	# if load_prefix == 'module':
		# new_key = key[7:]
		# new_state_dict[new_key] = value
	# else:
		# new_state_dict[key] = value

if args.online:
        net.load_state_dict(ckpt['net'])
        head.load_state_dict(ckpt['head'])
        print('In ONLINE mode')
else:
        print("In OFFLINE mode")

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(ssh.parameters(), lr=args.lr)


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

def count_conf_bad_pred(correctness, confidence): # size torch.Size([8]), size torch.Size([8, 2])
        indices = torch.argmax(confidence, axis=1)
        selected_confidences = confidence[torch.arange(indices.size(0)), indices]
        conf_bad_pred = (correctness == 0) & (selected_confidences >= 0.9)
        return conf_bad_pred.sum().item()

confident_bad_pred_seen = 0
if args.batch_examples > 1:
        print("In MULTIPLE online mode")
        for batch_idx, (images, labels, img_filenames) in enumerate(tqdm(teloader)):
                """
                image: torch.Size([4, 3, 224, 224])
                label: torch.Size([4])
                img_filename: ['020347.jpg', '062094.jpg', '174550.jpg', '049646.jpg...
                """
                n_examples = images.shape[0]

                if not args.online:
                        net.load_state_dict(ckpt['net'])
                        head.load_state_dict(ckpt['head'])
                
                sshconf.append(test_multiple(ssh, images, torch.zeros(n_examples))[1])

                net_og = net.state_dict().__str__()
                # if the self supervised head has low confidence, train (adapt) model  ## could check here 
                #if sshconf[-1] < args.threshold:
                pil_images =  [T.ToPILImage()(img) for img in images]
                adapt_multiple(ssh, pil_images, optimizer, criterion, args.niter, args.batch_size)

                 # check if net has been updated by adapt_single
                net_adapted = net.state_dict().__str__()
                if net_og != net_adapted:
                        print('Net adapted!')
                # testing the classification head
                correctness, confidence, predictions = test_multiple(net, images, labels)
                correct.append(correctness)
                
                for i in range(len(img_filenames)):
                        pred = predictions[i].item()
                        prediction = {'img_filename': img_filenames[i],
                                      'prediction': pred,
                                      'conf': confidence[i][pred].item(),
                                      'ground_truth': int(labels[i].item())
                                      }
                        result_ls.append(prediction)
                
                if args.visualize_pred > 0:
                        confident_bad_pred_seen += count_conf_bad_pred(correctness, confidence)
                        print("confident_bad_pred_seen: ", confident_bad_pred_seen)
                        if confident_bad_pred_seen >= args.visualize_pred:
                                break
                

else:
        print("In single mode (update 1 at a time)")
        for i in tqdm(range(1, len(teset)+1)):
                if not args.online:
                        net.load_state_dict(ckpt['net'])
                        head.load_state_dict(ckpt['head'])
                # _, valloader = prepare_test_data(args)
                # err_cls, one_hot_cls, losses_cls = test(valloader, net, verbose=True)
                # print('err_cls: ', err_cls)
                # breakpoint()

                image, label, img_filename = teset[i-1] # image: torch.Size([3, 224, 224]); label: int; img_filename: list
                # print()
                # print('image shape', image.shape)
                # print('label', label)
                # print('file name:', img_filename)
                # transform = T.ToPILImage()
                # image = transform(image)
                sshconf.append(test_single(ssh, image, 0)[1])

                net_og = net.state_dict().__str__()
                # if the self supervised head has low confidence, train (adapt) model  ## could check here 
                if sshconf[-1] < args.threshold:
                        pil_image =  T.ToPILImage()(image)
                        adapt_single(ssh, pil_image, optimizer, criterion, args.niter, args.batch_size)
                
                # check if net has been updated by adapt_single
                net_adapted = net.state_dict().__str__()
                if net_og != net_adapted:
                        print('Net adapted!')
                # testing the classification head
                correctness, confidence, prediction = test_single(net, image, label)
                correct.append(correctness)
                # trerror.append(trerr_single(ssh, image))
                # if (prediction == 1):
                #     print('label:', label)
                #     print('prediction:', prediction)
                #     print('confidence:', confidence)

                prediction = {'img_filename': img_filename, 'prediction': prediction}
                result_ls.append(prediction)

if args.batch_examples > 1:
        # Flatten & convert to cpu
        correct = [item.item() for tensor in correct for item in tensor]
        sshconf = torch.cat([tensor.cpu() for tensor in sshconf], dim=0).numpy()

print('Adapted test error cls %.2f' %((1-mean(correct))*100))
rdict = {'cls_correct': np.asarray(correct), 'ssh_confide': np.asarray(sshconf), 'cls_adapted':1-mean(correct), 'trerror': trerror}
torch.save(rdict, args.outf + '/%s_%d_ada.pth' %(args.corruption, args.level))

if args.visualize_pred > 0:
        predictions_filename = "/data/yusun/manuka_nicole_teddi/test-time-training-project/ttt_imagenet_release/"+ckpt_path_name+"_results_lr_visualize_predictions.json"
elif args.debug:
        # predictions_filename = "temp.json"
        predictions_filename = "/data/yusun/manuka_nicole_teddi/test-time-training-project/ttt_imagenet_release/"+ckpt_path_name+"_results_lr"+str(args.lr) + "_online_+"+str(args.online) + "_batchexamples_+"+str(args.batch_examples) + "_nonstationary_distrib_+"+str(args.nonstationary_distrib)+"_DEBUG_" + ".json"
else:
        # predictions_filename = "temp.json"
        predictions_filename = "/data/yusun/manuka_nicole_teddi/test-time-training-project/ttt_imagenet_release/"+ckpt_path_name+"_results_lr"+str(args.lr) + "_online_+"+str(args.online) + "_batchexamples_+"+str(args.batch_examples) + "_nonstationary_distrib_+"+str(args.nonstationary_distrib)+".json"

with open(predictions_filename, 'w') as fp: 
    json.dump(result_ls, fp)