import numpy as np
import PIL
import torch
import sys
import os
import pandas as pd
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.misc import *
from utils.adapt_helpers import *

import torchvision.transforms as T

from PIL import Image


###
# Referenced from:
# https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py
#
# Example usage:
# python get_cams.py --modelpath $MODELPATH --img_ids 535811 430054 554674
#
# --modelpath: path to the model to visualize
# --img_ids: COCOStuff image IDs (use the Explore tool on the COCO dataset website)
###

def get_heatmap(CAM_map, img):
    CAM_map = cv2.resize(CAM_map, (img.shape[0], img.shape[1]))
    CAM_map = CAM_map - np.min(CAM_map)
    CAM_map = CAM_map / np.max(CAM_map)
    CAM_map = 1.0 - CAM_map # make sure colormap is not reversed
    heatmap = cv2.applyColorMap(np.uint8(255 * CAM_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap + np.float32(img)
    heatmap = heatmap / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    return heatmap

def returnCAM(feature_conv, weight_softmax, class_labels, device):
    bz, nc, h, w = feature_conv.shape # (1, hidden_size, height, width)
    output_cam = torch.Tensor(0, 7, 7).to(device=device)
    for idx in class_labels:
        cam = torch.mm(weight_softmax.squeeze()[int(idx)].unsqueeze(0), feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - cam.min()
        cam_img = cam / cam.max()
        output_cam = torch.cat([output_cam, cam_img.unsqueeze(0)], dim=0)
    return output_cam

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelpath', type=str, default=None)
    parser.add_argument('--img_ids', type=str, nargs='+', default='022405')
    parser.add_argument('--outdir', type=str, default='cam_results/')
    parser.add_argument('--split', type=int, default=1024)
    parser.add_argument('--coco2014_images', type=str, default=None)
    parser.add_argument('--device', default=torch.device('cpu'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--test_csv', type=str, default='setting2_test.csv')
    parser.add_argument('--shared', default=None)
    parser.add_argument('--depth', default=18, type=int)
    parser.add_argument('--group_norm', default=0, type=int)
    
    args = parser.parse_args()

    net, ext, head, ssh = build_model(args)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    classifier_features = []
    def hook_classifier_features(module, input, output):
        classifier_features.append(output)

    ckpt = torch.load(args.modelpath, map_location=torch.device('cpu'))
    # ckpt = torch.load('%s/ckpt.pth' %(args.resume))

    net.load_state_dict(ckpt['net'])

    net.module.layer4.register_forward_hook(hook_classifier_features)
    classifier_params = list(net.parameters())
    classifier_softmax_weight = classifier_params[-2].squeeze(0)

    # TO DO CHANGE THIS TO ARGUMENT
    df = pd.read_csv(args.test_csv)

    onehot_to_humanlabels  = {0: 'Not Black Hair', 1: "Black Hair"}

    for img_id in args.img_ids:
        # Open image
        

        # img_path = '/data/yusun/manuka_nicole_teddi/test-time-training-project/celebAdata/img_align_celeba/{}.jpg'.format(img_id)
        img_path = '{}.jpg'.format(img_id)
        img_name = img_path.split('/')[-1][:-4]
        if not os.path.exists(img_path):
            print('WARNING: Could not find img {}'.format(img_id), flush=True)
            continue
        original_img = Image.open(img_path).convert('RGB')

        if args.outdir != None:
            outdir = '{}/{}'.format(args.outdir, img_id)
        else:
            outdir = str(img_id)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print('Processing img {}'.format(img_id), flush=True)

        # Get image class labels as torch ByteTensor TO DO!!! class_labels = torch.zeros(1)
        
        class_labels =torch.tensor([df[df['img_filename']==str(img_id)+'.jpg'].iloc[0]['Black_Hair']]).byte()
        # class_labels = torch.flatten(torch.nonzero(class_labels))

        classifier_features.clear()
        img = transform(original_img)
        norm_img = normalize(img)
        norm_img = norm_img.to(device=args.device, dtype=args.dtype)
        norm_img = norm_img.unsqueeze(0)
        x = net.forward(norm_img)

        CAMs = returnCAM(classifier_features[0], classifier_softmax_weight, class_labels, args.device)
        CAMs = CAMs.detach().cpu().numpy()

        img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
        class_labels = class_labels.cpu().detach().numpy()
        for i in range(len(class_labels)):
            heatmap = get_heatmap(CAMs[i], img)
            plt.figure()
            plt.imshow(heatmap)
            plt.axis('off')
            plt.title(onehot_to_humanlabels[class_labels[i]])
            humanlabel = onehot_to_humanlabels[class_labels[i]].replace(' ', '+')
            plt.savefig('{}/{}_{}.png'.format(outdir, img_name, humanlabel))
            plt.show()
            plt.close()

if __name__ == '__main__':
    main()