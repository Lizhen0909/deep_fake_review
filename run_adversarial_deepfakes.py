"""
File: evaluation.py -- Model Evaluation Script
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020
"""

# Script Parameters

DIR_NAMES = ['perturbed_cw/vgg_reg1000_cw']

MODELS_DIR = 'models/'
MODEL_NAMES = ['resnet_blur.pt']

OUTPUT_DIR = 'results.csv'

SOFTMAX = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
#import cv2
from torchvision.utils import save_image
from sklearn import metrics
import scipy
import torch.utils.data as data
import argparse
from utils import HHReLU
print('class', HHReLU)

from PIL import Image
def default_loader(path):
     return Image.open(path).convert('RGB')

class FileListDataset(data.Dataset):
    def __init__(self, list_file, transform):
        with open(list_file, 'rt') as f:
            all_paths=[x.strip() for x in f.readlines()];
        print('num of images=', len(all_paths))        
        self.dataset = all_paths
        self.transform = transform


    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = default_loader(item)
        # add transforms
        if self.transform is not None:
            img=self.transform(img)
        return img, 0
    
    

    def __len__(self):
        return len(self.dataset)
    
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input', dest='input',                        help='input_file_list')
parser.add_argument('-o', '--output', dest='output', help='Path to save outputs.',                        default='./output')

opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

def input_gradient(images, model, z=2, use_softmax=SOFTMAX, num_classes = 2):
    if use_softmax:
      repeated_images = images.repeat(num_classes, 1, 1, 1, 1)
      repeated_output = torch.stack([model(repeated_images[0]).sum(axis=0), model(repeated_images[1]).sum(axis=0)])
      grads = torch.autograd.grad(repeated_output, repeated_images, grad_outputs=torch.eye(num_classes).to(device), create_graph=False)[0]
    else:
      grads = torch.autograd.grad(model(images).sum(), images, create_graph=False)[0]
    return grads.abs().pow(z).mean().cpu().detach().numpy()

def evaluate(model, data, device, use_softmax=SOFTMAX):
  labels = []
  scores = []
  preds = []
  grads = []
  losses = []

  for img, label in tqdm(data):      
      img = img.to(device) 
      output = model(img)

      # Calculate Gradient With Respect to Input
      img.requires_grad = True
      grad = input_gradient(img, model, use_softmax)
      img.requires_grad = False
      grads.append(grad)

      # Calculate Loss 
      if use_softmax:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label.to(device))
      else:
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, label.unsqueeze(1).float())
      losses.append(loss.squeeze().cpu().detach().numpy())

      # Calculate prediction
      if use_softmax:
        #print(output)
        score, pred = torch.max(output, 1)
        try:
          score = np.exp(output[:, 1].cpu().detach().numpy())/(output.exp().sum(1).cpu().detach().numpy())
          #print(score)
        except:
          print('Error applying softmax to model output. Setting output to 0.5.')
          score = 0.5
        pred = pred.squeeze().cpu().detach().numpy()
      else:
        score = scipy.special.expit(output.squeeze().cpu().detach().numpy())
        pred = 1 if (score >= 0.5) else 0

      labels.extend(label)
      scores.extend(score)
      preds.extend(pred)
      del img, score, pred, loss, label, output
      #break
    
  return labels, scores, preds, grads, losses

if True: 

  transformation = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((256,256)),])
  imgfolder = FileListDataset(opt.input, transform=transformation)
  data = torch.utils.data.DataLoader(imgfolder, batch_size=64, shuffle=True)

  for mn in MODEL_NAMES:
    print('Evaluating model:', mn)

    model = torch.load(MODELS_DIR+mn, map_location=device)
    model.to(device)

    labels, scores, preds, grads, losses = evaluate(model, data, device)
    #print(list(zip(scores,preds)))
    paths=imgfolder.dataset
    with open(opt.output, 'wt') as f:
        for i in range(len(preds)):
            f.write("{},{:.4f},{}\n".format( paths[i],scores[i], float(preds[i])))
        
    break
