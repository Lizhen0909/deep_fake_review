
import argparse
import os
import csv
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
import torch.utils.data as data
from PIL import Image

from networks.resnet import resnet50

from tqdm import tqdm

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
parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-b','--batch_size', type=int, default=32)
parser.add_argument('-j','--workers', type=int, default=4, help='number of workers')
parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--size_only', action='store_true', help='only look at sizes of images in dataset')

opt = parser.parse_args()

# Load model
if(not opt.size_only):
  model = resnet50(num_classes=1)
  if(opt.model_path is not None):
      state_dict = torch.load(opt.model_path, map_location='cpu')
  model.load_state_dict(state_dict['model'])
  model.eval()
  if(not opt.use_cpu):
      model.cuda()

# Transform
trans_init = []
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset loader
data_loaders = []
if True:
  dataset = FileListDataset(opt.input, transform=trans)
  data_loaders+=[torch.utils.data.DataLoader(dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=False,
                                          num_workers=opt.workers),]

y_true, y_pred = [], []
Hs, Ws = [], []
with torch.no_grad():
  for data_loader in data_loaders:
    for data, label in tqdm(data_loader):
    # for data, label in data_loader:
      Hs.append(data.shape[2])
      Ws.append(data.shape[3])

      y_true.extend(label.flatten().tolist())
      if(not opt.size_only):
        if(not opt.use_cpu):
            data = data.cuda()
        y_pred.extend(model(data).sigmoid().flatten().tolist())
      #break

Hs, Ws = np.array(Hs), np.array(Ws)
y_true, y_pred = np.array(y_true), np.array(y_pred)

paths=dataset.dataset
with open(opt.output, 'wt') as f:
    for i in range(len(y_pred)):
        f.write("{},{:.4f}\n".format( paths[i], y_pred[i]))

#print('Average sizes: [{:2.2f}+/-{:2.2f}] x [{:2.2f}+/-{:2.2f}] = [{:2.2f}+/-{:2.2f} Mpix]'.format(np.mean(Hs), np.std(Hs), np.mean(Ws), np.std(Ws), np.mean(Hs*Ws)/1e6, np.std(Hs*Ws)/1e6))
#print('Num reals: {}, Num fakes: {}'.format(np.sum(1-y_true), np.sum(y_true)))

#print(y_pred)