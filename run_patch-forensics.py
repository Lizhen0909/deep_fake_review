from options.test_options import TestOptions
from models import create_model

import numpy as np
import os
import torch
import torch.utils.data as data

from utils import pidfile, util, imutil, pbar
from utils import util
from utils import imutil
from torch.utils.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
from PIL import Image
from IPython import embed

from data import transforms


torch.backends.cudnn.benchmark = False


def default_loader(path):
     return Image.open(path).convert('RGB')

class FileListDataset(data.Dataset):
    def __init__(self, list_file, opt, is_val=True):
        with open(list_file, 'rt') as f:
            all_paths=[x.strip() for x in f.readlines()];
        print('num of images=', len(all_paths))        
        self.dataset = all_paths
        self.transform = transforms.get_transform(opt, for_val=is_val)


    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = default_loader(item)
        # add transforms
        if self.transform is not None:
            img=self.transform(img)
        return {'img': img,
                'path': item
               }
    
    

    def __len__(self):
        return len(self.dataset)
    
def run_eval(opt):
    assert opt.model == 'patch_discriminator'
    model = create_model(opt)
    model.setup(opt)
    model.eval()


    # values to track
    paths = []
    labels = []
    
    prediction_voted = []
    prediction_avg_after_softmax = []
    prediction_avg_before_softmax = []
    prediction_raw = []
    
    print('gpuid=', opt.gpu_ids)
    if True:
        dset = FileListDataset(opt.input, opt, is_val=True)
        dl = DataLoader(dset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.nThreads, pin_memory=False)

        for i, data in enumerate(dl):
            # set model inputs
            ims = data['img']#.to(opt.gpu_ids[0])
            pred_labels = (torch.ones(ims.shape[0], dtype=torch.long)
                           * 1)#.to(opt.gpu_ids[0])
            inputs = dict(ims=ims, labels=pred_labels)

            # forward pass
            model.reset()
            model.set_input(inputs)
            model.test(True)
            predictions = model.get_predictions()

            # update counts
            labels.append(pred_labels.cpu().numpy())
            prediction_voted.append(predictions.vote)
            prediction_avg_before_softmax.append(predictions.before_softmax)
            prediction_avg_after_softmax.append(predictions.after_softmax)
            prediction_raw.append(predictions.raw)
            paths.extend(data['path'])
            #break;
    
    
    # compute and save metrics
    if opt.model == 'patch_discriminator':
        # save precision, recall, AP metrics on voted predictions
        prediction_voted=np.concatenate(prediction_voted)[:,0]
        prediction_avg_before_softmax=np.concatenate(prediction_avg_before_softmax)[:,0]
        prediction_avg_after_softmax=np.concatenate(prediction_avg_after_softmax)[:,0]
        
        #print(prediction_voted,prediction_avg_before_softmax,prediction_avg_after_softmax)
        with open(opt.output, 'wt') as f:
            for i in range(len(prediction_voted)):
                f.write("{},{:.4f},{:.4f},{:.4f}\n".format( paths[i], 
                prediction_voted[i], prediction_avg_before_softmax[i], prediction_avg_after_softmax[i] ))
        

    else:
        assert False
        # save precision, recall, AP metrics, on non-patch-based model
        print(np.concatenate(prediction_raw),
                        np.concatenate(labels))


if __name__ == '__main__':
    testoptions=TestOptions()
    parser=testoptions.parser
    parser.add_argument('-i', '--input', dest='input',                        help='input_file_list')
    parser.add_argument('-o', '--output', dest='output', help='Path to save outputs.',                        default='./output')    

    opt = testoptions.parse()
    
    print("Evaluating model: %s epoch %s" % (opt.name, opt.which_epoch))
    print("On dataset (real): %s" % (opt.real_im_path))
    print("And dataset (fake): %s" % (opt.fake_im_path))

    # check if checkpoint is out of date (e.g. if model is still training)
    ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_net_D.pth'
                             % opt.which_epoch)
    run_eval(opt)


