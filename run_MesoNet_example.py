from tqdm import tqdm
import numpy as np
import pandas as pd
from classifiers import *
#from pipeline import *

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def run(input_list_file, output):
    with open(input_list_file, 'rt') as f:
        all_paths=[x.strip() for x in f.readlines()];
    print('num of images=', len(all_paths), flush=True)        
    n_images=len(all_paths)
    df =pd.Series(all_paths).to_frame();
    df.columns=['filename']
    df['class']=[str(i%2) for i in range(len(df))]
    
    # 1 - Load the model and its pretrained weights
    classifier = Meso4()
    classifier.load('weights/Meso4_DF.h5')
    print('loaded model', flush=True)
    # 2 - Minimial image generator
    # We did use it to read and compute the prediction by batchs on test videos
    # but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

    dataGenerator = ImageDataGenerator(rescale=1./255)
    generator = dataGenerator.flow_from_dataframe(
        df,x_col='filename', y_col='class',
        target_size=(256, 256),
        class_mode='binary', 
        batch_size=128, 
        shuffle=False,
        subset='training'
    )
    
#     generator = dataGenerator.flow_from_directory(
#             'test_images',
#             target_size=(256, 256),
#             batch_size=1,
#             class_mode='binary',
#             subset='training')

    # 3 - Predict
    print("begin predit", flush=True)
    pred=[]
    count=0;
    for X, y in tqdm(generator):
        pred.append(classifier.predict(X))
        count += X.shape[0]
        if count>=n_images: break
        pass
        #print('Predicted :', pred[-1], '\nReal class :', y)
        #break
    print("processed {} images".format(count), flush=True)
    pred=np.concatenate(pred)
    df=df.iloc[:len(pred)]
    df['prob']=pred
    
    df[['filename','prob']].to_csv(output,header=None, index=None)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input',                        help='input_file_list')
    parser.add_argument('-o', '--output', dest='output', help='Path to save outputs.',                        default='./output')    
    opt = parser.parse_args()
    run(opt.input, opt.output)
