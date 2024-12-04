import os
import time
import pandas as pd
from code_package.comp2 import extract, print_feature_hook, reg_hook_on_module, init_from_model, init_from_onekey
from code_package.comp1 import compress_df_feature
from functools import partial

def main():
   #read image data in '*.png' or '*.jpg' format
   img_dir = r'./images'
   directory = os.path.expanduser(img_dir)
   test_samples = [os.path.join(directory, p) for p in os.listdir(directory) if p.endswith('.png') or p.endswith('.jpg')]

   #load the pre-trained resnet34-atttention model 
   model_path = r'./model'
   model, transformer, device = init_from_onekey(model_path)

   #extract the deep features and save it into the csv file
   feature_name = 'avgpool'
   with open('feature.csv', 'w') as outfile:
       hook = partial(print_feature_hook, fp=outfile)
       find_num = reg_hook_on_module(feature_name, model, hook)
       results = extract(test_samples, model, transformer, device, fp=outfile)

   #read extracted deep features
   features = pd.read_csv('feature.csv', header=None)
   features.columns = ['ID'] + list(features.columns[1:])

   #compress deep features
   cm_features = compress_df_feature(features=features, dim=64, prefix='DL_', not_compress='ID')
   cm_features.to_csv('compress_features.csv', header=True, index=False)

if __name__ == '__main__':
   start = time.time()
   main()
   end = time.time()
   print(end - start)


