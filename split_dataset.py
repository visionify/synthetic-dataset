
from pathlib import Path
from sklearn.model_selection import  StratifiedShuffleSplit
import shutil
import os, sys
import numpy as np
import shutil
import random
from pathlib import Path
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # az2yolo root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class split_data():
  def __init__(self):
    self.root_dir = ' ' #"class_dataset/" 
    self.classes_dir = []
    self.input_destination = ' ' #'splitted-sythetic-dataset/'
    self.train_ratio = 0.0
    self.val_ratio  = 0.0

  def main(self, opt):
    self.root_dir = opt.dataset_folder
    self.input_destination = opt.splitted_dataset
    self.train_ratio = opt.split_ratio
    self.classes_dir = os.listdir(self.root_dir)
    for cls in self.classes_dir:
        os.makedirs(self.input_destination+'/' +'train/' + cls, exist_ok=True)
        os.makedirs(self.input_destination+'/' +'test/' + cls, exist_ok=True)
        os.makedirs(self.input_destination+'/' +'val/' + cls, exist_ok=True)
        src = self.root_dir + '/' + cls
        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames, val_FileNames = np.split(np.array(allFileNames),[int(self.train_ratio * len(allFileNames)), int((1-self.val_ratio) * len(allFileNames))])
        train_FileNames = [src+'/'+ name  for name in train_FileNames.tolist()]
        test_FileNames  = [src+'/' + name for name in test_FileNames.tolist()]
        val_FileNames   = [src+'/' + name for name in val_FileNames.tolist()]
        print("Total images: ",cls, len(allFileNames),'  Training: ', len(train_FileNames),'  Testing: ', len(test_FileNames),
              '  Validation: ', len(val_FileNames),)
        for name in train_FileNames:
          shutil.copy(name, self.input_destination+'/' +'train/' + cls)
        for name in test_FileNames:
          shutil.copy(name, self.input_destination+'/' +'test/' + cls)
        for name in val_FileNames:
          shutil.copy(name, self.input_destination+'/' +'val/' + cls)

    # checking 
    # paths = ['train/', 'test/','val/']
    # for p in paths:
    #   for dir,subdir,files in os.walk(self.input_destination + p):
    #     print(dir,' ', p, str(len(files)))

  def parse_opt(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', default = 'class_dataset', type=str, help='Path to input datset folder')
    parser.add_argument('--splitted-dataset', default = 'splitted_dataset', type=str, help='path to the splitted dataset')
    parser.add_argument('--split-ratio', default = 0.8 , help='split ratio for train-test')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    split_d = split_data()
    opt = split_d.parse_opt()
    split_d.main(opt)