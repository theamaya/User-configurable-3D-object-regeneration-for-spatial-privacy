from path import Path
import os, shutil
import numpy as np
import random

def getDirNum(num):
    l= len(str(num))
    return '0'*(3-l) +str(num)

for i in range(100):
  parent_dir= '/home/ubuntu/Pointnet/chair_intra_class_epoch_dataset100/chair_'+getDirNum(i)
  dir1='plots'
  path= os.path.join(parent_dir, dir1) 
  shutil.rmtree(path) 