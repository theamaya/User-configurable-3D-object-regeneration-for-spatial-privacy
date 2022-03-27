from path import Path
import os, shutil
import numpy as np
import random

def getDirNum(num):
    l= len(str(num))
    return '0'*(3-l) +str(num)

# for i in range(100):
#   parent_dir= '/home/ubuntu/Pointnet/bathtub_intra_class_dataset100/bathtub_'+getDirNum(i)
#   dir1='Train'
#   dir2='Valid'
#   path= os.path.join(parent_dir, dir1) 
#   #shutil.rmtree(path) 
#   #!rm -r parent_dir+'/'+ dir1
#   os.mkdir(path)
#   path= os.path.join(parent_dir, dir2) 
#   #!rm -r parent_dir+'/'+ dir2
#   #shutil.rmtree(path) 
#   os.mkdir(path)


for i in range(0,100):
  print(i)
  parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/all_intra_class_dataset100/bathtub/bathtub_'+getDirNum(i)+'/coordinates'
  test_dir= '/home/ubuntu/Pointnet/bathtub_intra_class_dataset100/bathtub_'+getDirNum(i)+'/Valid'
  train_dir= '/home/ubuntu/Pointnet/bathtub_intra_class_dataset100/bathtub_'+getDirNum(i)+'/Train'
  a= np.asarray(random.sample(range(500), 100)) #length 100
  b= np.arange(500)
  c =list(filter(lambda x: x not in a, b))   ### length 400

  for j in range(100):
    m= a[j]
    n= '0'*(3-len(str(m)))+str(m)
    filepath= os.path.join(parent_dir, 'bathtub_'+getDirNum(i)+'_'+n+'_recon_coord.pt')
    shutil.copy(filepath, test_dir)
  for j in range(400):
    m= c[j]
    n= '0'*(3-len(str(m)))+str(m)
    filepath= os.path.join(parent_dir, 'bathtub_'+getDirNum(i)+'_'+n+'_recon_coord.pt')
    shutil.copy(filepath, train_dir)