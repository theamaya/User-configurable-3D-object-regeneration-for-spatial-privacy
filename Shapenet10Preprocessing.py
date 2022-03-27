from path import Path
import os, shutil
import numpy as np
import random

def getDirNum(num):
    l= len(str(num))
    return '0'*(3-l) +str(num)

for i in range(10):
  parent_dir= '/home/ubuntu/Pointnet/shapenet_dataset10/'+str(i)
  dir1='Train'
  dir2='Valid'
  path= os.path.join(parent_dir, dir1) 
  #shutil.rmtree(path) 
  #!rm -r parent_dir+'/'+ dir1
  if os.path.exists(path):
    pass
  else:
    os.mkdir(path)

  path= os.path.join(parent_dir, dir2) 
  #!rm -r parent_dir+'/'+ dir2
  #shutil.rmtree(path) 
  if os.path.exists(path):
    pass
  else:
    os.mkdir(path)

classes=['airplane','bathtub', 'bench', 'boat','cabinet', 'car','chair','monitor','sofa','table']

for i in range(0,10):
  print(i)
  parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/3d_point_cloud/dataset/shapenet_pointnet/'+classes[i]+'/'
  test_dir= '/home/ubuntu/Pointnet/shapenet_dataset10/'+str(i)+'/Valid'
  train_dir= '/home/ubuntu/Pointnet/shapenet_dataset10/'+str(i)+'/Train'
  a= np.asarray(random.sample(range(500), 100)) #length 100
  b= np.arange(500)
  c =list(filter(lambda x: x not in a, b))   ### length 400
  files= os.listdir(parent_dir)
  print(parent_dir, len(files))
  for j in range(100):
    m= a[j]
    n= '0'*(3-len(str(m)))+str(m)
    try:
        filepath= os.path.join(parent_dir, files[m])
    except :
        print (i, m)
    shutil.copy(filepath, test_dir)
  for j in range(400):
    m= c[j]
    n= '0'*(3-len(str(m)))+str(m)
    try:
        filepath= os.path.join(parent_dir, files[m])
    except :
        print (i, m)
    shutil.copy(filepath, train_dir)