# from path import Path
# import os, shutil
# import numpy as np
# import random

# classes=['bathtub',  'bed',  'bench',  'bookshelf',  'cabinet',  'chair',  'lamp',  'monitor',  'sofa',  'table']

# def getDirNum(num):
#     l= len(str(num))
#     return '0'*(3-l) +str(num)

# for c in classes:
#     #for i in range(100):
#     parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp2/'+c
#     dir1='Train'
#     dir2='Valid'
#     path= os.path.join(parent_dir, dir1) 
#     # if os.path.exists(path):
#     #     shutil.rmtree(path) 
#     #!rm -r parent_dir+'/'+ dir1
#     if os.path.exists(path):
#         pass
#     else:
#         os.mkdir(path)
#     path= os.path.join(parent_dir, dir2) 
#     #!rm -r parent_dir+'/'+ dir2
#     # if os.path.exists(path):
#     #     shutil.rmtree(path) 
#     if os.path.exists(path):
#         pass
#     else:
#         os.mkdir(path)

# for c in classes:
#     print(c)
#     parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp2/'+c
#     test_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp2/'+c+'/Valid'
#     train_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp2/'+c+'/Train'
    
#     a= np.asarray(random.sample(range(100), 20)) #length 100
#     b= np.arange(100)
#     w =list(filter(lambda x: x not in a, b))   ### length 400

#     for j in range(20):
#         m= a[j]
#         n= '0'*(3-len(str(m)))+str(m)

#         for i in range(100):
#             filepath= os.path.join(parent_dir, c+'_'+n+'/'+c+'_'+n+'_'+getDirNum(i)+'.pt')
#             shutil.copy(filepath, test_dir)

#     for j in range(80):
#         m= w[j]
#         n= '0'*(3-len(str(m)))+str(m)

#         for i in range(100):
#             filepath= os.path.join(parent_dir, c+'_'+n+'/'+c+'_'+n+'_'+getDirNum(i)+'.pt')
#             shutil.copy(filepath, train_dir)

    
from path import Path
import os, shutil
import numpy as np
import random

classes=['bathtub',  'bed',  'bench',  'bookshelf',  'cabinet',  'chair',  'lamp',  'monitor',  'sofa',  'table']

def getDirNum(num):
    l= len(str(num))
    return '0'*(3-l) +str(num)

for c in classes:
    #for i in range(100):
    parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp4/'+c
    dir1='Train'
    dir2='Valid'
    
    path= os.path.join(parent_dir, dir1) 
#     if os.path.exists(path):
#         shutil.rmtree(path) 
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        
    path= os.path.join(parent_dir, dir2) 
#     if os.path.exists(path):
#         shutil.rmtree(path) 
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

for c in classes:
    print(c)
    parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp4/'+c
    test_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp4/'+c+'/Valid'
    train_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp4/'+c+'/Train'
    
    a= np.asarray(random.sample(range(100), 20)) #length 100
    b= np.arange(100)
    w =list(filter(lambda x: x not in a, b))   ### length 400

    
    
    for j in range(20):
        m= a[j]
        n= '0'*(3-len(str(m)))+str(m)
        
        temp_path= os.path.join(parent_dir, c+'_'+n+'/coordinates/')
        all_files= os.listdir(temp_path)
        
        for i in range(100):
            filepath= os.path.join(temp_path, all_files[i])
            shutil.copy(filepath, test_dir)

    for j in range(80):
        m= w[j]
        n= '0'*(3-len(str(m)))+str(m)

        temp_path= os.path.join(parent_dir, c+'_'+n+'/coordinates/')
        all_files= os.listdir(temp_path)
        
        for i in range(100):
            filepath= os.path.join(temp_path, all_files[i])
            shutil.copy(filepath, train_dir)

    