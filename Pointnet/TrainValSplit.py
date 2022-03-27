# from path import Path
# import os, shutil
# import numpy as np
# import random

# classes=['bathtub',  'bed',  'bench',  'bookshelf',  'cabinet',  'chair',  'lamp',  'monitor',  'sofa',  'table']

# def getDirNum(num):
#     l= len(str(num))
#     return '0'*(3-l) +str(num)

# for c in classes:
#     for i in range(100):
#         parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/3d_point_cloud/dataset/shapenet_training_normalize_noise/'+c+'/'+c+'_'+getDirNum(i)
#         dir1='Train'
#         dir2='Valid'
#         path= os.path.join(parent_dir, dir1) 
#         # if os.path.exists(path):
#         #     shutil.rmtree(path) 
#         #!rm -r parent_dir+'/'+ dir1
#         if os.path.exists(path):
#             pass
#         else:
#             os.mkdir(path)
#         path= os.path.join(parent_dir, dir2) 
#         #!rm -r parent_dir+'/'+ dir2
#         # if os.path.exists(path):
#         #     shutil.rmtree(path) 
#         if os.path.exists(path):
#             pass
#         else:
#             os.mkdir(path)

# for c in classes:
#     for i in range(0,100):
#         print(c, i)
#         parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/3d_point_cloud/dataset/shapenet_training_normalize_noise/'+c+'/'+c+'_'+getDirNum(i)
#         test_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/3d_point_cloud/dataset/shapenet_training_normalize_noise/'+c+'/'+c+'_'+getDirNum(i)+'/Valid'
#         train_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/3d_point_cloud/dataset/shapenet_training_normalize_noise/'+c+'/'+c+'_'+getDirNum(i)+'/Train'
#         a= np.asarray(random.sample(range(100), 20)) #length 100
#         b= np.arange(100)
#         w =list(filter(lambda x: x not in a, b))   ### length 400

#         for j in range(20):
#             m= a[j]
#             n= '0'*(3-len(str(m)))+str(m)
#             filepath= os.path.join(parent_dir, c+'_'+getDirNum(i)+'_'+n+'.pt')
#             shutil.copy(filepath, test_dir)
#         for j in range(80):
#             m= w[j]
#             n= '0'*(3-len(str(m)))+str(m)
#             filepath= os.path.join(parent_dir, c+'_'+getDirNum(i)+'_'+n+'.pt')
#             shutil.copy(filepath, train_dir)


################################## for experiment 02

from path import Path
import os, shutil
import numpy as np
import random

classes=['bathtub',  'bed',  'bench',  'bookshelf',  'cabinet',  'chair',  'lamp',  'monitor',  'sofa',  'table']

def getDirNum(num):
    l= len(str(num))
    return '0'*(3-l) +str(num)

for c in classes:
    for i in range(100):
        parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp4/'+c+'/'+c+'_'+getDirNum(i)
        dir1='Train'
        dir2='Valid'
        
        path= os.path.join(parent_dir, dir1) 
#         if os.path.exists(path):
#             shutil.rmtree(path) 
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)


        path= os.path.join(parent_dir, dir2) 
#         if os.path.exists(path):
#             shutil.rmtree(path) 
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)

for c in classes:
    for i in range(0,100):
        print(c, i)
        parent_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp4/'+c+'/'+c+'_'+getDirNum(i)+'/coordinates'
        test_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp4/'+c+'/'+c+'_'+getDirNum(i)+'/Valid'
        train_dir= '/home/ubuntu/3d-point-clouds-HyperCloud/experiments_arpit/shapenet_training_exp4/'+c+'/'+c+'_'+getDirNum(i)+'/Train'
        a= np.asarray(random.sample(range(100), 20)) #length 100
        b= np.arange(100)
        w =list(filter(lambda x: x not in a, b))   ### length 400

        all_files= os.listdir(parent_dir)
        
#         all_files.remove('Train')
#         all_files.remove('Valid')
        #print(all_files)
        
        for j in range(20):
            m= a[j]
            n= '0'*(3-len(str(m)))+str(m)
            #filepath= os.path.join(parent_dir, c+'_'+getDirNum(i)+'_'+n+'.pt')
            filepath= os.path.join(parent_dir, all_files[m])
            #print(filepath)
            shutil.copy(filepath, test_dir)
        for j in range(80):
            m= w[j]
            n= '0'*(3-len(str(m)))+str(m)
            #filepath= os.path.join(parent_dir, c+'_'+getDirNum(i)+'_'+n+'.pt')
            filepath= os.path.join(parent_dir, all_files[m])
            #print(filepath)
            shutil.copy(filepath, train_dir)