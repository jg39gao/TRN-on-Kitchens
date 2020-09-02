#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 06:51:43 2020

@author: gaojiejun
"""

import sys, getopt, re, time, io
import os
import numpy as np
from collections import Counter
from scipy.special import comb, perm, factorial
from PIL import Image
import torch
import torchvision 
import transforms 
import matplotlib.pyplot as plt
import tarfile
from dataset import TSNDataSet
import datasets_video

import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from transforms import *
from tools import *


#---------------------------------------------------------------------------------
print('-'*10,'tools','-'*20)

path= '/Users/gaojiejun/Sheffield/_00_Dissertation/Code/TRN-pytorch-master'
#frame_path= '/Users/gaojiejun/Sheffield/_00_Dissertation/Code/TRN-pytorch-master/sample_data/juggling_frames'
frame_path='/Users/gaojiejun/Sheffield/_06_Industrial_teamProject(COM6911)/Data/frames'
root_path= '/data/acq18jg/epic/frames_rgb_flow'
#
if Tools.path_exists(path):

    frames= Tools.load_frames(frame_path)
    
    x= len(frames)
    print(len(frames),type(frames[0]))
    
    t_gos= transforms.GroupOverSample(crop_size= (211,300), scale_size=(200,500))
    t1= t_gos(frames)
    print('t1',len(t1), type(t1[0]))
    t_stack= transforms.Stack(roll=1) # rgb-> bgr
    t2= t_stack(t1)
    print('t2', len(t2),  t2.shape)
    #    
    
    
    #----------------------------
    t_start= time.time()
    
    t_torch= transforms.ToTorchFormatTensor(div=False)# chw
    t3= t_torch(t2)
    print('t3', t3.shape)
    
    t_end = time.time()
    t_diff = t_end - t_start
    print('cost {}s'.format(t_diff))  #0.055351972579956055s
    #----------------------------
    t_norm= transforms.GroupNormalize(mean= [104, 117, 128], std=[1] )
    t4= t_norm(t3)
    print('t4', t4.shape)
    
    #imshow(t2[:,:,:3])
    #imshow(t1[0])
    #print(t3[:3,:8,:8])
    #plt.imshow(np.transpose(t3[:3,:,:], (1, 2,0)))
    print(t3.size(0))


#---------------------------------------------------------------------------------
#print('-'*10,'pickle','-'*20)
#
xy_train= EpicLoad.read_pickle('epic_kitchens/train.pkl')
xy_test= EpicLoad.read_pickle('epic_kitchens/test.pkl')
xy_val= EpicLoad.read_pickle('epic_kitchens/val.pkl')

#df=xy_train[ (xy_train.video_id== 'P08_13') | (xy_train.video_id== 'P08_25')]
#df.to_pickle("epic_kitchens/train_01.pkl")
#df1=xy_train[  (xy_train.video_id== 'P08_25')]
#df1.to_pickle("epic_kitchens/val_01.pkl")
#print(len(EpicLoad.read_pickle('epic_kitchens/train_01.pkl')))
#





#EA= EpicActions(pickle_file= '../LXY_reepickitchens/train.pkl')
#A= EA.actions
#L= EA.actions_num
#print(type(A[0]), len(A), L)
#
#def members(tf):
#    l = len("subfolder/")
#    for member in tf.getmembers():
#        if member.path.startswith("subfolder/"):
#            member.path = member.path[l:]
#            yield member
#
#with tarfile.open("sample.tar") as tar:
#    tar.extractall(members=members(tar))
    

#cnt =10
#with tarfile.open(os.path.join(root_path,'flow/train/P08/P08_13.tar')) as tar:
##    tar.extractall(members=members(tar)) 
#    for tarinfo in tar.getmembers():
#        cnt-=1
#        if cnt<0: break
#        print(tarinfo.path, tarinfo.path.endswith('./v/'))
#        
#
#def _load_image_fromEpic( modality, video_id, idx):
#    '''
#    FROM EPIC KITCHEN dataset : Load frames(RGB or FLOW) from directory with idx. 
#    return:
#        [PIL.Image.Image]  # RGB
#        or [PIL.Image.Image ,PIL.Image.Image ] # Flow
#    '''
#    
#    image_tmpl= 'frame_{:010d}.jpg'
#    
#    
    
    
#    if modality == 'RGB' or modality == 'RGBDiff':
#        tarpath= os.path.join(root_path,'rgb/train',video_id.split('_')[0], '{}.tar'.format( video_id) )
#        filenames= ['./'+image_tmpl.format(idx)]
#        try: 
#            return Tools.extract_tar(tarpath, filenames,convertTO='RGB')
#        except Exception:
#            print('error loading image {} from {}'.format(filenames, tarpath))
##                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
#    elif modality == 'Flow':
#        idx_skip = 1 + (idx-1)*5
#        tarpath= os.path.join(root_path,'flow/train',video_id.split('_')[0], '{}.tar'.format( video_id)  )
#        filenames= ['./u/'+image_tmpl.format(idx_skip),'./v/'+image_tmpl.format(idx_skip)] 
#        try:
#            
#            flow = Tools.extract_tar(tarpath, filenames,convertTO='L')
#        except Exception:
#            print('error loading image {} from {}'.format(filenames, tarpath))
##                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
#        # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
#
#        return flow


#----for hpc test-------------
#
#print('+'*20)
#file= os.path.join(root_path,"rgb/train/P08/P08_13.tar") # 'frame_0000000256.jpg'
#print('file:',file)
#
#EL= EpicLoad( num_segs=8 )
#x= EpicLoad.read_pickle("epic_kitchens/train.pkl")
#print(x[(x.video_id=='P08_25') | (x.video_id=='P08_13')][['video_id','start_frame','stop_frame','frame_num']])
#
#
#images= EL( pkl= "epic_kitchens/train.pkl", videoid_list= ['P08_25','P08_13'], #actionid_list= [2761],
#                         frame_type='RGB', new_length=None, ifprint= 0)
#


#------select frames---------
##num_frames=280
##num_segments=8
##new_length=1
##start_frame=1
##print('_get_random_indices:', Tools._get_random_indices(num_frames, num_segments, new_length,0))
##print('_get_val_indices:',(Tools._get_val_indices(num_frames, num_segments, new_length, 0)) )
##print('_get_test_indices:', Tools._get_test_indices(num_frames, num_segments, new_length, 0))
#
##print(Tools.select_frames(list(range(1,num_frames+1)),8, 1, mode='random'))
#
##-----------------
##videoid_list= None 
##actionid_list= None
##
#print('-'*10,' tar file test ','-'*20)
#print('*'*5)
##if videoid_list :
##    print('has sth')
##else: 
##    print('none')
##    
#mx= 0
#with tarfile.open(os.path.join(root_path,'rgb/train/P22/P22_07.tar')) as tar:
#    for tarinfo in tar.getmembers():
#        
#        if mx<=15 :#and tarinfo.path>='P08_25/u/frame_0000000150.jpg' and tarinfo.path<'P08_25/v/':
#            print(tarinfo.path)
#            mx+=1
#            





#print(filenames)
#with tarfile.open(file) as tar:
#    subdir_and_files = [
#        tarinfo for tarinfo in tar.getmembers()
#        if tarinfo.path in filenames #.startswith("./u/")#
#    ]
#    # make sure after sorting the frame paths have the correct temporal order
#    subdir_and_files= sorted(subdir_and_files, key=lambda x: x.path )
##    for i in subdir_and_files: print(i.path)
#    frames= list(map(lambda f: Image.open(io.BytesIO(tar.extractfile(f).read())).convert('RGB'), subdir_and_files))
#frames= Tools.extract_tar(tarpath= file, filenames=filenames )
#frames= _load_image_fromEpic(modality ='RGB', video_id='P08_13', idx=21)
#frames=list(map(lambda f: f.convert('RGB'), frames))

#print('-'*10 ,len(frames), type(frames[0]))
#df= (xy_train[xy_train.video_id=='P08_25'])
#
##df=df.append(df.iloc[0])
#print('-'*20,len(df), type(df))
#num_f=0 
#for i in range(len(df)):
#    n= df.iloc[i].stop_frame-df.iloc[i].start_frame+1
#    num_f+= n
#    print( df.iloc[i].start_frame, df.iloc[i].stop_frame ,n,num_f )
#
#print(type(df.iloc[0].video_id))

#
#
##print(unique(xy_train.video_id))
#
#print(len(images), len(images[0].frames) , type(images[0].frames[0]), images[0].verb)


#---------------------------------------------------------------------------------
print('-'*10,'dataset','-'*20)


image_tmpl= 'frame_{:010d}.jpg'
train_list='epic_kitchens/train_01.pkl'#'../LXY_reepickitchens/train.pkl'
#
#d= EpicActions(train_list)
#print(d.actions_num)

DS= EpicDataSet( train_list,
                 #frame_rootfolder= '../../frames_rgb_flow',
                 num_segments=8, #new_length=2, 
                 modality='Flow',
                 transform=torchvision.transforms.Compose([
                       Stack(roll=1),
                       ToTorchFormatTensor(div=1),
                       IdentityTransform()]),
                 random_shift=True, test_mode=False
                 )

print('len', len(DS))
print('type:{} (seg * framechannel* new_length *wh), class: {}'.format((DS[0][0].size()), (DS[0][1])))


DS= EpicDataSet( train_list,
                 #frame_rootfolder= '../../frames_rgb_flow',
                 num_segments=8, new_length=2, 
                 modality='Flow',
                 transform=torchvision.transforms.Compose([
                       Stack(roll=1),
                       ToTorchFormatTensor(div=1),
                       IdentityTransform()]),
                 random_shift=True, test_mode=False
                 )

print('len', len(DS))
print('type:{} (seg * framechannel* new_length *wh), class: {}'.format((DS[0][0].size()), (DS[0][1])))

#DS= TSNDataSet(root_path, train_list, num_segments= 8,
#                   new_length=1,
#                   modality='Flow',
#                   image_tmpl=image_tmpl,
#                   transform=torchvision.transforms.Compose([
##                       train_augmentation,
#                       Stack(roll=1),
#                       ToTorchFormatTensor(div=1),
#                       IdentityTransform(),
#                   ]),fromEpic= 1
#            )
    


#
#train_ds= TSNDataSet(root_path, train_list, num_segments=8,
#                   new_length=1,
#                   modality='RGBDiff',
#                   image_tmpl=image_tmpl,
#                   random_shift=False,
#                   transform=torchvision.transforms.Compose([
#                       GroupScale((341, 241)),
#                       GroupCenterCrop((299,511)),
#                       Stack(True),
#                       ToTorchFormatTensor(div=False),
#                       IdentityTransform(),
#                   ]),fromEpic= 1
#                    )
##
##print('tl0:', train_ds[0][0].size(),train_ds[0][1])
##print( 'ds len:', len(train_ds))
##
#train_loader = torch.utils.data.DataLoader(
#        train_ds,  
#        batch_size= 2, shuffle=True,
#        num_workers=30, pin_memory=True)
#tl_len= len(train_loader)
#print( 'trainloader len:', tl_len)
#
#
#
##test_i = 5
##for i, ( target) in enumerate(list(range(tl_len)) ):
##    print('teste p3: ',i,'=',  target)
##    test_i-=1
##    if test_i<0 : 
##        break 
##
##print('not crashed hereA  len')
##---------------------------------------------------------------------------------
#print('-'*10,'-'*20)
#
#test_i = 5
#for i, (ip, target) in enumerate(train_loader):
#    print('*'*5,i)
#    print('start training p3: ',i, ip.size(),  target)
#
#    print('test_i: ',test_i)
#    test_i= test_i-1 
#    if test_i<0 : break 
#print('crashed here ___enumerate')
#
#test_i = 5
#print('get here')
#for i, (ip, target) in enumerate(train_loader):
#    print(' p3: ',i, ip.size(),  target)
#    test_i-=1
#    if test_i<0 : break 
#print('crashed here ___enumerate2')
#
#print('get here2')
#tl_len= len(train_loader)
#print( 'trainloader len2:', tl_len)
#
#test_i = 5
#print('get here')  ### abortion at here. so enumerate(train_loader) can be done only once? 
#for i, (ip, target) in enumerate(train_loader):
#    print(' p3: ',i, ip.size(),  target)
#    test_i-=1
#    if test_i<0 : break 
#print('get here3')   
#
#j = 5 
#for i in range(min(5, len(DS))) :
#   
#    x= DS[i]
#    print(type(x[0]),x[0].size(), x[1] )
#    
#for i, (input, target) in enumerate(DS):
#    print('start training p3')
#    print(i, input.size(), target)
#    
#    j-=1
#    if j<0 : break
#---------------------------------------------------------------------------------
print('-'*10,' time eval ','-'*20)



image_tmpl= 'frame_{:010d}.jpg'
train_list='epic_kitchens/train_02.pkl'#eval_efficiency.pkl'#'../LXY_reepickitchens/train.pkl'


transform=torchvision.transforms.Compose([
                       Stack(roll=1),
                       ToTorchFormatTensor(div=1),
                       IdentityTransform()])
    

def eval_dl(ds, end):
    test_loader = torch.utils.data.DataLoader(
                DS,
                batch_size= 8, 
                shuffle=False,
                num_workers=1, 
                pin_memory=True)

    btchs= len(test_loader)
    t_len= time.time()- end
    print('len:', len(DS), 'batch:', btchs, 'time:', t_len)
    for i, (input, target) in enumerate(test_loader):
        if i==0:
            t1= time.time()- end
        if i==1: t2= time.time()- end
        if i==2: t3= time.time()- end
    t_end= time.time()- end

    evl= [{'ds_init':t_len,'t1': t1,'t2':t2, 't3':t3, 't_end':t_end}]
    df= pd.DataFrame(evl)
    print('dataloader[0]:',(DS[0][0].size()),type(DS[0][1]))
    print(df)
    return df 

##--------original loader-------------
#end = time.time()
#DS= TSNDataSet(root_path, train_list, 
#               num_segments=8,
#           new_length= 5,
#           modality="Flow",
#           image_tmpl=image_tmpl,
#           random_shift=True,
#
#           transform=transform,
#           fromEpic= 1
#            )
#
#d1= eval_dl(DS, end)

#------epicloader---------------
#end = time.time()
#DS= EpicDataSet( train_list,
#                 #videoid_list = ['P08_25','P08_13'],
#                 num_segments=8, 
#                modality='Flow',
#                 transform= transform,
#                 random_shift=True, test_mode=False
#                 )
#d2= eval_dl(DS, end)
#
#print(d2)
#    

#-----------------only in hpc-------------
#print('-'*10,' time eval full set ','-'*20)
#end = time.time()
#DS= EpicDataSet( 'epic_kitchens/val.pkl',
##             videoid_list = ['P08_25','P08_13'],
#             num_segments=8, 
#            modality='RGB',
#             transform= transform,
#             random_shift=True, test_mode=False
#             )
#test_loader = torch.utils.data.DataLoader(
#                DS,
#                batch_size= 16, 
#                shuffle=False,
#                num_workers=1, 
#                pin_memory=True)
#
#btchs= len(test_loader)
#t_len= time.time()- end
#tim=[]
#print('len:', len(DS), 'batch:', btchs, 'time:', t_len)
#for i, (input, target) in enumerate(test_loader):
#    tim.append((i, time.time()- end))
#    print(i,',',time.time()- end)
#
#print(tim)


###########---------------------------------------
#index_pd_dataframe= EpicLoad.read_pickle('epic_kitchens/train_02.pkl')
##print( [v in ['P08_13','P08_25'] for v in index_pd_dataframe.video_id ])
##print(index_pd_dataframe.index)
##index_pd_dataframe=index_pd_dataframe.iloc[[v in ['P08_13','P08_25'] for v in index_pd_dataframe.video_id ]]
#print(index_pd_dataframe.index)
#print(len(index_pd_dataframe.index),',', index_pd_dataframe.index[1])
##print( [i for i in [1670,1787,16711] if i in index_pd_dataframe.index] )
##print((index_pd_dataframe.loc[np.array([1670,0])]))
#
#
#EL= EpicLoad()
#
#for action in EL(pkl= 'epic_kitchens/train_02.pkl', videoid_list = ['P08_13','P08_25']
#                #,ifprint= 1, printsummary= 1
##                 actionid_list=[2761,2762]
#                 ):
#    print(action.index, action.narration)
#
#
#DS= EpicDataSet( 'epic_kitchens/train_02.pkl',
#             videoid_list = ['P08_13','P08_25'],
#             num_segments=8, 
#            modality='Flow',
#             transform= transform,
#             random_shift=True, test_mode=False #,ifprint= 1#, printsummary= 1
#             )
#print(len(DS))
#x= DS[0]
#print(type(x),',', x[0].size(),',', x[1] )
#print('-'*20)
#test_loader = torch.utils.data.DataLoader(
#                DS,
#                batch_size= 16, 
#                shuffle=True,
#                num_workers=1, 
#                pin_memory=True)
#for i, (input, target) in enumerate(test_loader):
#    print(i,'-',input.size(), '-', target)

