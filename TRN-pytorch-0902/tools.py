#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 07:03:14 2020

@author: gaojiejun
"""
import io
import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import moviepy.editor as mpy

import torchvision
import torch.nn.parallel
import torch.optim

import transforms
from torch.nn import functional as F
import tarfile
from numpy.random import randint
import torch.utils.data as data




class Tools:
    
    def round(x, digit):
        return np.round(x,digit) if x else x
    
    def printdf(df, column=[], keep_n=20, title= None, fmt= None):
        '''print a dataframe '''
        N= len(df)
        if len(column)==0: column= df.columns
        if (title==None) | (title==''): title= column
        if (fmt==None) | (fmt==''): fmt= ' | '.join(['{}']*(len(column)))
        if keep_n==0: keep_n= N
        df= pd.DataFrame(df, columns=column)
        fmt= '{:2d} | '+fmt
        print('index | '+' | '.join(title))
    
        ellipsis= True
        for index, row in df.iterrows():
            if (index<keep_n) | (index>= N-keep_n): 
                print(fmt.format(index,*row))
            elif ellipsis: 
                print('...')
                ellipsis= False
                
    def path_exists(path):
        if os.path.exists(path): return True
        else :
            print('【Path Error 】{} not exist'.format(path))
            return False
    
    def extract_frames(video_file, num_frames=8):
        try:
            os.makedirs(os.path.join(os.getcwd(), 'frames'))
        except OSError:
            pass
    
        output = subprocess.Popen(['ffmpeg', '-i', video_file], 
                                  stderr=subprocess.PIPE).communicate()
        # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
        re_duration = re.compile(r"Duration: (\d+:\d+:\d+)")
        m= re_duration.search(str(output[1]))
        if m :
            duration = re_duration.search(str(output[1])).groups()[0]
            print('    video Duration:{}'.format(duration))
        else: 
            print('Error. None duration searched in : ', output) 
            return 
    
        seconds = functools.reduce(lambda x, y: x * 60 + y,
                                   map(int, duration.split(':')))
        rate = num_frames / float(seconds)
    
        output = subprocess.Popen(['ffmpeg', '-i', video_file,
                                   '-vf', 'fps={}'.format(rate),
                                   '-vframes', str(num_frames),
                                   '-loglevel', 'panic',
                                   'frames/%d.jpg']).communicate()
        frame_paths = sorted([os.path.join('frames', frame)
                              for frame in os.listdir('frames')])
    
        frames = Tools.load_frames(frame_paths)
        subprocess.call(['rm', '-rf', 'frames'])
        return frames
    
    def select_frames(frames, num_segs, new_length=1, mode=None):
        '''if num_segs= -1, then no selection. return all frames
            Mode: 'random', 'test' or None
        '''
        if num_segs== -1:
            return frames[::1]
        elif len(frames)-new_length+1  >= num_segs:
            if mode=='test': get_indice= Tools._get_test_indices
            elif mode=='random': get_indice= Tools._get_random_indices
            else:
                if mode: # not none
                    print('[warning]: mode can only be [test, random], otherwise taken as test mode(no random),default None')
                get_indice= Tools._get_test_indices # include start and stop frame
            idx= functools.reduce(lambda x,y: x+y, [list(range(i,i+new_length)) for i in get_indice(len(frames),num_segs, new_length, start_frame=0 )])
            return [frames[i]  for i in idx]
            
        else:
            raise ValueError('Video must have at least {} frames'.format(num_segs+new_length-1))
            
    def _get_random_indices(num_frames, num_segments, new_length=1, start_frame=0):
        """
        Generate RANDOM offsets  for all segments. 
        
        PARAMS: 
             num_frames, total number of frames the action has
             num_segments, 
             new_length,  
             start_frame
        RETURN: 
            A list of offset(frame index) for all segments
        """
        average_duration = (num_frames - new_length + 1) // num_segments
        if average_duration > 0:#(num_frames - new_length + 1)>=  num_segments  # ramdom sampling 
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration, size=num_segments)
        elif num_frames > num_segments: # dense sample. such as [0, 1, 1, 2, 3, 3, 4, 4]
            offsets = np.sort(randint(num_frames - new_length + 1, size= num_segments))
        else:
            offsets = np.zeros((num_segments),dtype=np.int8)
        return offsets + start_frame  
    
    def _get_val_indices(num_frames, num_segments, new_length=1, start_frame=0):
        '''Generate uniform offsets  for all segments'''
        if num_frames - new_length +1 >=  num_segments :
            duration = (num_frames - new_length + 1) / float( num_segments)
            # sample not randomly, but uniformly.  duration / 2.0 is to set the offset in the middle of duration.
            offsets = np.array([int(duration / 2.0 + duration * x) for x in range( num_segments)])
        else:
            offsets = np.zeros((num_segments),dtype=np.int8) # why give up when num_frames-new_length+1<num_segments?
        return offsets + start_frame  
    
    def _get_test_indices(num_frames, num_segments, new_length=1, start_frame=0):
        ''' the difference with _get_val_indices is :it can get both the start and stop frames selected. 
        '''
#        duration = (num_frames - new_length + 1) / float( num_segments)
#        offsets = np.array([int(duration / 2.0 + duration * x) for x in range(num_segments)])
        ###  to make sure the start and stop frames are both selected.
        duration = (num_frames - new_length + 1) / float( num_segments-1)
        offsets = np.array([int( duration * x) for x in range(num_segments-1)]+ [num_frames - new_length] ) 
        return offsets + start_frame
    
    
    
    def load_frames(frame_folder_path, num_frames=8):
        ''' load all the frames from a directory. 
        return a list of PIL.image.image class. 
        '''
        print('Loading frames in {}'.format(frame_folder_path))
        import glob
        # Here, make sure after sorting the frame paths have the correct temporal order
        frame_paths = sorted(glob.glob(os.path.join(frame_folder_path, '*.jpg')))
        frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
        return Tools.select_frames(frames, num_frames)
    
    
    def render_frames(frames, prediction):
        rendered_frames = []
        for frame in frames:
            img = np.array(frame)
            height, width, _ = img.shape
            cv2.putText(img, prediction,
                        (1, int(height / 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            rendered_frames.append(img)
        return rendered_frames
    
#   
    def parameter_desc(cnn, ifprint= False):
        '''summary the parameters of a net 
        '''
## for checking :
#        for idx, m in enumerate(net.modules()):
#            if idx==0:
#                print(type(m), len(m))
#            else: 
#                print(idx, '->', m, len(m.state_dict().items()), m.type )
#                print(list(m.parameters())[0].size())
        
        smr= []
        s= re.compile(r'(([^\s]*)\([^)]*\))') # search from '<bound method Module.type of Linear(in_features=2, out_features=3, bias=True)>'
        types=[]
        for idx, m in enumerate(cnn.modules()):
            if idx==0: # container
                try: 
                    print('-'*20,'\n total {} layers in {}'.format(len(m), type(m)))
                    
                except (RuntimeError, TypeError, NameError):
                    print(f'{type(m)} has no len() mothod')

                continue
            else:
                m_type= s.search(str(m.type) )
                types.append((str(m),  re.compile(r'conv\dd',re.I).search(str(m)))) #m_type.group(1)
                for k,v in m.state_dict().items(): #m_type.group(2)
                    smr.append((m_type.group(2)+str(len(types)), k, v.size(), v.size().numel(),v.dtype ))
        
        df= pd.DataFrame(smr, columns=['module','keys','weights','params','dtype'])

        
#            print('-'*20,'\n',df)
        if(ifprint): print('-'*20,' {} module types:'.format(len(types)))
        cn= 0 
        for i,(l, isConvNet) in enumerate(types):
            if isConvNet: cn+=1 
            if(ifprint): print(i, ' '*20 if re.compile(r'(batch|relu|maxpool|avgpool)').search(str(l).lower()) else cn ,l)
        print('-'*20,'{:,} layers ({} ConvNets) ,Total number of parameters: {:,}'.format(idx, cn, df.params.sum()) )
        
        return df
    
    def extract_tar(tarpath,  filenames, convertTO='RGB'):
        '''
        filenames should be a list of filename with are in the tarfile, even with only 1 element.
                 ['./frame_{}.jpg'.format(str(n).zfill(10))]
        return:
            a list of PIL.Image.Image# 
        '''
        frames=[]
        if len(filenames)>0:
            with tarfile.open(tarpath) as tar:
                subdir_and_files = [
                            tarinfo for tarinfo in tar.getmembers()
                            if tarinfo.path in filenames #.startswith("./u/")#
                        ]
                tbload= len(filenames)
                loaded= len(subdir_and_files)
                if loaded!= tbload:
                    print('【Warning】: {}(/{})frames not exist in {}'.format(tbload- loaded, tbload, tarpath))
                ### make sure after sorting the frame paths have the correct temporal order
                if loaded>1 : subdir_and_files= sorted(subdir_and_files, key=lambda x: x.path )
                frames= list(map(lambda f: Image.open(io.BytesIO(tar.extractfile(f).read())).convert(convertTO), subdir_and_files))
        else: 
            raise ValueError(" parameter [filenames] shouldn't be empty! ")
        return frames



class EpicAction(object):
    def __init__(self, pd_series):
        '''parameter pd_series obtained from EpicLoad.read_pickle() 
        '''
        self.index= pd_series.name# uid
        self.participant_id= pd_series.participant_id
        self.video_id= pd_series.video_id
        self.narration= pd_series.narration
        self.start_timestamp= pd_series.start_timestamp
        self.stop_timestamp= pd_series.stop_timestamp
        self.start_frame= pd_series.start_frame
        self.stop_frame= pd_series.stop_frame
        self.verb= pd_series.verb
        self.verb_class= pd_series.verb_class
        self.noun= pd_series.noun
        self.noun_class= pd_series.noun_class
        self.all_nouns= pd_series.all_nouns
        self.all_noun_classes= pd_series.all_noun_classes
        self.frame_num= self.stop_frame- self.start_frame+1
        
        self.frames= None #/For a list of Image.images
class EpicActions(object):
    def __init__(self,pickle_file):
        
        self.actions=[EpicAction(row) for index, row in EpicLoad.read_pickle(pickle_file).iterrows()]
        self.actions_num= len(self.actions)
class EpicLoad(object):
    
    def __init__(self, frame_rootfolder= '/data/acq18jg/epic/frames_rgb_flow',frame_tmpl='frame_{:010d}.jpg', num_segs=8):
        '''load designated actions from epic-kitchen dataset  using a pickle file.
        The epic-kitchen dataset should be filed in such way:
            frames_rgb_flow (the "frame_rootfolder")
            |--- rgb/train
            |    |--- P01
            |    |    |--- P01_01.tar
            |    |    |    |-- frame_1234567890.jpg
            |    |    |    ...
            |    |    |--- P01_02.tar
            |    ...
            |--- flow/train
            |    |--- P01
            |    |    |--- P01_01.tar
            |    |    |    |--- u
            |    |    |    |    |-- frame_1234567890.jpg
            |    |    |    |    ...
            |    |    |    |--- v
            |    |    |--- P01_02.tar
            ...
        
        PARAMS:
            frame_type= 'RGB', or 'FLOW'
        
        '''
        self.frame_rootfolder = frame_rootfolder
        self.num_segs = num_segs
        self.frame_tmpl= frame_tmpl



    
    def read_pickle(pkl):
        if not os.path.exists(pkl):
            raise OSError("{} not exists".format(pkl))
        else:
            a= pd.read_pickle(pkl)
            a.columns= ['participant_id', 'video_id', 'narration', 'start_timestamp',
               'stop_timestamp', 'start_frame', 'stop_frame', 'verb', 'verb_class',
               'noun', 'noun_class', 'all_nouns', 'all_noun_classes','frame_num']
            return a 
    
    def __call__(self, pkl, pkl_dataframe=None,  videoid_list= None, actionid_list= None, frame_type='RGB', new_length= None, mode=None, 
                 ifprint= False, printsummary= False):
        ''' 
        Params:
            pkl,  pkl file path. 
            videoid_list , should be a list. ['P08_01']
            actionid_list, should be a list of uid (pkl.index)
            pkl_dataframe, only for EpicDataSet.
            
        Return:
            a list of EpicAction[ea]. 
                ea.frames, a list of Image.image
                ea.attrs...
        '''
        if  pkl_dataframe is not None and len(pkl_dataframe)>0:
            index_pd_dataframe= pkl_dataframe 
        else:
            index_pd_dataframe= EpicLoad.read_pickle(pkl)
        
        if not len(index_pd_dataframe)>=1:
            raise ValueError("pickle file ({} ) is empty".format( pkl))

        if actionid_list :
            if not isinstance(actionid_list, list):
                raise ValueError("actionid_list should be a list of integer, got {}".format(actionid_list))
            actionid_list= [i for i in actionid_list if i in index_pd_dataframe.index]
            index_pd_dataframe= index_pd_dataframe.loc[np.array(actionid_list)]
        
        if videoid_list and len(videoid_list)>=1 :
            if not isinstance(videoid_list, list):
                raise ValueError("videoid_list should be a list of string(like['P08_01']), got {}".format(videoid_list))
            index_pd_dataframe= index_pd_dataframe.iloc[[v in videoid_list for v in index_pd_dataframe.video_id ]]
            
        
        self.videos= np.unique(index_pd_dataframe['video_id'])
            
        if frame_type not in ['RGB','Flow']: 
            raise ValueError("frame_type should be 'RGB' or 'Flow' ")
        if new_length: 
            self.new_length= new_length
        else: self.new_length = 1 if frame_type=='RGB' else 5
        self.mode= mode 
        
            
        
        self.videos_num= len(self.videos)
        self.actions_list=[]
        self.loading_frames_num= 0
        
        for vid in self.videos:
            if ifprint: print('*'*3, vid)
            index_df= index_pd_dataframe[index_pd_dataframe.video_id==vid]
            
            tarpath= os.path.join( self.frame_rootfolder,
                                  'rgb/train' if frame_type=='RGB' else 'flow/train',
                                  vid.split('_')[0],
                                  '{}.tar'.format(vid))
            with tarfile.open(tarpath) as tar:
                for i in range(len(index_df)):
                    i_series= index_df.iloc[i]

                    ea= EpicAction(i_series)
                    if ifprint: print('-'*3, i_series.name, '-'*3, i_series.narration)
                    
                    if frame_type=='RGB':
                        start_frame= i_series.start_frame
                        stop_frame = i_series.stop_frame
                    else: # 'FLOW'
                        start_frame= i_series.start_frame//2 
                        if start_frame==0: 
                            start_frame= 1 
                        stop_frame = i_series.stop_frame//2

                    filenames= [self.frame_tmpl.format(n) for n in range(start_frame,stop_frame+1,1)]
                    filenames= Tools.select_frames(filenames, self.num_segs, self.new_length, mode= self.mode)
                    
                    if frame_type=='RGB':
                        filenames=['./'+f for f in filenames]
                        read_mode='RGB'
                    else: # FLOW
                        filenames=functools.reduce(lambda x,y: x+y,[['./u/'+f, './v/'+f] for f in filenames]) 
                        read_mode='L'

#                    print('actionid {} (rgbframe{} -> {}) \n'.format(i_series.name, i_series.start_frame,i_series.stop_frame),filenames)
                    
                    
                    ## extract from tar----
                    subdir_and_files = [
                        tarinfo for tarinfo in tar.getmembers()
                        if tarinfo.path in filenames #.startswith("./u/")#
                        # for rgb:  './frame_0000000247.jpg'
                        # for flow: './v/frame_0000000247.jpg' or './u/frame_0000000247.jpg'
                    ]
                    tbload= len(np.unique(filenames)) # for sake of overlaping between segments
                    loaded= len(subdir_and_files)
                    if loaded!= tbload:

                        print('【 Error 】 actionid {} (rgbframe{} -> {}) : {}(/{})frames  not exist in {} : \n {}'.format(
                              i_series.name, i_series.start_frame,i_series.stop_frame,
                              tbload- loaded, tbload, tarpath, [f for f in np.unique(filenames) if f not in [x.path for x in subdir_and_files] ]))
                    
                    ### make sure after sorting the frame paths have the correct temporal order
                    #subdir_and_files= sorted(subdir_and_files, key=lambda x: x.path.split('/')[-2:][::-1] ) 
                    #ea.frames= list(map(lambda f: Image.open(io.BytesIO(tar.extractfile(f).read())).convert(read_mode), subdir_and_files))
                    iamge_dict= { f.path : Image.open(io.BytesIO(tar.extractfile(f).read())).convert(read_mode) for f in subdir_and_files}
                    ea.frames= list(map(lambda f: iamge_dict[f] ,filenames ))
                    
                    if ifprint: 
                        for f in filenames: print(' '*5, f)
                        
                    self.actions_list.append(ea)
                    self.loading_frames_num+= len(ea.frames)
        if printsummary:print('Load {} actions( {} {} frames)({}) from {} videos '.format(
                len(self.actions_list), self.loading_frames_num, frame_type,
                index_pd_dataframe.index, self.videos_num ))
        return self.actions_list
        
class EpicDataSet(data.Dataset):
    ''' for torch.utils.data.DataLoader
    
    '''
    def __init__(self,  pickle_file,
                 num_segments=8, new_length=None, modality='RGB',
                 transform=None,
                 random_shift=True, test_mode=False, 
                 **kws
                 ):
        '''videoid_list= ['P08_01',...] , 
           actionid_list= [111,112,...] 
           frame_rootfolder= ../'''
        self.pickle_file= pickle_file
        self.num_segments= num_segments
        self.modality= modality
        self.new_length= new_length
        self.transform= transform


        if random_shift:
            self.mode= 'random'
        if test_mode:
            self.mode= 'test'
        
        index_pd_dataframe = EpicLoad.read_pickle(pickle_file)
        
        actionid_list= kws.get('actionid_list',None)
        videoid_list= kws.get('videoid_list',None)
        frame_rootfolder= kws.get('frame_rootfolder',None)
        self.printsummary= kws.get('printsummary',0)
        if actionid_list :
            if not isinstance(actionid_list, list):
                raise ValueError("actionid_list should be a list of integer, got {}".format(actionid_list))
            actionid_list= [i for i in actionid_list if i in index_pd_dataframe.index]
            index_pd_dataframe= index_pd_dataframe.loc[np.array(actionid_list)]
        
        if videoid_list and len(videoid_list)>=1 :
            if not isinstance(videoid_list, list):
                raise ValueError("videoid_list should be a list of string(like['P08_01']), got {}".format(videoid_list))
            index_pd_dataframe= index_pd_dataframe.iloc[[v in videoid_list for v in index_pd_dataframe.video_id ]]
        
        self.pkl_dataframe= index_pd_dataframe
        self.actions_id = index_pd_dataframe.index
        self.EL= EpicLoad( frame_rootfolder= frame_rootfolder,  num_segs= self.num_segments) if frame_rootfolder else EpicLoad( num_segs= self.num_segments)
        print('-'*20, '{} actions to go...'.format(len(self.actions_id)))
        
#        self.actions = EL(pkl= pickle_file, 
#                          frame_type= self.modality,
#                          new_length= self.new_length,
#                          mode= self.mode, **kws
#                          )
        
        
    def __getitem__(self, index):
        uid = self.actions_id[index]
        action= self.EL( pkl= '', #only for its position parameter, meaningless
                        pkl_dataframe= self.pkl_dataframe, 
                        actionid_list= [uid],
                        frame_type= self.modality,
                        new_length= self.new_length,
                        mode= self.mode,
                        printsummary = self.printsummary
                        )[0]

        data, label= self.transform(action.frames), action.verb_class
    
        return data, label
    
    def __len__(self):
        return len(self.actions_id) 
    