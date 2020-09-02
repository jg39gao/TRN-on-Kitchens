import torch.utils.data as data

from PIL import Image
import os,sys,re
import os.path
import numpy as np
from numpy.random import randint
from tools import *


class VideoRecord(object):
    def __init__(self, row):
        self._data = row
        self.start_frame= row[3] # default 0 

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        if isinstance(self._data[2], list): return self._data[2][0] # [verb_class, noun_class]
        return int(self._data[2])
    



class TSNDataSet(data.Dataset):
    def __init__(self, root_path , list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, 
                 fromEpic= 0
                 ):
        '''
        PARAMS:
            root_path： frame_path= os.path.join(self.root_path, list_file.video(path), self.image_tmpl.format(idx))
            list_file.  a list of video info: [video(path)-num_frames-class_idx] 
            Epic:  dataset is EPIC-kitchen, default=0 
            modality:  RGB,Flow
            
        '''
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        
        self.fromEpic= fromEpic

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff
        if self.fromEpic:
            self._parse_list_epic()
            self.image_tmpl='frame_{:010d}.jpg'

        else: 
            self._parse_list()

    def _load_image(self, directory, idx):
        '''
        Load frames(RGB or FLOW) from directory with idx. 
        PARAMS:
            idx: rgb_frame_idx
        return:
            PIL.Image.Image  # RGB
            or (PIL.Image.Image ,PIL.Image.Image ) # Flow
        '''
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            try:
                idx_skip = 1 + (idx-1)*5
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert('RGB')
            except Exception:
                print('error loading flow file:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
            # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            flow_x, flow_y, _ = flow.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')

            return [x_img, y_img]
        
    def _load_image_fromEpic(self, video_id, idx):
        '''
        FROM EPIC KITCHEN dataset : Load frames(RGB or FLOW) from directory with idx. 
        PARAMS:
            video_id:  P08_13
            idx: rgb_frame_idx
        return:
            [PIL.Image.Image]  # RGB
            or [PIL.Image.Image ,PIL.Image.Image ] # Flow
        '''
        root_path=self.root_path 
        image_tmpl= self.image_tmpl

        images=[]
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            tarpath= os.path.join(root_path,'rgb/train',video_id.split('_')[0], '{}.tar'.format( video_id) )
            if Tools.path_exists(tarpath):
                filenames= ['./'+image_tmpl.format(idx)]
                try: 
                    images= Tools.extract_tar(tarpath, filenames,convertTO='RGB')
                except Exception:
                    print('error loading image {} from {}'.format(filenames, tarpath))

        elif self.modality == 'Flow':
            idx_skip =( 2 if idx==1 else idx )// 2 # to make sure the first frame also have a corresponding flow frame
            tarpath= os.path.join(root_path,'flow/train',video_id.split('_')[0], '{}.tar'.format( video_id)  )
            if Tools.path_exists(tarpath):
                filenames= ['./u/'+image_tmpl.format(idx_skip),'./v/'+image_tmpl.format(idx_skip)] 
                try:
                    
                    images = Tools.extract_tar(tarpath, filenames,convertTO='L')
                    
                except Exception:
                    print('error loading image {} from {}'.format(filenames, tarpath))
        
#        print('loading {} {} images'.format(len(images), self.modality))
        return images


    def _parse_list_epic(self, least_frames= 3):
        '''
        'participant_id', 'video_id', 'narration', 'start_timestamp',
           'stop_timestamp', 'start_frame', 'stop_frame', 'verb', 'verb_class',
           'noun', 'noun_class', 'all_nouns', 'all_noun_classes','frame_num'
        '''
        EA= EpicActions(pickle_file= self.list_file) #'../LXY_reepickitchens/train.pkl'
        self.video_list= [ VideoRecord([a.video_id, 
                                        a.frame_num, 
                                        [a.verb_class, a.noun_class], 
                                        a.start_frame]) 
                            for  a in EA.actions if a.frame_num >= least_frames]
        print('actions(num_frames ≥ {} ) number:{}'.format( least_frames,len(self.video_list)))

    def _parse_list(self, least_frames= 3):
        ''' Check and select videos of which frame number is larger or equal to 3:
        RETURN:
            a list of VideoRecord (which has attributes: path(video), num_frames, label (class_idx))
        '''
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=least_frames]
        tmp[3]=0 # for compatibility of non epic datasets
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video(num_frames ≥ {} ) number:{}'.format( least_frames,len(self.video_list)))

    def _sample_indices(self, record):
        """
        Generate RANDOM offsets  for all segments. 
        !!! return the RGB frame index, when with flow, it will change the idx in _load_image
        PARAMS: 
            record: VideoRecord
        RETURN: 
            A list of offset(frame index) for all segments
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:#(record.num_frames - self.new_length + 1)>= self.num_segments  # ramdom sampling 
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments: # dense sample. such as [0, 1, 1, 2, 3, 3, 4, 4]
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + record.start_frame if self.fromEpic else 1  

    def _get_val_indices(self, record):
        '''Generate uniform offsets  for all segments'''
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            # sample not randomly, but uniformly. 
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + record.start_frame if self.fromEpic else 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + record.start_frame if self.fromEpic else 1

    
    
    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        video_path= os.path.join(self.root_path, record.path, self.image_tmpl.format(1))
        while  (not self.fromEpic)and (not os.path.exists(video_path)) :
            print(video_path,'video not exist!')
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        
        data, label= self.get(record, segment_indices)
        
        #------ progress monitor, every 4 videos.
#        if index%4==0: 
#            sys.stdout.write("{}".format('.' ))
#            sys.stdout.flush()

        return data, label

    def get(self, record, indices):
        ''' get images from record object
        PARAMS: 
            record: VideoRecord
        '''
        images = list()
        for seg_ind in indices: # for all segment_indices
            p = int(seg_ind)
            for i in range(self.new_length):#  frame lenth of each segment
                loadimage= self._load_image_fromEpic if self.fromEpic else self._load_image
                seg_imgs = loadimage(record.path, p) 
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1 # each new_length, image move forward 1 frame
        if len(images)==0:
            raise ValueError('【Error】no {} frame(rgb_index {}) found in {}'.format(self.modality, indices, record.path))
        process_data = self.transform(images)

        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
