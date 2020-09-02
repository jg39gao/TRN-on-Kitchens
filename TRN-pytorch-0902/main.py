import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import datasets_video
from tools import Tools, EpicDataSet
import pandas as pd
import numpy as np
from main import *


best_prec1 = 0


def check_rootfolders():
    """Create log and model folder"""
    folders_util = ['log', 'model', args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


def main():
    global args
    args = parser.parse_args()
    check_rootfolders() 

    if not args.new_length:# none
        data_length= 1 if args.modality== 'RGB' else 5
    else :
        data_length= args.new_length

    trnmodel= TRN(        dataset = args.dataset,
                num_segments = args.num_segments,
                modality = args.modality,
                new_length=data_length, #

                lr = args.lr, 
                loss_type = args.loss_type, # default="nll",
                weight_decay = args.weight_decay, # default=5e-4,
                lr_steps = args.lr_steps, # default=[50, 100],
                momentum= args.momentum, # default=0.9,
                gpus = args.gpus, 
                clip_gradient = args.clip_gradient,
                base_model= args.arch, #"resnet50",
                dropout= args.dropout, #0.7,
                img_feature_dim=args.img_feature_dim ,#default 256,
                partial_bn= not args.no_partialbn ,# default=False 
                consensus_type= args.consensus_type, #'TRN', # MTRN

                batch_size= args.batch_size,# default=64
                workers= args.workers, #default=2
                
                resume = args.resume ,  #  pretained model 
                epochs= args.epochs,
                start_epoch = args.start_epoch, #
                

                ifprintmodel= args.print_model in [1, 'True'], # default =1
                print_freq =1,
                eval_freq =1

        )
    #---------if evaluation: ----------------------------------------------------
    print('evalutate=',args.evaluate)
    if str(args.evaluate).lower()=='true' or args.evaluate=='1':
        logits= trnmodel(args.test_pickle)
        logits= np.array(logits)
        print('output size: ',logits.shape)
        with np.printoptions(threshold=np.inf):
            print(logits)
    else:
        trnmodel.do_training( ifprint= args.print_training_in_terminal)

if __name__ == '__main__':
    main()

class TRN():
    def __init__(self, 
                num_segments,
                modality,

                lr =0.001, 
                loss_type = 'nll', # cross entropy
                weight_decay=5e-4, #weight_decay: L2 penalty #default
                lr_steps =[30, 60], # epochs to decay learning rate by 10
                momentum= 0.9, 
                gpus= None, 
                clip_gradient =20, 

                new_length=None,
                base_model="resnet50",
                dropout=0.7,
                img_feature_dim=256, #The dimensionality of the features used for relational reasoning. 
                partial_bn=True,
                consensus_type= 'TRN', # MTRN
                
                dataset = 'epic',
                batch_size= 1,
                workers= 2,
                
                resume = None,  #  pretained model (path)
                epochs= None,
                start_epoch = None, #
                
                
                ifprintmodel= 0, # print the model structure
                print_freq =1,
                eval_freq =1,
                ):


        self.num_segments= num_segments
        self.modality= modality
        self.base_model= base_model
        self.new_length= new_length
        self.img_feature_dim= img_feature_dim
        self.consensus_type= consensus_type
        self.dataset= dataset

        self.resume = resume 
        self.epochs = epochs
        self.start_epoch= start_epoch

        self.lr= lr  
        self.loss_type= loss_type
        self.weight_decay = weight_decay
        self.lr_steps= lr_steps
        self.momentum= momentum
        self.partial_bn= partial_bn
        self.dropout= dropout

        self.batch_size = batch_size
        self.workers= workers
        self.gpus=  gpus
        self.eval_freq= eval_freq
        self.print_freq= print_freq

        
        self.num_class, self.train_list, self.val_list, self.root_path, self.prefix = datasets_video.return_dataset(self.dataset, self.modality)
        self.store_name = '_'.join(['TRN', self.dataset, self.modality, self.base_model, self.consensus_type, 'segment%d'% self.num_segments, 'K%d'% self.new_length])      
        self.best_prec1= 0 
        self.clip_gradient= clip_gradient
        
        
        
        self.model = TSN(self.num_class, self.num_segments, self.modality,
                new_length= self.new_length,
                base_model= self.base_model,
                consensus_type= self.consensus_type,
                dropout=self.dropout,
                img_feature_dim= self.img_feature_dim,
                partial_bn= self.partial_bn)

        self.crop_size =  self.model.crop_size 
        self.scale_size = self.model.scale_size
        self.input_mean = self.model.input_mean
        self.input_std = self.model.input_std
        self.model_policies = self.model.get_optim_policies()
        self.augmentation= self.model.get_augmentation()
        
        print('we have {} GPUs found'.format(torch.cuda.device_count()))
        self.model = torch.nn.DataParallel(self.model #, device_ids=self.gpus
                                           ).cuda()

        print(f'''  
+-------------------------------------------------------+
               num_class : {self.num_class}
                modality : {self.modality}
              base_model : {self.base_model}
              new_length : {self.new_length}
          consensus_type : {self.consensus_type}
         img_feature_dim : {self.img_feature_dim}

                  resume : {self.resume}
                  epochs : {self.epochs }
             start_epoch : {self.start_epoch }
                      lr : {self.lr }
               loss_type : {self.loss_type }
            weight_decay : {self.weight_decay }
                lr_steps : {self.lr_steps }
                momentum : {self.momentum }
              partial_bn : {self.partial_bn}
           clip_gradient : {self.clip_gradient }
                 dropout : {self.dropout}

              batch_size : {self.batch_size}
                 workers : {self.workers}
                    gpus : {self.gpus } ( no use now)
               eval_freq : {self.eval_freq }
              print_freq : {self.print_freq }
              
               crop_size : {self.crop_size}
              scale_size : {self.scale_size}
+-------------------------------------------------------+
construct a network named : {self.store_name}''')
        
        #---- checkpoint------load model ---- 
        if self.resume:
            if os.path.isfile(self.resume):
                print(("=> loading checkpoint '{}'".format(self.resume)))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                print(("=> loaded checkpoint '{}' (epoch {}) (epochs={})"
                      .format(self.resume, checkpoint['epoch'], self.epochs)))
            else:
                print(("=> no checkpoint found at '{}'".format(self.resume)))

        cudnn.benchmark = True

        # Data loading code
        if self.modality != 'RGBDiff':
            self.normalize = GroupNormalize(self.input_mean, self.input_std)
        else:
            self.normalize = IdentityTransform()
        

        #------- define loss function (criterion) and optimizer-------
        if self.loss_type == 'nll':
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            raise ValueError("Unknown loss type")

        
        #---------=========describe parameters:========= ----------------------
        
        print('*'*20,'TSN parameters:')
        Tools.parameter_desc(self.model, ifprint= ifprintmodel)
        #------parameter  way2-----
        print('-'*30)
        for group in self.model_policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        
        toal_params=0 
        for p in self.model_policies:
    #        print('-'*10,'{} ( num: {})'.format(p['name'],len(p['params'])))
            for i, param in enumerate(p['params']):
                toal_params+= param.size().numel()
    #            if i< 5 :
    #                print(param.size(), param.size().numel())
    #            elif i==5 : 
    #                print('...')
        print('*'*20, 'count from policies, total parameters: {:,}'.format(toal_params))
        print('TRN initialised \n')

        

    def __call__(self, input_pickle):
        '''
        input_list: pickle file ,like ' epic_kitchens/val_02.pkl'
        '''
        end= time.time()
        DS= EpicDataSet( pickle_file= input_pickle,
                 num_segments= self.num_segments, 
                 new_length= self.new_length, 
                 modality=self.modality,
                 transform=torchvision.transforms.Compose([
                           GroupScale(int(self.scale_size)), #hw 
                           GroupCenterCrop(self.crop_size), #hw
                           Stack(roll=(self.base_model in ['BNInception','InceptionV3'])),
                           ToTorchFormatTensor(div=(self.base_model not in ['BNInception','InceptionV3'])),
                           self.normalize,
                       ]),
                 random_shift=False, test_mode= True
                 )
        
        test_loader = torch.utils.data.DataLoader(
            DS,
            batch_size= self.batch_size, 
            shuffle=False,
            num_workers=self.workers, 
            pin_memory=True)
        time1= time.time()- end
        print('testmode: load {} actions ({} batchs) '.format(len(DS) ,len(test_loader) ))
        logits= self.validate(self.model, test_loader, self.criterion)[0]
        print('cost time: {:.1f} mins, {:.1f} mins'.format(time1,(time.time()- end)/60))
        return logits


#--------- training: ------------------------------------------------------
    
    def do_training(self, ifprint=1):
        
        
        # initial logging file 
        model= self.model
        self.log_training = os.path.join('log/{}.txt'.format(self.store_name))
        with open(self.log_training, 'w') as log:
            log.write('initial logfile. {} \n'.format(time.strftime(" %d.%B %H:%M:%S")))
        self.logging("Start Training {}epochs (start from epoch {}) at: {}".format(self.epochs, self.start_epoch  ,time.strftime(" %d.%B %H:%M:%S")))
        
        
        
        #---------load data----------------------------------------------------
        print('======batch_size=', self.batch_size )
        transform_train= torchvision.transforms.Compose([
                           self.augmentation, # GroupMultiScaleCrop+ GroupRandomHorizontalFlip
                           Stack(roll=(self.base_model in ['BNInception','InceptionV3'])),
                           ToTorchFormatTensor(div=(self.base_model not in ['BNInception','InceptionV3'])),
                           self.normalize,
                       ])
        train_loader = torch.utils.data.DataLoader(
#            TSNDataSet(self.root_path, self.train_list, num_segments=self.num_segments,
#                       new_length=self.new_length,modality=self.modality,image_tmpl=self.prefix,
#                       transform=transform_train,
#                       fromEpic= 1
#                        ),
            EpicDataSet( pickle_file= self.train_list,
                 num_segments= self.num_segments, 
                 new_length= self.new_length, 
                 modality=self.modality,
                 transform= transform_train,
                 random_shift=True, test_mode= False
                 ),
            
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True)

        transform_val= torchvision.transforms.Compose([
                           GroupScale(int(self.scale_size)), #hw 
                           GroupCenterCrop(self.crop_size), #hw
                           Stack(roll=(self.base_model in ['BNInception','InceptionV3'])),
                           ToTorchFormatTensor(div=(self.base_model not in ['BNInception','InceptionV3'])),
                           self.normalize,
                       ])
        val_loader = torch.utils.data.DataLoader(
#            TSNDataSet(self.root_path, self.val_list, num_segments=self.num_segments,
#                       new_length= self.new_length,modality=self.modality,image_tmpl=self.prefix,
#                       random_shift=False,
#                       transform= transform_val,
#                       fromEpic= 1
#                        ),
            EpicDataSet( pickle_file= self.val_list,
                 num_segments= self.num_segments, 
                 new_length= self.new_length, 
                 modality=self.modality,
                 transform= transform_val,
                 random_shift=False, test_mode= True
                 ),
        
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True)
        
        optimizer = torch.optim.SGD(self.model_policies, #---params
                                    self.lr,
                                    momentum= self.momentum,
                                    weight_decay=self.weight_decay) 
        

        val_monitor=[]
        
        end= time.time()
        for epoch in range(self.start_epoch, self.epochs):
            print('\n','-'*10,'epoch:{}, lr {}'.format(epoch, self.lr),'-'*10, time.strftime(" %d.%B %H:%M:%S"))
            self.adjust_learning_rate(optimizer, epoch, self.lr_steps)

            # train for one epoch
            tr_loss= self.train(train_loader,  model, self.criterion, optimizer, epoch, self.log_training, ifprint)

            #----------------validation ----------------------------
            # evaluate on validation set        
            val= ((epoch- self.start_epoch +1) % self.eval_freq == 0 or epoch== self.epochs- 1 )
            if (val):
                print('-'*5,'begin evaluating.. ')
                val_prec1, val_prec5, val_loss = self.validate( model, val_loader, self.criterion, epoch, self.log_training)
                val_monitor.append((epoch, (time.time()-end)/60, tr_loss.item(), val_loss.item(), val_prec1.item(), val_prec5.item()))# get single item from tensor
                
                is_best = val_prec1 > self.best_prec1
                self.best_prec1 = max(val_prec1, self.best_prec1)
                #---- checkpoint------save--------
                print('-'*5,'save checkpoint.. ')
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.base_model,
                    'state_dict': model.state_dict(),
                    'best_prec1': self.best_prec1,
                }, is_best)
                print('-'*30, '\n')
            #--------------log loss--------------------------------
        
        print('*'*10, 'training completed!')
        print('Best Prec@1: %.3f'%(self.best_prec1))
        
        df= pd.DataFrame(val_monitor, columns=['epoch','epochtime(min)','train_loss','val_loss','val_prec1','val_prec5'])
        print(df)
    
# ----------------------------------------------------------------------------------------------------------------        





    def train(self, train_loader, model, criterion, optimizer, epoch, outputfile, ifprint):
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        if not self.partial_bn:
            model.module.partialBN(False)
        else:
            model.module.partialBN(True)
        # switch to train mode
        model.train()

        #  num_segments*new_length* channel (rgb 3, flow 2)* crop_size
        if ifprint: print('begin train(), train_loader  len:{}'.format( len(train_loader)))
        tr_end=time.time()
        for i, (input, target) in enumerate(train_loader):

            if i==0 and ifprint:
                print('input size: {} (batchsize ,num_segments*new_length* channel (rgb 3, flow 2), [crop_size])'.format(input.size()))  
            
            # measure data loading time
            data_time.update(time.time() - tr_end)

    ##        target = target.cuda(async=True) #async is now a reserved word in Python >= 3.7 so use non_blocking instead.
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(output.data, target, topk=(1,5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))
            top5.update(prec5, input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            if self.clip_gradient is not None:
                total_norm = clip_grad_norm(model.parameters(), self.clip_gradient) ##torch.nn.utils.clip_grad_norm is now deprecated in favor of torch.nn.utils.clip_grad_norm_.
                if total_norm > self.clip_gradient:
                    print("clipping gradient: {} with coef {}".format(total_norm, self.clip_gradient / total_norm))

            optimizer.step()

            # measure elapsed time
            batcht0= time.time() - tr_end
            batch_time.update(batcht0)
            
            idx= epoch * len(train_loader) + i+1
            
            op= 'epoch:{:^5d}, batch: {:2d} /{}, lr: {:.5f}, time(min): {:.1f}, loss: {:.4f}, prec1: {:.2f}, prec5: {:.2f}'.format(
                          epoch, i, len(train_loader), optimizer.param_groups[-1]['lr'],
                          batcht0/60, loss.data , prec1, prec5
                           )
            if outputfile: self.logging(op)
            if (ifprint and idx % self.print_freq == 0) or (i== len(train_loader)-1) : 
                print(op)
                
        return losses.avg 


    def validate(self, model, val_loader,  criterion, iter= None, outputfile=None):
        
    #    batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode: same as model.train(mode=False)
        model.eval() 
        val_end = time.time()
        logits=[]
        for i, (input, target) in enumerate(val_loader):
            
            if i==0 :
                print('input size: {} (batchsize ,num_segments*new_length* channel (rgb 3, flow 2), [crop_size])'.format(input.size()))  
                  
            #print('validating enumerate batch=', i )
            
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            #print('###{}: input batchs*size: {}*{} output size:{}'.format(i, len(val_loader), input.size(),output.size()))
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = self.accuracy(output.data, target, topk=(1,5))

            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0)) #input.size(0)= batch_size
            top5.update(prec5, input.size(0))
            if not outputfile:# testing
                logits.extend(output.tolist())
            
        
        val_time= time.time() - val_end
        op = ( (('-'*60 +'(epoch{}), validating'.format(iter)) if outputfile else 'testing') +
              'Results: time(min): {:.1f},  Loss {loss.avg:.5f}, Prec1 {top1.avg:.3f}( best:{best_p:.3f} ), Prec5 {top5.avg:.3f}'
              .format( val_time/60, loss=losses, top1=top1, best_p= self.best_prec1, top5=top5))
        print(op)
        
        if outputfile: # validating
            self.logging(op)
            return top1.avg, top5.avg, losses.avg
        else:  # testing
            return logits, top1.avg, top5.avg, losses.avg


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, '%s/%s_checkpoint.pth.tar' % ('model', self.store_name))
        if is_best:
            shutil.copyfile('%s/%s_checkpoint.pth.tar' % ('model', self.store_name),'%s/%s_best.pth.tar' % ('model', self.store_name))


    def adjust_learning_rate(self, optimizer, epoch, lr_steps):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = self.lr * decay
        if epoch in np.array(lr_steps).astype(int):
            text= 'learning_rate decayed({}*{}) to {}'.format(self.lr, decay, lr)
            print(text)
            self.logging(text)


        decay = self.weight_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']


    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    def logging(self, text):
        with open(self.log_training, 'a') as log:
            log.write(text +  '\n')
            log.flush()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



