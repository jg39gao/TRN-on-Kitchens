from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant

import TRNmodule

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=1,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True):
        ''' 
        Single-scale Temporal Relational Network
        See https://arxiv.org/abs/1711.08496 for more details.
        and see https://github.com/epic-kitchens/action-models/blob/master/tsn.py
        Args:
            num_class:
                Number of classes, can be either a single integer,
                or a 2-tuple for training verb+noun multi-task models
            num_segments:
                Number of frames/optical flow stacks input into the model
            modality:
                Either ``RGB`` or ``Flow``.
            base_model:
                Backbone model architecture one of ``resnet18``, ``resnet30``,
                ``resnet50``, ``BNInception``, ``InceptionV3``, ``VGG16``.
                ``BNInception`` and ``resnet50`` are the most thoroughly tested.
            new_length:
                The number of frame(channel inputs) per snippet
            consensus_type:
                The consensus function used to combined information across segments.
                One of ``avg``, ``max``, ``TRN``, ``TRNMultiscale``.
            before_softmax:
                Whether to output class score before or after softmax.
            dropout:
                The dropout probability. The dropout layer replaces the backbone's
                classification layer.
            img_feature_dim:
                Only for TRN/MTRN models. The dimensionality of the features used for
                relational reasoning.
            partial_bn:
                Whether to freeze all BN layers beyond the first 2 layers.
            pretrained:
                Either ``'imagenet'`` for ImageNet initialised models,
                or ``'epic-kitchens'`` for weights pretrained on EPIC-Kitchens.
        '''
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments 
        self.new_length = new_length
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        
        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5 #
        else:
            self.new_length = new_length
        
        
        self._prepare_base_model(base_model)
        feature_dim = self._prepare_tsn(num_class)
        
        if print_spec == True:
            print(("""
+-------------------------------------------------------+
Initializing TSN with 
               base model:  {}
Configurations:  
           input_modality:  {}
             num_segments:  {}
               new_length:  {}
         consensus_module:  {}
            dropout_ratio:  {}
feature_dim(for fc_layer):  {}
 img_feature_dim（[M]TRN):  {}
          before_softmax:  {}
+-------------------------------------------------------+
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, 
            self.dropout, feature_dim, self.img_feature_dim, self.before_softmax)))



        if self.modality in('RGB', 'Flow'):
            print("Converting the ImageNet model according to the snippet length")
            self.base_model = self._construct_RGBflow_model(self.base_model)
            print("Done. base model(for RGB or Flow) ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")
            
            
        if 'trn' in consensus_type.lower() :#['TRN', 'MTRN']:
            # plug in the Temporal Relation Network Module
            self.consensus = TRNmodule.return_TRN(consensus_type, self.img_feature_dim, self.num_segments, num_class)
        else:
            self.consensus = ConsensusModule(consensus_type) # here lies a bug to be fixed."TypeError: SegmentConsensus.forward: expected Variable (got NoneType) for return value 0"

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        ''' prepare tsn after base model prepared. return feature_dim 
        PARAMS: 
            num_class,  number of target classes 
        
        RETURN:
            feature_dim,  the size of each input sample of fc_layer(base_model.last_layer_name)
        '''
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class)) #finetune
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.consensus_type in ['TRN','TRNmultiscale']:
                # create a new linear layer as the frame feature
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            else:
                # the default consensus types in TSN
                self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        # normalisation @GJJ tb concise
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
            
        return feature_dim

    def _prepare_base_model(self, base_model):
        '''  Prepare base model ， set the hyperparameters for different models.
        (base_model.last_layer_name), input_size, mean, std, modality. 
        PARAMS:
            base_model,  str like 'resnet','vgg','BNInception', 'InceptionV3'
        '''

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True) #pretrained= True
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif base_model == 'InceptionV3':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_mean = [104,117,128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1+self.new_length)

        elif 'inception' in base_model:
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters 
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):#BatchNorm2d: Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
                        
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        '''
        PARAMS:
            input.size()= batchsize ,num_segments*new_length* channel, [scaled size]
        '''
        
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

#        print('j----basemodel: input:{} {}, output:{} {}'.format(type(input), input.size(), type(base_out), base_out.size()))
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
#            print('after new_fc{}:{}'.format(self.new_fc, base_out.size()))

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)# base_out is segmental. 

        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2 # input channel = 2 when optical flow
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_RGBflow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]#Conv2d.parameters():  ([out_channels, in_channels/groups, *kenelsize tuple], bias)
        kernel_size = params[0].size()
        
        new_kernel_size = kernel_size[:1] + (sample_len, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(sample_len, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()] #Conv2d.parameters():   ([out_channels, in_channels, *kenelsize tuple], bias)
        kernel_size = params[0].size()
        
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0].split('.')[0]
#        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224  #@why?

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
