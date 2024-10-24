import importlib

import torch
import torch.nn as nn

from unet3d.buildingblocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv
from unet3d.utils import create_feature_maps


class Position_AM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(Position_AM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=self.chanel_in, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=self.chanel_in, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C,thickness, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, thickness*width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, thickness*width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, thickness*width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C,thickness, height, width)

        #out = self.gamma*out + x
        out = out + x
        return out


class Block_self_attention_inter_intra_3D(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim=64,block_width=8,depth_width=2,stride=2,kernel=3):
        super(Block_self_attention_inter_intra_3D, self).__init__()
        self.chanel_in = in_dim

        self.block_width=block_width
        self.depth_width=depth_width
        self.inter_block_SA=Position_AM_Module(in_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        
        self.stride=stride
        self.kernel=kernel
        #self.split_size_H=[]
        #self.split_size_W=[]
        #for k in range (int(self.block_num)):
        #    self.split_size_H.append(self.block_width)
        #    self.split_size_W.append(self.block_width)


    def forward(self, x):

        #print (x.size())
        _, _, height, width,thickness = x.size()

        self.block_num=height/self.block_width
        self.block_num_depth=thickness/self.depth_width

        self.scane_x_max_num=height/(self.block_width*self.stride)
        self.scane_y_max_num=self.scane_x_max_num

        self.scane_z_max_num=thickness/(self.depth_width*self.stride)

        x_clone=x.clone()   
        #print (x.size())
        for i in range(int(self.scane_x_max_num)+1):
            for j in range (int(self.scane_y_max_num)+1):
                for z in range (int(self.scane_z_max_num)+1):

                    start_x=i*self.block_width*self.stride
                    end_x=i*self.block_width*self.stride+self.block_width*self.kernel

                    start_y=j*self.block_width*self.stride
                    end_y=j*self.block_width*self.stride+self.block_width*self.kernel

                    start_z=z*self.depth_width*self.stride
                    end_z=z*self.depth_width*self.stride+self.depth_width*self.kernel


                    if end_y>height:
                        end_y=height
                    if end_x>height:
                        end_x=height
                    if end_z>thickness:
                        end_z=thickness
                    #print (start_x)
                    #print (end_x)
                    #print (start_y)
                    #print (end_y)
                    #print (start_z)
                    #print (end_z)                    
                    if start_x<height and start_y<height and start_z<thickness:
                        tep_=x[:,:,start_z:end_z,start_x:end_x,start_y:end_y]
                        #print (tep_.size())
                        if len(tep_.size())>1:

                            #x_clone[:,:,start_z:end_z,start_x:end_x,start_y:end_y]=self.inter_block_SA(x[:,:,start_z:end_z,start_x:end_x,start_y:end_y])
                            x_clone[:,:,start_x:end_x,start_y:end_y,start_z:end_z]=self.inter_block_SA(x[:,:,start_x:end_x,start_y:end_y,start_z:end_z])

        return x_clone


class Position_AM_Module_3D(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_channel,inter_channel):
        super(Position_AM_Module_3D, self).__init__()
        
        self.chanel_in = inter_channel

        self.query_conv = nn.Conv3d(in_channels=in_channel, out_channels=inter_channel, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_channel, out_channels=inter_channel, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """

        m_batchsize, C, thickness, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, thickness*width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, thickness*width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        #print (x.size())
        proj_value = self.value_conv(x).view(m_batchsize, -1, thickness*width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, thickness, height, width)

        #out = self.gamma*out + x
        out = out + x
        return out


class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

        #self.attention_module_3D=Position_AM_Module_3D(64,16)  # original channel is 64 and reduced to 16
        self.Block_SA1=Block_self_attention_inter_intra_3D(64,8,4,2,3) #(in_dim=64,block_width=8,depth_width=2,stride=2,kernel=3)  
        self.Block_SA2=Block_self_attention_inter_intra_3D(64,8,4,2,3) #(in_dim=64,block_width=8,depth_width=2,stride=2,kernel=3)  
        #self.Block_SA2=Block_self_attention_inter_intra_change_second_layer(64,12,2,3)    # (64,block_width=16,stride=2,kernel=3)        
    def forward(self, x):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        import numpy as np
        #x=torch.tensor(np.ndarray(shape=(1,1,32,128,128), dtype=float, order='F'))
        #x=x.cuda().float()
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            
            x = encoder(x)
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        decoder_index=1
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
            #if decoder_index==3:
            #    x=self.Block_SA1(x)
            #    x=self.Block_SA2(x)
            #print ('decoders size are:',x.size())
            #print ('encoder_features size are:',encoder_features.size())
            decoder_index=decoder_index+1
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class ResidualUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        conv_layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, conv_layer_order='crb', num_groups=8,
                 **kwargs):
        super(ResidualUNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses ExtResNetBlock as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses ExtResNetBlock as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            decoder = Decoder(reversed_f_maps[i], reversed_f_maps[i + 1], basic_module=ExtResNetBlock,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
        #self.attention_module_3D=Position_AM_Module_3D(64,16)  # original channel is 64 and reduced to 16
        # works for (64,3,2,3) of (32,128,128) input for residual 3D 
        self.Block_SA1=Block_self_attention_inter_intra_3D(64,8,3,2,3) #(in_dim=64,block_width=8,depth_width=2,stride=2,kernel=3)  
        self.Block_SA2=Block_self_attention_inter_intra_3D(64,8,3,2,3) #(in_dim=64,block_width=8,depth_width=2,stride=2,kernel=3)  
    def forward(self, x):
        # encoder part
        import numpy as np
        #x=torch.cat((x,x[:,:,:,0:32,0:32]),3)
        #x=torch.cat((x,x[:,:,:,0:32,0:32]),4)        
        #x=np.zeros(1,1,64,160,160))
        #x=torch.tensor(np.ndarray(shape=(1,1,32,128,128), dtype=float, order='F'))
        #x=x.cuda().float()
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            #print ('inserted x size is ',x.size())
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        

        # decoder part
        decoder_index=1
        #print (encoders_features)
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            #print ('feature size ',encoder_features.size())
            #print ('x size ',x.size())
            x = decoder(encoder_features, x)
            if decoder_index==3:
                #print ('feature size is ',x.size())
                x=self.Block_SA1(x)
                x=self.Block_SA2(x)
            decoder_index=decoder_index+1

        x = self.final_conv(x)
        #print ('final feature size is ',x.size())
        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class Noise2NoiseUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, f_maps=16, num_groups=8, **kwargs):
        super(Noise2NoiseUNet3D, self).__init__()

        # Use LeakyReLU activation everywhere except the last layer
        conv_layer_order = 'clg'

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # 1x1x1 conv + simple ReLU in the final convolution
        self.final_conv = SingleConv(f_maps[0], out_channels, kernel_size=1, order='cr', padding=0)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        return x


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('unet3d.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    
    return model_class(**model_config)


###############################################Supervised Tags 3DUnet###################################################

class TagsUNet3D(nn.Module):
    """
    Supervised tags 3DUnet
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels; since most often we're trying to learn
            3D unit vectors we use 3 as a default value
        output_heads (int): number of output heads from the network, each head corresponds to different
            semantic tag/direction to be learned
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `DoubleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
    """

    def __init__(self, in_channels, out_channels=3, output_heads=1, conv_layer_order='crg', init_channel_number=32,
                 **kwargs):
        super(TagsUNet3D, self).__init__()

        # number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups)
        ])

        self.final_heads = nn.ModuleList(
            [FinalConv(init_channel_number, out_channels, num_groups=num_groups) for _ in
             range(output_heads)])

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # apply final layer per each output head
        tags = [final_head(x) for final_head in self.final_heads]

        # normalize directions with L2 norm
        return [tag / torch.norm(tag, p=2, dim=1).detach().clamp(min=1e-8) for tag in tags]


################################################Distance transform 3DUNet##############################################
class DistanceTransformUNet3D(nn.Module):
    """
    Predict Distance Transform to the boundary signal based on the output from the Tags3DUnet. Fore training use either:
        1. PixelWiseCrossEntropyLoss if the distance transform is quantized (classification)
        2. MSELoss if the distance transform is continuous (regression)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        final_sigmoid (bool): 'sigmoid'/'softmax' whether element-wise nn.Sigmoid or nn.Softmax should be applied after
            the final 1x1 convolution
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, init_channel_number=32, **kwargs):
        super(DistanceTransformUNet3D, self).__init__()

        # number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order='crg',
                    num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, pool_type='avg', conv_layer_order='crg',
                    num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(3 * init_channel_number, init_channel_number, conv_layer_order='crg', num_groups=num_groups)
        ])

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        # allow multiple heads
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # apply final 1x1 convolution
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class EndToEndDTUNet3D(nn.Module):
    def __init__(self, tags_in_channels, tags_out_channels, tags_output_heads, tags_init_channel_number,
                 dt_in_channels, dt_out_channels, dt_final_sigmoid, dt_init_channel_number,
                 tags_net_path=None, dt_net_path=None, **kwargs):
        super(EndToEndDTUNet3D, self).__init__()

        self.tags_net = TagsUNet3D(tags_in_channels, tags_out_channels, tags_output_heads,
                                   init_channel_number=tags_init_channel_number)
        if tags_net_path is not None:
            # load pre-trained TagsUNet3D
            self.tags_net = self._load_net(tags_net_path, self.tags_net)

        self.dt_net = DistanceTransformUNet3D(dt_in_channels, dt_out_channels, dt_final_sigmoid,
                                              init_channel_number=dt_init_channel_number)
        if dt_net_path is not None:
            # load pre-trained DistanceTransformUNet3D
            self.dt_net = self._load_net(dt_net_path, self.dt_net)

    @staticmethod
    def _load_net(checkpoint_path, model):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model_state_dict'])
        return model

    def forward(self, x):
        x = self.tags_net(x)
        return self.dt_net(x)
