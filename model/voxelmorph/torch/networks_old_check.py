import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args




import math
class GetNCCImg:
    """
    Local (over window) normalized cross correlation loss.forward_seg_training_with_grid_deformation_dose
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [5] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        NCCImg = cross * cross / (I_var * J_var + 1e-5)

        return NCCImg
import numpy as np
class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf
 
    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x




class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_nearest = layers.SpatialTransformer(inshape)
        #self.transformer_nearest = layers.SpatialTransformer(inshape,mode='nearest')

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out



class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """
        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf
 
    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x
        



class VxmDense_3D_LSTM(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_3D_LSTM(
        #self.unet_model = Unet_3D_LSTM(    
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_nearest = layers.SpatialTransformer(inshape)
        self.transformer_nearest = layers.SpatialTransformer(inshape,mode='nearest')



        self.grid_template=torch.zeros(1, 1,192, 192, 48)
        grid_w=12
        for i in range(0,32):
                    
            self.grid_template[:,:,:,i*6,:]=1

        for i in range(0,32):
                    
            self.grid_template[:,:,i*6,:,:]=1
        self.grid_template=self.grid_template.cuda()        
        self.get_NCCimg=GetNCCImg().loss

    def forward(self, source, target,source_m,flow_num):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        h=None
        c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x,h,c = self.unet_model(x,states=None)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = self.transformer(source, pos_flow)
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            


        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list


    def forward_seg_training(self, source, target,source_m,flow_num,states):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        #states=[h,c]
        x,h,c = self.unet_model(x,states)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = self.transformer(source, pos_flow)
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            

        states=[h.detach(),c.detach()]
        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,states,source_m


    def forward_seg_training_val_DVF(self, source, target,source_m,flow_num,states):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        #states=[h,c]
        x,h,c = self.unet_model(x,states)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = self.transformer(source, pos_flow)
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            

        states=[h.detach(),c.detach()]
        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,states,source_m


    def forward_seg_training_with_grid_deformation(self, source, target,source_m,flow_num,states,template_in,flow_in,ori_plan_C):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        #states=[h,c]
        x,h,c = self.unet_model(x,states)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        
        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None
        #print (pos_flow.size())

        flow_acc=pos_flow+flow_in

        pos_flow_show=torch.abs(pos_flow)+flow_in
        pos_flow_show_cur=torch.abs(pos_flow)
        
        #pos_flow_show=torch.sum(pos_flow_show, dim=1) 
        #pos_flow_show_cur=torch.sum(pos_flow_show_cur, dim=1) 
        # warp image with flow field
        #source = self.transformer(source, pos_flow)

        source = self.transformer(ori_plan_C, flow_acc)
        #source_m = self.transformer_nearest(source_m, pos_flow)
        #print (source_m.size())
        #print (self.grid_template.size())
        grid_template=self.transformer_nearest(self.grid_template, flow_acc)

        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            

        states=[h.detach(),c.detach()]
        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,states,source_m,grid_template,pos_flow_show,pos_flow_show_cur,flow_acc



    def forward_seg_training_with_grid_deformation_NCC(self, source, target,source_m,flow_num,states,template_in,flow_in,ori_plan_C):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        #states=[h,c]
        x,h,c = self.unet_model(x,states)

        ncc_img1=self.get_NCCimg(target,source)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        
        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None
        #print (pos_flow.size())

        flow_acc=pos_flow+flow_in

        pos_flow_show=torch.abs(pos_flow)+flow_in
        pos_flow_show_cur=torch.abs(pos_flow)
        
        #pos_flow_show=torch.sum(pos_flow_show, dim=1) 
        #pos_flow_show_cur=torch.sum(pos_flow_show_cur, dim=1) 
        # warp image with flow field
        #source = self.transformer(source, pos_flow)

        source = self.transformer(ori_plan_C, flow_acc)
        #source_m = self.transformer_nearest(source_m, pos_flow)
        #print (source_m.size())
        #print (self.grid_template.size())
        grid_template=self.transformer_nearest(self.grid_template, flow_acc)

        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            

        states=[h.detach(),c.detach()]
        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,states,source_m,grid_template,pos_flow_show,pos_flow_show_cur,flow_acc,ncc_img1





    def forward_seg_training_with_grid_deformation_dose(self, source, target,source_m,flow_num,states,template_in,flow_in,ori_plan_C,ori_dose):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        #states=[h,c]
        x,h,c = self.unet_model(x,states)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        
        preint_flow = pos_flow
        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None
        #print (pos_flow.size())
        flow_acc=pos_flow+flow_in
        pos_flow_show=torch.abs(pos_flow)+flow_in
        pos_flow_show_cur=torch.abs(pos_flow)
        
        #pos_flow_show=torch.sum(pos_flow_show, dim=1) 
        #pos_flow_show_cur=torch.sum(pos_flow_show_cur, dim=1) 
        # warp image with flow field
        #source = self.transformer(source, pos_flow)

        source = self.transformer(ori_plan_C, flow_acc)
        dose_warp= self.transformer(ori_dose, flow_acc) 
        #source_m = self.transformer_nearest(source_m, pos_flow)
        #print (source_m.size())
        #print (self.grid_template.size())
        grid_template=self.transformer_nearest(self.grid_template, flow_acc)

        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            

        states=[h.detach(),c.detach()]
        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,states,source_m,grid_template,pos_flow_show,pos_flow_show_cur,flow_acc,dose_warp

    def forward_only_img(self, source, target,flow_num):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        h=None
        c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x,h,c = self.unet_model(x,states=None)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = self.transformer(source, pos_flow)
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            


        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list



class VxmDense_3D_LSTM_Step_Reg(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet_3D_LSTM(
        #self.unet_model = Unet_3D_LSTM(    
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.transformer_nearest = layers.SpatialTransformer(inshape)
        #self.transformer_nearest = layers.SpatialTransformer(inshape,mode='nearest')

    def forward(self, source, source_m,target,states):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        x = torch.cat([source, target], dim=1)

        #print (x.size())
        #print (states.size())
        x,h,c = self.unet_model(x,states)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None
        source = self.transformer(source, pos_flow)
        source_m = self.transformer_nearest(source_m, pos_flow)
        states=[h.detach(),c.detach()]
        return source, source_m, pos_flow,states


    def forward_seg_training(self, source, target,source_m,flow_num,h,c):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        if h is None:
            states=None
        else:

            states=[h,c]
        x,h,c = self.unet_model(x,states)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = self.transformer(source, pos_flow)
        #print (source_m.size())
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            


        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,h,c,source_m




    def forward_seg_training_DVF(self, source, target,source_m,flow_num,h,c,flow_in):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        if h is None:
            states=None
        else:

            states=[h,c]
        x,h,c = self.unet_model(x,states)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None
        neg_flow = -pos_flow
        
        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow)
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        flow_acc=torch.abs(pos_flow)+flow_in

        #flow_acc=flow_in+ neg_flow#pos_flow
        #flow_acc_abs=torch.abs(pos_flow)+flow_in
        # warp image with flow field
        source = self.transformer(source, pos_flow)
        #print (source_m.size())
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            


        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,h,c,source_m,flow_acc

    def forward_only_img(self, source, target,flow_num):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        h=None
        c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x,h,c = self.unet_model(x,states=None)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = self.transformer(source, pos_flow)
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            


        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list





class VxmDense_3D_LSTM_Step_Reg_All_Encoder_LSTM(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        range_flow=1,
        bidir=False,
        use_probs=False,
        pre_train=False,
        device='cuda'):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True
        self.device = device
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        
        self.range_flow=range_flow
        self.unet_model = Unet_All_Encoder_3D_LSTM(
         
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            device = self.device
        )



        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        #self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.integrate = layers.VecInt_range_flow(down_shape, int_steps) if int_steps > 0 else None
        # configure transformer

        #self.transformer = layers.SpatialTransformer(inshape)
        #self.transformer_nearest = layers.SpatialTransformer(inshape)

        self.transformer = layers.SpatialTransformer_range_flow(inshape)
        self.transformer_nearest = layers.SpatialTransformer_range_flow(inshape,mode='nearest')



        self.grid_template=torch.zeros(1, 1,192, 192, 48)
        grid_w=12
        for i in range(0,32):
                    
            self.grid_template[:,:,:,i*6,:]=1

        for i in range(0,32):
                    
            self.grid_template[:,:,i*6,:,:]=1
        self.grid_template=self.grid_template.to(self.device) 

    def save_model(self, name):
        
        self.save_network(self.unet_model, name)

    def save_network(self, network, name):
        
        save_path = name#os.path.join(self.save_dir, save_filename)
        
        torch.save(network.cpu().state_dict(), save_path)
        network.to(self.device)
        
    def load_model_Jue(self,name):
        self.load_network(self.unet_model, name)

    def load_network(self, network, name):
        #network.load_state_dict(torch.load(name))
        #print ('info name is !!!!!!!!!!!!!!!! ',name)
        #name='/lila/data/deasy/Eric_Data/Registration/voxlmorph_3D_development/all_encoder_lstm_sv_again/sv_reg_model_reg.pt'
        self.unet_model.load_state_dict(torch.load(name))


    def forward(self, source, source_m,target,state_h_in,state_c_in):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x,state_h,state_c = self.unet_model(x,state_h_in,state_c_in)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
           
            #pos_flow = self.integrate(pos_flow)
            pos_flow = self.integrate(pos_flow,self.range_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        #source = self.transformer(source, pos_flow)
        #source_m = self.transformer_nearest(source_m, pos_flow)

        source = self.transformer(source, pos_flow,self.range_flow)
        source_m = self.transformer_nearest(source_m, pos_flow,self.range_flow)

        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        
        #return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,state_h,state_c,source_m
        return source, source_m, pos_flow,state_h,state_c


    def inference_img(self, img,flow):
        
        img,flow = img.to(self.device),flow.to(self.device)
        transformer_img = layers.SpatialTransformer_range_flow(torch.squeeze(img).shape).to(self.device)
        

        img_def = transformer_img(img, flow,self.range_flow)
        
        return img_def
    
    def inference_msk(self, msk,flow):
        
        msk,flow = msk.to(self.device),flow.to(self.device)
        transformer_msk = layers.SpatialTransformer_range_flow(torch.squeeze(msk).shape,mode='nearest').to(self.device)

        msk_def = transformer_msk(msk, flow,self.range_flow)
        
        
        return msk_def

    def forward_seg_training_all_enc_lstm(self, source, target,source_m,state_h_in,state_c_in):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x,state_h,state_c = self.unet_model(x,state_h_in,state_c_in)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            #pos_flow = self.integrate(pos_flow)
            pos_flow = self.integrate(pos_flow,self.range_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        #source = self.transformer(source, pos_flow)
        #source_m = self.transformer_nearest(source_m, pos_flow)
        
        source = self.transformer(source, pos_flow,self.range_flow)
        source_m = self.transformer_nearest(source_m,pos_flow,self.range_flow)

        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        
        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,state_h,state_c,source_m
        #return source, source_m, pos_flow,state_h,state_c
        

    def forward_seg_training_all_enc_lstm_accu_dvf(self, source, target,source_m,state_h_in,state_c_in,flow_in,plan_ori_img,plan_ori_msk):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x,state_h,state_c = self.unet_model(x,state_h_in,state_c_in)


        # transform into flow field
        flow_field = self.flow(x)
        # resize flow for integration
        pos_flow = flow_field
        pos_flow=flow_in+pos_flow
        
        if self.resize:
            pos_flow = self.resize(pos_flow)


        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            #pos_flow = self.integrate(pos_flow)
            pos_flow = self.integrate(pos_flow,self.range_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        #source = self.transformer(source, pos_flow)
        #source_m = self.transformer_nearest(source_m, pos_flow)

        pos_flow_cur=pos_flow

        
        source = self.transformer(plan_ori_img, pos_flow,self.range_flow)
        source_m = self.transformer_nearest(plan_ori_msk,pos_flow,self.range_flow)
        #source_m = self.transformer_nearest(source_m,pos_flow_cur,self.range_flow)



        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        
        #return source, source_m, pos_flow,state_h,state_c
        return source, source_m, pos_flow,state_h,state_c


    def forward_seg_training_all_enc_lstm_DVF(self, source, target,source_m,flow_num,state_h_in,state_c_in,flow_in):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x,state_h,state_c = self.unet_model(x,state_h_in,state_c_in)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None
        flow_acc=flow_in+pos_flow
        # warp image with flow field
        source = self.transformer(source, pos_flow)
        #print (source_m.size())
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,state_h,state_c,source_m,flow_acc


    def forward_seg_training_All_Enc_Lstm(self, source, target,source_m,flow_num,h,c):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        #x = torch.cat([x, source_m], dim=1)

        if h is None:
            x,h,c = self.unet_model(x,h,c)
        else:

            x,h,c = self.unet_model(x,h,c)
        


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = self.transformer(source, pos_flow)
        #print (source_m.size())
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            


        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,h,c,source_m


    def forward_only_img(self, source, target,flow_num):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        h=None
        c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        x,h,c = self.unet_model(x,states=None)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        source = self.transformer(source, pos_flow)
        source_m = self.transformer_nearest(source_m, pos_flow)
        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
        for iter_i in range (flow_num):
            #print (iter_i)
            
            x = torch.cat([source, target], dim=1)


            states=[h,c]
            #states=None
            x,h,c = self.unet_model(x,states)

            # transform into flow field
            #print ('x.size ',x.size())  [1,16,256,256]
            flow_field = self.flow(x)

            # resize flow for integration
            pos_flow = flow_field

            if self.resize:
                pos_flow = self.resize(pos_flow)


            # integrate to produce diffeomorphic warp
            if self.integrate:
                #pos_flow = self.integrate_nearest(pos_flow)
                pos_flow = self.integrate(pos_flow)
            

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(pos_flow)
                
            
            # warp image with flow field
            source = self.transformer(source, pos_flow) # source wraped image
            source_m = self.transformer_nearest(source_m, pos_flow) # source wraped image
            positive_deform_list.append(pos_flow)
            source_deformed_list.append(source)
            source_m_deformed_list.append(source_m)
            
            


        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list

    
    def forward_seg_training_with_grid_deformation(self, source, target,source_m,flow_num,states_h_in,states_c_in,template_in,flow_in,ori_plan_C):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''


        'First iteration'
        'ITERATION'

        source_deformed_list=[]
        source_m_deformed_list=[]
        #source_img_deformed_list=[]
        positive_deform_list=[]


        #h=None
        #c=None
        
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        
        x,states_h,states_c = self.unet_model(x,states_h_in,states_c_in)


        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        
        preint_flow = pos_flow



        # negate flow for bidirectional model
        #neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            #print ('pos flow size is ',pos_flow.size())
            pos_flow = self.integrate(pos_flow)
            
            #neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                #neg_flow = self.fullsize(neg_flow) if self.bidir else None
        #print (pos_flow.size())

        flow_acc=pos_flow+flow_in
        #print (flow_acc.size())
        pos_flow_show=torch.abs(pos_flow)+flow_in
        pos_flow_show_cur=torch.abs(pos_flow)
        

        source = self.transformer(source, pos_flow)
        #source = self.transformer(ori_plan_C, flow_acc)
        #source_m = self.transformer_nearest(source_m, pos_flow)
        #print (source_m.size())
        #print (self.grid_template.size())
        grid_template=self.transformer_nearest(self.grid_template, flow_acc)

        positive_deform_list.append(pos_flow)
        source_deformed_list.append(source)
        source_m_deformed_list.append(source_m)

        #y_target = self.transformer(target, neg_flow) if self.bidir else None

        #flow_num=8
        # start the LSTM Iteration
        
            
            

        #states=[h.detach(),c.detach()]
        return source, pos_flow,source_deformed_list,source_m_deformed_list,positive_deform_list,states_h,states_c,source_m,grid_template,pos_flow_show,pos_flow_show_cur,flow_acc
    

class Unet_3D_LSTM(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        input_img_channel=2
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += input_img_channel
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        #CBCT Tumor    
        self.conv_lstm=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,12,12,3), kernel_size=3,)  #[16,16,16]  [x,y,z]
        #CBCT Eso
        self.conv_lstm=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,10,10,5), kernel_size=3,)  #[16,16,16]  [x,y,z]
        self.dropout=torch.nn.Dropout(0.7)

    def forward(self, x, states):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            #print (x_enc[-1].size())
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()  #[1,32,16,16]
        #h=x
        #print (x.size())
        if states==None:
            h,c=self.conv_lstm(x,None)
        else:
            h,c=self.conv_lstm(x,states)

        x=h
        for layer in self.uparm: # [x channel size is 32]
            #print ('x size is up-sampling is ',x.size())
            #x=self.dropout(x)
            x = layer(x)
            #print (x.size())
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            #x=self.dropout(x)
            #print ('x size is extras is ',x.size())
            x = layer(x)
            #print (x.size())

        return x,h,c


class Unet_All_Encoder_3D_LSTM(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, device='cuda'):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        input_img_channel=2
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += input_img_channel
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        #CBCT Tumor    
        # self.conv_lstm=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,8,8,4), kernel_size=3,)  #[16,16,16]  [x,y,z]
        
        # self.conv_lstm4=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,16,16,8), kernel_size=3,device=device)
        # self.conv_lstm3=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,32,32,16), kernel_size=3,device=device)
        # self.conv_lstm2=ConvLSTM3D(input_channel=16, num_filter=16, b_h_w=(1,64,64,32), kernel_size=3,device=device)
        # self.conv_lstm1=ConvLSTM3D(input_channel=2, num_filter=2, b_h_w=(1,256,256,64), kernel_size=3,device=device)

        self.conv_lstm=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,8,8,4), kernel_size=3,)  #[16,16,16]  [x,y,z]
        self.conv_lstm4=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,16,16,8), kernel_size=3,device=device)
        self.conv_lstm3=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,32,32,16), kernel_size=3,device=device)
        self.conv_lstm2=ConvLSTM3D(input_channel=16, num_filter=16, b_h_w=(1,64,64,32), kernel_size=3,device=device)
        self.conv_lstm1=ConvLSTM3D(input_channel=2, num_filter=2, b_h_w=(1,128,128,64), kernel_size=3,device=device)

        # self.conv_lstm=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,16,16,8), kernel_size=3,)  #[16,16,16]  [x,y,z]
        # self.conv_lstm4=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,32,32,16), kernel_size=3,device=device)
        # self.conv_lstm3=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,64,64,32), kernel_size=3,device=device)
        # self.conv_lstm2=ConvLSTM3D(input_channel=16, num_filter=16, b_h_w=(1,128,128,64), kernel_size=3,device=device)
        # self.conv_lstm1=ConvLSTM3D(input_channel=2, num_filter=2, b_h_w=(1,256,256,128), kernel_size=3,device=device)

        #self.dropout=torch.nn.Dropout(0.7)

    def forward(self, x, state_h_in,state_c_in):
        
        #print ('x in size is ',x.size())
        # get encoder activations
        x_enc = [x]

        # layer  1
        # torch.Size([1, 2, 192, 192, 48])
        # layer  2
        # torch.Size([1, 16, 96, 96, 24])
        # layer  3
        #torch.Size([1, 32, 48, 48, 12])
        # layer  4
        # torch.Size([1, 32, 24, 24, 6])
        # layer 5
        # torch.Size([1, 32, 12, 12, 3])


        enc_lay_ct=0

        state_h=[]
        state_c=[]
        #print ('*'*50)
        #if state_h_in == None:
        #    print ('info print state_h_in None ')
        #else:
        #    print ('state_h_in size is ',len(state_h_in))
        for layer in self.downarm:
            enc_lay_ct=enc_lay_ct+1
            
            #print (x_enc[-1].size())
            #print (len(x_enc))
#
            if enc_lay_ct==1:
                state_h_in=None
                if state_h_in == None:
                    h1,c1=self.conv_lstm1(x_enc[-1],None)
                else:
                    h1,c1=self.conv_lstm1(x_enc[-1],[state_h_in[enc_lay_ct-1],state_c_in[enc_lay_ct-1]])
                
                #print (' layer ',enc_lay_ct)
                #print (x_enc[-1].size())
                x_enc[-1]=h1
                state_h.append(h1.detach())
                state_c.append(c1.detach())
                
                #print (h1.size())
            if enc_lay_ct==2:
                state_h_in=None
                if state_h_in == None:
                    h2,c2=self.conv_lstm2(x_enc[-1],None)
                else:
                    h2,c2=self.conv_lstm2(x_enc[-1],[state_h_in[enc_lay_ct-1],state_c_in[enc_lay_ct-1]])

                
                state_h.append(h2.detach())
                state_c.append(c2.detach())
                #print (' layer ',enc_lay_ct)
                #print (x_enc[-1].size())
                #print (h2.size())
                x_enc[-1]=h2
            if enc_lay_ct==3:
                state_h_in=None
                if state_h_in == None:
                    h3,c3=self.conv_lstm3(x_enc[-1],None)
                else:
                    h3,c3=self.conv_lstm3(x_enc[-1],[state_h_in[enc_lay_ct-1],state_c_in[enc_lay_ct-1]])

                
                state_h.append(h3.detach())
                state_c.append(c3.detach())
                #print (' layer ',enc_lay_ct)
                #print (x_enc[-1].size())
                x_enc[-1]=h3
            if enc_lay_ct==4:
                #state_h_in=None
                if state_h_in == None:
                    h4,c4=self.conv_lstm4(x_enc[-1],None)
                else:
                    h4,c4=self.conv_lstm4(x_enc[-1],[state_h_in[enc_lay_ct-1],state_c_in[enc_lay_ct-1]])
                
                state_h.append(h4.detach())
                state_c.append(c4.detach())
                #print (' layer ',enc_lay_ct)
                #print (x_enc[-1].size())
                x_enc[-1]=h4

            x_enc.append(layer(x_enc[-1]))

        #print (enc_lay_ct)
        # conv, upsample, concatenate series
        x = x_enc.pop()  #[1,32,16,16]
        #h=x
        # x last size is  torch.Size([1, 32, 12, 12, 3])
        if state_h_in==None:
            # print("X:",x.shape)
            h,c=self.conv_lstm(x,None)
        else:
            h,c=self.conv_lstm(x,[state_h_in[4],state_c_in[4]])

        state_h.append(h.detach())
        state_c.append(c.detach())
        x=h
        for layer in self.uparm: # [x channel size is 32]
            #print ('x size is up-sampling is ',x.size())
            
            x = layer(x)
            #print (x.size())
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            
            #print ('x size is extras is ',x.size())
            x = layer(x)
            #print (x.size())

        return x,state_h,state_c



class Unet_All_Encoder_3D_LSTM_Small(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        input_img_channel=2
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += input_img_channel
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        #CBCT Tumor    
        self.conv_lstm=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,12,12,3), kernel_size=3,)  #[16,16,16]  [x,y,z]
        self.conv_lstm4=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,24,24,6), kernel_size=3,)
        self.conv_lstm3=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,48,48,12), kernel_size=3,)
        self.conv_lstm2=ConvLSTM3D(input_channel=16, num_filter=16, b_h_w=(1,96,96,24), kernel_size=3,)
        self.conv_lstm1=ConvLSTM3D(input_channel=2, num_filter=2, b_h_w=(1,192,192,48), kernel_size=3,)

        #self.dropout=torch.nn.Dropout(0.7)

    def forward(self, x, state_h_in,state_c_in):
        
        #print ('x in size is ',x.size())
        # get encoder activations
        x_enc = [x]

        # layer  1
        # torch.Size([1, 2, 192, 192, 48])
        # layer  2
        # torch.Size([1, 16, 96, 96, 24])
        # layer  3
        #torch.Size([1, 32, 48, 48, 12])
        # layer  4
        # torch.Size([1, 32, 24, 24, 6])
        # layer 5
        # torch.Size([1, 32, 12, 12, 3])


        enc_lay_ct=0

        state_h=[]
        state_c=[]
        #print ('*'*50)
        #if state_h_in == None:
        #    print ('info print state_h_in None ')
        #else:
        #    print ('state_h_in size is ',len(state_h_in))
        for layer in self.downarm:
            enc_lay_ct=enc_lay_ct+1
            
            #print (x_enc[-1].size())
            #print (len(x_enc))
#
            if enc_lay_ct==1:
                
                
                state_h.append(x_enc[-1])
                state_c.append(x_enc[-1])
                #print (h1.size())
            if enc_lay_ct==2:
                if state_h_in == None:
                    h2,c2=self.conv_lstm2(x_enc[-1],None)
                else:
                    h2,c2=self.conv_lstm2(x_enc[-1],[state_h_in[enc_lay_ct-1],state_c_in[enc_lay_ct-1]])

                
                state_h.append(h2.detach())
                state_c.append(c2.detach())
                #print (' layer ',enc_lay_ct)
                #print (x_enc[-1].size())
                #print (h2.size())
                x_enc[-1]=h2
            if enc_lay_ct==3:
                if state_h_in == None:
                    h3,c3=self.conv_lstm3(x_enc[-1],None)
                else:
                    h3,c3=self.conv_lstm3(x_enc[-1],[state_h_in[enc_lay_ct-1],state_c_in[enc_lay_ct-1]])

                
                state_h.append(h3.detach())
                state_c.append(c3.detach())
                #print (' layer ',enc_lay_ct)
                #print (x_enc[-1].size())
                x_enc[-1]=h3
            if enc_lay_ct==4:
                #if state_h_in == None:
                #    h4,c4=self.conv_lstm4(x_enc[-1],None)
                #else:
                #    h4,c4=self.conv_lstm4(x_enc[-1],[state_h_in[enc_lay_ct-1],state_c_in[enc_lay_ct-1]])
                
                state_h.append(x_enc[-1])
                state_c.append(x_enc[-1])
                #print (' layer ',enc_lay_ct)
                #print (x_enc[-1].size())
                #x_enc[-1]=h4

            x_enc.append(layer(x_enc[-1]))

        #print (enc_lay_ct)
        # conv, upsample, concatenate series
        x = x_enc.pop()  #[1,32,16,16]
        #h=x
        # x last size is  torch.Size([1, 32, 12, 12, 3])
        #if state_h_in==None:
        #    h,c=self.conv_lstm(x,None)
        #else:
        #    h,c=self.conv_lstm(x,[state_h_in[4],state_c_in[4]])

        state_h.append(x.detach())
        state_c.append(x.detach())
        
        for layer in self.uparm: # [x channel size is 32]
            #print ('x size is up-sampling is ',x.size())
            
            x = layer(x)
            #print (x.size())
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            
            #print ('x size is extras is ',x.size())
            x = layer(x)
            #print (x.size())

        return x,state_h,state_c


class Unet_All_Encoder_3D_No_LSTM(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        input_img_channel=2
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += input_img_channel
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

        #CBCT Tumor    
        #self.conv_lstm=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,12,12,3), kernel_size=3,)  #[16,16,16]  [x,y,z]
        #self.conv_lstm4=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,24,24,6), kernel_size=3,)
        #self.conv_lstm3=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,48,48,12), kernel_size=3,)
        #self.conv_lstm2=ConvLSTM3D(input_channel=16, num_filter=16, b_h_w=(1,96,96,24), kernel_size=3,)
        #self.conv_lstm1=ConvLSTM3D(input_channel=2, num_filter=2, b_h_w=(1,192,192,48), kernel_size=3,)

        #self.dropout=torch.nn.Dropout(0.7)

    def forward(self, x, state_h_in,state_c_in):
        
        #print ('x in size is ',x.size())
        # get encoder activations
        x_enc = [x]

        # layer  1
        # torch.Size([1, 2, 192, 192, 48])
        # layer  2
        # torch.Size([1, 16, 96, 96, 24])
        # layer  3
        #torch.Size([1, 32, 48, 48, 12])
        # layer  4
        # torch.Size([1, 32, 24, 24, 6])
        # layer 5
        # torch.Size([1, 32, 12, 12, 3])


        enc_lay_ct=0

        state_h=[]
        state_c=[]
        #print ('*'*50)
        #if state_h_in == None:
        #    print ('info print state_h_in None ')
        #else:
        #    print ('state_h_in size is ',len(state_h_in))
        for layer in self.downarm:
            enc_lay_ct=enc_lay_ct+1
            
            #print (x_enc[-1].size())
            #print (len(x_enc))
#
            if enc_lay_ct==1:
                
                
                state_h.append(x_enc[-1])

            if enc_lay_ct==2:
               
                
                state_h.append(x_enc[-1])

            if enc_lay_ct==3:
               
                state_h.append(x_enc[-1])

            if enc_lay_ct==4:
               
                
                state_h.append(x_enc[-1])


            x_enc.append(layer(x_enc[-1]))

        #print (enc_lay_ct)
        # conv, upsample, concatenate series
        x = x_enc.pop()  #[1,32,16,16]
        #h=x
        # x last size is  torch.Size([1, 32, 12, 12, 3])


        state_h.append(x)
        state_c.append(x)
        #x=h
        for layer in self.uparm: # [x channel size is 32]
            #print ('x size is up-sampling is ',x.size())
            
            x = layer(x)
            #print (x.size())
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            
            #print ('x size is extras is ',x.size())
            x = layer(x)
            #print (x.size())

        return x,state_h,state_c

class ConvLSTM3D(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1, device='cuda'):
        super().__init__()
        self._conv = nn.Conv3d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self._batch_size, self._state_height, self._state_width,self._state_depth = b_h_w
        self.device = device
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width,self._state_depth)).to(self.device)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width,self._state_depth)).to(self.device)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width,self._state_depth)).to(self.device)
        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W  #squence,batch,channel,height,width
    def forward(self, inputs=None, states=None, seq_len=1):

        if states is None: # initiallize the hidden states of h and c
            c = torch.zeros((inputs.size(0), self._num_filter, self._state_height,
                                  self._state_width,self._state_depth), dtype=torch.float).to(self.device)
            h = torch.zeros((inputs.size(0), self._num_filter, self._state_height,
                             self._state_width,self._state_depth), dtype=torch.float).to(self.device)
        else:
            h, c = states

        #outputs = []
        for index in range(seq_len):
            
            x=inputs
            # print("X:",x.shape)
            # print("H:",h.shape)
            #print ('x size is ',x.size())
            #print ('h size is ',h.size())
            #print ('c size is ',c.size())
            cat_x = torch.cat([x, h], dim=1)
            #print (cat_x.size())
            conv_x = self._conv(cat_x)

            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)

            i = torch.sigmoid(i+self.Wci*c) # used information
            f = torch.sigmoid(f+self.Wcf*c) # forget information
            c = f*c + i*torch.tanh(tmp_c) # output control
            o = torch.sigmoid(o+self.Wco*c) # ouput information
            h = o*torch.tanh(c) # output hidden
            #outputs.append(h)
        return h,c   



from unet3d.buildingblocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv
from unet3d.buildingblock_lstm import Encoder_LSTM, Decoder_LSTM, FinalConv_LSTM, DoubleConv_LSTM, SingleConv_LSTM
from unet3d.utils import create_feature_maps
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='cr', num_groups=8,
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


class UNet3D_Seg_LSTM(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_LSTM, self).__init__()
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


        self.conv_lstm=ConvLSTM3D(input_channel=512, num_filter=512, b_h_w=(1,8,8,4), kernel_size=3,)  #[16,16,16]  [x,y,z]
        
                
    def forward(self, x,states):
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
        # print ('x size is ',x.size())
        if states==None:
            h,c=self.conv_lstm(x,None)
            #print ('after the fist h size ', h.size())
            #print ('after the fist c size ', c.size())
        else:
            h,c=self.conv_lstm(x,states)

        x=h

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

        return x,h,c        



class UNet3D_Seg_LSTM_Small(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_LSTM_Small, self).__init__()

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

        # for CBCT Tumor    
        #self.conv_lstm=ConvLSTM3D(input_channel=512, num_filter=512, b_h_w=(1,12,12,3), kernel_size=3,)  #[16,16,16]  [x,y,z]
        self.conv_lstm=ConvLSTM3D(input_channel=256, num_filter=256, b_h_w=(1,12,12,3), kernel_size=3,)  #[16,16,16]  [x,y,z]
        # for Eso
        self.conv_lstm=ConvLSTM3D(input_channel=512, num_filter=512, b_h_w=(1,10,10,5), kernel_size=3,)  #[16,16,16]  [x,y,z]
                
    def forward(self, x,states):
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
        #print ('x size is ',x.size())
        if states==None:
            h,c=self.conv_lstm(x,None)
            #print ('after the fist h size ', h.size())
            #print ('after the fist c size ', c.size())
        else:
            h,c=self.conv_lstm(x,states)

        x=h

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

        return x,h,c     

class UNet3D_Seg_All_Encoder_LSTM(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=16, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_All_Encoder_LSTM, self).__init__()

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

        # for CBCT Tumor    
        self.conv_lstm5=ConvLSTM3D(input_channel=256, num_filter=256, b_h_w=(1,12,12,3), kernel_size=3,)  #[16,16,16]  [x,y,z]
        self.conv_lstm4=ConvLSTM3D(input_channel=128, num_filter=128, b_h_w=(1,24,24,6), kernel_size=3,)
        self.conv_lstm3=ConvLSTM3D(input_channel=64, num_filter=64, b_h_w=(1,48,48,12), kernel_size=3,)
        self.conv_lstm2=ConvLSTM3D(input_channel=32, num_filter=32, b_h_w=(1,96,96,24), kernel_size=3,)
        self.conv_lstm1=ConvLSTM3D(input_channel=16, num_filter=16, b_h_w=(1,192,192,48), kernel_size=3,)

    def forward(self, x,state_h_in,state_c_in):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        import numpy as np
        #x=torch.tensor(np.ndarray(shape=(1,1,32,128,128), dtype=float, order='F'))
        #x=x.cuda().float()
        # encoder part
        encoders_features = []

        lstm_ct=0

        state_h=[]
        state_c=[]
        for encoder in self.encoders:
            lstm_ct=lstm_ct+1
            x = encoder(x)

            if lstm_ct==1:
                #print ('x1 size is ',x.size())
                if state_h_in == None:
                    h1,c1=self.conv_lstm1(x,None)
                else:
                    h1,c1=self.conv_lstm1(x,[state_h_in[lstm_ct-1],state_c_in[lstm_ct-1]])
                    
                state_h.append(h1.detach())
                state_c.append(c1.detach())

                x=h1
            if lstm_ct==2:
                if state_h_in == None:
                    h2,c2=self.conv_lstm2(x,None)
                else:
                    h2,c2=self.conv_lstm2(x,[state_h_in[lstm_ct-1],state_c_in[lstm_ct-1]])
                state_h.append(h2.detach())
                state_c.append(c2.detach())
                


                x=h2
            if lstm_ct==3:
                if state_h_in == None:
                    h3,c3=self.conv_lstm3(x,None)
                else:
                    h3,c3=self.conv_lstm3(x,[state_h_in[lstm_ct-1],state_c_in[lstm_ct-1]])
                state_h.append(h3.detach())
                state_c.append(c3.detach())


                x=h3
            if lstm_ct==4:
                if state_h_in == None:
                    h4,c4=self.conv_lstm4(x,None)
                else:
                    h4,c4=self.conv_lstm4(x,[state_h_in[lstm_ct-1],state_c_in[lstm_ct-1]])
                state_h.append(h4.detach())
                state_c.append(c4.detach())

                x=h4
            if lstm_ct==5:
                if state_h_in == None:
                    h5,c5=self.conv_lstm5(x,None)    
                else:
                    h5,c5=self.conv_lstm5(x,[state_h_in[lstm_ct-1],state_c_in[lstm_ct-1]])  
                state_h.append(h5.detach())
                state_c.append(c5.detach())


                x=h5          

            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        #print ('lstm number is ',lstm_ct)
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        #print ('x size is ',x.size())
        #if states==None:
        #    h,c=self.conv_lstm(x,None)
            #print ('after the fist h size ', h.size())
            #print ('after the fist c size ', c.size())
        #else:
        #    h,c=self.conv_lstm(x,states)

        #x=h

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

        return x,state_h,state_c

class UNet3D_Seg_LSTM_New(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_LSTM_New, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_LSTM(in_channels, out_feature_num//2, apply_pooling=False, basic_module=DoubleConv_LSTM,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_LSTM(f_maps[i - 1]//2, out_feature_num//2, basic_module=DoubleConv_LSTM,
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
            decoder = Decoder_LSTM(in_feature_num//2, out_feature_num//2, basic_module=DoubleConv_LSTM,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0]//2, out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

                
    def forward(self, x,states):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        import numpy as np
        #x=torch.tensor(np.ndarray(shape=(1,1,32,128,128), dtype=float, order='F'))
        #x=x.cuda().float()
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            #print (x.size())
            x = encoder(x,states)
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
            x = decoder(encoder_features, x,states)
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



class UNet3D_Seg_LSTM_New_use_State(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=48, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_LSTM_New_use_State, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_LSTM(in_channels, out_feature_num//2, apply_pooling=False, basic_module=DoubleConv_LSTM,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_LSTM(f_maps[i - 1]//2, out_feature_num//2, basic_module=DoubleConv_LSTM,
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
            decoder = Decoder_LSTM(in_feature_num//2, out_feature_num//2, basic_module=DoubleConv_LSTM,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0]//2, out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

                
    def forward(self, x,states_in):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        import numpy as np
        #x=torch.tensor(np.ndarray(shape=(1,1,32,128,128), dtype=float, order='F'))
        #x=x.cuda().float()
        # encoder part
        encoders_features = []
        states_list=[]
        
        state_count=0
        for encoder in self.encoders:
            #print (x.size())
            if states_in is None:
                x,en_state = encoder(x,None)
            else:
                test_st=states_in[state_count]
                #H_,c_=test_st[0]
                #print (H_.size())
                #print (c_.size())
                x,en_state = encoder(x,states_in[state_count])
            #print ('en_state is ',len(en_state[0]))
            #print ('en_state is ',len(en_state[0][0]))  #h
            #print ('en_state is ',len(en_state[0][1]))  #c
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            states_list.append(en_state)
            state_count=state_count+1

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        decoder_index=1
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            if states_in is None:
                x,dec_state = decoder(encoder_features, x,None)
            else:
                x,dec_state = decoder(encoder_features, x,states_in[state_count])

            decoder_index=decoder_index+1
            states_list.append(dec_state)
            state_count=state_count+1
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)
        #print (len(states_list))
        #print (len(states_list[0]))
        #print (states_list[0].size())
        #print (states_list[1].size())
        #print (states_list[2].size())
        #print (states_list[3].size())
        #print (states_list[4].size())
        #print (states_list[5].size())
        #print (states_list[6].size())
        return x,states_list



class UNet3D_Seg_LSTM_New_use_State_Only_Encoder(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=8, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_LSTM_New_use_State_Only_Encoder, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_LSTM(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv_LSTM,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_LSTM(f_maps[i - 1], out_feature_num, basic_module=DoubleConv_LSTM,
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

                
    def forward(self, x,states_in):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        import numpy as np
        #x=torch.tensor(np.ndarray(shape=(1,1,32,128,128), dtype=float, order='F'))
        #x=x.cuda().float()
        # encoder part
        encoders_features = []
        states_list=[]
        
        state_count=0
        for encoder in self.encoders:
            #print ('x size is ',x.size())
            x = torch.reshape(x, (1, 1, x.size(1),
                                        x.size(2), x.size(3), x.size(4)))

            if states_in is None:
                x,en_state = encoder(x,None)
            else:
                test_st=states_in[state_count]
                #H_,c_=test_st[0]
                #print (H_.size())
                #print (c_.size())
                x,en_state = encoder(x,states_in[state_count])
            #print ('en_state is ',len(en_state[0]))
            #print ('en_state is ',len(en_state[0][0]))  #h
            #print ('en_state is ',len(en_state[0][1]))  #c
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            states_list.append(en_state)
            state_count=state_count+1

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        decoder_index=1
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            
            x = decoder(encoder_features, x)

            decoder_index=decoder_index+1
            
            
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)
        #print (len(states_list))
        #print (len(states_list[0]))
        #print (states_list[0].size())
        #print (states_list[1].size())
        #print (states_list[2].size())
        #print (states_list[3].size())
        #print (states_list[4].size())
        #print (states_list[5].size())
        #print (states_list[6].size())
        return x,states_list



class UNet3D_Seg_LSTM_New_use_State_Only_Decoder(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_LSTM_New_use_State_Only_Decoder, self).__init__()

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
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv_LSTM,
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

                
    def forward(self, x,states_in):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        import numpy as np
        #x=torch.tensor(np.ndarray(shape=(1,1,32,128,128), dtype=float, order='F'))
        #x=x.cuda().float()
        # encoder part
        encoders_features = []
        states_list=[]
        
        state_count=0
        for encoder in self.encoders:
            #print (x.size())
            
            x= encoder(x)
            #print ('en_state is ',len(en_state[0]))
            #print ('en_state is ',len(en_state[0][0]))  #h
            #print ('en_state is ',len(en_state[0][1]))  #c
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            #states_list.append(en_state)
            #state_count=state_count+1

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        decoder_index=1
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            
            if states_in is None:
                x,en_state = decoder(x,None)
            else:
                test_st=states_in[state_count]
                #H_,c_=test_st[0]
                #print (H_.size())
                #print (c_.size())
                x,en_state = encoder(x,states_in[state_count])
            
            states_list.append(en_state)
            state_count=state_count+1

            decoder_index=decoder_index+1
            
            
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)
        #print (len(states_list))
        #print (len(states_list[0]))
        #print (states_list[0].size())
        #print (states_list[1].size())
        #print (states_list[2].size())
        #print (states_list[3].size())
        #print (states_list[4].size())
        #print (states_list[5].size())
        #print (states_list[6].size())
        return x,states_list

class UNet3D_Seg_Split_LSTM(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_Split_LSTM, self).__init__()

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

        encoders_ct = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoders_ct1 = Encoder(2, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoders_ct1 = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders_ct.append(encoders_ct1)

        self.encoders_ct = nn.ModuleList(encoders_ct)
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
        self.conv_lstm=ConvLSTM3D(input_channel=512, num_filter=512, b_h_w=(1,12,12,3), kernel_size=3,)  #[16,16,16]  [x,y,z]
        self.channel_adjust=nn.Conv3d(in_channels=1024,
                               out_channels=512,
                               kernel_size=1,
                               stride=1,
                               padding=0)            
    def forward(self, x,x_ct,states):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        
        encoders_features = []
        for encoder in self.encoders:
            
            x = encoder(x)
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        for encoder in self.encoders_ct:
            
            x_ct = encoder(x_ct)
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder

        'concat x and x_ct'

        x=torch.cat([x,x_ct],dim=1)    
        x=self.channel_adjust(x)
        #print ('x size is ',x.size())
        if states==None:
            h,c=self.conv_lstm(x,None)
            #print ('after the fist h size ', h.size())
            #print ('after the fist c size ', c.size())
        else:
            h,c=self.conv_lstm(x,states)

        x=h

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

        return x,h,c            


class UNet3D_Seg_Share_LSTM(nn.Module):
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

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_Seg_Share_LSTM, self).__init__()

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

        encoders_ct = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoders_ct1 = Encoder(2, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoders_ct1 = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders_ct.append(encoders_ct1)

        self.encoders_ct = nn.ModuleList(encoders_ct)
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
        self.conv_lstm=ConvLSTM3D(input_channel=512, num_filter=512, b_h_w=(1,12,12,3), kernel_size=3,)  #[16,16,16]  [x,y,z]
             
    def forward(self, x,x_ct,states):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        
        encoders_features = []
        for encoder in self.encoders:
            
            x = encoder(x)
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        #print (x.size())
        encoders_features = encoders_features[1:]

        encoders_features_ct = []
        for encoder in self.encoders_ct:
            
            x_ct = encoder(x_ct)
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features_ct.insert(0, x_ct)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        
        encoders_features_ct = encoders_features_ct[1:]

        if states==None:
            h,c=self.conv_lstm(x_ct,None)
            #print ('after the fist h size ', h.size())
            #print ('after the fist c size ', c.size())
        else:
            h,c=self.conv_lstm(x_ct,states)

        x_ct=h
        #print (x.size())
        #print (x_ct.size())
        # decoder part
        decoder_index=1
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            #print ('x size are:',x.size())
            #print ('encoder_features size are:',encoder_features.size())            
            x = decoder(encoder_features, x)
            
            
            decoder_index=decoder_index+1
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        # decoder part for CT
        decoder_index=1
        for decoder, encoder_features in zip(self.decoders, encoders_features_ct):
            
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x_ct = decoder(encoder_features, x_ct)
            #if decoder_index==3:
            #    x=self.Block_SA1(x)
            #    x=self.Block_SA2(x)
            #print ('decoders size are:',x.size())
            #print ('encoder_features size are:',encoder_features.size())
            decoder_index=decoder_index+1
        x_ct = self.final_conv(x_ct)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x_ct = self.final_activation(x_ct)        

        return x,x_ct,h,c                   


    def forward_CT(self, x_ct,states):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        

        encoders_features_ct = []
        for encoder in self.encoders_ct:
            
            x_ct = encoder(x_ct)
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features_ct.insert(0, x_ct)

        encoders_features_ct = encoders_features_ct[1:]

        if states==None:
            h,c=self.conv_lstm(x_ct,None)
            #print ('after the fist h size ', h.size())
            #print ('after the fist c size ', c.size())
        else:
            h,c=self.conv_lstm(x_ct,states)

        x_ct=h
        
        decoder_index=1
        for decoder, encoder_features in zip(self.decoders, encoders_features_ct):
            
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x_ct = decoder(encoder_features, x_ct)
            #if decoder_index==3:
            #    x=self.Block_SA1(x)
            #    x=self.Block_SA2(x)
            #print ('decoders size are:',x.size())
            #print ('encoder_features size are:',encoder_features.size())
            decoder_index=decoder_index+1
        x_ct = self.final_conv(x_ct)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x_ct = self.final_activation(x_ct)        

        return x_ct,h,c  

    def forward_CBCT(self, x,states):
        #print ('the input image size is:',x.size())
        #(1,32,64,128,128)
        
        

        encoders_features = []
        for encoder in self.encoders:
            
            x = encoder(x)
            #print ('encoders size are:',x.size())
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        if states==None:
            h,c=self.conv_lstm(x,None)
            #print ('after the fist h size ', h.size())
            #print ('after the fist c size ', c.size())
        else:
            h,c=self.conv_lstm(x,states)

        x=h
        
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

        return x,h,c  

import functools
class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator3D, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        norm_layer=nn.InstanceNorm3d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return self.model(input)        