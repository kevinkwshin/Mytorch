import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import monai
import segmentation_models_pytorch as smp

# def init_weights(net, init_type='normal', gain=0.02):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)


import torch.nn as nn

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import deprecated_arg, ensure_tuple_rep

import torch
from torch import nn
from torch.nn import functional as F

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


# if __name__ == '__main__':
#     import torch

#     for bn_layer in [True, False]:
#         img = torch.zeros(2, 3, 20)
#         net = NLBlockND(in_channels=3, mode='concatenate', dimension=1, bn_layer=bn_layer)
#         out = net(img)
#         print(out.size())
        
class TwoConv(nn.Sequential):
    """two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()

        if dim is not None:
            spatial_dims = dim
        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    @deprecated_arg(name="dim", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead.")
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        dim: Optional[int] = None,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.

        .. deprecated:: 0.6.0
            ``dim`` is deprecated, use ``spatial_dims`` instead.
        """
        super().__init__()
        if dim is not None:
            spatial_dims = dim
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x
    
class monai_unet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        net_inputch: int = 1,
        net_outputch: int = 2,
        net_bayesian = 0,
        # features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        features: Sequence[int] = (32, 32, 64, 128, 256, 512, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("group", {"num_groups": 8}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
        bottleneck_channels = None,
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        .. deprecated:: 0.6.0
            ``dimensions`` is deprecated, use ``spatial_dims`` instead.

        Examples::

            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        fea = ensure_tuple_rep(features, 7)
        print(f"BasicUNet features: {fea}.")
        in_channels = net_inputch
        out_channels = net_outputch
        
        self.NLBlock_1 = NLBlockND(in_channels=fea[1], mode='dot', dimension=2)
        self.NLBlock_2 = NLBlockND(in_channels=fea[2], mode='dot', dimension=2)
        self.NLBlock_3 = NLBlockND(in_channels=fea[3], mode='dot', dimension=2)
        self.NLBlock_4 = NLBlockND(in_channels=fea[4], mode='dot', dimension=2)
        self.NLBlock_5 = NLBlockND(in_channels=fea[5], mode='dot', dimension=2)
        
        self.skipAtt = attention_block(F_g=fea[4],F_l=fea[4],F_int=fea[4])
        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.down_5 = Down(spatial_dims, fea[4], fea[5], act, norm, bias, dropout)

        self.upcat_5 = UpCat(spatial_dims, fea[5], fea[4], fea[4], act, norm, bias, dropout, upsample)
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        # self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[6], act, norm, bias, dropout, upsample, halves=False)

        # self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        self.final_conv = Conv["conv", spatial_dims](fea[6], out_channels, kernel_size=1)
        self.bottleneck_channels = bottleneck_channels 
        
        if self.bottleneck_channels is not None:
            pool = nn.AdaptiveAvgPool1d(1)
            flatten = nn.Flatten()
            dropout = nn.Dropout(p=.2, inplace=True) if dropout else nn.Identity()
            linear = nn.Linear(fea[5], regression_channels, bias=True)
            activation = nn.Sigmoid() # nn.ReLU()
            self.bottleneck_head = nn.Sequential(pool,flatten,dropout,linear,activation)
    
        if net_bayesian!=0:
            self.MCDropout = MCDropout(p=net_bayesian)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        # x1 = self.NLBlock_1(x1)
        
        x2 = self.down_2(x1)
        x2 = self.NLBlock_2(x2)
        
        x2 = self.MCDropout(x2)
        x3 = self.down_3(x2)
        x3 = self.NLBlock_3(x3)
        
        x3 = self.MCDropout(x3)
        x4 = self.down_4(x3)
        x4 = self.NLBlock_4(x4)
        
        x4 = self.MCDropout(x4)
        x5 = self.down_5(x4)
        x5 = self.NLBlock_5(x5)
        
        u5 = self.upcat_5(x5, x4)
        u4 = self.upcat_4(u5, x3)
        u4 = self.NLBlock_3(u4)
        
        u3 = self.upcat_3(u4, x2)
        u3 = self.NLBlock_2(u3)
        
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        x = self.final_conv(u1)
        # x = F.sigmoid(x)
        
        if self.bottleneck_channels is None:
            return x
        else:
            y = self.bottleneck_head(x4)            
            return x, y

class attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(attention_block,self).__init__()
        inplace= False

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=inplace)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi  
    

class MCDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


class monai_unetr(nn.Module):
    def __init__(self, net_inputch=3, net_outputch=2, net_norm='batch', net_bayesian=0):
        super(monai_unetr, self).__init__()        

        self.net = monai.networks.nets.UNETR(
        in_channels=net_inputch,
        out_channels=net_outputch,
        spatial_dims=2,
        img_size=128,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0)
        
    def forward(self,x):
        return self.net(x)    


def bn2instance(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.InstanceNorm2d(module.num_features,
                                                module.eps, module.momentum,
                                                module.affine,
                                                module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn2instance(child))

    del module
    return module_output

def bn2group(module):
#     num_groups = 16 # hyper_parameter of GroupNorm
    num_groups = 8 # hyper_parameter of GroupNorm
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.num_features/num_groups <1:
            module_output = torch.nn.GroupNorm(1,
                                           module.num_features,
                                           module.eps, 
                                           module.affine,
                                          )
        else:
            module_output = torch.nn.GroupNorm(num_groups,
                               module.num_features,
                               module.eps, 
                               module.affine,
                                          )

        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn2group(child))

    del module
    return module_output

def relu2lrelu(module):
    """
    relu2lrelu(net)
    """
    module_output = module
    if isinstance(module, torch.nn.modules.ReLU):
        module_output = torch.nn.LeakyReLU(0.2)
        
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, relu2lrelu(child))

    del module
    return module_output

def relu2gelu(module):
    """
    relu2gelu(net)
    """
    module_output = module
    if isinstance(module, torch.nn.modules.ReLU):
        module_output = torch.nn.GELU()
        
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, relu2gelu(child))

    del module
    return module_output

# def pooling2wavelet(module):
#     module_output = module
#     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#         module_output = torch.nn.InstanceNorm2d(module.num_features,
#                                                 module.eps, module.momentum,
#                                                 module.affine,
#                                                 module.track_running_stats)
#         if module.affine:
#             with torch.no_grad():
#                 module_output.weight = module.weight
#                 module_output.bias = module.bias
#         module_output.running_mean = module.running_mean
#         module_output.running_var = module.running_var
#         module_output.num_batches_tracked = module.num_batches_tracked
#         if hasattr(module, "qconfig"):
#             module_output.qconfig = module.qconfig

#     for name, child in module.named_children():
#         module_output.add_module(name, bn2instance(child))

#     del module
#     return module_output

# def conv2ws(module):
#     module_output = module
#     if isinstance(module, torch.nn.modules.BatchNorm):
#         module_output = torch.nn.InstanceNorm2d(module.num_features,
#                                                 module.eps, module.momentum,
#                                                 module.affine,
#                                                 module.track_running_stats)
#         if module.affine:
#             with torch.no_grad():
#                 module_output.weight = module.weight
#                 module_output.bias = module.bias
#         module_output.running_mean = module.running_mean
#         module_output.running_var = module.running_var
#         module_output.num_batches_tracked = module.num_batches_tracked
#         if hasattr(module, "qconfig"):
#             module_output.qconfig = module.qconfig

#     for name, child in module.named_children():
#         module_output.add_module(name, bn2instance(child))

#     del module
#     return module_output

# class smp_deeplabplus(nn.Module):
#     def __init__(self, encoder_name='resnet50', net_inputch=3, net_outputch=2, net_norm='batch', bayesian=0):
#         super(smp_deeplabplus, self).__init__()        
#         self.net = smp.DeepLabV3Plus(encoder_name = encoder_name,
#                                     encoder_depth = 5,
#                                     encoder_weights = 'imagenet',
#                                     encoder_output_stride = 16,
#                                     decoder_channels = 256,
#                                     decoder_atrous_rates = (12, 24, 36),
#                                     in_channels = net_inputch,
#                                     classes= net_outputch,
#                                     activation = None,
#                                     upsampling = 4,
#                                     aux_params = None)

#         if net_norm=='group':
#             self.net = bn2group(self.net)
#         elif net_norm == 'instance':
#             self.net = bn2instance(self.net)
#         if bayesian != 0:
#             self.MCDropout = MCDropout(bayesian)

#             if 'resne' in encoder_name or 'densenet' in encoder_name:
#                 list(self.net.encoder.children())[-1][-1] = nn.Sequential(self.MCDropout,list(self.net.encoder.children())[-1][-1]) # bottleneck
#                 print(list(self.net.encoder.children())[-1][-1])
#             elif 'efficientnet' in encoder_name:
#                 self.net.encoder.bn2 =  nn.Sequential(self.MCDropout, self.net.encoder.bn2) # efficientnet
#                 print(self.net.encoder.bn2)
#             elif 'regnet' in encoder_name:                
#                 self.net.encoder.s4 = nn.Sequential(self.MCDropout,self.net.encoder.s4) # regnet
# #                 self.net.encoder.s4.b1 = nn.Sequential(self.MCDropout,self.net.encoder.s4.b1) # regnety
#                 print(self.net.encoder.s4)
                
#     def forward(self,x):
#         return self.net(x)

# class smp_manet(nn.Module):
#     def __init__(self, net_encoder_name='resnet50', net_inputch=3, net_outputch=2, net_norm='batch', net_bayesian=0):
#         super(smp_manet, self).__init__()        
#         self.net = smp.MAnet(
#                         encoder_name=net_encoder_name,
#                         encoder_depth=5,
#                         encoder_weights='imagenet',
#                         decoder_use_batchnorm=True,
#                         decoder_channels=(256, 128, 64, 32, 16),
#                         decoder_pab_channels=64,
#                         in_channels=net_inputch, 
#                         classes=net_outputch,
#                         activation=None,
#                         aux_params=None)
        
#         if net_norm=='group':
#             self.net = bn2group(self.net)
#         elif net_norm == 'instance':
#             self.net = bn2instance(self.net)
# #         if nnblock:
# #             self.nnblock = self.conv1.out_channels
#         if net_bayesian!=0:
#             self.MCDropout = MCDropout(bayesian)

#             if 'resne' in encoder_name or 'densenet' in net_encoder_name:
#                 list(self.net.encoder.children())[-1][-1] = nn.Sequential(self.MCDropout,list(self.net.encoder.children())[-1][-1]) # bottleneck
#                 print(list(self.net.encoder.children())[-1][-1])
#             elif 'efficientnet' in encoder_name:
#                 self.net.encoder.bn2 =  nn.Sequential(self.MCDropout, self.net.encoder.bn2) # efficientnet
#                 print(self.net.encoder.bn2)
#             elif 'regnet' in encoder_name:                
#                 self.net.encoder.s4 = nn.Sequential(self.MCDropout,self.net.encoder.s4) # regnet
# #                 self.net.encoder.s4.b1 = nn.Sequential(self.MCDropout,self.net.encoder.s4.b1) # regnety
#                 print(self.net.encoder.s4)                
        
#     def forward(self,x):
#         return self.net(x)

class smp_unet(nn.Module):
    def __init__(self, net_encoder_name='resnet50', net_inputch=3, net_outputch=2, net_norm='batch', net_bayesian=0):
        super(smp_unet, self).__init__()        
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch
        
        self.net = smp.Unet(
                        encoder_name=net_encoder_name,
                        decoder_use_batchnorm = True,
                        decoder_attention_type ='scse',
                        encoder_weights='imagenet',
                        encoder_depth=5,
                        in_channels=self.net_inputch, 
                        classes=self.net_outputch)
        
        if net_norm=='group':
            self.net = bn2group(self.net)
        elif net_norm == 'instance':
            self.net = bn2instance(self.net)
#         if nnblock:
#             self.conv1.out_channels
#             self.nnblock = NONLocalBlock2D()        
        if net_bayesian!=0:
            self.MCDropout = MCDropout(p=net_bayesian)
            if 'resne' in net_encoder_name or 'densenet' in net_encoder_name:
                # self.net.encoder.layer1 = nn.Sequential(self.MCDropout, self.net.encoder.layer1)
                # self.net.encoder.layer2 = nn.Sequential(self.MCDropout, self.net.encoder.layer2)
                # self.net.encoder.layer3 = nn.Sequential(self.MCDropout, self.net.encoder.layer3)
                self.net.encoder.layer4 = nn.Sequential(self.MCDropout, self.net.encoder.layer4)
            elif 'efficientnet' in net_encoder_name:
                self.net.encoder.bn2 =  nn.Sequential(self.MCDropout, self.net.encoder.bn2) # efficientnet
            elif 'regnet' in net_encoder_name:                                
#                 self.net.encoder.s1 = nn.Sequential(self.MCDropout,self.net.encoder.s1) # regnet
#                 self.net.encoder.s2 = nn.Sequential(self.MCDropout,self.net.encoder.s2) # regnet
#                 self.net.encoder.s3 = nn.Sequential(self.MCDropout,self.net.encoder.s3) # regnet
                self.net.encoder.s4 = nn.Sequential(self.MCDropout,self.net.encoder.s4) # regnet
                
    def forward(self,x):
        return self.net(x)    
    
class smp_unetplusplus(nn.Module):
    def __init__(self, net_encoder_name='resnet50', net_inputch=3, net_outputch=2, net_norm='batch', net_bayesian=0):
        super(smp_unetplusplus, self).__init__()        
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch
        
        self.net = smp.UnetPlusPlus(
                        encoder_name=net_encoder_name,
                        decoder_use_batchnorm = True,
                        decoder_attention_type ='scse',
                        encoder_weights='imagenet',
                        encoder_depth=5,
                        in_channels=self.net_inputch, 
                        classes=self.net_outputch)
        
        if net_norm=='group':
            self.net = bn2group(self.net)
        elif net_norm == 'instance':
            self.net = bn2instance(self.net)
#         if nnblock:
#             self.conv1.out_channels
#             self.nnblock = NONLocalBlock2D()        
        if net_bayesian!=0:
            self.MCDropout = MCDropout(p=net_bayesian)
            if 'resne' in net_encoder_name or 'densenet' in net_encoder_name:
                # self.net.encoder.layer1 = nn.Sequential(self.MCDropout, self.net.encoder.layer1)
                # self.net.encoder.layer2 = nn.Sequential(self.MCDropout, self.net.encoder.layer2)
                # self.net.encoder.layer3 = nn.Sequential(self.MCDropout, self.net.encoder.layer3)
                self.net.encoder.layer4 = nn.Sequential(self.MCDropout, self.net.encoder.layer4)
            elif 'efficientnet' in net_encoder_name:
                self.net.encoder.bn2 =  nn.Sequential(self.MCDropout, self.net.encoder.bn2) # efficientnet
            elif 'regnet' in net_encoder_name:                                
#                 self.net.encoder.s1 = nn.Sequential(self.MCDropout,self.net.encoder.s1) # regnet
#                 self.net.encoder.s2 = nn.Sequential(self.MCDropout,self.net.encoder.s2) # regnet
#                 self.net.encoder.s3 = nn.Sequential(self.MCDropout,self.net.encoder.s3) # regnet
                self.net.encoder.s4 = nn.Sequential(self.MCDropout,self.net.encoder.s4) # regnet
                
    def forward(self,x):
        return self.net(x)    
    
    
class smp_FPN(nn.Module):
    def __init__(self, net_encoder_name='resnet50', net_inputch=3, net_outputch=2, net_norm='batch', bayesian=0):
        super(smp_FPN, self).__init__()        
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch  
        self.net = smp.FPN(
                        encoder_name=net_encoder_name,
                        encoder_weights='imagenet',
                        decoder_merge_policy= 'cat',
                        encoder_depth=5,
                        in_channels=self.net_inputch, 
                        classes=self.net_outputch,)
        if net_norm=='group':
            self.net = bn2group(self.net)
        elif net_norm == 'instance':
            self.net = bn2instance(self.net)
            
        if bayesian!=0:
            self.MCDropout = MCDropout(p=bayesian)
            if 'resne' in net_encoder_name or 'densenet' in net_encoder_name:
                # self.net.encoder.layer1 = nn.Sequential(self.MCDropout, self.net.encoder.layer1)
                # self.net.encoder.layer2 = nn.Sequential(self.MCDropout, self.net.encoder.layer2)
                # self.net.encoder.layer3 = nn.Sequential(self.MCDropout, self.net.encoder.layer3)
                self.net.encoder.layer4 = nn.Sequential(self.MCDropout, self.net.encoder.layer4)
            elif 'efficientnet' in net_encoder_name:
                self.net.encoder.bn2 =  nn.Sequential(self.MCDropout, self.net.encoder.bn2) # efficientnet
            elif 'regnet' in net_encoder_name:                                
#                 self.net.encoder.s1 = nn.Sequential(self.MCDropout,self.net.encoder.s1) # regnet
#                 self.net.encoder.s2 = nn.Sequential(self.MCDropout,self.net.encoder.s2) # regnet
#                 self.net.encoder.s3 = nn.Sequential(self.MCDropout,self.net.encoder.s3) # regnet
                self.net.encoder.s4 = nn.Sequential(self.MCDropout,self.net.encoder.s4) # regnet
                
    def forward(self,x):
        return self.net(x)    

import torch
import torch.nn as nn
from typing import Optional, Union, List

class smp_unetRecSoft(nn.Module):
    def __init__(self,
        net_encoder_name = 'resnet50',
        encoder_depth = 5,
        encoder_weights  = "imagenet",
        decoder_use_batchnorm = True,
        decoder_channels = (256, 128, 64, 32),
        decoder_attention_type = 'scse',
        net_inputch=3, 
        net_outputch=2,         
        activation = None,
        mode = 'soft',
        upsample_type = 'upsample',
        net_norm='batch', net_bayesian=0,
        **kwargs):
        super(smp_unetRecSoft, self).__init__()
        
        self.mode = mode
        self.base_net = smp.Unet(encoder_name=net_encoder_name, in_channels=net_inputch, classes=net_outputch, decoder_attention_type=decoder_attention_type)
        if self.mode == 'soft':
            self.recon_decoder = AE_Decoder(
                encoder_channels=self.base_net.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                upsample_type = upsample_type,
                use_batchnorm=decoder_use_batchnorm,
                center=True if net_encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type
            )
            self.recon_head = ReconstructionHead(
                in_channels=decoder_channels[-1]//2,
                out_channels=net_inputch,
                activation=activation,
                kernel_size=3
            )
        elif self.mode == 'hard':
            self.recon_head = ReconstructionHead(
                in_channels=decoder_channels[-1]//2,
                out_channels=net_inputch,
                activation=activation,
                kernel_size=3
            )
            
        if net_norm=='group':
            self.base_net = bn2group(self.base_net)
        elif net_norm == 'instance':
            self.base_net = bn2instance(self.base_net)
        if net_bayesian!=0:
            self.MCDropout = MCDropout(p=net_bayesian)
            if 'resne' in net_encoder_name or 'densenet' in net_encoder_name:
                # self.net.encoder.layer1 = nn.Sequential(self.MCDropout, self.net.encoder.layer1)
                # self.net.encoder.layer2 = nn.Sequential(self.MCDropout, self.net.encoder.layer2)
                # self.net.encoder.layer3 = nn.Sequential(self.MCDropout, self.net.encoder.layer3)
                self.base_net.encoder.layer4 = nn.Sequential(self.MCDropout, self.base_net.encoder.layer4)
            elif 'efficientnet' in net_encoder_name:
                self.base_net.encoder.bn2 =  nn.Sequential(self.MCDropout, self.base_net.encoder.bn2) # efficientnet
            elif 'regnet' in net_encoder_name:                                
#                 self.net.encoder.s1 = nn.Sequential(self.MCDropout,self.net.encoder.s1) # regnet
#                 self.net.encoder.s2 = nn.Sequential(self.MCDropout,self.net.encoder.s2) # regnet
#                 self.net.encoder.s3 = nn.Sequential(self.MCDropout,self.net.encoder.s3) # regnet
                self.base_net.encoder.s4 = nn.Sequential(self.MCDropout,self.base_net.encoder.s4) # regnet
                
    def forward(self,x):
        yhat = self.base_net(x)
        bottleneck = self.base_net.encoder(x)
        if self.mode=='soft':
            xhat = self.recon_decoder(bottleneck[-1])
            xhat = self.recon_head(xhat)
        elif self.mode == 'hard':
            xhat = self.base_net.decoder(*bottleneck) 
            xhat = self.recon_head(xhat)
        return yhat, xhat
    
class smp_unetRecHard(nn.Module):
    def __init__(self,
        encoder_name = 'resnet50',
        encoder_depth = 5,
        encoder_weights  = "imagenet",
        decoder_use_batchnorm = True,
        decoder_channels = (256, 128, 64, 32),
        decoder_attention_type = 'scse',
        net_inputch=3, 
        net_outputch=2,         
        activation = None,
        mode = 'hard',
        upsample_type = 'upsample',
        net_norm='batch', bayesian=0,
        **kwargs):
        super(smp_unetRecHard, self).__init__()
        
        self.mode = mode
        self.base_net = smp.Unet(encoder_name=encoder_name, in_channels=net_inputch, classes=net_outputch, decoder_attention_type=decoder_attention_type)
        if self.mode == 'soft':
            self.recon_decoder = AE_Decoder(
                encoder_channels=self.base_net.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                upsample_type = upsample_type,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type
            )
            self.recon_head = ReconstructionHead(
                in_channels=decoder_channels[-1]//2,
                out_channels=net_inputch,
                activation=activation,
                kernel_size=3
            )
        elif self.mode == 'hard':
            self.recon_head = ReconstructionHead(
                in_channels=decoder_channels[-1]//2,
                out_channels=net_inputch,
                activation=activation,
                kernel_size=3
            )
            
        if net_norm=='group':
            self.net = bn2group(self.net)
        elif net_norm == 'instance':
            self.net = bn2instance(self.net)
#         if nnblock:
#             self.nnblock = self.conv1.out_channels
        if bayesian!=0:
            self.MCDropout = MCDropout(p=bayesian)
            if 'resne' in encoder_name or 'densenet' in encoder_name:
                # self.net.encoder.layer1 = nn.Sequential(self.MCDropout, self.net.encoder.layer1)
                # self.net.encoder.layer2 = nn.Sequential(self.MCDropout, self.net.encoder.layer2)
                # self.net.encoder.layer3 = nn.Sequential(self.MCDropout, self.net.encoder.layer3)
                self.net.encoder.layer4 = nn.Sequential(self.MCDropout, self.net.encoder.layer4)
            elif 'efficientnet' in encoder_name:
                self.net.encoder.bn2 =  nn.Sequential(self.MCDropout, self.net.encoder.bn2) # efficientnet
            elif 'regnet' in encoder_name:                                
#                 self.net.encoder.s1 = nn.Sequential(self.MCDropout,self.net.encoder.s1) # regnet
#                 self.net.encoder.s2 = nn.Sequential(self.MCDropout,self.net.encoder.s2) # regnet
#                 self.net.encoder.s3 = nn.Sequential(self.MCDropout,self.net.encoder.s3) # regnet
                self.net.encoder.s4 = nn.Sequential(self.MCDropout,self.net.encoder.s4) # regnet
            
    def forward(self,x):
        yhat = self.base_net(x)
        bottleneck = self.base_net.encoder(x)
        if self.mode=='soft':
            xhat = self.recon_decoder(bottleneck[-1])
            xhat = self.recon_head(xhat)
        elif self.mode == 'hard':
            xhat = self.base_net.decoder(*bottleneck) 
            xhat = self.recon_head(xhat)
        return yhat, xhat
    
class smp_FPNRecSoft(nn.Module):
    def __init__(self,
        encoder_name = 'resnet50',
        encoder_depth= 5,
        encoder_weights= 'imagenet',
        decoder_pyramid_channels = 256,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32),
        decoder_attention_type: Optional[str] = 'scse',
        decoder_segmentation_channels = 128,
        decoder_merge_policy: str = 'cat',
        decoder_dropout: float = 0.2,
        net_inputch = 3,
        net_outputch = 2,
        activation = None,
        upsampling = 4,
        mode = 'soft',
        upsample_type = 'upsample',
        **kwargs):
        super(smp_FPNRecSoft, self).__init__()
        
        self.mode = mode
        self.base_net = smp.FPN(encoder_name=encoder_name, in_channels=net_inputch, classes=net_outputch, decoder_merge_policy=decoder_merge_policy)
        if self.mode == 'soft':
            self.recon_decoder = AE_Decoder(
                encoder_channels=self.base_net.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                upsample_type = upsample_type,
                use_batchnorm=decoder_use_batchnorm,
                center=True if encoder_name.startswith("vgg") else False,
                attention_type=decoder_attention_type
            )
            self.recon_head = ReconstructionHead(
                in_channels=decoder_channels[-1]//2,
                out_channels=net_inputch,
                activation=activation,
                kernel_size=3
            )
        elif self.mode == 'hard':
            self.recon_head = ReconstructionHead(
                in_channels=decoder_channels[-1]//2,
                out_channels=net_inputch,
                activation=activation,
                kernel_size=3
            )
                
    def forward(self,x):
        yhat = self.base_net(x)
        bottleneck = self.base_net.encoder(x)
        if self.mode=='soft':
            xhat = self.recon_decoder(bottleneck[-1])
            xhat = self.recon_head(xhat)
        elif self.mode == 'hard':
            xhat = self.base_net.decoder(*bottleneck) 
            xhat = self.recon_head(xhat)
        return yhat, xhat
    
import pywt
import torch
from torch.autograd import Variable

w=pywt.Wavelet('db1')
# w=pywt.Wavelet('haar')
# w=pywt.Wavelet('rbio1.1')
dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)
inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)
inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

def wt(vimg):
    padded = vimg
    res = torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))
    res = res.cuda()
    for i in range(padded.shape[1]):
        res[:,4*i:4*i+4] = torch.nn.functional.conv2d(padded[:,i:i+1], Variable(filters[:,None].cuda(),requires_grad=True),stride=2)
    return res

def iwt(vres):
    res = torch.zeros(vres.shape[0],int(vres.shape[1]/4),int(vres.shape[2]*2),int(vres.shape[3]*2))
    res = res.cuda()
    for i in range(res.shape[1]):
        temp = torch.nn.functional.conv_transpose2d(vres[:,4*i:4*i+4], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)
        res[:,i:i+1,:,:] = temp
    return res

class unet(nn.Module):
    def __init__(self, net_inputch=3, net_outputch=2, net_wavelet=False, net_skipatt=False, net_rcnn=False, net_nnblock=False, net_supervision=False, net_bayesian = 0):
        super(unet,self).__init__()
        num_c=32
        
        self.attention = net_skipatt
        self.rcnn = net_rcnn
        self.nnblock = net_nnblock
        self.supervision = net_supervision
        self.wavelet = net_wavelet
        self.net_bayesian = net_bayesian
        
        if net_bayesian!=0:
            self.MCDropout = MCDropout(p=net_bayesian)
    
        if self.rcnn == False:
            self.Conv1 = conv_block(ch_in=net_inputch, ch_out=num_c)
            self.Conv2 = conv_block(ch_in=num_c*4,ch_out=num_c*2)
            self.Conv3 = conv_block(ch_in=num_c*8,ch_out=num_c*4)
            self.Conv4 = conv_block(ch_in=num_c*16,ch_out=num_c*8)
            self.Conv5 = conv_block(ch_in=num_c*32,ch_out=num_c*32)

            self.Up_conv5 = conv_block(ch_in=num_c*16, ch_out=num_c*16)
            self.Up_conv4 = conv_block(ch_in=num_c*8, ch_out=num_c*8)
            self.Up_conv3 = conv_block(ch_in=num_c*4, ch_out=num_c*4)
            self.Up_conv2 = conv_block(ch_in=num_c*2,ch_out=num_c)
        else:
            t = 2
            self.RRCNN1 = RRCNN_block(ch_in=net_inputch,ch_out=num_c,t=t)
            self.RRCNN2 = RRCNN_block(ch_in=num_c*4,ch_out=num_c*2,t=t)
            self.RRCNN3 = RRCNN_block(ch_in=num_c*8,ch_out=num_c*4,t=t) 
            self.RRCNN4 = RRCNN_block(ch_in=num_c*16,ch_out=num_c*8,t=t)      
            self.RRCNN5 = RRCNN_block(ch_in=num_c*32,ch_out=num_c*32,t=t)

            self.Up_RRCNN5 = RRCNN_block(ch_in=num_c*16, ch_out=num_c*16,t=t)
            self.Up_RRCNN4 = RRCNN_block(ch_in=num_c*8, ch_out=num_c*8,t=t)
            self.Up_RRCNN3 = RRCNN_block(ch_in=num_c*4, ch_out=num_c*4,t=t)
            self.Up_RRCNN2 = RRCNN_block(ch_in=num_c*2,ch_out=num_c,t=t)
            
        if self.attention:
            self.Att5 = attention_block(F_g=num_c*8,F_l=num_c*8,F_int=num_c*8)
            self.Att4 = attention_block(F_g=num_c*4,F_l=num_c*4,F_int=num_c*4)
            self.Att3 = attention_block(F_g=num_c*2,F_l=num_c*2,F_int=num_c*2)
            self.Att2 = attention_block(F_g=num_c,F_l=num_c,F_int=num_c)

        if self.wavelet == False:
            self.Maxpool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Conv2d(num_c,num_c*4,1))
            self.Maxpool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Conv2d(num_c*2,num_c*8,1))
            self.Maxpool4 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Conv2d(num_c*4,num_c*16,1))
            self.Maxpool5 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),nn.Conv2d(num_c*8,num_c*32,1))
            
            self.Up5 = up_conv(ch_in=num_c*32, ch_out=num_c*8)
            self.Up4 = up_conv(ch_in=num_c*16, ch_out=num_c*4)
            self.Up3 = up_conv(ch_in=num_c*8, ch_out=num_c*2)
            self.Up2 = up_conv(ch_in=num_c*4, ch_out=num_c*1)

        if self.nnblock:        
            # self.nnblock2 = NONLocalBlock2D(num_c*2)
            # self.nnblock4 = NONLocalBlock2D(num_c*4)
            self.nnblock8 = NONLocalBlock2D(num_c*8)
            self.nnblock16 = NONLocalBlock2D(num_c*16)
            self.nnblock32 = NONLocalBlock2D(num_c*32)
            
        if self.supervision:
            self.Conv_final = nn.Sequential(
                    nn.Conv2d(int(num_c+num_c+num_c/2+num_c/4), num_c, kernel_size=3,stride=1,padding=1,bias=True), # all layers
                    nn.BatchNorm2d(num_c),
                    nn.ReLU(),
                    nn.Conv2d(num_c, num_c, kernel_size=3,stride=1,padding=1,bias=True),
                    nn.BatchNorm2d(num_c),
                    nn.ReLU(),
                    nn.Conv2d(num_c, net_outputch, kernel_size=1,stride=1,padding=0,bias=True),
            )
        else:
            self.Conv_final = nn.Sequential(
                    nn.Conv2d(num_c, num_c, kernel_size=3,stride=1,padding=1,bias=True),
                    nn.BatchNorm2d(num_c),
                    nn.ReLU(),
                    nn.Conv2d(num_c, num_c, kernel_size=3,stride=1,padding=1,bias=True),
                    nn.BatchNorm2d(num_c),
                    nn.ReLU(),
                    nn.Conv2d(num_c, net_outputch, kernel_size=1,stride=1,padding=0,bias=True),
            )
               
                
    def forward(self,x):
#         print('x',x.shape)

        # encoding path
        x1 = self.RRCNN1(x) if self.rcnn else self.Conv1(x)
#         print('x1',x1.shape)

        x2 = wt(x1) if self.wavelet else self.Maxpool2(x1)
#         x2 = self.nnblock4(x2) if self.nnblock else x2
        x2 = self.RRCNN2(x2) if self.rcnn else self.Conv2(x2)
#         print('x2',x2.shape)

        x3 = wt(x2) if self.wavelet else self.Maxpool3(x2)
        x3 = self.MCDropout(x3) if self.net_bayesian!=0 else x3  # MCDropout!
        x3 = self.nnblock8(x3) if self.nnblock else x3        
        x3 = self.RRCNN3(x3) if self.rcnn else self.Conv3(x3)

#         print('x3',x3.shape)

        x4 = wt(x3) if self.wavelet else self.Maxpool4(x3)
        x4 = self.MCDropout(x4) if self.net_bayesian!=0 else x4  # MCDropout!
        x4 = self.nnblock16(x4) if self.nnblock else x4
        x4 = self.RRCNN4(x4) if self.rcnn else self.Conv4(x4)
#         print('x4',x4.shape)

        x5 = wt(x4) if self.wavelet else self.Maxpool5(x4)
        x5 = self.MCDropout(x5) if self.net_bayesian!=0 else x5  # MCDropout!
        x5 = self.nnblock32(x5) if self.nnblock else x5
        x5 = self.RRCNN5(x5) if self.rcnn else self.Conv5(x5)
#         print('x5',x5.shape)

        # decoding + concat path
        d5 = iwt(x5) if self.wavelet else self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4) if self.attention else x4
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.nnblock16(d5) if self.nnblock else d5
        # d5 = self.Up_conv5(d5) if self.rcnn == False else self.Up_RRCNN5(d5)
        d5 = self.Up_RRCNN5(d5) if self.rcnn else self.Up_conv5(d5)
#         print('d5',d5.shape)

        d4=iwt(d5) if self.wavelet else self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3) if self.attention else x3
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.nnblock8(d4) if self.nnblock else d4
        # d4 = self.Up_conv4(d4) if self.rcnn == False else self.Up_RRCNN4(d4)
        d4 = self.Up_RRCNN4(d4) if self.rcnn else self.Up_conv4(d4)

#         print('d4',d4.shape)

        d3=iwt(d4) if self.wavelet else self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2) if self.attention else x2
        d3 = torch.cat((x2,d3),dim=1)
        # d3 = self.nnblock4(d3) if self.nnblock else d3
        # d3 = self.Up_conv3(d3) if self.rcnn == False else self.Up_RRCNN3(d3)
        d3 = self.Up_RRCNN3(d3) if self.rcnn else self.Up_conv3(d3)

        #         print('d3',d3.shape)

        d2=iwt(d3) if self.wavelet else self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1) if self.attention else x1
        d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.nnblock2(d2) if self.nnblock else d2
        # d2 = self.Up_conv2(d2) if self.rcnn == False else self.Up_RRCNN2(d2)
        d2 = self.Up_RRCNN2(d2) if self.rcnn else self.Up_conv2(d2)
#         print('d2',d2.shape)

        if self.supervision:

            s2 = d2
            s3 = iwt(d3)
            s4 = iwt(iwt(d4))
            s5 = iwt(iwt(iwt(d5)))
            
            d2 = torch.cat((s2,s3,s4,s5),dim=1)
            d1 = self.Conv_final(d2)
            
        else:
            d1 = self.Conv_final(d2)            

        return d1

# weight standardization
class Conv2d(nn.Conv2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
#         inplace = True
        inplace = False
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
#         inplace = True
        inplace = False

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace)
        )
    
    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        inplace = False
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
#         inplace = True
        inplace = False

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=inplace)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(attention_block,self).__init__()
#         inplace= True
        inplace= False

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=inplace)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi  
    

class MCDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,t=2,norm='batch'):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t,norm=norm)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t,norm=norm)
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t,norm=norm)        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t,norm=norm)        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t,norm=norm)
        
        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t,norm=norm)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t,norm=norm)
        
        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=2,ws=False,net_bayesian=0):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64,norm=norm)
        self.Conv2 = conv_block(ch_in=64,ch_out=128,norm=norm)
        self.Conv3 = conv_block(ch_in=128,ch_out=256,norm=norm)
        self.Conv4 = conv_block(ch_in=256,ch_out=512,norm=norm)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024,norm=norm)

        self.Up5 = up_conv(ch_in=1024,ch_out=512,norm=norm)
        self.Att5 = attention_block(F_g=512,F_l=512,F_int=256,norm=norm)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512,norm=norm)

        self.Up4 = up_conv(ch_in=512,ch_out=256,norm=norm)
        self.Att4 = attention_block(F_g=256,F_l=256,F_int=128,norm=norm)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256,norm=norm)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128,norm=norm)
        self.Att3 = attention_block(F_g=128,F_l=128,F_int=64,norm=norm)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128,norm=norm)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64,norm=norm)
        self.Att2 = attention_block(F_g=64,F_l=64,F_int=32,norm=norm)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64,norm=norm)
        
        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0) if ws == False else Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0) 
        self.net_bayesian = net_bayesian
        if self.net_bayesian !=0:
            self.MCDropout = MCDropout(p=self.net_bayesian)
        
    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
#         x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
#         x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
#         x3 = self.MCDropout(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
#         x4 = self.nnblock512(x4) 
#         x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.MCDropout(x5) if self.net_bayesian!=0 else x5
        x5 = self.Conv5(x5)
#         x5 = self.nnblock1024(x5) 

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
#         d5 = self.nnblock512(d5) 
#         d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
#         d4 = self.MCDropout(d4)
        
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
#         d3 = self.MCDropout(d3)
        
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
#         d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class R2AttU_Net(nn.Module):
    def __init__(self,net_inputch=3,net_outputch=2,t=2,net_bayesian=0):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=net_inputch,ch_out=64,t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Conv_1x1 = nn.Conv2d(64,net_outputch,kernel_size=1,stride=1,padding=0)  
        if net_bayesian !=0:
            self.MCDropout = MCDropout(p=net_bayesian)
            
#         self.nnblock256 = NONLocalBlock2D(256)
#         self.nnblock512 = NONLocalBlock2D(512)
#         self.nnblock1024 = NONLocalBlock2D(1024)

    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)
#         x1 = self.MCDropout(x1)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
#         x2 = self.MCDropout(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
#         x3 = self.nnblock256(x3) 
#         x3 = self.MCDropout(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
#         x4 = self.nnblock512(x4) 
#         x4 = self.MCDropout(x4)

        x5 = self.Maxpool(x4)
        x5 = self.MCDropout(x5)
        x5 = self.RRCNN5(x5)
#         x5 = self.nnblock1024(x5) 
        
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)        
#         d5 = self.nnblock512(d5) 
#         d5 = self.MCDropout(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)        
#         d4 = self.nnblock256(d4)
#         d4 = self.MCDropout(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)
#         d3 = self.MCDropout(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)
#         d2 = self.MCDropout(d2)

        d1 = self.Conv_1x1(d2)

        return d1
    
import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = max_pool_layer

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
        
        
class DiscriminateNet(nn.Module):
    def __init__(self, n_class=2):
        super(StanfordBNet, self).__init__()

        self.conv1_1 = nn.Conv2d(n_class, 16, 5, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(8, 16, 5, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, 2, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, segmented_input, original_input):
        res1 = self.relu(self.conv1_1(segmented_input))

        res2 = self.relu(self.conv2_1(original_input))
        res2 = F.max_pool2d(res2, 2, stride=1, padding=1)
        res2 = self.relu(self.conv2_2(res2))
        res2 = F.max_pool2d(res2, 2, stride=1, padding=1)

        res3 = torch.cat((res1, res2), 1)

        res3 = self.relu(self.conv3_1(res3))
        res3 = F.max_pool2d(res3, 2, stride=1)
        res3 = self.relu(self.conv3_2(res3))
        res3 = F.max_pool2d(res3, 2, stride=1)
        res3 = self.relu(self.conv3_3(res3))
        res3 = self.conv3_4(res3)
        
        # return res
        out = F.avg_pool2d(res3, (res3.shape[2],res3.shape[3]))
        out = F.softmax(out)
        n , _ , _ ,_  = segmented_input.size()
        return out.view(n,-1).transpose(0,1)[0]
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,net_inputch=3,net_outputch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, net_inputch=3, mid_ch=12, net_outputch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, net_inputch=3, mid_ch=12, net_outputch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, net_inputch=3, mid_ch=12, net_outputch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, net_inputch=3, mid_ch=12, net_outputch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, net_inputch=3, mid_ch=12, net_outputch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self,net_inputch=3,net_outputch=1):
        super(U2NET,self).__init__()

        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,net_inputch=3,net_outputch=1):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
    
    
#####
import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale)

    def forward(self, input):
        return self.upsample(input)
    
class PixelshuffleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(PixelshuffleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels*(scale**2), kernel_size=1, stride=1, padding=ksize//2),
            nn.PixelShuffle(upscale_factor=scale)
        )
        
    def forward(self, input):
        return self.upsample(input)
    
class IWTBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(IWTBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels*(scale**2), kernel_size=1, stride=1, padding=ksize//2)
        self.iwt = iwt

    def forward(self, input):
        return self.iwt(self.conv(input))
    
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            upsample_type = 'upsample',
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        if upsample_type=='upsample':
            self.upsample   = UpsampleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        elif upsample_type=='pixelshuffle':
            self.upsample   = PixelshuffleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        elif upsample_type=='iwt':
            self.upsample   = IWTBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        
        self.conv1      = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention1 = Attention(attention_type, in_channels=in_channels)
        self.conv2      = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, in_channels=out_channels)
        

    def forward(self, x):
        x = self.upsample(x)

        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class Last_DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            upsample_type = 'upsample',
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        if upsample_type=='upsample':
            self.upsample   = UpsampleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        elif upsample_type=='pixelshuffle':
            self.upsample   = PixelshuffleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        elif upsample_type=='iwt':
            self.upsample   = IWTBlock(scale=2, input_channels=in_channels, output_channels=in_channels)

        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class AE_Decoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            upsample_type='upsample'
    ):
        super().__init__()
        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels    # (256, 128, 64, 32)

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [ DecoderBlock(in_ch, out_ch, upsample_type, **kwargs) for in_ch, out_ch in zip(in_channels, out_channels) ]
        self.blocks = nn.ModuleList(blocks)
        self.last_block = Last_DecoderBlock(in_channels=32, out_channels=16, upsample_type=upsample_type, use_batchnorm=True, attention_type='scse')

    def forward(self, features):
        x = self.center(features)

        for decoder_block in self.blocks:
            x = decoder_block(x)

        x = self.last_block(x)

        return x
    
import torch
import torch.nn as nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)
    

class ReconstructionHead(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)
        