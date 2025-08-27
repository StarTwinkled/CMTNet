"Enhancing Real-Time Semantic Segmentation: A Dual-Branch Architecture with Mamba-Transformer Synergy"
""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from IPython import embed
import math

from torch.autograd import Variable
from mamba_ssm import Mamba





class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)




class BRUModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):  #
        super().__init__()
        #
        self.bn_relu_1 = BNPReLU(nIn)  #

        self.conv1x1_init = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=True)  #
        self.ca0 = eca_layer(nIn // 2)
        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)

        self.dconv1x3_l = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.dconv3x1_l = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)

        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3_r = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1_r = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.ca11 = eca_layer(nIn // 2)
        self.ca22 = eca_layer(nIn // 2)
        self.ca = eca_layer(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle_end = ShuffleBlock(groups=nIn // 2)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_init(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)

        c1 = self.shuffle_end(br1)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        c2 = self.shuffle_end(br2)


        b1 = self.ca11(br1)
        br1 = self.dconv1x3_l(b1 + c2)
        br1 = self.dconv3x1_l(br1)

        b2 = self.ca22(br2)
        br2 = self.ddconv1x3_r(b2 + c1)
        br2 = self.ddconv3x1_r(br2)


        output = br1 + br2 + self.ca0(output )+ b1 + b2

        output = self.bn_relu_2(output)

        output = self.conv1x1(output)
        output = self.ca(output)
        out = output + input
        return out



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool],
                               1)

        output = self.bn_prelu(output)

        return output



class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace= True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output
    
class ExternalAttention(nn.Module):

    def __init__(self, d_model=256, S=64):
        super().__init__()

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        queries = x.view(b, c, n)  # 即bs，n，d_model
        queries = queries.permute(0, 2, 1)
        attn = self.mk(queries)  # bs,n,S  # torch.Size([6, 2700, 256])
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / (1e-9 + torch.sum(attn, dim=2, keepdim=True))  # bs,n,S
        attn = self.mv(attn)  # bs,n,d_model
        attn = attn.permute(0, 2, 1)
        x_attn = attn.view(b, c, h, w)
        x = x + x_attn
        x = F.relu(x)
        return x
    
class CrossResolutionAttention(nn.Module):

    def __init__(self, chann_high=128, chann_low=128):
        super().__init__()
        self.EAlayer = ExternalAttention(chann_high + chann_low)
        self.channel_high = chann_high
        self.channel_low = chann_low
        self.CBNReLU = ConvBNReLU(128, 128, 3, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x_h, x_l):  # torch.Size([6, 128, 90, 120])  torch.Size([6, 128, 45, 60])
        x_h_in = self.CBNReLU(x_h)
        x1 = torch.cat((x_h_in, x_l), 1)
        x_l_in = self.up(x_l)   # torch.Size([6, 128, 90, 120])
        x2 = torch.cat((x_h, x_l_in), 1)
        x_l = self.EAlayer(x1)
        x_h = self.EAlayer(x2)
        x_high = x_h[:, :self.channel_high, :, :]
        x_low = x_l[:, self.channel_high:(self.channel_high + self.channel_low), :, :]
        return x_high, x_low
    
class CARAFE(nn.Module):
    def __init__(self, c, k_enc=3, k_up=5, c_mid=64, scale=2):
        """ The unofficial implementation of the CARAFE module.
        The details are in "https://arxiv.org/abs/1905.02188".
        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.
        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernelsize=1, stride=1, padding=0, dilation=1)
        self.enc = nn.Conv2d(c_mid, (scale * k_up) ** 2, kernel_size=k_enc, stride=1, padding=k_enc // 2, dilation=1)
        self.bn = nn.BatchNorm2d((scale * k_up) ** 2)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.bn(W)
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = torch.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X
    
# """分割头"""
class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
            ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat

# '''基本的CBR层'''
class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernelsize=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channel, out_channel, kernel_size=kernelsize, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feature_map = self.conv(x)
        feature_map = self.bn(feature_map)
        feature_map = self.relu(feature_map)
        return feature_map
    
class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        # self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        # x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        # x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        # x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        # x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        # x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        # x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.mamba(x_norm)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out
    
class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        
    def forward(self, t1, t2, t3):
        r1, r2, r3 = t1, t2, t3

        satt1, satt2, satt3 = self.satt(t1, t2, t3)
        t1, t2, t3 = satt1 * t1, satt2 * t2, satt3 * t3

        r1_, r2_ , r3_ = t1, t2, t3
        t1, t2, t3= t1 + r1, t2 + r2, t3 + r3

        catt1, catt2, catt3= self.catt(t1, t2, t3)
        t1, t2, t3 = catt1 * t1, catt2 * t2, catt3 * t3

        return t1 + r1_, t2 + r2_, t3 + r3_
    
class Channel_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        # c_list_sum = sum(c_list) - c_list[-1]
        c_list_sum = sum(c_list) - c_list[0]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[1]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[2]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[3]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, t1, t2, t3):
        att = torch.cat((self.avgpool(t1), 
                         self.avgpool(t2),
                         self.avgpool(t3),     
                         ), dim=1)
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            
        return att1, att2, att3

class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                          nn.Sigmoid())
    
    def forward(self, t1, t2, t3):
        t_list = [t1, t2, t3]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
        return att_list[0], att_list[1], att_list[2]




class CMTNet(nn.Module):
    def __init__(self, classes=11, block_1=5, block_2=5, block_3 = 16, block_4 = 3, block_5 = 3,
                 c_list=[16,32,64,128], split_att='fc'):
        super().__init__()

        # ---------- Encoder -------------#
        self.init_conv = nn.Sequential(
            Conv(3, 16, 3, 2, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True),
            Conv(16, 16, 3, 1, padding=1, bn_acti=True),
        )
        # 1/2
        self.bn_prelu_1 = BNPReLU(16)

        # Branch 1
        # Attention 1
        self.attention1_1 = eca_layer(16)

        # BRU Block 1
        dilation_block_1 = [1, 1, 1, 1, 1]
        self.BRU_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.BRU_Block_1.add_module("BRU_Module_1_" + str(i) ,BRUModule(16, d=dilation_block_1[i]))
        self.bn_prelu_2 = BNPReLU(16)
        # Attention 2
        self.attention2_1 = eca_layer(16)



        # Down 1  1/4
        self.downsample_1 = DownSamplingBlock(16, 64)
        # BRU Block 2
        dilation_block_2 = [1, 2, 5, 9, 17]
        self.BRU_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.BRU_Block_2.add_module("BRU_Module_2_" + str(i) ,BRUModule(64, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(64)
        # Attention 3
        self.attention3_1 = eca_layer(64)


        # Down 2  1/8
        self.downsample_2 = DownSamplingBlock(64, 128)
        # BRU Block 3
        dilation_block_3 = [1, 2, 5, 9, 1, 2, 5, 9,       2, 5, 9, 17, 2, 5, 9, 17]
        self.BRU_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.BRU_Block_3.add_module("BRU_Module_3_" + str(i), BRUModule(128, d=dilation_block_3[i]))
        self.bn_prelu_4 = BNPReLU(128)
        # Attention 4
        self.attention4_1 = eca_layer(128)





        # --------------Decoder   ----------------- 
        # trans
        self.Cross_Atten = CrossResolutionAttention(128, 128)
        self.up2times1 = CARAFE(128)
        # self.up2times2 = CARAFE(256)
        self.Head = SegmentHead(256, 384, classes, up_factor=4, aux=False)

        # mamba
        self.encoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[0], output_dim=c_list[1])
        )
        self.ebn1 = nn.GroupNorm(1, c_list[1])

        self.encoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[2])
        )
        self.ebn2 = nn.GroupNorm(1, c_list[2])

        self.encoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.ebn3 = nn.GroupNorm(1, c_list[3])
        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[3], 3, stride=1, padding=1),
        )
        self.ebn4 = nn.GroupNorm(1, c_list[3])

        # self.scab = SC_Att_Bridge(c_list, split_att)




    def forward(self, input):

        output0 = self.init_conv(input)
        output0 = self.bn_prelu_1(output0)

        # Detail Branch
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(output0)),2,2))  # torch.Size([6, 32, 90, 120])
        t1 = out
        out = F.gelu(self.ebn2(self.encoder2(out))) # torch.Size([6, 64, 90, 120])
        t2 = out
        out = F.gelu(self.ebn3(self.encoder3(out)))  # torch.Size([6, 128, 90, 120]) 高分辨率
        
        out = F.gelu(self.ebn4(self.encoder4(out)))
        t3 = out

        # t1, t2, t3 = self.scab(t1, t2, t3)

        

        # Branch1
        output1 = self.attention1_1(output0)

        # block1
        output1 = self.BRU_Block_1(output1)  # torch.Size([6, 16, 180, 240])
        # output1 = torch.add(output1, t1)
        output1 = self.bn_prelu_2(output1)
        output1 = self.attention2_1(output1)

        # down1
        output1 = self.downsample_1(output1)

        # block2
        output1 = self.BRU_Block_2(output1)  # torch.Size([6, 64, 90, 120])
        output2 = torch.add(output1, t2)
        output1 = self.bn_prelu_3(output1)
        output1 = self.attention3_1(output1)

        # down2
        output1 = self.downsample_2(output1)

        # block3
        output2 = self.BRU_Block_3(output1)  # torch.Size([6, 128, 45, 60])
        output2 = self.bn_prelu_4(output2)
        output2 = self.attention4_1(output2)   #  低分辨率




        att_output_high, att_output_low = self.Cross_Atten(t3, output2)  # 分别输出给高分辨分支和低分辨分支的注意力图  torch.Size([6, 128, 90, 120])  torch.Size([6, 128, 45, 60])
        # 融合部分
        feature_h = t3 + att_output_high  # 高分辨分支加上注意力

        # 低分辨分支的逐元素加（融合）
        feature_l = output2 + att_output_low  # 低分辨分支加上注意力
        feature_l = self.up2times1(feature_l)  # torch.Size([6, 128, 48, 64])
        # feature_end = self.bga(feature_h, feature_l)
        feature_end = torch.cat((feature_h, feature_l), 1)
        # feature_end = self.up2times2(feature_end)  # torch.Size([6, 256, 96, 128])
        output = self.Head(feature_end)  # 分割头  torch.Size([6, 11, 360, 480])
        return output




"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMTNet(classes=19).to(device)
    summary(model, (3, 512, 1024))
