#gcn的A是可以学习的，拼接unet、gcn、transformer的特征
import torch.nn as nn
import torch
import os,sys
os.chdir(sys.path[0])
import torch.nn.functional as F
from CTrans import ChannelTransformer
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.parameter import Parameter
import numpy as np
import copy
from torch.nn.modules.utils import _pair
import math
print("gcn的A是可以学习的,拼接unet、gcn、transformer的特征")
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        #将大小变为，4*512，-1表示根据前面的一个参数进行计算
        return x.view(x.size(0), -1)

class CCA_Threefeature(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_x2 = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x,x2):
        # channel-wise attention。g是unet的，x是transformer学习之后的
        #x,[4,512,28,28]
        #二维平均池化操作，窗口大小是28*28，每次移动28个位置，相当于把最后两个维度给全部弄成一维的了。
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))#[4,512,1,1]
        #[4,512]，先展平，然后再进入线性层
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_x2 = F.avg_pool2d( x2, (x2.size(2), x2.size(3)), stride=(x2.size(2), x2.size(3)))#[4,512,1,1]
        #[4,512]，先展平，然后再进入线性层
        channel_att_x2 = self.mlp_x2(avg_pool_x2)
        #对unet的特征也进行相同的操作
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        #[4,512]
        channel_att_g = self.mlp_g(avg_pool_g)
        #[4,512]
        #这个地方为什么不加一个sigmoid呢？后面已经加了sigmoid了
        channel_att_sum = (channel_att_x + channel_att_g+channel_att_x2)/3.0
        #上面的操作，对应公式5
        #[4,512,28,28],在第三个和第四个维度上按照x的大小进行扩展
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        #扩展之后进行点乘，更新trans特征
        x_after_channel = g * scale
        #[4,512,28,28]
        out = self.relu(x_after_channel)
        return out
class UpBlock_attention_threeeFeature(nn.Module):
    def __init__(self,in_channels,out_channels,nb_Conv,activation="ReLU") -> None:
        super().__init__()
        self.up=nn.Upsample(scale_factor=2)
        self.concat_three=CCA_Threefeature(in_channels//2,in_channels//2)
        self.nConvs=_make_nConv(3*in_channels//2,out_channels,nb_Conv,activation)
    def forward(self, x, skip_x1,skip_x2):
        #[4,512,14,14],[4,512,28,28]
        up = self.up(x)
        #需要升维的矩阵放在最前面
        skip_x_att = self.concat_three(up, skip_x1,skip_x2)
        x = torch.cat([skip_x_att, skip_x1,skip_x2], dim=1)  # dim 1 is the channel dimension
        #最后相当于通过
        return self.nConvs(x)


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        #1024,512
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        #不改变通道数的卷积层
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        #[4,512,14,14],[4,512,28,28]
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        #最后相当于通过
        return self.nConvs(x)
class ConvBatchNorm_one(nn.Module):
    """(convolution => [BN] => ReLU)"""
    #使用1x1的卷积改变通道的大小
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm_one, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, padding=0)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
class oneConv(nn.Module):
    def __init__(self,config,in_channels,out_channels):
        super(oneConv,self).__init__()    
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.layer=nn.ModuleList()
        #向下采样了多少次，就要有几个1x1的卷积层，用来调整通道的数量，默认为5
        # for _ in range(config.feature_num):
        #     layer=ConvBatchNorm_one(in_channels,out_channels)
        #     self.layer.append(copy.deepcopy(layer))
        self.layer1=ConvBatchNorm_one(in_channels,out_channels)
        self.layer2=ConvBatchNorm_one(in_channels*2,out_channels)
        self.layer3=ConvBatchNorm_one(in_channels*4,out_channels)
        self.layer4=ConvBatchNorm_one(in_channels*8,out_channels)
        self.layer5=ConvBatchNorm_one(in_channels*8,out_channels)
    def forward(self,x1,x2,x3,x4,x5):
        em1=self.layer1(x1)
        em2=self.layer2(x2)
        em3=self.layer3(x3)
        em4=self.layer4(x4)
        em5=self.layer5(x5) 
        return em1,em2,em3,em4,em5
class Channel_embed_samecchannel(nn.Module):
    def __init__(self,config,in_channels,imgsize,patchsize):
        super().__init__()
        img_size=_pair(imgsize)
        patch_size=_pair(patchsize)
        num_patch=(img_size[0]//patch_size[0]*(img_size[1]//patch_size[1]))
        self.imgsize=imgsize
        self.patchsize=patchsize
        self.patch_embed=Conv2d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=patchsize,
                                stride=patch_size
                                )
        #[1.196,64]
        self.para_emb=nn.Parameter(torch.zeros(1,in_channels,num_patch))
        self.dropout=Dropout(config.dropout_rate)
    def forward(self,x):
        if x is None:
            return None
        #[4,64,196]
        x=self.patch_embed(x).flatten(2)
        embed=x+self.para_emb
        embed=self.dropout(embed)
        return embed
class gcn(nn.Module):
    def __init__(self,config,in_feature_num,out_feature_num ,batch_size,use_bias=True):
        super(gcn,self).__init__()
        self.infeatures=in_feature_num
        self.outfeatures=out_feature_num
        self.use_bias=use_bias
        #直接用全连接层，nn.linear
        self.linear=nn.Linear(self.infeatures,self.outfeatures,bias=False)
        # self.weight=Parameter(torch.FloatTensor(self.infeatures,self.outfeatures))
        # if use_bias:
        #     self.bias=Parameter(torch.FloatTensor(self.outfeatures))
        # self.reset_parameters()
        self.out=nn.Linear(out_feature_num, in_feature_num)
        self.drop=Dropout(config.transformer["attention_dropout_rate"])
    def forward(self,inputs,adj):
        #全连接
        support=self.linear(inputs)
        outputs=torch.matmul(adj,support)
        outputs=self.out(outputs)
        outputs=self.drop(outputs)        
        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = self.weight.shape
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
class gcn_encoder(nn.Module):
    def __init__(self,config,in_channels,out_channels,patch_size,imgsize,activation='ReLU'):
        super(gcn_encoder,self).__init__()
        self.patchsize=patch_size
        self.oneconv=oneConv(config,in_channels,out_channels)
        # self.layer1=ConvBatchNorm_one(in_channels,out_channels)
        # self.layer2=ConvBatchNorm_one(in_channels*2,out_channels)
        # self.layer3=ConvBatchNorm_one(in_channels*4,out_channels)
        # self.layer4=ConvBatchNorm_one(in_channels*8,out_channels)
        # self.layer5=ConvBatchNorm_one(in_channels*8,out_channels)
        self.channelembed1=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize,patchsize=self.patchsize)
        self.channelembed2=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize//2,patchsize=self.patchsize//2)
        self.channelembed3=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize//4,patchsize=self.patchsize//4)
        self.channelembed4=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize//8,patchsize=self.patchsize//8)
        self.channelembed5=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize//16,patchsize=self.patchsize//16)
        self.infeature=in_channels*(imgsize//self.patchsize*(imgsize//self.patchsize))
        self.gcn=gcn(config,in_feature_num=self.infeature,out_feature_num=self.infeature//2,batch_size=4,use_bias=config.usebias)
        self.activation = get_activation(activation)
        self.adj=Parameter(torch.FloatTensor(5,5))
        torch.nn.init.kaiming_uniform_(self.adj, a=math.sqrt(5))
    def forward(self,x1,x2,x3,x4,x5):
        #将各层卷积的通道数设置成一样
        emb1,emb2,emb3,emb4,emb5=self.oneconv(x1,x2,x3,x4,x5)        
        #将上述的各层弄成统一大小，4*in_channels*196
        en1=self.channelembed1(emb1)
        en2=self.channelembed2(emb2)
        en3=self.channelembed3(emb3)
        en4=self.channelembed4(emb4)
        en5=self.channelembed5(emb5)
        #将上述模型展平，拼接成5*（in_channels*196）
        en1=en1.flatten(1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        en2=en2.flatten(1)
        en3=en3.flatten(1)
        en4=en4.flatten(1)
        en5=en5.flatten(1)
        emball=torch.stack([en1,en2,en3,en4,en5],dim=1)
        embed_gcn=self.gcn(inputs=emball,adj=self.adj)
        self.activation(embed_gcn)
        return embed_gcn
def create_adj(num):
    adj=np.zeros([num,num])
    adj[0,0],adj[0,1]=1,1
    adj[1,0],adj[1,1],adj[1,2]=1,1,1
    adj[2,1],adj[2,2],adj[2,3]=1,1,1
    adj[3,2],adj[3,3],adj[3,4]=1,1,1
    adj[4,3],adj[4,4]=1,1
    return(torch.FloatTensor(adj).cuda())

class reconstruct_same(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,B,Feature_len_reconsruct,scale):
        super(reconstruct_same,self).__init__()
        self.B,self.Feature_len_reconsruct,self.in_channels=B,Feature_len_reconsruct,in_channels
        self.Feature_len=int(self.Feature_len_reconsruct*self.Feature_len_reconsruct)
        self.scale=scale
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    def forward(self,x):
        #4,5,64*196(12544)
        #[4,64,196]#[4,64,14,14]
        Batch,_,_=x.size()
        # print(x.size())
        # print(Batch,self.Feature_len_reconsruct)
        em1=torch.squeeze(x)
        em1=em1.reshape(Batch,self.in_channels,self.Feature_len_reconsruct,self.Feature_len_reconsruct)
        # em1=em.view(Batch,self.in_channels,self.Feature_len_reconsruct,self.Feature_len_reconsruct)
        em1=nn.Upsample(scale_factor=self.scale)(em1)
        em1=self.conv1(em1)
        out = self.norm(em1)
        out = self.activation(em1)
        return out
class reconstruct(nn.Module):
    def __init__(self,batch,imgsize,in_channels,patchsize,kernel_size):
        super(reconstruct,self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.in_channels=in_channels
        self.patch_size=patchsize
        self.scale=(self.patch_size,self.patch_size)
        Feature_len_reconsruct=imgsize//patchsize
        #self.infeature=in_channels*(Feature_len_reconsruct)
        self.conv1=reconstruct_same(in_channels,in_channels,kernel_size,padding,batch,Feature_len_reconsruct,(self.patch_size,self.patch_size))
        self.conv2=reconstruct_same(in_channels,in_channels*2,kernel_size,padding,batch,Feature_len_reconsruct,(self.patch_size//2,self.patch_size//2))
        self.conv3=reconstruct_same(in_channels,in_channels*4,kernel_size,padding,batch,Feature_len_reconsruct,(self.patch_size//4,self.patch_size//4))
        self.conv4=reconstruct_same(in_channels,in_channels*8,kernel_size,padding,batch,Feature_len_reconsruct,(self.patch_size//8,self.patch_size//8))
        self.conv5=reconstruct_same(in_channels,in_channels*8,kernel_size,padding,batch,Feature_len_reconsruct,(self.patch_size//16,self.patch_size//16))
    def forward(self,x):
        #4,5,64*196
        em1,em2,em3,em4,em5=torch.split(x,split_size_or_sections=1,dim=1)
        em1=self.conv1(em1)
        em2=self.conv2(em2)
        em3=self.conv3(em3)
        em4=self.conv4(em4)
        em5=self.conv5(em5)
        return em1,em2,em3,em4,em5
        
        
class UCTransNet(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.patch=config.patch_size
        self.inc = ConvBatchNorm(n_channels, in_channels)#64
        # self.down1=ConvBatchNorm(n_channels,in_channels)
        # self.down2=ConvBatchNorm(n_channels,in_channels)
        # self.down3=ConvBatchNorm(n_channels,in_channels)
        # self.down4=ConvBatchNorm(n_channels,in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)#128
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)#256
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)#512
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)#512
        self.gcn=gcn_encoder(config,in_channels,in_channels,self.patch,img_size)
        self.recons=reconstruct(config.batchsize,img_size,in_channels,self.patch,config.kernersize)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.concat=ConvBatchNorm(in_channels*16,in_channels*8)
        self.up4 = UpBlock_attention_threeeFeature(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention_threeeFeature(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention_threeeFeature(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention_threeeFeature(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss
        self.duolei_activattion=Softmax(dim=1)
    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)#[4,64,224,224]
        x2 = self.down1(x1)#[4,128,112,112]
        x3 = self.down2(x2)#[4,256,56,56]
        x4 = self.down3(x3)#[4,512,28,28]
        x5 = self.down4(x4)#[4,512,14,14]
        embed=self.gcn(x1,x2,x3,x4,x5)
        de1,de2,de3,de4,de5=self.recons(embed)

        x5=torch.cat([x5,de5],dim=1)

        x5=self.concat(x5)
        x1,x2,x3,x4,att_weights = self.mtc(x1,x2,x3,x4)

        # 或者把上面的特征融合成为一层，和底层特征concat
        x = self.up4(x5, x4,de4)
        x = self.up3(x, x3,de3)
        x = self.up2(x, x2,de2)
        x = self.up1(x, x1,de1)
        if self.n_classes ==1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1
            logits=self.duolei_activattion(logits)
        if self.vis: # visualize the attention maps
            return logits, att_weights
        else:
            return logits




