# -*- coding: utf-8 -*-
#MLMSeg for ultrasound nodule segmentation
import torch.nn as nn
import torch
import os,sys
os.chdir(sys.path[0])
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.parameter import Parameter
import numpy as np
import copy
from torch.nn.modules.utils import _pair
import math
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
class MCTS(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config,self.patchSize_1, img_size=img_size,    in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config,self.patchSize_2, img_size=img_size//2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config,self.patchSize_3, img_size=img_size//4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config,self.patchSize_4, img_size=img_size//8, in_channels=channel_num[3])
        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,scale_factor=(self.patchSize_1,self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,scale_factor=(self.patchSize_2,self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,scale_factor=(self.patchSize_3,self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,scale_factor=(self.patchSize_4,self.patchSize_4))

    def forward(self,en1,en2,en3,en4):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1,emb2,emb3,emb4)  # (B, n_patch, hidden)
        #[4,64,224,224]
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None#上采样成原来大小，放入一个1*1的卷积层，增加一些科学系的参数
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None
        x1 = x1 + en1  if en1 is not None else None#返回unet的卷积核encoder1相加的结果
        x2 = x2 + en2  if en2 is not None else None
        x3 = x3 + en3  if en3 is not None else None
        x4 = x4 + en4  if en4 is not None else None

        return x1, x2, x3, x4, attn_weights

class Cgat(nn.Module):
    # channel-wise graph attention network
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
        
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))#[4,512,1,1]
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_x2 = F.avg_pool2d( x2, (x2.size(2), x2.size(3)), stride=(x2.size(2), x2.size(3)))#[4,512,1,1]
        channel_att_x2 = self.mlp_x2(avg_pool_x2)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g+channel_att_x2)/3.0

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = g * scale
        out = self.relu(x_after_channel)
        return out
class Cgatblock(nn.Module):
    #cgatblock
    def __init__(self,in_channels,out_channels,nb_Conv,activation="ReLU") -> None:
        super().__init__()
        self.up=nn.Upsample(scale_factor=2)
        self.concat_three=Cgat(in_channels//2,in_channels//2)
        self.nConvs=_make_nConv(3*in_channels//2,out_channels,nb_Conv,activation)
    def forward(self, x, skip_x1,skip_x2):
        up = self.up(x)
        skip_x_att = self.concat_three(up, skip_x1,skip_x2)
        x = torch.cat([skip_x_att, skip_x1,skip_x2], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)
class ConvBatchNorm_one(nn.Module):
    #using 1x1 convs
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
        self.linear=nn.Linear(self.infeatures,self.outfeatures,bias=False)
        self.out=nn.Linear(out_feature_num, in_feature_num)
        self.drop=Dropout(config.transformer["attention_dropout_rate"])
    def forward(self,inputs,adj):
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
class Clgcn(nn.Module):
    def __init__(self,config,in_channels,out_channels,patch_size,imgsize,activation='ReLU'):
        super(Clgcn,self).__init__()
        self.patchsize=patch_size
        self.oneconv=oneConv(config,in_channels,out_channels)
        self.channelembed1=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize,patchsize=self.patchsize)
        self.channelembed2=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize//2,patchsize=self.patchsize//2)
        self.channelembed3=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize//4,patchsize=self.patchsize//4)
        self.channelembed4=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize//8,patchsize=self.patchsize//8)
        self.channelembed5=Channel_embed_samecchannel(config,in_channels,imgsize=imgsize//16,patchsize=self.patchsize//16)
        self.infeature=in_channels*(imgsize//self.patchsize*(imgsize//self.patchsize))
        self.gcn=clgcn(config,in_feature_num=self.infeature,out_feature_num=self.infeature//2,batch_size=4,use_bias=config.usebias)
        self.activation = get_activation(activation)
        self.adj=Parameter(torch.FloatTensor(5,5))
        torch.nn.init.kaiming_uniform_(self.adj, a=math.sqrt(5))
    def forward(self,x1,x2,x3,x4,x5):
        emb1,emb2,emb3,emb4,emb5=self.oneconv(x1,x2,x3,x4,x5)        
        en1=self.channelembed1(emb1)
        en2=self.channelembed2(emb2)
        en3=self.channelembed3(emb3)
        en4=self.channelembed4(emb4)
        en5=self.channelembed5(emb5)
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
        Batch,_,_=x.size()
        em1=torch.squeeze(x)
        em1=em1.reshape(Batch,self.in_channels,self.Feature_len_reconsruct,self.Feature_len_reconsruct)
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
    
class Channel_Embeddings(nn.Module):
    def __init__(self,config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size2=img_size
        self.patch_size2=patch_size
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x) 
        x = x.flatten(2)
        x = x.transpose(-1, -2) 
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings 

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #14,14
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        #[4,64,196]
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)#[4,64,14,14]
        x = nn.Upsample(scale_factor=self.scale_factor)(x)#[上采样16倍，变成[4,64,224,224]

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class channel_attention(nn.Module):
    def __init__(self, config, vis,channel_num):
        super(channel_attention, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            query3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
            query4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
            key = nn.Linear( self.KV_size,  self.KV_size, bias=False)
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.dropout_attn = Dropout(config.attn_dropout)
        self.proj_dropout = Dropout(config.proj_dropout)



    def forward(self, emb1,emb2,emb3,emb4, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []
        multi_head_Q4_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2)
                multi_head_Q2_list.append(Q2)
        if emb3 is not None:
            for query3 in self.query3:
                Q3 = query3(emb3)
                multi_head_Q3_list.append(Q3)
        if emb4 is not None:
            for query4 in self.query4:
                Q4 = query4(emb4)
                multi_head_Q4_list.append(Q4)
        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)
        # print(len(multi_head_Q4_list))
        #print(multi_head_Q1_list[1].shape)
        #[4,4,196,64]
        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1)  
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1)  
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1)  
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1)  
        #[4,4,196,960]
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)
        #print(multi_head_Q1.shape)
        #[4,4,64,196]
        multi_head_Q1 = multi_head_Q1.transpose(-1, -2)  
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2)  
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2)  
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2)  
        #[4,4,64,960]
        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K)  
        #[4,4,128,960]
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K)  
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K)  
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_K)  

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size)  
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size)  
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size)  
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size)  
        #[4,4,64,960]
        attention_probs1 = self.softmax(self.psi(attention_scores1))  
        attention_probs2 = self.softmax(self.psi(attention_scores2))  
        attention_probs3 = self.softmax(self.psi(attention_scores3))  
        attention_probs4 = self.softmax(self.psi(attention_scores4))  
        # print(attention_probs4.size())

        if self.vis:
            weights =  []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))
            weights.append(attention_probs4.mean(1))
        else: weights=None
        #[4,4,64,960]
        attention_probs1 = self.dropout_attn(attention_probs1)  
        attention_probs2 = self.dropout_attn(attention_probs2)  
        attention_probs3 = self.dropout_attn(attention_probs3)  
        attention_probs4 = self.dropout_attn(attention_probs4)  
        #[4,4,960,196]
        multi_head_V = multi_head_V.transpose(-1, -2)
        #[4,4,64,196]
        context_layer1 = torch.matmul(attention_probs1, multi_head_V)  
        context_layer2 = torch.matmul(attention_probs2, multi_head_V)  
        context_layer3 = torch.matmul(attention_probs3, multi_head_V)  
        context_layer4 = torch.matmul(attention_probs4, multi_head_V)  
        #[4,196,64,4]
        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous()  
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous()  
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous()  
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous()  
        #[4,196,64]
        context_layer1 = context_layer1.mean(dim=3)  
        context_layer2 = context_layer2.mean(dim=3)  
        context_layer3 = context_layer3.mean(dim=3)  
        context_layer4 = context_layer4.mean(dim=3)  
        O1 = self.out1(context_layer1)  
        O2 = self.out2(context_layer2)  
        O3 = self.out3(context_layer3)  
        O4 = self.out4(context_layer4)  
        O1 = self.proj_dropout(O1)  
        O2 = self.proj_dropout(O2)  
        O3 = self.proj_dropout(O3)  
        O4 = self.proj_dropout(O4)  
        return O1,O2,O3,O4, weights




class Mlp(nn.Module):
    def __init__(self,config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.attn_norm =  LayerNorm(config.KV_size,eps=1e-6)
        self.atten = channel_attention(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.ffn1 = Mlp(config,channel_num[0],channel_num[0]*expand_ratio)
        self.ffn2 = Mlp(config,channel_num[1],channel_num[1]*expand_ratio)
        self.ffn3 = Mlp(config,channel_num[2],channel_num[2]*expand_ratio)
        self.ffn4 = Mlp(config,channel_num[3],channel_num[3]*expand_ratio)


    def forward(self, emb1,emb2,emb3,emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb"+str(i+1)

            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat,dim=2)
        cx1 = self.attn_norm1(emb1)
        cx2 = self.attn_norm2(emb2) 
        cx3 = self.attn_norm3(emb3) 
        cx4 = self.attn_norm4(emb4)
        emb_all = self.attn_norm(emb_all)
        cx1,cx2,cx3,cx4, weights = self.atten(cx1,cx2,cx3,cx4,emb_all)
        #[4,196,64]
        cx1 = org1 + cx1  
        cx2 = org2 + cx2  
        cx3 = org3 + cx3  
        cx4 = org4 + cx4  
        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1)  
        x2 = self.ffn_norm2(cx2)  
        x3 = self.ffn_norm3(cx3)  
        x4 = self.ffn_norm4(cx4)  
        x1 = self.ffn1(x1)  
        x2 = self.ffn2(x2)  
        x3 = self.ffn3(x3)  
        x4 = self.ffn4(x4)  
        x1 = x1 + org1  
        x2 = x2 + org2  
        x3 = x3 + org3  
        x4 = x4 + org4  
        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2,emb3,emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1,emb2,emb3,emb4, weights = layer_block(emb1,emb2,emb3,emb4)
            if self.vis:
                attn_weights.append(weights)
        #[4,196,64]
        emb1 = self.encoder_norm1(emb1)
        emb2 = self.encoder_norm2(emb2)
        emb3 = self.encoder_norm3(emb3)
        emb4 = self.encoder_norm4(emb4)
        return emb1,emb2,emb3,emb4




class MLMSeg(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.patch=config.patch_size
        self.inc = ConvBatchNorm(n_channels, in_channels)#64
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)#128
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)#256
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)#512
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)#512
        self.gcn=Clgcn(config,in_channels,in_channels,self.patch,img_size)
        self.recons=reconstruct(config.batchsize,img_size,in_channels,self.patch,config.kernersize)
        self.mcts = MCTS(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.concat=ConvBatchNorm(in_channels*16,in_channels*8)
        self.up4 = Cgatblock(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = Cgatblock(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = Cgatblock(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = Cgatblock(in_channels*2, in_channels, nb_Conv=2)
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
        x1,x2,x3,x4 = self.mcts(x1,x2,x3,x4)
        x = self.up4(x5, x4,de4)
        x = self.up3(x, x3,de3)
        x = self.up2(x, x2,de2)
        x = self.up1(x, x1,de1)
        logits = self.last_activation(self.outc(x))
        return logits




