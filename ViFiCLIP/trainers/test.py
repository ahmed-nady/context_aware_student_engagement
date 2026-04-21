import torch
from torch import nn


class TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=7,padding=int(7/2),bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x_ord = x
        bs,t,c = x.shape
        x = self.gap(x)
        x = self.conv(x.transpose(1,2))
        attention = self.sigmoid(x.view(-1,t,1))
        return attention* x_ord

class multiRangesTemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        short_range,mid_range,long_range = 3,7,11
        self.conv_short = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=short_range,padding=int(short_range/2),bias=False)
        self.conv_mid = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=mid_range,padding=int(mid_range/2),bias=False)
        self.conv_long = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=long_range,padding=int(long_range/2),bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x_ord = x
        bs,t,c = x.shape
        x = self.gap(x)
        x= x.transpose(1, 2)
        x_short = self.conv_short(x)
        x_mid = self.conv_mid(x)
        x_long = self.conv_long(x)
        x = torch.cat((x_short,x_mid,x_long),dim=1)
        x = x.transpose(1,2)
        x= self.gap(x)
        attention = self.sigmoid(x.view(-1, t, 1))
        return attention * x_ord

class TopKPooling(nn.Module):
    def __init__(self,k=4):
        super().__init__()
        self.k = k
    def forward(self,x):
        scores  = torch.norm(x,dim=-1)
        topk_scores,topk_idx = scores.topk(self.k,dim=1)
        idx = topk_idx.unsqueeze(-1).expand(-1,-1,x.size(-1))
        topk_feats = torch.gather(x,1,idx)
        out = topk_feats.mean(dim=1)
        return out

if __name__=="__main__":
    x = torch.rand(1,32,512)
    # tepAttention =multiRangesTemporalAttention()
    # out = tepAttention(ten)
    topkPool = TopKPooling()
    out = topkPool(x)
    print(out.shape)