import os
import torch
import torch.nn as nn
import torch.nn.functional as F 

##### SpatialBlock
class SpatialBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        """
        Args:
            in_dim; the dimension of the feature map g(v) ex) 512
            hidden_dim: the dimension of the output of 1x1 conv (0.25 * in_dim) ex) 128
            out_dim: the dimension of the output of ChannelBlock (same as in_dim) ex) 512
        Returns:
            output of the SpatialBlock h or \hat{h}
        """
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1)
    
    def forward(self, x, shuffle=False):
        """
        self.layer1 helps stable training. It is not shuffled in the training stage.
        Args:
            x: output of the ChannelBlock g(v)
            shuffle: set to True if \hat{h} is used. set to False if h is used
        Returns:
            output of the ChannelBlock h or \hat{h}
        """

        x = self.layer1(x)

        if shuffle:
            i, o, h, w = self.layer2.weight.shape
            l_rand = self.layer2.weight.view(i,o,h*w)
            idx = torch.randperm(h*w).cuda()
            l_land = l_rand[:,:, idx]
            l_rand = l_rand.view(i,o,h,w)
            x = F.conv2d(x, l_rand, bias=self.layer2.bias, padding=1)
        else:
            x = self.layer2(x)
        return x
    
##### ChannelBlock
    
class ChannelBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        """
        Args:
            in_dim: the dimension of the feature map v ex) c3 of Resnet: 512
            hidden_dim: the dimension of the output of 1x1 conv (same as in_dim)
            out_dim: the dimension of the output of ChannelBlock (same as in_dim)
        Returns:
            output of the ChannelBlock g or \hat{g}
        """

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x, shuffle=False):
        """
        self.layer1 helps stable training. It is not shuffled during training.
        Args:
            x: input feature map ex) c3 or c5 of ResNet
            shuffle: set to True if \hat{g} is used. set to False if g is used
        Returns:
            output of the ChannelBlock g or \hat{g}
        """

        x = self.layer1(x)

        if shuffle: # channel shuffle
            i,o,h,w = self.conv2.weight.shape
            l_rand = self.conv2.weight.view(i,o,h*w)
            l_rand = l_rand[:, torch.randperm(l_rand.size()[1])]
            l_rand = l_rand.view(i,o,h,w)
            x = F.conv2d(x, l_rand, bias=self.conv2.bias)

        else:
            x = self.conv2(x)

        x = self.bn2(x)
        return x
    
class CROFFLE(nn.Module):
    def __init__(self, feature_dim, temperature=10):
        super(CROFFLE, self).__init__()
        """
        Args:
            feature_dim: the dimension of the input feature map ex) 512 for C3, 2048 for C5
            temperature: the temperature for the InfoNCE loss ex) 10
        """
        self.channelblock = ChannelBlock(feature_dim, feature_dim, feature_dim)
        self.spatialblock = SpatialBlock(feature_dim, feature_dim // 4, feature_dim)
        self.temp = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, v, v_prime):
        z1_v, z1_v_prime = self.channelblock(v), self.channelblock(v_prime)
        p1_v, p1_v_prime = self.spatialblock(z1_v), self.spatialblock(z1_v_prime)
        z1_fake, z1_fake_prime = self.channelblock(v, shuffle=True), self.channelblock(
            v_prime, shuffle=True
        )
        p1_fake, p1_fake_prime = self.spatialblock(
            z1_fake, shuffle=True
        ), self.spatialblock(z1_fake_prime, shuffle=True)

        pos1 = F.cosine_similarity(p1_v, z1_v.detach()).reshape(-1, 1)
        neg1 = F.cosine_similarity(p1_fake, z1_v.detach()).reshape(-1, 1)
        neg2 = F.cosine_similarity(p1_v, z1_fake.detach()).reshape(-1, 1)

        input = torch.stack((pos1, neg1, neg2), dim=1) / self.temp
        target = torch.zeros(input.shape[0], dtype=torch.long).cuda()
        Loss_croffle = self.ce(input, target)

        pos2 = F.cosine_similarity(p1_v_prime, z1_v_prime.detach()).reshape(-1, 1)
        neg3 = F.cosine_similarity(p1_fake_prime, z1_v_prime.detach()).reshape(-1, 1)
        neg4 = F.cosine_similarity(p1_v_prime, z1_fake_prime.detach()).reshape(-1, 1)

        input = torch.stack((pos2, neg3, neg4), dim=1) / self.temp
        target = torch.zeros(input.shape[0], dtype=torch.long).cuda()
        Loss_croffle = Loss_croffle + self.ce(input, target)

        return Loss_croffle / 2
