import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50



def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def feature_loss_function(fea, target_fea):
    target_fea = target_fea.detach() # 고민 (detach를 안하면 trivial 빠질거 같은데 하면 의미적으로 괜찮은지?)
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class projection_conv(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512, num_layer=2):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim,kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(hidden_dim, out_dim,kernel_size=1),
        #     nn.BatchNorm2d(out_dim)
        # )
    
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x, shuffle=False):
        x = self.layer1(x)

        if shuffle==True:
            i,o,h,w = self.conv2.weight.shape
            l_rand = self.conv2.weight.view(i,o,h*w) # [128, 32, 1]
            l_rand = l_rand[:,torch.randperm(l_rand.size()[1])]
            l_rand = l_rand.view(i,o,h,w)
            x = F.conv2d(x, l_rand,bias=self.conv2.bias)
        else:
            x = self.conv2(x)
        x = self.bn2(x)


        return x 

class prediction_conv(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, out_dim=512): # bottleneck structure
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # self.idxes = [torch.tensor([8,5,2,7,4,1,6,3,0]).cuda(),
        #     torch.tensor([0,3,6,1,4,7,2,5,8]).cuda(),
        #     torch.tensor([6,7,8,3,4,5,0,1,2]).cuda(),
        #     torch.tensor([2,1,0,5,4,3,8,7,6]).cuda()
        #     ]
        self.layer2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1)
    
    def forward(self, x, shuffle=False):
        x = self.layer1(x)

        if shuffle:
            i,o,h,w = self.layer2.weight.shape
            l_rand = self.layer2.weight.view(i,o,h*w)
            idx = torch.randperm(h*w).cuda()

            # idx = torch.tensor(idx).cuda()
            l_rand = l_rand[:,:, idx]
            l_rand = l_rand.view(i,o,h,w)

            x = F.conv2d(x, l_rand,bias=self.layer2.bias, padding=1)

            return x

        else:
            x = self.layer2(x)
            return x


class SimSiam(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential( # f encoder
            # self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()


        self.c3_projector = projection_conv(512,512,512)
        self.c3_predictor = prediction_conv(512,128,512)


        self.ce= nn.CrossEntropyLoss()


    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor

        c2,c3,c4,c5, x1 = self.backbone(x1)
        c2_prime, c3_prime, c4_prime, c5_prime,x2 = self.backbone(x2)
        
        
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2 
        L = L.mean()

        z1_c3, z1_c3_prime = self.c3_projector(c3), self.c3_projector(c3_prime)
        p1_c3, p1_c3_prime = self.c3_predictor(z1_c3), self.c3_predictor(z1_c3_prime)

        z1_fake, z1_fake_prime = self.c3_projector(c3,shuffle=True), self.c3_projector(c3_prime,shuffle=True)
        p1_fake , p1_fake_prime = self.c3_predictor(z1_fake, shuffle=True), self.c3_predictor(z1_fake_prime, shuffle=True)

        pos1 = F.cosine_similarity(p1_c3, z1_c3.detach())
        neg1 = F.cosine_similarity(p1_fake, z1_c3.detach())
        neg2 = F.cosine_similarity(p1_c3, z1_fake.detach())
        # print('neg:',neg.mean())
        # print('neg2:',neg2.mean())
        # print('pos:',pos.shape, 'neg:',neg.shape) # [64, 28, 28]
        pos1 = pos1.reshape(-1,) 
        neg1 = neg1.reshape(-1,)
        neg2 = neg2.reshape(-1,)

        input = torch.stack((pos1, neg1, neg2), dim=1) / 10
        # print('input:',input.shape) # [50176,2]
        target = torch.zeros(input.shape[0], dtype=torch.long).cuda()
        L4 = self.ce(input, target)

        pos2 =  F.cosine_similarity(p1_c3_prime, z1_c3_prime.detach())
        neg3 = F.cosine_similarity(p1_fake_prime, z1_c3_prime.detach())
        neg4 = F.cosine_similarity(p1_c3_prime, z1_fake_prime.detach())
        # print('neg3:',neg.mean())
        # print('neg4:',neg2.mean())
        # print('-'*20)
        pos2 = pos2.reshape(-1,)
        neg3 = neg3.reshape(-1,)
        neg4= neg4.reshape(-1,)
        
        input = torch.stack((pos2,neg3,neg4),dim=1) / 10
        target = torch.zeros(input.shape[0],dtype=torch.long).cuda()
        L4 = L4 + self.ce(input,target)

        return {'loss': L,'L_rand':L4/2}, {'pos1':pos1.mean(), 'neg1':neg1.mean(),
        'neg2':neg2.mean(),'pos2':pos2.mean(),'neg3':neg3.mean(),'neg4':neg4.mean()} 

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")

    z1 = torch.randn((200, 2560))
    z2 = torch.randn_like(z1)
    import time
    tic = time.time()
    print(D(z1, z2, version='original'))
    toc = time.time()
    print(toc - tic)
    tic = time.time()
    print(D(z1, z2, version='simplified'))
    toc = time.time()
    print(toc - tic)

# Output:
# tensor(-0.0010)
# 0.005159854888916016
# tensor(-0.0010)
# 0.0014872550964355469












