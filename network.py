import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F

#### two convolution operations one that doubles the number of channels from in_ch to out_ch and another goes from out_ch to out_ch
#### 2D convolutions with kernel size 3, stride 1, padding 1 followed by ReLU
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

#### self.enc_block is a list of block operations(doubling the input channels), to the output of every block is performed MaxPool2d
#### with stride 2 and padding 0
### MaxPool2d is performed between two Block operations
class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2, 2, 0)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


#### self.dec_blocks  list of Decoder blocks that perform two conv operations + ReLU
#### The self.upconvs is a list of ConvTranspose2d operations  with stride 2 and padding 0, that perform the up-convolution operations
#### the forward function, the decoder accepts the encoder_features which were output by the Encoder to 
#### perform the concatenation operation before passing the result to Block
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2, 0) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            #enc_ftrs = self.crop(encoder_features[i], x)
            enc_ftrs = encoder_features[i]
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x
    
    #def crop(self, enc_ftrs, x):
        #_, _, H, W = x.shape
        #enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        #return enc_ftrs


#encoder = Encoder()
# input image
#x = torch.randn(1, 3, 572, 572)
#ftrs = encoder(x)
#for ftr in ftrs: print("Encoder: ",ftr.shape)
    
#decoder = Decoder()
#x_1 = torch.randn(1, 1024, 35, 35)
#print("Decoder: ",decoder(x_1, ftrs[::-1][1:]).shape)


#### last convolutional layer Conv 1-1, with Encoder and Decoder parts
class U_Net(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(512, 384)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        return out
### testing to see if its correct architecure
#unet = UNet()
#x = torch.randn(1, 3, 512, 384)
#print(unet(x).shape)
