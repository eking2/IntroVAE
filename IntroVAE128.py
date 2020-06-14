import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# resblock conv
def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     stride=1,
                     padding=0)


def conv3x3(in_channels, out_channels):

    # same padding
    p = (kernel_size - 1) / 2

    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=1,
                     padding=p)


class ResBlock_133(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv1x1(in_channels, out_channels)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)
        #self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = conv1x1(in_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        #out = self.bn2(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv3(out)
        #out = self.bn3(out)

        # reshape residual to have same out_channels as out
        if self.in_channels != self.out_channels:
            residual = self.downsample(residual)

        # add residual to output and activate
        out += residual

        return F.leaky_relu(out, 0.2)


class ResBlock_33(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(in_channels, out_channels)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = conv1x1(in_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        #out = self.bn2(out)

        if self.in_channels != self.out_channels:
            residual = self.downsample(residual)

        out += residual

        return F.leaky_relu(out, 0.2)


class Encoder(nn.Module):
    def __init__(self, z_size):
        super().__init__()

        # latent dim
        self.z_size = z_size

        # avgpool reduces img dimensions (HxW) by 1/2
        self.avgPool = nn.AvgPool2d(2)

        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels = 16,
                               kernel_size=5,
                               stride=1,
                               padding=2)

        self.res1 = ResidualBlock_133(16, 32)
        self.res2 = ResidualBlock_133(32, 64)
        self.res3 = ResidualBlock_133(64, 128)
        self.res4 = ResidualBlock_33(128, 256)
        self.res5 = ResidualBlock_33(256, 256)

        # mu, logvar
        self.fc1 = nn.Linear(4096, self.z_size)
        self.fc2 = nn.Linear(4096, self.z_size)

    def forward(self, x):

        # input is (batch, 3, 128, 128)

        out = F.leaky_relu(self.conv1(x), 0.2)
        out = self.avgPool(out)      # out = (batch, 16, 64, 64)
        out = self.res1(out)
        out = self.avgPool(out)      # out = (batch, 32, 32, 32)
        out = self.res2(out)
        out = self.avgPool(out)      # out = (batch, 64, 16, 16)
        out = self.res3(out)
        out = self.avgPool(out)      # out = (batch, 128, 8, 8)
        out = self.res4(out)
        out = self.avgPool(out)      # out = (batch, 256, 4, 4)
        out = self.res5(out)         # out = (batch, 256, 4, 4)

        out = out.view(-1, 256*4*4)  # reshape to (batch, 4096)

        mu = self.fc1(out)
        logvar = self.fc2(out)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_size):
        super(self).__init__()

        self.z_size = z_size

        self.upsample = nn.Upsample(scale_factor=2)

        self.fc1 = nn.Linear(self.z_size, 8192)

        self.res2 = ResidualBlock_33(512, 256)
        self.res3 = ResidualBlock_33(256, 128)
        self.res4 = ResidualBlock_33(128, 64)
        self.res5 = ResidualBlock_133(64, 32)
        self.res6 = ResidualBlock_133(32, 16)
        self.res7 = ResidualBlock_33(16, 16)

        self.conv8 = nn.Conv2d(in_channels = 16,
                               out_channels = 3,
                               kernel_size=5,
                               stride=1,
                               padding=2)


    def forward(self, x):

    	# input is (batch, 256)

        out = F.relu(self.fc1(x))      # out = (batch, 8192)
        out = out.view(-1, 512, 4, 4)  # out = (batch, 512, 4, 4)

        out = self.res2(out)
        out = self.upsample(out)       # out = (batch, 256, 8, 8)
        out = self.res3(out)
        out = self.upsample(out)       # out = (batch, 128, 16, 16)
        out = self.res4(out)
        out = self.upsample(out)       # out = (batch, 64, 32, 32)
        out = self.res5(out)
        out = self.upsample(out)       # out = (batch, 32, 64, 64)
        out = self.res6(out)
        out = self.upsample(out)       # out = (batch, 16, 128, 128)
        out = self.res7(out)           # out = (batch, 16, 128, 128)

        # no activation
        out = self.conv8(out)          # out = (batch, 3, 128, 128)

        return out


class IntroVAE(nn.Module):
    def __init__(self, z_size=256):

        super(self).__init__()

        self.z_size = z_size

        self.encoder = Encoder(z_size)
        self.decoder = Decoder(z_size)


    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std


    def forward(self, x):

        # encode
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        # reconstruct
        rec_x = self.decoder(z)

        return rec_x

    def ae_loss(self, x, x_rec):

        '''reconstruction loss'''

        criterion = nn.MSELoss(reduction='sum')
        loss = 0.5 * criterion(x, x_rec)

        return loss/x.shape[0]

    def reg_loss(self, mu, logvar):

        '''kl-divergence to gaussian'''

        loss = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))

        return loss/mu.shape[0]
