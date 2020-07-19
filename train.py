import torch
from torchvision.utils import make_grid

import utils
import IntroVAE128
import datasets

def load_data():

    '''read in param data'''
    pass

def train(model, optimizer, train_loader, margin, alpha, beta):

    '''run train'''

    model.train()

    for batch, images in enumerate(train_loader):

        # numbers correspond to pseudocode on page 6

        # decoder = generator
        # encoder (inference model) = discriminator

        # 3, get minibatch
        images = images.to(device)

        # begin by training encoder
        # freeze decoder params
        for param in model.encoder.parameters():
            param.requires_grad = True
        for param in model.decoder.parameters():
            param.requires_grad = False

        # 4, encode true data to latent
        z_mu, z_log_var = model.encoder(images)
        z = model.reparameterize(z_mu, z_log_var)

        # 5, random sample from prior p(z)
        z_p = torch.randn_like(z)

        # 6, decode encoded sample and from random
        x_p = model.decoder(z_p)
        x_r = model.decoder(z)

        # 7, mse reconstruction loss on real
        l_AE = model.lossAE(images, x_r)

        # 8, encode fake images, do not backprop through decoder
        mu_r, logVarR = model.encoder(x_r.detach())
        mu_pp, logVarPP = model.encoder(x_p.detach())

        # 9, discriminate, get loss on fake images
        lossEncReg = model.lossReg(z_mu, z_log_var) + alpha \
            * (torch.clamp((margin - model.lossReg(mu_r, logVarR)), 0) \
            +  torch.clamp((margin - model.lossReg(mu_pp, logVarPP)), 0))

        # 10, update encoder
        lossEnc = beta*l_AE + lossEncReg
        optimizer.zero_grad()
        lossEnc.backward()
        optimizer.step()

        # freeze encoder params
        # train decoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = True

        # 11, recompute x_r, x_p, and lossAE with detached z and z_p
        # not supposed to backprop through encoder
        # only backprop up to z and z_p. see appendix C
        x_r = model.decoder(z.detach())
        x_p = model.decoder(z_p.detach())
        l_AE = model.lossAE(images, x_r)
        mu_r, logVarR = model.encoder(x_r)
        mu_pp, logVarPP = model.encoder(x_p)

        # 12, decoder loss
        lossGenReg = alpha * (model.lossReg(mu_r, logVarR) + model.lossReg(mu_pp, logVarPP))

        # 13, update decoder
        lossGen = beta*l_AE + lossGenReg
        optimizer.zero_grad()
        lossGen.backward()
        optimizer.step()


def save_output():

    '''save reconstructed and generated images'''
    pass


def main():

    # load params

    # init logger

    # load data

    # train

    # output images

    pass

if __name__ == '__main__':


    main()
    pass
