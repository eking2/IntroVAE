import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn.functional as F

import utils
import IntroVAE128
from pathlib import Path
import matplotlib.pyplot as plt
import datasets
import argparse
import logging
import time

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config path')

    return parser.parse_args()


def load_data(batch_size, transforms=None):

    '''setup dataloader'''

    dataset = datasets.CelebA(transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def train(model, optimizer, dataloader, margin, alpha, beta, log_interval, epoch, device):

    '''run train'''

    model.train()

    for batch, images in enumerate(dataloader):

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

        # 6, decode encoded sample and from random sample
        x_p = model.decoder(z_p)
        x_r = model.decoder(z)

        # 7, mse reconstruction loss on real
        ae_loss = model.ae_loss(images, x_r)

        # 8, encode fake images, do not backprop through decoder
        z_mu_r, z_log_var_r = model.encoder(x_r.detach())
        z_mu_pp, z_log_var_pp = model.encoder(x_p.detach())

        # 9, discriminate, get loss on fake images
        e_real = model.reg_loss(z_mu, z_log_var)
        e_rec = model.reg_loss(z_mu_r, z_log_var_r)
        e_sample = model.reg_loss(z_mu_pp, z_log_var_pp)

        enc_reg_loss = e_real + alpha * (F.relu(margin - e_rec) +  F.relu(margin - e_sample))

        # 10, update encoder
        enc_loss = beta*ae_loss + enc_reg_loss
        optimizer.zero_grad()
        enc_loss.backward()
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
        ae_loss = model.ae_loss(images, x_r)
        z_mu_r, z_log_var_r = model.encoder(x_r)
        z_mu_pp, z_log_var_pp = model.encoder(x_p)

        # 12, decoder loss
        g_rec = model.reg_loss(z_mu_r, z_log_var_r)
        g_sample = model.reg_loss(z_mu_pp, z_log_var_pp)

        gen_reg_loss = alpha * (g_rec + g_sample)

        # 13, update decoder
        gen_loss = beta*ae_loss + gen_reg_loss
        optimizer.zero_grad()
        gen_loss.backward()
        optimizer.step()

        #logging.info(f'{enc_loss.item():.2f}, {gen_loss.item():.2f}')

        # log losses
        if batch % log_interval == 0:
            logging.info(f'Epoch: {epoch}, Batch: {batch}, Encoder Loss: {enc_loss.item():.3f}, Generator Loss: {gen_loss.item():.3f}')

    # epoch loss decomposition
    logging.info(f'ae_loss: {beta*ae_loss.item():.3f}, e_real: {e_real.item():.3f}, e_rec: {e_rec.item():.3f}, e_sample: {e_sample.item():.3f}, g_rec: {g_rec.item():.3f}, g_sample: {g_sample.item():.3f}')


def save_output(model, images, z_dim, epoch, device, z_p, num_images):

    '''save reconstructed and generated images'''

    # make output dir if missing
    if not Path('output').exists():
        Path('output').mkdir()

    model.eval()
    with torch.no_grad():

        # images sampled from p(z)
        # should use a fixed sample vector or random each time?
        #z_p = torch.randn(num_images, z_dim).to(device)
        x_p = model.decoder(z_p)
        img_grid = vutils.make_grid(x_p, nrow=8).to('cpu')
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Epoch {epoch}\nSampled')
        plt.savefig(f'output/sampled_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
        plt.close()

        # reconstructed images
        x_r = model(images)
        recon_samples = torch.cat([images[:num_images,...], x_r[:num_images,...]], dim=0)
        img_grid = vutils.make_grid(recon_samples, nrow=8).to('cpu')
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Epoch {epoch}\nReconstructed')
        plt.savefig(f'output/recon_epoch_{epoch}.png', bbox_inches='tight', dpi=300)
        plt.close()


# to access attributes of the wrapped module
# https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975
#class CustomDataParallel(nn.DataParallel):
#    def __getattr__(self, name):

#        try:
#            return super().__getattr__(name)
#        except AttributeError:
#            return getattr(self.module, name)


def main():


    # parse args
    args = parse_args()
    assert Path(args.config).exists(), f'invalid config file: {args.config}'

    # load params
    params = utils.parse_params(args.config)

    # init logger
    utils.init_logger(params.experiment_name)
    logging.info(params)

    # load data
    trans = transforms.ToTensor()
    dataloader = load_data(transforms=trans, batch_size=params.batch_size)

    # init model and optimizer
    # setup for multigpu
    model = IntroVAE128.IntroVAE(z_size=params.z_dim)

    #if torch.cuda.device_count() > 1:
    #    logging.info('Running DataParallel')
    #    model = CustomDataParallel(model)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Number of GPUs: {torch.cuda.device_count()}')
    logging.info(f'Device: {device}')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    train_start = time.time()
    z_p = torch.randn(params.num_images*2, params.z_dim).to(device)

    for epoch in range(1, params.num_epochs + 1):

        epoch_start = time.time()
        logging.info(f'Epoch: {epoch} start')
        train(model, optimizer, dataloader, params.margin, params.alpha, params.beta,
              params.log_interval, epoch, device)

        # save model every checkpoint interval
        if epoch % params.checkpoint_interval == 0:
            utils.save_checkpoint(model, optimizer, epoch, params.experiment_name)

            # output images
            # draw images to reconstruct from dataloader
            images = next(iter(dataloader))
            images = images.to(device)
            save_output(model, images, params.z_dim, epoch, device, z_p, params.num_images)

        epoch_time = (time.time() - epoch_start) / 60
        logging.info(f'Epoch {epoch} time: {epoch_time:.2f}m')

    total_train = (time.time() - train_start) / 60
    logging.info(f'Trained over {total_train:.2f}m')


if __name__ == '__main__':

    main()
