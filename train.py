import torch
import torch.optim as optim
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
        enc_reg_loss = model.reg_loss(z_mu, z_log_var) \
            + alpha * (F.relu(margin - model.reg_loss(z_mu_r, z_log_var_r))
                    +  F.relu(margin - model.reg_loss(z_mu_pp, z_log_var_pp)))

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
        gen_reg_loss = alpha * (model.reg_loss(z_mu_r, z_log_var_r) + model.reg_loss(z_mu_pp, z_log_var_pp))

        # 13, update decoder
        gen_loss = beta*ae_loss + gen_reg_loss
        optimizer.zero_grad()
        gen_loss.backward()
        optimizer.step()

        #logging.info(f'{enc_loss.item():.2f}, {gen_loss.item():.2f}')

        # log losses
        if batch % log_interval == 0:
            logging.info(f'Epoch: {epoch}, Batch: {batch}, Encoder Loss: {enc_loss.item():.3f}, Generator Loss: {gen_loss.item():.3f}')


def save_output(model, images, z_dim, epoch, device, num_images=8):

    '''save reconstructed and generated images'''

    model.eval()
    with torch.no_grad():

        # images sampled from p(z)
        # should use a fixed sample vector or random each time?
        z_p = torch.randn(num_images, z_dim).to(device)
        x_p = model.decoder(z_p)
        img_grid = vutils.make_grid(x_p, nrow=4).to('cpu')
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'output/sampled_epoch_{epoch}.png')
        plt.close()

        # reconstructed images
        x_r = model(images)
        recon_samples = torch.cat([images[:num_images,...], x_r[:num_images,...]], dim=0)
        img_grid = vutils.make_grid(recon_samples, nrow=4).to('cpu')
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'output/recon_epoch_{epoch}.png')
        plt.close()


def main():

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse args
    args = parse_args()
    assert Path(args.config).exists(), f'invalid config file: {args.config}'

    # load params
    params = utils.parse_params(args.config)

    # init logger
    utils.init_logger(params.experiment_name)
    logging.info(params)
    logging.info(f'Device: {device}')

    # load data
    trans = transforms.ToTensor()
    dataloader = load_data(transforms=trans, batch_size=params.batch_size)

    # init model and optimizer
    # setup for multigpu
    model = IntroVAE128.IntroVAE(z_size=params.z_dim)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(IntroVAE128.IntroVAE(z_size=params.z_dim))

    logging.info(f'Number of GPUs: {torch.cuda.device_count()}')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    train_start = time.time()
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
            save_output(model, images, params.z_dim, epoch, device)

        epoch_time = (time.time() - epoch_start) / 60
        logging.info(f'Epoch {epoch} time: {epoch_time:.2f}m')

    total_train = (time.time() - train_start) / 60
    logging.info('Trained over {total_train:.2f}m')


if __name__ == '__main__':

    main()
