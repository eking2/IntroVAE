#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from IntroVAE128 import IntroVAE
import torch
import matplotlib.pyplot as plt
import utils
import torchvision.utils as vutils
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import yaml

#import pygifsicle


def parse_args():

    parser = argparse.ArgumentParser()

    # make gifs
    parser.add_argument('-g', '--gif', choices=['sample', 'recon'],
                        help='make gifs of selected image set from output')
    parser.add_argument('-s', '--subsample', type=int, default=5,
            help='subsample stride for images (default: 5)')

    # plot losses
    parser.add_argument('-p', '--plot', type=str,
                        help='path for log file to plot losses')

    # interpolate
    parser.add_argument('-i', '--interpolate', type=str,
                        help='checkpoint to interpolate in latent dim')
    parser.add_argument('-c', '--count', type=int, default=6,
            help='number of samples to generate with interpolation (default: 6)')
    parser.add_argument('-w', '--windows', type=int, default=8,
            help='number of intermediate windows/steps to sample (default: 8)')

    # ms-ssim score to measure diversity
    parser.add_argument('-m', '--ms_ssim', type=str,
                        help='checkpoint to calculate mssim score')
    parser.add_argument('-n', '--num_ms_ssim', type=int, default=1000,
            help='number of images to generate for ms-ssim scoring (default: 1000)')

    return parser.parse_args()


def parse_log(log):

    '''parse losses from log file

    Args:
        log (str) : log file stem name

    Returns:
        loss_df (dataframe) : df with epoch losses parsed from log file
    '''

    lines = Path(f'logs/{log}.log').read_text().splitlines()

    data = []
    for line in lines:
        if 'ae_loss' in line:

            ae_loss = line.split()[4][:-1]
            e_real = line.split()[6][:-1]
            e_rec = line.split()[8][:-1]
            e_sample = line.split()[10][:-1]
            g_rec = line.split()[12][:-1]
            g_sample = line.split()[14][:-1]

            data.append([ae_loss, e_real, e_rec, e_sample, g_rec, g_sample])

    loss_df = pd.DataFrame(data, columns=['ae_loss', 'e_real', 'e_rec', 'e_sample', 'g_rec', 'g_sample'])
    loss_df['epoch'] = range(1, len(loss_df) + 1)

    loss_df.to_csv(f'assets/{log}_losses.csv', index=False)


def plot_losses(log):

    '''plot losses over training epochs

    Args:
        log (str) : log file stem name
    '''

    loss_df = f'assets/{log}_losses.csv'
    df = pd.read_csv(loss_df)

    fig = plt.figure(figsize=(6, 4))

    plt.plot(df['epoch'], df['ae_loss'], label='MSE autoencoder')
    plt.plot(df['epoch'], df['e_real'], label='KLD enc_real')
    plt.plot(df['epoch'], df['e_rec'], label='KLD enc_recon')
    plt.plot(df['epoch'], df['e_sample'], label='KLD enc_sample')
    plt.plot(df['epoch'], df['g_rec'], label='KLD gen_recon')
    plt.plot(df['epoch'], df['g_sample'], label='KLD gen_sample')

    plt.grid(alpha=0.2)
    plt.title(log)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'assets/{log}_losses.png', bbox_inches='tight', dpi=300)

def make_gif(glob_pat, subsample):

    '''make gif for set of images

    Args:
        glob_pat (str) : glob pattern to match image names on
        subsample (int) : stepsize for images to keep
    '''

    # glob all images and sort by epoch
    # subsample to make smaller gif size
    out_images = list(Path('output').glob(f'*{glob_pat}*'))
    out_images = sorted(out_images, key=lambda x: int(x.stem.split('_')[2]))[::subsample]

    images = []
    for out_image in out_images:
        # resize
        im = Image.open(out_image)
        images.append(im)

    # duration in ms
    images[0].save(f'assets/{glob_pat}.gif', save_all=True, append_images=images[1:],
                   loop=0, duration=1000)
    #pygifsicle.optimize(f'{glob_pat}.gif')


def run_single_interp(dataloader, model, steps):

    '''interpolate between points in latent space from trained model

    Args:
        dataloader (dataloader) : loader with real images, select 2 images from here
        model (model) : vae model to encode and decode image tensors
        steps (int) : number of intermediate steps to sample images

    Returns:
        recon (tensor) : reconstructed images from interpolated sequence
    '''

    # get random images from dataloader
    images = next(iter(dataloader))

    # real images
    image_1 = images[0].unsqueeze(0)
    image_2 = images[1].unsqueeze(0)

    # to latent space
    z_mu, z_log_var = model.encoder(images)

    # select positions in latent dim
    # must have at least 2 images in batch
    z_1 = z_mu[0]
    z_2 = z_mu[1]

    # sample at equally spaced steps between selected points
    z_interp = []

    for i in range(0, steps):
        interp = z_1 * (steps - i)/steps + z_2 * (i/steps)
        z_interp.append(interp)

    # each image goes to batch dim
    z_interp = torch.stack(z_interp, dim=0)

    # decode
    recon = model.decoder(z_interp)

    # real images on ends
    recon = torch.cat([image_1, recon, image_2], dim=0)

    return recon


def plot_interp(n_samples, model, checkpoint, steps):

    '''plot interpolated images

    Args:
        n_samples (int) : number of samples
        model (model) : vae model to encode and decode image tensors
        checkpoint (str) : path to model checkpoint
        steps (int) : number of intermediate steps to sample images
    '''

    # setup dataloader to get images
    trans = transforms.ToTensor()
    dataset = datasets.CelebA(transforms=trans)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # load checkpoint
    utils.load_checkpoint(f'checkpoints/{checkpoint}.pt', model)

    # interp
    interps = []
    for i in range(n_samples):
        recon = run_single_interp(dataloader, model, steps)
        interps.append(recon)

    # plot
    interps = torch.cat(interps, dim=0)
    grid = vutils.make_grid(interps, nrow=steps + 2)
    plt.imshow(grid.permute(1, 2, 0).detach().numpy())
    plt.axis('off')
    name = Path(checkpoint).stem
    plt.savefig(f'assets/{name}_interp.png', bbox_inches='tight', dpi=300)


def calc_msssim(n_images, model, checkpoint):

    '''calculate image diversity with ms-ssim (multi-scale structural similarity) metric and reconstruction quality with rmse.
       reconstuctions (x_r), samples (x_p).

    Args:
        n_images (int) : number of images to generate to compare against real
        model (model) : vae model to encode and decode image tensors
        checkpoint (str) : stem for checkpoint file to load

    Returns:
        avg_msssim (float) : average ms-ssim score for generated images (smaller is more diverse)
        avg_rmse (float) : average root mean square error for recon to real (smaller is more accurate)
    '''

    # dataloader
    # normalizes to [0-1]
    trans = transforms.ToTensor()
    dataset = datasets.CelebA(transforms = trans)
    dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.load_checkpoint(f'checkpoints/{checkpoint}.pt', model)
    model = model.to(device)

    # loop over loader and calc ms-ssim and rmse for each batch until hits n_images
    image_count = 0
    m_scores = []
    rmse_scores = []

    while image_count < n_images:
        for images in dataloader:

            images = images.to(device)

            # real image recon
            z_mu, z_log_var = model.encoder(images)
            z = model.reparameterize(z_mu, z_log_var)
            x_r = model.decoder(z)

            # calc rmse
            # sum of all pixel loss per image
            rmse = torch.sqrt(F.mse_loss(x_r, images, reduction='sum')) / len(images)
            rmse_scores.append(rmse.item())

            # sampled
            z_p = torch.randn_like(z)
            x_p = model.decoder(z_p)

            # calc ms-ssim from recon to sampled
            # good model will have low similarity between recon and sampled, indicates not memorizing
            m = ms_ssim(x_r, x_p, data_range=1.0, size_average=True, win_size=7)
            m_scores.append(m.item())

            # up counter
            image_count += len(images)

    # averaged results
    avg_msssim = sum(m_scores) / len(m_scores)
    avg_rmse = sum(rmse_scores) / len(rmse_scores)

    # write
    output = {'checkpoint' : checkpoint,
              'ms_ssim' : avg_msssim,
              'avg_rmse' : avg_rmse}

    Path(f'assets/{checkpoint}.yaml').write_text(yaml.dump(output))


def main():

    args = parse_args()

    if not Path('assets').exists():
        Path('assets').mkdir()

    # make gifs
    if args.gif:
        make_gif(args.gif, args.subsample)

    # plot losses
    if args.plot:
        parse_log(args.plot)
        plot_losses(args.plot)

    # interpolate
    if args.interpolate:
        model = IntroVAE()
        plot_interp(args.count, model, args.interpolate, args.windows)

    # ms-ssim
    if args.ms_ssim:
        model = IntroVAE()
        calc_msssim(args.num_ms_ssim, model, args.ms_ssim)


if __name__ == '__main__':

    main()


