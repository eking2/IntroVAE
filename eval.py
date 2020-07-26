#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
#import pygifsicle 


def parse_args():

    # setup subparsers

    # make gifs
    # plot losses
    # interpolate

    pass


def parse_log(log):

    '''parse losses from log file

    Args:
        log (str) : path to log file
    '''
    
    lines = Path(log).read_text().splitlines()

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

    df = pd.DataFrame(data, columns=['ae_loss', 'e_real', 'e_rec', 'e_sample', 'g_rec', 'g_sample'])
    df['epoch'] = range(1, len(df) + 1)

    name = Path(log).stem
    df.to_csv(f'logs/{name}_losses.csv')


def plot_losses(loss_df):

    '''plot losses over training epochs

    Args:
        loss_df (str) : csv file with losses
    '''

    name = Path(loss_df).stem.rsplit('_', 1)[0]
    df = pd.read_csv(loss_df)

    fig = plt.figure(figsize=(6, 4))

    plt.plot(df['epoch'], df['ae_loss'], label='MSE autoencoder')
    plt.plot(df['epoch'], df['e_real'], label='KLD enc_real')
    plt.plot(df['epoch'], df['e_rec'], label='KLD enc_recon')
    plt.plot(df['epoch'], df['e_sample'], label='KLD enc_sample')
    plt.plot(df['epoch'], df['g_rec'], label='KLD gen_recon')
    plt.plot(df['epoch'], df['g_sample'], label='KLD gen_sample')

    plt.grid(alpha=0.2)
    plt.title(f'{name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'logs/{name}_losses.png', bbox_inches='tight', dpi=300)

def make_gif(glob_pat):

    '''make gif for set of images

    Args:
        glob_pat (str) : glob pattern to match image names on
    '''

    # glob all images and sort by epoch
    # subsample to make smaller gif size
    out_images = list(Path('output').glob(f'*{glob_pat}*'))
    out_images = sorted(out_images, key=lambda x: int(x.stem.split('_')[2]))[::5]

    images = []
    for out_image in out_images:
        # resize
        im = Image.open(out_image)
        images.append(im)

    # duration in ms
    images[0].save(f'{glob_pat}.gif', save_all=True, append_images=images[1:], 
                   loop=0, duration=1000) 
    #pygifsicle.optimize(f'{glob_pat}.gif')



#parse_log('logs/introvae_128.log')
#plot_losses('logs/introvae_128_losses.csv')
make_gif('recon')




