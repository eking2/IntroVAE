from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image


class CelebA(Dataset):

    def __init__(self, train=True, transforms=None):

        # download from
        # https://www.dropbox.com/s/ayz2roywuq253l0/celebA-HQ-128x128.tar.gz?dl=0
        if train:
            self.images = list(Path('../celebA-HQ-128x128/train').rglob('*.npy'))
        else:
            self.images = list(Path('../celebA-HQ-128x128/test').rglob('*.npy'))

        self.transforms = transforms


    def __getitem__(self, idx):

        # npy to image
        img = np.load(self.images[idx]).squeeze().astype(np.uint8)
        img = Image.fromarray(img)

        if self.transforms:
            return self.transforms(img)

        return img

    def __len__(self):

        return len(self.images)
