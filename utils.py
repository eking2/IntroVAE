import yaml
import logging
import torch
from pathlib import Path

class dotdict(dict):

    '''dot.notation access to dict attributes'''

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_params(yaml_path):

    '''Read hyperparameters from config yaml'''

    content = Path(yaml_path).read_text()
    params = yaml.safe_load(content)

    return dotdict(params)


def init_logger(log_file):

    '''setup logger to print to screen and save to file'''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handlers = [logging.StreamHandler(),
                logging.FileHandler(f'logs/{log_file}.log', 'a')]

    # do not print millisec
    fmt = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s',
                            "%Y-%m-%d %H:%M:%S")

    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)

    return logger


def save_checkpoint(model, optimizer, epoch, file_name, delete=True):

    '''save model state dict and optimizer state dict'''

    # remove last checkpoint file
    if delete:

        # first checkpoint will not have file
        try:
            old_check = list(Path('checkpoint').glob('*.pt'))[0]
            old_check.unlink()
        except:
            pass

    torch.save({'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()},
        f'checkpoints/{file_name}_{epoch}.pt')


def load_checkpoint(checkpoint, model, optimizer=None):

    '''load model state dict to continue training or evaluate'''

    check = Path(f'checkpoints/{checkpoint}.pt')
    if not check.exists():
        raise(f'File does not exist {check}')

    check = torch.load(check)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


