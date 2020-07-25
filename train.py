# Training

# Importing Required Libraries

from train_utils import imshow,save_checkpoint_location,load_data train_model

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import shutil, argparse
import matplotlib.pyplot as plt

def main():

    test_loaders=False

    inputs = argparse.Argumentinputs(description='Train a new network on a data set')

    inputs.add_argument('data_dir', type=str, \
       )
    inputs.add_argument('--save_dir', type=str, \
        help='Directory to save checkpoints')
    inputs.add_argument('--arch', type=str, \
       )
    inputs.add_argument('--learning_rate', type=float, \
        )
    inputs.add_argument('--hidden_units', type=int, \
        )
    inputs.add_argument('--epochs', type=int, \
      )
    inputs.add_argument('--gpu', action='store_true', \
        )
    inputs.add_argument('--save_every', type=int, \
       )
    
    args, _ = inputs.parse_known_args()

    data_dir = args.data_dir

    save_dir = './'
    if args.save_dir:
        save_dir = args.save_dir

    arch = 'densenet121'
    if args.arch:
        arch = args.arch

    learning_rate = 0.01
    if args.learning_rate:
        learning_rate = args.learning_rate

    hidden_units = 200
    if args.hidden_units:
        hidden_units = args.hidden_units

    epochs = 7
    if args.epochs:
        epochs = args.epochs

    save_every = 50
    if args.save_every:
        save_every = args.save_every

    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("Warning! GPU flag was set however no GPU is available in \
                the machine")

    trainloader, validloader, testloader = load_data(data_dir)

    # Test loaders
    if test_loaders:
        images, labels = next(iter(trainloader))
        imshow(images[2])
        plt.show()

        images, labels = next(iter(validloader))
        imshow(images[2])
        plt.show()

        images, labels = next(iter(testloader))
        imshow(images[2])
        plt.show()

    train_model(trainloader, validloader, arch=arch, hidden_units=hidden_units,\
     learning_rate=learning_rate, cuda=cuda, epochs=epochs, save_dir=save_dir, \
     save_every=save_every)



if __name__ == '__main__':
    main()