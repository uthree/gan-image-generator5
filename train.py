from model import GAN
from dataset import ImageDataset
import torch
import os
import sys
from augmentations import *
from augmentor import ImageDataAugmentor

aug = ImageDataAugmentor()
aug.add_function(flip, probability=0.5)
aug.add_function(random_roll, probability=0.5)

ds = ImageDataset(sys.argv[1:], max_len=5000, background_resize=False)
if os.path.exists('model.pt'):
    model = torch.load('model.pt')
    print("Loaded model")
else:
    print("Creating new model...")
    model = GAN(style_dim=512)
    print("Created new model")
model.train(ds, num_epoch=20, dtype=torch.float32, augment_func=aug, batch_size=256)
