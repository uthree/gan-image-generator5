from model import GAN
from dataset import ImageDataset
import torch
import os
import sys

if os.path.exists('model.pt'):
    model = torch.load('model.pt')
    print("Loaded model")
else:
    print("Creating new model...")
    model = GAN()
    print("Created new model")

model.generate_random_image_to_directory(int(sys.argv[1]))
