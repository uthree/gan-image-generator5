import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

def flip(image):
    return torch.flip(image, [3])

def contrast(image):
    return image * (1.0 + random.uniform(-0.2, 0.2))

def brightness(image):
    return image * (1.0 + random.uniform(-0.2, 0.2))

def gaussian_noise(image):
    return image + torch.randn(*image.shape).to(image.device) * 0.2

def random_roll(image):
    H, W = image.shape[-2:]
    p = H // 8
    h, w = random.randint(-p, p), random.randint(-p, p)
    image = torch.roll(image, (h, w), (-2, -1))
    return image
