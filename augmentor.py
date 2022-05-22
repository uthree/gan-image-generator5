import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

class ImageDataAugmentor:
    def __init__(self):
        self.functions = []
        self.probabilities = []
    
    def add_function(self, func, probability=0.125):
        self.functions.append(func)
        self.probabilities.append(probability)
    
    @torch.no_grad()
    def __call__(self, image):
        # image: B, C, H, W
        for func, probability in zip(self.functions, self.probabilities):
            if random.random() < probability:
                image = func(image)
        image = image.to(torch.float)
        return image
    
