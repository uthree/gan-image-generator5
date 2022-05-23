import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import revtorch as rv

import multiprocessing
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import random

class ForwardPassSettingWrapper(nn.Module):
    def __init__(self, module):
        super(ForwardPassSettingWrapper, self).__init__()
        self.module = module
        self.settings = {}
    def forward(self, *args):
        return self.module(*args, **self.settings)


class Conv2dMod(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, eps=1e-8, groups=1, demodulation=True):
        super(Conv2dMod, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels // groups, kernel_size, kernel_size, dtype=torch.float32))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') # initialize weight
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.demodulation = demodulation
        self.groups = groups

    def forward(self, x, y):
        # x: (batch_size, input_channels, H, W) 
        # y: (batch_size, output_channels)
        # self.weight: (output_channels, input_channels, kernel_size, kernel_size)
        N, C, H, W = x.shape
        
        # reshape weight
        w1 = y[:, None, :, None, None]
        w1 = w1.swapaxes(1, 2)
        w2 = self.weight[None, :, :, :, :]
        # modulate
        weight = w2 * (w1 + 1)

        # demodulate
        if self.demodulation:
            d = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weight = weight * d
        # weight: (batch_size, output_channels, input_channels, kernel_size, kernel_size)
        
        # reshape
        x = x.reshape(1, -1, H, W)
        _, _, *ws = weight.shape
        weight = weight.reshape(self.output_channels * N * self.groups, *ws)
        
        
        # padding
        x = F.pad(x, (self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2), mode='replicate')
        
        # convolution
        x = F.conv2d(x, weight, stride=1, padding=0, groups=N)
        x = x.reshape(N, self.output_channels, H, W)

        return x

class EqualLinear(nn.Module):
    def __init__(self, input_dim, output_dim, lr_mul=0.1):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.lr_mul = lr_mul
    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, self.bias *  self.lr_mul)

class MappingNetwork(nn.Module):
    def __init__(self, style_dim, num_layers=6):
        super(MappingNetwork, self).__init__()
        blocks = [rv.ReversibleBlock(
                nn.Sequential(
                    EqualLinear(style_dim, style_dim),
                    nn.GELU()
                    ),
                nn.Sequential(
                    EqualLinear(style_dim, style_dim),
                    nn.GELU()
                    ),
                split_along_dim=1
                ) for _ in range(num_layers)]
        self.seq = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x):
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        return x

class ChannelMLPMod(nn.Module):
    def __init__(self, channels, style_dim):
        super(ChannelMLPMod, self).__init__()
        self.c_mlp = Conv2dMod(channels, channels, 1)
        self.affine = nn.Linear(style_dim, channels)

    def forward(self, x, style=None):
        x = self.c_mlp(x, self.affine(style))
        return x

class Blur(nn.Module):
    def __init__(self):
        super(Blur, self).__init__()
        self.kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32)
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel[None, None, :, :]
    def forward(self, x):
        shape = x.shape
        # padding
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        # reshape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        # convolution
        x = F.conv2d(x, self.kernel.to(x.device), stride=1, padding=0, groups=x.shape[1])
        # reshape
        x = x.reshape(shape)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, style_dim, num_layers=6, upscale=True):
        super(GeneratorBlock, self).__init__()
        blocks = []
        self.mod_layers = []
        self.ch_conv = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.upscale = nn.Upsample(scale_factor=2) if upscale else nn.Identity()
        self.to_rgb = nn.Conv2d(output_channels, 3, 1, 1, 0)
        self.blur = Blur()
        for i in range(num_layers):
            mod_layer = ForwardPassSettingWrapper(ChannelMLPMod(output_channels, style_dim))
            b = rv.ReversibleBlock(
                    nn.Conv2d(output_channels, output_channels, 3, 1, 1),
                    nn.Sequential(
                        mod_layer,
                        nn.GELU(),
                        ),
                    split_along_dim=1
                    )
            blocks.append(b)
            self.mod_layers.append(mod_layer)
        self.seq = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x, y):
        x = self.ch_conv(x)
        x = self.upscale(x)
        for ml in self.mod_layers:
            ml.settings = {'style' : y}
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        x = self.blur(x)
        return x


class Generator(nn.Module):
    def __init__(self, style_dim=512, channels=[512, 512, 256, 256, 128, 64, 64, 32], num_layers_per_block=4):
        super(Generator, self).__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.num_layers_per_block = num_layers_per_block
        self.initial_image = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.last_layer_channels = channels[0]
        self.layers = nn.ModuleList([])
        self.add_layer(False)

    def forward(self, style):
        if type(style) != list:
            style = [style] * len(self.layers)
        x = self.initial_image.repeat(style[0].shape[0], 1, 1, 1)
        for i, l in enumerate(self.layers):
            x = l(x, style[i])
        rgb = self.layers[-1].to_rgb(x)
        return rgb

    def add_layer(self, upscale=True):
        idx = len(self.layers)
        ic, oc = self.channels[idx], self.channels[idx+1]
        self.layers.append(GeneratorBlock(ic, oc, self.style_dim, self.num_layers_per_block, upscale))
        self.last_layer_channels = oc

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=4, downscale=True):
        super(DiscriminatorBlock, self).__init__()
        self.downscale = nn.AvgPool2d(kernel_size=2) if downscale else nn.Identity()
        self.ch_conv = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.from_rgb = nn.Conv2d(3, input_channels, 1, 1, 0)
        blocks = nn.ModuleList([rv.ReversibleBlock(
                 nn.Conv2d(output_channels, output_channels, 3, 1, 1, groups=output_channels),
                 nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, 1, 1, 0),
                    nn.GELU()
                    ),
                split_along_dim=1
                ) for _ in range(num_layers)])
        self.seq = rv.ReversibleSequence(blocks)

    def forward(self, x):
        x = self.ch_conv(x)
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        x = self.downscale(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, channels=[32, 64, 64, 128, 256, 256, 512, 512], num_layers_per_block=4):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.layers = nn.ModuleList([])
        self.last_layer_channels = channels[-1]
        self.num_layers_per_block = num_layers_per_block
        self.judge = nn.Sequential(
                nn.Linear(channels[-1] + 2, 64),
                nn.GELU(),
                nn.Linear(64, 1)
                )
        self.pool4x = nn.AvgPool2d(kernel_size=4)
        self.add_layer(False)
        self.downscale = nn.Sequential(Blur(), nn.AvgPool2d(kernel_size=2))
    
    def add_layer(self, downscale=True):
        nl = len(self.layers)
        ic, oc = self.channels[-1-nl-1], self.channels[-1-nl]
        self.layers.insert(0, DiscriminatorBlock(ic, oc, self.num_layers_per_block, downscale))
        self.last_layer_channels = oc

    def forward(self, rgb):
        co_std = torch.std(rgb, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(rgb.shape[0], 1) # Color Std.

        x = self.layers[0].from_rgb(rgb)
        for i, l in enumerate(self.layers):
            if i == 1:
                x += l.from_rgb(self.downscale(rgb))
            x = l(x)
        mb_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1) # Minibatch Std.
        x = self.pool4x(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, mb_std, co_std], dim=1)
        x = self.judge(x)
        return x

class GAN(nn.Module):
    def __init__(self, style_dim=512):
        super(GAN, self).__init__()
        self.mapping_network = MappingNetwork(style_dim)
        self.generator = Generator(style_dim)
        self.discriminator = Discriminator()
        self.style_dim = style_dim
            
    def train_epoch(self, dataloader, optimizers, device, dtype=torch.float32,augment_func=nn.Identity):
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        self.mapping_network = self.mapping_network.to(device)
        opt_m, opt_g, opt_d = optimizers
        self.to(device)
        for i, real in enumerate(dataloader):
            if real.shape[0] < 2:
                continue
            real = real.to(device).to(dtype)
            N = real.shape[0]
            G, M, D = self.generator, self.mapping_network, self.discriminator
            G.train()
            M.train()
            D.train()
            T = augment_func
            L = len(G.layers)
            # train generator
            M.zero_grad()
            G.zero_grad()
            z1 = torch.randn(N, self.style_dim, device=device, dtype=dtype)
            z2 = torch.randn(N, self.style_dim, device=device, dtype=dtype)
            mid = random.randint(1, L)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            w = [w1] * mid + [w2] * (L-mid)
            fake = G(w)

            generator_loss = -D(fake).mean()
            generator_loss.backward()
            opt_m.step()
            opt_g.step()

            # train discriminator
            fake = fake.detach()
            D.zero_grad()
            discriminator_fake_loss = -torch.minimum(-D(T(fake))-1, torch.zeros(N, 1).to(device)).mean()
            discriminator_real_loss = -torch.minimum(D(T(real))-1, torch.zeros(N, 1).to(device)).mean()
            discriminator_loss = discriminator_fake_loss + discriminator_real_loss
            discriminator_loss.backward()

            # update parameter
            opt_d.step()
            
            tqdm.write(f"Batch: {i} G: {generator_loss.item():.4f} D: {discriminator_loss.item():.4f}")

    def train_resolution(self, dataset, num_epoch, batch_size, device, dtype=torch.float32, result_dir='./results/', model_path='./model.pt', augment_func=nn.Identity()):
        optimizers = (
                optim.Adam(self.mapping_network.parameters(), lr=1e-5),
                optim.Adam(self.generator.parameters(), lr=1e-5),
                optim.Adam(self.discriminator.parameters(), lr=1e-5),
                )
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
        for i in tqdm(range(num_epoch)):
            self.train_epoch(dataloader, optimizers, device, dtype=dtype, augment_func=augment_func)
            torch.save(self, model_path)

            # save image
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            path = os.path.join(result_dir, f"{i}.jpg")
            img = self.generator(self.mapping_network(torch.randn(1, self.style_dim, dtype=dtype, device=device)))
            img = img.cpu().detach().numpy() * 127.5 + 127.5
            img = img[0].transpose(1, 2, 0)
            img = img.astype(np.uint8)
            img = Image.fromarray(img, mode='RGB')
            img.save(path)

    def train(self, dataset,  num_epoch=100, batch_size=32, max_resolution=256, device=None, dtype=torch.float32, augment_func=nn.Identity()):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        resolution = -1
        while resolution <= max_resolution:
            num_layers = len(self.generator.layers)
            bs = batch_size // (2 ** max(num_layers-1, 0))
            if bs < 4:
                bs = 4
            resolution = 4 * (2 ** (num_layers-1))
            dataset.set_size(resolution)
            self.to(device)
            print(f"Start training with batch size: {bs}")
            self.train_resolution(dataset, num_epoch, bs, device, dtype, augment_func=augment_func)
            if resolution >= max_resolution:
                break
            self.generator.add_layer()
            self.discriminator.add_layer()

    def generate_random_image(self, num_images):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images = []
        for i in range(num_images):
            z1 = torch.randn(1, self.style_dim).to(device)
            z2 = torch.randn(1, self.style_dim).to(device)
            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)
            L = len(self.generator.layers)
            mid = random.randint(1, L)
            style = [w1] * mid + [w2] * (L-mid)
            image = self.generator(style)
            image = image.detach().cpu().numpy()
            images.append(image[0])
        return images

    def generate_random_image_to_directory(self, num_images, dir_path="./tests"):
        images = self.generate_random_image(num_images)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for i in range(num_images):
            image = images[i]
            image = np.transpose(image, (1, 2, 0))
            image = image * 127.5 + 127.5
            image = image.astype(np.uint8)
            image = Image.fromarray(image, mode='RGB')
            image = image.resize((256, 256))
            image.save(os.path.join(dir_path, f"{i}.png"))
