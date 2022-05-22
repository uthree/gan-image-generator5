import torch
import torch.nn as nn
import torch.nn.functional as F
import revtorch as rv

class ForwardPassSettingWrapper(nn.Module):
    def __init__(self, module):
        super(ForwardPassSettingWrapper, self).__init__()
        self.module = module
        self.settings = {}
    def forward(self, *args):
        return self.module(*args, **self.settings)


# for S2-MLP type models.
# input: [batch_size, channels, height, width]
class SpatialShift2d(nn.Module):
    def __init__(
            self,
            shift_directions=[[1,0], [-1,0], [0,1], [0,-1]], # List of shift directions. [X, Y], if you need to identity mapping, set this value [0, 0]
            padding_mode='replicate',
            ):
        super(SpatialShift2d, self).__init__()
        # caluclate padding range. I like one-line code lol.
        (l, r), (t, b) = [[f(d) for f in [lambda a:abs(min(a+[0])), lambda b:abs(max(*b+[0]))]] for d in [list(c) for c in zip(*shift_directions)]]
        self.pad_size = [l, r, t, b]
        self.num_directions = len(shift_directions)
        self.shift_directions = shift_directions
        self.padding_mode = padding_mode

    def forward(
            self,
            x,
            ):
        x = F.pad(x, self.pad_size, mode=self.padding_mode) # pad
        # caluclate channels of each section
        c = x.shape[1]
        sections = [c//self.num_directions]*self.num_directions
        # concatenate remainder of channels to last section
        sections[-1] += c % self.num_directions
        # save height and width
        h, w = x.shape[2], x.shape[3]

        # split
        x = torch.split(x, sections, dim=1)

        l,r,t,b = self.pad_size
        # clip each sections.
        x = torch.cat([s[:, :, t:h-b, l:w-r] for (s, d) in zip(x, self.shift_directions)], dim=1)
        return x

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

class MappingNetwork(nn.Module):
    def __init__(self, style_dim, num_layers=4):
        super(MappingNetwork, self).__init__()
        blocks = [rv.ReversibleBlock(
                nn.Sequential(
                    nn.Linear(style_dim, style_dim),
                    nn.GELU()
                    ),
                nn.Sequential(
                    nn.Linear(style_dim, style_dim),
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

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, style_dim, num_layers=4, upscale=True):
        super(GeneratorBlock, self).__init__()
        blocks = []
        self.mod_layers = []
        self.ch_conv = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.upscale = nn.Upsample(scale_factor=2) if upscale else nn.Identity()
        self.to_rgb = nn.Conv2d(output_channels, 3, 1, 1, 0)
        for i in range(num_layers):
            mod_layer = ForwardPassSettingWrapper(ChannelMLPMod(output_channels, style_dim))
            b = rv.ReversibleBlock(
                    SpatialShift2d(),
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
        return x

class Generator(nn.Module):
    def __init__(self, style_dim=512, channels=[512, 512, 256, 256, 128, 64, 32], num_layer_per_blocks=4):
        super(Generator, self).__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.num_layers_per_blocks = num_layer_per_blocks
        self.initial_image = nn.Parameter(torch.rand(1, channels[0], 4, 4))
        self.last_layer_channels = channels[0]
        self.layers = nn.ModuleList([])
        self.add_layer(False)

    def forward(self, style):
        x = self.initial_image.repeat(style.shape[0], 1, 1, 1)
        for l in self.layers:
            x = l(x, style)
        return self.layers[-1].to_rgb(x)

    def add_layer(self, upscale=True):
        idx = len(self.layers)
        ic, oc = self.channels[idx], self.channels[idx+1]
        self.layers.append(GeneratorBlock(ic, oc, self.style_dim, self.num_layers_per_blocks, upscale))
        self.last_layer_channels = oc

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, num_layers=4, downscale=True):
        super(DiscriminatorBlock, self).__init__()
        self.downscale = nn.AvgPool2d(kernel_size=2) if downscale else nn.Identity()
        self.ch_conv = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        self.from_rgb = nn.Conv2d(3, input_channels, 1, 1, 0)
        blocks = nn.ModuleList([rv.ReversibleBlock(
                SpatialShift2d(),
                nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, 1, 1, 0),
                    nn.GELU()
                    ),
                split_along_dim=1
                ) for _ in range(num_layers)])
        self.seq = rv.ReversibleSequence(blocks)

    def forward(self, x):
        x = self.ch_conv(x)
        x = self.downscale(x)
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        return x

class Discriminator(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 256, 512, 512], num_layers_per_block=4):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.layers = nn.ModuleList([])
        self.last_layer_channels = channels[-1]
        self.num_layers_per_block = num_layers_per_block
        self.judge = nn.Sequential(
                nn.Linear(channels[-1] + 1, 64),
                nn.GELU(),
                nn.Linear(64, 1)
                )
        self.pool4x = nn.AvgPool2d(kernel_size=4)
        self.add_layer(False)
    
    def add_layer(self, downscale=False):
        nl = len(self.layers)
        ic, oc = self.channels[-1-nl], self.channels[-1-nl-1]
        self.layers.insert(0, DiscriminatorBlock(ic, oc, self.num_layers_per_block, downscale))
        self.last_layer_channels = oc

    def forward(self, x):
        x = self.layers[0].from_rgb(x)
        for l in self.layers:
            x = l(x)
        mb_std = torch.std(x, dim=[0], keepdim=False).mean().unsqueeze(0).repeat(x.shape[0], 1) # Minibatch Std.
        x = self.pool4x(x)
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, mb_std], dim=1)
        x = self.judge(x)
        return x


