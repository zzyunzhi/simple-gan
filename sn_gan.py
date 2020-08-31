import torch.nn as nn
import torch


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output


# Spatial Upsampling with Nearest Neighbors
class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super().__init__()
        self.depth_to_space = DepthToSpace(block_size=2)
        self.conv_block = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)  # (b, c, h, w) -> (b, c*4, h, w)
        x = self.depth_to_space(x)  # (b, c, h*2, w*2)
        out = self.conv_block(x)
        return out


# Spatial Downsampling with Spatial Mean Pooling
class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super().__init__()
        self.space_to_depth = SpaceToDepth(2)
        self.conv_block = nn.Conv2d(in_dim, out_dim, kernel_size,
                                    stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.space_to_depth(x)
        x = torch.sum(torch.stack(x.chunk(4, dim=1), dim=0), dim=0) / 4.0
        out = self.conv_block(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=1),
        )

    def forward(self, x):
        residual = self.residual_block(x)
        shortcut = x
        return residual + shortcut


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            Upsample_Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=1)
        )
        self.shortcut = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        residual = self.residual_block(x)
        shortcut = self.shortcut(x)
        return residual + shortcut


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            Downsample_Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=1),
        )
        self.shortcut = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        residual = self.residual_block(x)
        shortcut = self.shortcut(x)
        return residual + shortcut


# Architectures follow Table 4 of SN-GAN

class Generator(nn.Module):
    def __init__(self, latent_dim=128, n_filters=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 4 * 4 * 256)
        self.res_blocks = nn.Sequential(
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(-1, 256, 4, 4)
        x = self.res_blocks(x)
        out = self.conv_block(x)  # (b, 3, h, w)
        return out

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, dtype=torch.float32)


class Discriminator(nn.Module):
    def __init__(self, n_filters=128):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResnetBlockDown(in_dim=3, n_filters=n_filters),
            ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
            ResBlock(in_dim=n_filters, n_filters=n_filters),
            ResBlock(in_dim=n_filters, n_filters=n_filters),
            nn.ReLU(),
        )
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x):
        x = self.res_blocks(x)
        # global sum pooling
        x = torch.sum(x, dim=(2, 3))
        out = self.fc(x)
        return out
