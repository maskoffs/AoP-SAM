import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MutiScale_Conv_Brach(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(3, 64, kernel_size=1, bias=False),
            LayerNorm2d(64),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList()
        self.conv_branch = nn.ModuleList()

        kernel = [7, 5, 3, 7, 5]
        for i in range(3):
            in_c = 64 * (2 ** i)
            out_c = 64 * (2 ** (i + 1))
            block = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_c, out_c, kernel_size=kernel[i], padding=(kernel[i] - 1) // 2, bias=False),
                LayerNorm2d(out_c),
                nn.GELU(),
            )
            self.blocks.append(block)
            conv_branch1 = nn.ModuleList()
            conv_branch1.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_c, out_c, kernel_size=kernel[i + 1], padding=(kernel[i + 1] - 1) // 2, bias=False),
                    LayerNorm2d(out_c),
                    nn.GELU(),
                )
            )
            conv_branch1.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_c, out_c, kernel_size=kernel[i + 2], padding=(kernel[i + 2] - 1) // 2, bias=False),
                    LayerNorm2d(out_c),
                    nn.GELU(),
                )
            )
            self.conv_branch.append(conv_branch1)

        self.cat = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(out_c, out_c * 2, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_c * 2),
            nn.GELU(),
            nn.ConvTranspose2d(out_c * 2, out_c, kernel_size=2, stride=2),
            LayerNorm2d(out_c),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(2 * out_c, 1024, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        for i in range(3):
            x1 = self.blocks[i](x)
            x2 = self.conv_branch[i][0](x)
            x3 = self.conv_branch[i][1](x)
            x = x1 + x2 + x3
        x_hat = self.cat(x)
        x = self.out_conv(torch.cat([x, x_hat], dim=1))
        return x.permute(0, 2, 3, 1)