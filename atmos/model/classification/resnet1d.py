"""
The code is based on the original ResNet implementation from torchvision.models.resnet
"""

import torch
import torch.nn as nn
from atmos.model.basemodel import BaseModel
import torch.nn.functional as F
from atmos.utils.aggregation_zeropad import LocalConvolution


class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, dilation=1):
        super(CotLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv1d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv1d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm1d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm1d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm1d(dim)
        act = nn.SiLU
        self.act = act(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv1d(dim, attn_chs, 1),
            nn.BatchNorm1d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv1d(attn_chs, self.radix * dim, 1)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        k = self.key_embed(x)  # torch.Size([20, 16, 50]) --> torch.Size([20, 16, 50])
        qk = torch.cat([x, k], dim=1)  # torch.Size([20, 32, 50])
        b, c, qk_hh = qk.size()

        w = self.embed(qk)  # torch.Size([20, 32, 50]) -->  torch.Size([20, 18, 50])
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh)  # torch.Size([20, 1, 2, 9, 50])

        x = self.conv1x1(x)  # torch.Size([20, 16, 50]) --> torch.Size([20, 16, 50])
        x = self.local_conv(x.unsqueeze(-1), w.unsqueeze(-1)).squeeze(-1)  # torch.Size([20, 16, 50])
        x = self.bn(x)
        x = self.act(x)

        B, C, H = x.shape
        x = x.view(B, C, 1, H)  # torch.Size([20, 16, 1, 50])
        k = k.view(B, C, 1, H)  # torch.Size([20, 16, 1, 50])
        x = torch.cat([x, k], dim=2)  # torch.Size([20, 16, 2, 50])

        x_gap = x.sum(dim=2)  # torch.Size([20, 16, 50])
        x_gap = x_gap.mean(axis=2, keepdim=True)  # torch.Size([20, 16, 1])
        x_attn = self.se(x_gap)  # torch.Size([20, 32, 1])
        x_attn = x_attn.view(B, C, self.radix)  # torch.Size([20, 16, 2])
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1))).sum(dim=2)  # torch.Size([20, 16, 50])

        out = self.fusion(out)

        return out.contiguous()


class CoXtLayer(nn.Module):
    def __init__(self, dim, kernel_size, stride=1, dilation=1):
        super(CoXtLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv1d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=8, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )

        self.dw_group = 2
        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv1d(2 * dim, dim // factor, 1, groups=self.dw_group, bias=False),
            nn.BatchNorm1d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1, groups=self.dw_group),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.dw_group, bias=False),
            nn.BatchNorm1d(dim)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.SiLU(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv1d(dim, attn_chs, 1),
            nn.BatchNorm1d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv1d(attn_chs, self.radix * dim, 1)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        batch_size, channels, height = x.size()  # torch.Size([20, 16, 50])
        k = self.key_embed(x)  # torch.Size([20, 16, 50])
        qk = torch.cat([x.unsqueeze(2), k.unsqueeze(2)], dim=2)  # torch.Size([20, 16, 2, 50])
        qk = qk.view(batch_size, -1, height)  # torch.Size([20, 32, 50])

        w = self.embed(qk)  # torch.Size([20, 18, 50])
        # torch.Size([40, 1, 1, 9, 50])
        w = w.view(batch_size * self.dw_group, 1, -1, self.kernel_size * self.kernel_size, height)

        x = self.conv1x1(x)  # torch.Size([20, 16, 50])
        x = x.view(batch_size * self.dw_group, -1, height)  # torch.Size([40, 8, 50])
        x = self.local_conv(x.unsqueeze(-1), w.unsqueeze(-1)).squeeze(-1)
        x = x.view(batch_size, -1, height)
        x = self.bn(x)
        x = self.act(x)

        B, C, H = x.shape
        x = x.view(B, C, 1, H)
        k = k.view(B, C, 1, H)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean(axis=2, keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1))).sum(dim=2)

        out = self.fusion(out)

        return out.contiguous()


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16, act_layer=nn.ReLU, min_channels=8, reduction_channels=None):
        super(SEModule, self).__init__()
        reduction_channels = reduction_channels or max(channels // reduction, min_channels)
        self.fc1 = nn.Conv1d(channels, reduction_channels, kernel_size=1, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv1d(reduction_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean(axis=2, keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


def conv3(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, args, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.args = args
        self.conv1 = conv3(in_planes, out_planes, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        if args.resnet_local_atts == 'CotLayer':
            self.conv2 = CotLayer(out_planes, kernel_size, stride, dilation)
        elif args.resnet_local_atts == 'CoXtLayer':
            self.conv2 = CoXtLayer(out_planes, kernel_size, stride, dilation)
        else:
            self.conv2 = conv3(out_planes, out_planes, kernel_size)
            self.bn2 = nn.BatchNorm1d(out_planes)

        self.stride = stride

        if args.resnet_atts and args.resnet_atts == 'se':
            self.se = SEModule(out_planes * self.expansion)
        else:
            self.se = None

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.args.resnet_local_atts == 'Conv1d':
            out = self.conv2(out)
            out = self.bn2(out)
        else:
            out = self.conv2(out)  # torch.Size([20, 16, 50])

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, args, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)

        if args.resnet_local_atts == 'CotLayer':
            self.conv2 = CotLayer(out_planes, kernel_size, stride, dilation)
        elif args.resnet_local_atts == 'CoXtLayer':
            self.conv2 = CoXtLayer(out_planes, kernel_size, stride, dilation)
        else:
            self.conv2 = conv3(out_planes, out_planes, kernel_size, stride, dilation)
            self.bn2 = nn.BatchNorm1d(out_planes)

        self.conv3 = nn.Conv1d(out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if args.resnet_atts and args.resnet_atts == 'se':
            self.se = SEModule(out_planes * self.expansion)
        else:
            self.se = None

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.args.resnet_local_atts == 'Conv1d':
            # Conv1d(16, 16, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
            out = self.conv2(out)  # torch.Size([20, 16, 50])
            out = self.bn2(out)
            out = self.relu(out)
        else:
            out = self.conv2(out)  # torch.Size([20, 64, 99])

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)  # torch.Size([20, 256, 99])

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FCOutputModule(nn.Module):
    """
    Fully connected output module.
    """

    def __init__(self, in_planes, num_outputs, **kwargs):
        """
        Constructor for a fully connected output layer.

        Args:
          in_planes: number of planes (channels) of the layer immediately proceeding the output module.
          num_outputs: number of output predictions.
          fc_dim: dimension of the fully connected layer.
          dropout: the keep probability of the dropout layer
          trans_planes: (optional) number of planes of the transition convolutional layer.
        """
        super(FCOutputModule, self).__init__()
        fc_dim = kwargs.get('fc_dim', 1024)
        dropout = kwargs.get('dropout', 0.5)
        in_dim = kwargs.get('in_dim', 7)
        trans_planes = kwargs.get('trans_planes', None)
        if trans_planes is not None:
            self.transition = nn.Sequential(
                nn.Conv1d(in_planes, trans_planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(trans_planes))
            in_planes = trans_planes
        else:
            self.transition = None

        self.fc = nn.Sequential(
            nn.Linear(in_planes * in_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_outputs))

    def get_dropout(self):
        return [m for m in self.fc if isinstance(m, torch.nn.Dropout)]

    def forward(self, x):
        if self.transition is not None:
            x = self.transition(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


class GlobAvgOutputModule(nn.Module):
    """
    Global average output module.
    """

    def __init__(self, in_planes, num_outputs):
        super(GlobAvgOutputModule, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_planes, num_outputs)

    def get_dropout(self):
        return []

    def forward(self, x):
        x = self.avg()
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResNet1D(nn.Module):
    def __init__(self, config, num_inputs, num_outputs, block_type, group_sizes, base_plane=64, output_block=None,
                 zero_init_residual=False, **kwargs):
        super(ResNet1D, self).__init__()
        self.config = config
        self.base_plane = base_plane
        self.inplanes = self.base_plane

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(num_inputs, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual groups
        self.planes = [self.base_plane * (2 ** i) for i in range(len(group_sizes))]
        kernel_size = kwargs.get('kernel_size', 3)
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        groups = [self._make_residual_group1d(block_type,
                                              self.planes[i],
                                              kernel_size,
                                              group_sizes[i],
                                              strides[i],
                                              dilations[i]) for i in range(len(group_sizes))]
        self.residual_groups = nn.Sequential(*groups)

        # Output module
        if output_block is None:
            self.output_block = GlobAvgOutputModule(self.planes[-1] * block_type.expansion, num_outputs)
        else:
            self.output_block = output_block(self.planes[-1] * block_type.expansion, num_outputs, **kwargs)

        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block_type, planes, kernel_size, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block_type.expansion))
        layers = [block_type(self.config, self.inplanes, planes, kernel_size=kernel_size,
                             stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.config, self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    # src.shape: torch.Size([1024, 6, 200])
    def forward(self, src, tgt=None):
        x = self.input_block(src)  # x.shape: torch.Size([128, 13, 3])  --> torch.Size([128, 64, 1])
        x = self.residual_groups(x)  # x.shape: torch.Size([128, 64, 1]) --> torch.Size([128, 512, 1])
        x = self.output_block(x)  # x.shape: torch.Size([128, 4])
        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# def build_model(model_type, input_channel, output_channel):
#     if model_type == 'resnet18':
#         network = ResNet1D(args.input_channel, args.output_channel, BasicBlock1D, [2, 2, 2, 2],
#                            base_plane=64, output_block=FCOutputModule, kernel_size=3, **args.fc_config)
#         args.logger.info("ResNet18 Network")
#     elif model_type == 'resnet50':
#         # For 1D network, the Bottleneck structure results in 2x more parameters, therefore we stick to BasicBlock.
#         args.fc_config['fc_dim'] = 1024
#         network = ResNet1D(args.input_channel, args.output_channel, BasicBlock1D, [3, 4, 6, 3],
#                            base_plane=64, output_block=FCOutputModule, kernel_size=3, **args.fc_config)
#         args.logger.info("ResNet50 Network")
#     elif args.type == 'resnet101':
#         args.fc_config['fc_dim'] = 1024
#         network = ResNet1D(args.input_channel, args.output_channel, BasicBlock1D, [3, 4, 23, 3],
#                            base_plane=64, output_block=FCOutputModule, **args.fc_config)
#         args.logger.info("ResNet101 Network")

# Source and target input embedding
class Embeddings(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_col):
        super(Embeddings, self).__init__()
        self.cov1D = nn.ConvTranspose1d(input_dim, n_col, output_dim)
        self.batchNorm = nn.BatchNorm1d(n_col)

    def forward(self, x):
        """
        :param x: The input 3D dimension (batch_size, windows_size, nums_channel)
        :return: The output with 3D (batch_size, windows_size, d_model)
        """
        x = self.cov1D(x.unsqueeze(dim=-1))

        return self.batchNorm(x)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # y = A*x + B,
        # where x.Size = d_model is input dimension,
        # and y.size = vocab is output dimension
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # return F.softmax(self.proj(x), dim=-1)
        return F.log_softmax(self.proj(x), dim=-1)


class ResNet18(BaseModel):
    def __init__(self, ctx, feature_size, label_size, **kwargs):
        super(ResNet18, self).__init__(ctx, feature_size, label_size)
        self.config = ctx.config
        step_size = kwargs.get("step_size", 3)
        fc_config = {
            'fc_dim': 512,
            'in_dim': step_size // 32 + 1,
            'dropout': 0.5,
            'trans_planes': 128
        }
        num_outputs = 256
        block_type = BasicBlock1D if self.config.resnet_block == 'Block' else Bottleneck1D

        self.network = ResNet1D(self.config, feature_size, num_outputs, block_type, self.config.resnet_layers,
                                base_plane=64, output_block=FCOutputModule,
                                kernel_size=3, **fc_config)

        self.gen1 = Generator(num_outputs, label_size[0])

        self.gen2 = Generator(num_outputs, label_size[1])

    def forward(self, x, **kwargs):

        if x.ndim == 2:
            x = x.unsqueeze(1)
        out = self.network(x.transpose(1, -1))  # torch.Size([128, 256])
        a = self.gen1(out)
        b = self.gen2(out)
        return a, b
