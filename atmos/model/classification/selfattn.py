from torch import nn
from atmos.utils.transformer import SelfAttention
from atmos.model.basemodel import BaseModel
import torch.nn.functional as F


class SelfAttnClassification(BaseModel):
    def __init__(self, ctx, feature_size, label_size, **kwargs):
        super(SelfAttnClassification, self).__init__(ctx, feature_size, label_size)
        self.n_hidden = 128
        layers = [256, 128, 64]
        p = 0.4
        all_layers = []

        self.self_attn = SelfAttention(self.feature_size, self.config.d_model, 15, self.config.head)
        self.flatten = nn.Flatten()
        input_size = 15 * self.config.d_model

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            # all_layers.append(nn.BatchNorm1d(i))
            # all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], self.labels_size[0]))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x, **kwargs):
        x = F.relu(self.self_attn(x))  # torch.Size([32, 32, 15])
        x = self.flatten(x)  # torch.Size([32, 480])
        x = self.layers(x.squeeze(1))  # torch.Size([32, 4])
        # x = torch.argmax(x, dim=1) # torch.Size([32])
        # x = torch.sigmoid(x)
        return x

