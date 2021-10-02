from torch import nn
import torch.nn.functional as F
from atmos.model.basemodel import BaseModel


class CNNClassification(BaseModel):
    def __init__(self, ctx, feature_size, label_size, **kwargs):
        super(CNNClassification, self).__init__(ctx, feature_size, label_size)
        self.n_hidden = 128
        layers = [256, 128, 64]
        p = 0.4
        all_layers = []

        self.conv1d = nn.Conv1d(1, 32, 3)
        self.flatten = nn.Flatten()
        self.l1 = nn.Lin
        input_size = (self.feature_size - 3 + 1) * 32

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], self.labels_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x, **kwargs):
        x = F.relu(self.conv1d(x.unsqueeze(1)))  # torch.Size([32, 32, 15])
        x = self.flatten(x)  # torch.Size([32, 480])
        x = self.layers(x)  # torch.Size([32, 4])
        # x = torch.argmax(x, dim=1) # torch.Size([32])
        # x = torch.sigmoid(x)
        return x
