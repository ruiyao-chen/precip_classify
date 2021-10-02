from atmos.utils.transformer import TransformerEncoder
from atmos.model.basemodel import BaseModel


class TransformerClassification(BaseModel):
    def __init__(self, ctx, feature_size, label_size, **kwargs):
        super(TransformerClassification, self).__init__(ctx, feature_size, label_size)
        self.labels_size = label_size

        self.encoder = TransformerEncoder(self.feature_size, self.labels_size, self.feature_size, self.config)

    def forward(self, x, **kwargs):
        output = self.encoder(x)
        return output


