from atmos.model.classification.mlp import MLPClassification
from atmos.model.classification.selfattn import SelfAttnClassification
from atmos.model.classification.transformer import TransformerClassification
from atmos.model.classification.cnn import CNNClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from atmos.model.classification.lstm import Res_Bidir_LSTMModel
from atmos.model.classification.resnet1d import ResNet18

str2classification = {
    "mlp": MLPClassification,
    "selfattn": SelfAttnClassification,
    "transformer": TransformerClassification,
    "cnn": CNNClassification,
    "random_forest": RandomForestClassifier,
    "gaussian_nb": GaussianNB,
    "lstm": Res_Bidir_LSTMModel,
    "resnet18": ResNet18
}

__all__ = ["str2classification"]
