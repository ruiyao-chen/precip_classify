from atmos.model.dlRunner import DeepLearningRunner
from atmos.model.mlRunner import MachineLearningRunner
RunnerModel = {
    "selfattn": DeepLearningRunner,
    "mlp": DeepLearningRunner,
    "transformer": DeepLearningRunner,
    "lstm": DeepLearningRunner,
    "resnet18": DeepLearningRunner,
    "random_forest": MachineLearningRunner,
    "gaussian_nb": MachineLearningRunner
}


def build_runner(ctx):
    return RunnerModel[ctx.classification](ctx)


__all__ = ['build_runner', 'DeepLearningRunner', 'MachineLearningRunner']


