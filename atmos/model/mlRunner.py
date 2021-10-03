import os

from atmos.model.basemodel import Runner
from atmos.utils.inputter import load_data, save_hdf5_data
from os.path import join, basename
import numpy as np


_MODELS = {
    "random_forest": {
        'n_estimators': 1000,
        'n_jobs': 5,
        'verbose': 1
    },
    "gaussian_nb": {

    }
}


class MachineLearningRunner(Runner):
    def __init__(self, ctx):
        super(MachineLearningRunner, self).__init__(ctx)

    def build_model(self):
        model = self.load_model(self.context.checkpoint)

        if model is None:
            self.logging.info(f"Build a model ... ")
            model = self.context.get_model()(**_MODELS[self.context.classification])

        self.logging.info(f"The model architecture: \n {model}")
        return model

    def train(self):
        train_sample, train_label = self.train_dataset
        self.logging.info("Training the model ...")
        self.model.fit(train_sample, train_label)

        self.save_model(join(self.context.checkpoint_dir,
                             f"checkpoint_latest_{self.context.current_date}"))

        self.evaluate()

    def evaluate(self, epoch=0):
        self.logging.info("Evaluating ...")
        xtest, ytest = self.valid_dataset
        ypred = self.model.predict(xtest).reshape(-1, 1)
        self.performance_evaluate(ypred, ytest)

    def predict(self):
        self.logging.info("Predicting ...")
        for (test_data, test_label), test_data_name in zip(self.test_dataset, self.context.data_test):
            save_path = join(self.context.predict_dir,
                             f"predict_{self.context.version}_{basename(test_data_name)}")
            ypred = self.model.predict(test_data)
            self.logging.info(f"Save predict test result: {save_path}")
            save_hdf5_data(save_path, ypred)

            if np.unique(test_label).shape[0] > 1:
                self.performance_evaluate(ypred, test_label)

    def save_model(self, model_path, **kwargs):
        from joblib import dump
        model_path = f"{model_path}.joblib"
        self.logging.info(f"Save best model: {model_path}")
        dump(self.model, model_path)

    def load_model(self, model_path, model=None, **kwargs):
        if model_path and os.path.exists(model_path):
            from joblib import load
            self.logging.info(f"Loading the model from checkpoint: {self.context.checkpoint}")
            return load(model_path)
        else:
            return None
