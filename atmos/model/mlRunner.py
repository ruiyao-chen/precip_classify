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
        if self.context.checkpoint and os.path.exists(self.context.checkpoint):
            self.logging.info(f"Loading the model from checkpoint: {self.context.checkpoint}")
            model = self.load_model(self.context.checkpoint)
        else:
            model = self.context.get_model()(**_MODELS[self.context.classification])

        self.logging.info(f"The model architecture: \n {model}")
        return model

    def get_test_data(self):
        for test_path in self.context.data_test:
            yield load_data(test_path, 'test', self.context.data_test_bias)

    def load_data(self):
        valid_sample, valid_label, self.valid_dim = load_data(self.context.data_valid, 'test')      # (46991, 16), (46991, 1)

        self.feature_size = valid_sample.shape[-1]
        self.labels_size = []

        for idx in range(valid_label.shape[-1]):
            self.labels_size.append(np.unique(valid_label[:, idx]).shape[0])

        if self.task == "train":
            # (187963, 16), (187963, 1)
            train_sample, train_label, self.train_dim = load_data(self.context.data_train, 'train')

            # Get input dataset feature size
            if train_sample.shape[1] != valid_sample.shape[1]:
                self.logging.error(f"The size feature of training[{train_sample.shape[1]}] and "
                                   f"testing[{valid_sample.shape[1]}] dataset does not match")
                raise ValueError

            # Build training iterated datasets for training process.
            train_ds = (train_sample, train_label)

            valid_ds = (valid_sample, valid_label)
        else:
            train_ds = None
            valid_ds = None

        return train_ds, valid_ds, self.get_test_data()

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
        for (test_data, test_label, test_dim), test_data_name in zip(self.test_dataset, self.context.data_test):
            save_path = join(self.context.predict_dir,
                             f"predict_{self.context.version}_{basename(test_data_name)}")
            ypred = self.model.predict(test_data)
            self.logging.info(f"Save predict test result: {save_path}")
            save_hdf5_data(save_path, ypred.reshape(test_dim))

    def save_model(self, model_path, **kwargs):
        from joblib import dump
        model_path = f"{model_path}.joblib"
        self.logging.info(f"Save best model: {model_path}")
        dump(self.model, model_path)

    def load_model(self, model_path, model=None, **kwargs):
        from joblib import load
        return load(model_path)
