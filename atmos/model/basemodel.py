from torch import nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from atmos.utils.inputter import load_data
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, ctx, feature_size, label_size):
        super(BaseModel, self).__init__()
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.feature_size = feature_size
        self.labels_size = label_size
        self.multi_label = self.config.multi_label


class Runner:
    def __init__(self, ctx):
        super(Runner, self).__init__()
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.task = ctx.task
        self.device = ctx.device

        self.multi_label = self.config.multi_label

        self.feature_size = 1
        self.labels_size = 1
        self.train_dim = 0
        self.valid_dim = 0

        # The current training step count
        self._step = 0
        self.start_epoch = 0

        # Loading the datasets and make them work in [self.device]
        self.train_dataset, self.valid_dataset, self.test_dataset = self.load_data()

        # build model and make it work using [self.device]
        self.model = self.build_model()

        self.best_model = None
        self.latest_model = None

    def build_model(self):
        raise NotImplementedError

    def get_labels_stats(self, task_labels, data="train"):
        self.logging.info(f"The stats of label for {data} dataset")
        for idx in range(task_labels.shape[-1]):
            labels = task_labels[:, idx]
            nums_instance = len(labels)
            for label in np.unique(labels):
                cnt = np.count_nonzero(labels==label)
                self.logging.info(f"[{data}]-[{idx}] Percentage of Label {label}: {cnt / nums_instance: .4f}% ({cnt} / {nums_instance})")

    def get_test_data(self):
        for test_path in self.context.data_test:
            test_sample, test_label, test_dim = load_data(test_path, 'test',
                                                          self.context.data_test_bias,
                                                          multi_label=self.multi_label)
            yield test_sample, test_label

    def load_data(self):

        train_ds = None
        valid_sample, valid_label, self.valid_dim = load_data(self.context.data_valid,
                                                              'valid',
                                                              multi_label=self.multi_label)
        valid_ds = (valid_sample, valid_label)
        self.feature_size = valid_sample.shape[-1]
        self.labels_size = []
        for idx in range(valid_label.shape[-1]):
            self.labels_size.append(np.unique(valid_label[:, idx]).shape[0])

        self.step_size = valid_sample.shape[1]

        self.get_labels_stats(valid_label, "valid")

        if self.task in ["train", "run"]:
            train_sample, train_label, self.train_dim = load_data(self.context.data_train, 'train', multi_label=self.multi_label)

            self.get_labels_stats(train_label, "train")

            # Get input dataset feature size
            if train_sample.shape[1] != valid_sample.shape[1]:
                self.logging.error(f"The size feature of training[{train_sample.shape[1]}] and "
                                   f"testing[{valid_sample.shape[1]}] dataset does not match")
                raise ValueError

            # Build training iterated datasets for training process.
            train_ds = (train_sample, train_label)

        return train_ds, valid_ds, self.get_test_data()

    def train(self):
        raise NotImplementedError

    def evaluate(self, valid_dataset, epoch, **kwargs):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def performance_evaluate(self, ypred, ytest, idx=0):
        ypred = ypred.reshape(-1, 1)
        ytest = ytest.reshape(-1, 1)

        class_correct = list(0. for _ in range(self.labels_size[idx]))
        class_total = list(0. for _ in range(self.labels_size[idx]))

        c = (ypred == ytest)
        for i in range(ypred.shape[0]):
            label = ytest[i, 0]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        accuracy = accuracy_score(ytest, ypred)
        if self.labels_size[idx] > 2:
            f1 = f1_score(ytest, ypred, labels=None, average="macro")
        else:
            f1 = f1_score(ytest, ypred, labels=None, average="binary")

        recall = recall_score(ytest, ypred, average='weighted')
        precision = precision_score(ytest, ypred, average='weighted')

        self.logging.info(f"Task {idx}, Accuracy: {accuracy:.4f}, "
                          f"F1 Score: {f1:.4f}, "
                          f"Recall Score: {recall:.4f}, "
                          f"Precision Score: {precision:.4f}")

        for i in range(self.labels_size[idx]):
            self.logging.info(f"\t|- Accuracy of '{i}': "
                              f"{int(class_correct[i])} / "
                              f"{int(class_total[i])} ({100 * class_correct[i] / class_total[i]:0.4f} %)")

        stats = {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

        return stats

    def submit(self):

        if self.context.task == 'run':
            self.train()

            self.load_model(self.best_model, self.model)

            self.predict()

        if self.context.task == 'train':
            self.train()

        if self.context.task == 'predict':
            self.predict()



    def save_model(self, model_path, **kwargs):
        raise NotImplementedError

    def load_model(self, model_path, model=None, **kwargs):
        raise NotImplementedError
