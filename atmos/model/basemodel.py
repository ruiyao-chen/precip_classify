from torch import nn
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from os.path import join
import h5py



class BaseModel(nn.Module):
    def __init__(self, ctx, feature_size, label_size):
        super(BaseModel, self).__init__()
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.feature_size = feature_size
        self.labels_size = label_size


class Runner:
    def __init__(self, ctx):
        super(Runner, self).__init__()
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.task = ctx.task
        self.device = ctx.device

        self.multi_label = True if self.config.loss_func == 'multi_loss' else False

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

    def load_data(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self, epoch):
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
