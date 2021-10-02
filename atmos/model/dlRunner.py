from atmos.model.basemodel import Runner
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from atmos.utils.inputter import load_data, save_hdf5_data
from os.path import join, basename
import numpy as np
from torch import Tensor
import time
from torch.autograd import Variable
import torch.nn.functional as F
from atmos.utils.earlystop import EarlyStopping, scorers_from_opts
import h5py
import os


class CustomMultiLossLayer(nn.Module):
    def __init__(self, device, nb_outputs=2, l1_loss=None, l2_loss=None, **kwargs):
        super(CustomMultiLossLayer, self).__init__()
        self.nb_outputs = nb_outputs
        self.log_vars = []
        self.lossL1 = torch.nn.CrossEntropyLoss() if l1_loss is None else l1_loss
        self.lossL2 = torch.nn.CrossEntropyLoss() if l2_loss is None else l2_loss
        self.device = device

        # Weight definition for each loss
        for i in range(self.nb_outputs):
            self.log_vars += [torch.nn.Parameter(torch.zeros(1), requires_grad=True).to(self.device)]

    def multi_loss(self, ys_pred, ys_true):
        # Relative velocity loss
        precision = torch.exp(-self.log_vars[0][0]).to(self.device)
        loss = precision * self.lossL1(ys_pred[0], ys_true[:, 0]) + self.log_vars[0][0]

        # Relative quaternion loss
        precision = torch.exp(-self.log_vars[1][0]).to(self.device)
        loss += precision * self.lossL2(ys_pred[1], ys_true[:, 1]) + self.log_vars[1][0]
        return loss

    def forward(self, ys_preds, ys_true):
        return self.multi_loss(ys_preds, ys_true)


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, ignore_index, alpha=None, gamma=2, isLearnable=True, reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        super(FocalLoss, self).__init__()
        if isLearnable:
            self.alpha = Variable(torch.ones(class_num, 1), requires_grad=True)
        else:
            self.alpha = Variable(torch.ones(class_num, 1).fill_(alpha), requires_grad=False)
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.class_num = class_num
        self.reduction = reduction

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'sum':
            loss = batch_loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class DeepLearningRunner(Runner):
    def __init__(self, ctx):
        super(DeepLearningRunner, self).__init__(ctx)

        if self.task in ['train', 'run']:
            # The optimizer for training the model
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            # The regression loss function to calcuate the loss
            self.criterion = self.get_criterion(ctx.config)

            if ctx.config.early_stopping > 0:
                # initialize the early_stopping object
                self.early_stopping = EarlyStopping(tolerance=ctx.config.early_stopping,
                                                    logger=ctx.logger,
                                                    scorers=scorers_from_opts(ctx.config))

    def get_criterion(self, opt):

        if opt.loss_func == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
            self.logging.info("Loss function: cross entropy")
        elif opt.loss_func == 'multi_loss':
            criterion = CustomMultiLossLayer(opt.device)
            self.logging.info("Loss function: Multiple Task cross entropy")

        elif opt.loss_func == "focal_loss":
            criterion = FocalLoss(class_num=self.labels_size,
                                  ignore_index=-1,
                                  alpha=opt.focal_loss,
                                  gamma=opt.focal_loss_gamma,
                                  isLearnable=opt.focal_loss_learnable,
                                  reduction='sum')
            self.logging.info("Loss function: Focal Loss")
        else:
            criterion = nn.CrossEntropyLoss()
            self.logging.info("Loss function: Cross Entropy")
        return criterion

    def build_model(self):
        model = self.context.get_model()(self.context,
                                         self.feature_size,
                                         self.labels_size,
                                         step_size=self.step_size).to(self.device)
        self.logging.info(f"The model architecture: \n {model}")
        if self.config.classification == "lstm":
            from atmos.model.classification.lstm import init_weights
            model.apply(init_weights)

        if self.context.checkpoint and os.path.exists(self.context.checkpoint):
            self.start_epoch = self.load_model(self.context.checkpoint, model)
        return model

    @property
    def get_current_step(self):
        return self._step

    def get_test_data(self):
        for test_path in self.context.data_test:
            test_sample, test_label, test_dim = load_data(test_path, 'test',
                                                          self.context.data_test_bias, multi_label=self.multi_label)
            yield (DataLoader(TensorDataset(Tensor(test_sample).to(self.device),
                                            Tensor(test_label).type(torch.int64).squeeze(-1).to(self.device)),
                              batch_size=self.config.batch_size, shuffle=False), test_dim)

    def get_labels_stats(self, task_labels, data="train"):
        self.logging.info(f"The stats of label for {data} dataset")
        for idx in range(task_labels.shape[-1]):
            labels = task_labels[:, idx]
            nums_instance = len(labels)
            for label in np.unique(labels):
                cnt = np.count_nonzero(labels==label)
                self.logging.info(f"[{data}]-[{idx}] Percentage of Label {label}: {cnt / nums_instance: .4f}% ({cnt} / {nums_instance})")

    def load_data(self):

        valid_sample, valid_label, self.valid_dim = load_data(self.context.data_valid, 'valid', multi_label=self.multi_label)
        self.feature_size = valid_sample.shape[-1]
        self.labels_size = []
        for idx in range(valid_label.shape[-1]):
            self.labels_size.append(np.unique(valid_label[:, idx]).shape[0])

        self.step_size = valid_sample.shape[1]

        self.get_labels_stats(valid_label, "valid")
        train_ds = None
        valid_ds = None
        if self.task in ["train", "run"]:
            train_sample, train_label, self.train_dim = load_data(self.context.data_train, 'train', multi_label=self.multi_label)

            self.get_labels_stats(train_label, "train")

            # Get input dataset feature size
            if train_sample.shape[1] != valid_sample.shape[1]:
                self.logging.error(f"The size feature of training[{train_sample.shape[1]}] and "
                                   f"testing[{valid_sample.shape[1]}] dataset does not match")
                raise ValueError

            # Build training iterated datasets for training process.
            train_ds = DataLoader(TensorDataset(Tensor(train_sample).to(self.device),
                                                Tensor(train_label).type(torch.int64).squeeze(-1).to(self.device)),
                                  batch_size=self.config.batch_size,
                                  shuffle=True)
            valid_ds = DataLoader(TensorDataset(Tensor(valid_sample).to(self.device),
                                                Tensor(valid_label).type(torch.int64).squeeze(-1).to(self.device)),
                                  batch_size=self.config.batch_size,
                                  shuffle=False)


        return train_ds, valid_ds, self.get_test_data()

    def train(self):
        train_time = []
        self.logging.info("Traing model ...")
        hidden = None
        epoch = 0
        if self.config.classification == "lstm":
            hidden = self.model.init_hidden(self.config.batch_size)
        try:
            early_stopping = False
            for epoch in range(self.start_epoch, self.config.epochs):
                start_t = time.time()

                training_loss = 0.0
                # x: torch.Size([32, 16, 1]), y: torch.Size([32, 1, 1])
                for i, batch in enumerate(self.train_dataset):
                    inputs, labels = batch  # torch.Size([128, 3, 13]), torch.Size([128])

                    # clear previous training gradients
                    self.optimizer.zero_grad()
                    # Invoke the model on [x] to get the prediction values
                    if hidden is not None:
                        hidden = tuple([each.data for each in hidden])

                    outputs = self.model(inputs, hidden=hidden)

                    # Using loss function to calculate loss
                    loss = self.criterion(outputs, labels)

                    # backpropagation, compute gradients
                    loss.backward()

                    # apply gradients
                    self.optimizer.step()
                    self._step += 1

                    training_loss += loss.item()
                    # # Evaluate the model every [self.config.valid_steps]
                    if self.get_current_step % self.config.valid_steps == 0:
                        stats, ytest, ypred = self.evaluate(epoch, hidden=hidden)
                        accuracy = stats['accuracy']

                        if self.early_stopping is not None:
                            self.early_stopping(stats, self._step)

                            if self.early_stopping.is_improving():
                                save_path = join(self.context.evaluate_dir,
                                                 f"valid_best_pred_{self.context.version}_"
                                                 f"{self.get_current_step}_{accuracy:0.4f}.HDF")
                                self.logging.info(f"Save best valid result: {save_path}")
                                with h5py.File(save_path, 'w') as hf:
                                    hf.create_dataset('Lable', data=ytest.reshape(self.valid_dim))
                                    hf.create_dataset('Predict', data=ypred.reshape(self.valid_dim))

                                self.best_model = join(self.context.checkpoint_dir,
                                                       f"checkpoint_best_{self.get_current_step}_{accuracy:0.4f}.pt")
                                self.save_model(self.best_model, epoch=epoch, loss=accuracy)

                            # If the patience has reached the limit, stop training
                            if self.early_stopping.has_stopped():
                                early_stopping = True
                                break

                    # Print training information every [self.config.report_every] steps
                    if self.get_current_step % self.config.report_every == 0:
                        self.logging.info(f"Epoch {epoch}/{self.config.epochs}, Step {self.get_current_step}, "
                                          f"Training Loss: {loss.item():.4f}")

                end_t = time.time()
                training_loss /= len(self.train_dataset)
                train_epoch_time = end_t - start_t
                train_time.append(train_epoch_time)
                self.logging.info(f'Epoch {epoch}/{self.config.epochs}, Step {self.get_current_step}, '
                                  f'Epoch loss: {training_loss:.4f}, '
                                  f'time usage: {train_epoch_time:.3f}s, '
                                  f'time left: {np.average(train_time) * (self.config.epochs - epoch) / 3600:.3f}h')

                if early_stopping:
                    self.logging.info("Early Stop...")
                    break
        except KeyboardInterrupt:
            self.logging.info('-' * 60)
            self.logging.info('Early terminate')

        self.logging.info('Training completed')
        self.latest_model = join(self.context.checkpoint_dir,
                               f"checkpoint_latest_{self.get_current_step}.pt")
        self.save_model(self.latest_model, epoch=epoch, loss=float('inf'))

    def evaluate(self, epoch, **kwargs):
        self.logging.info(f"Epoch {epoch}/{self.config.epochs}, step {self.get_current_step}: Performance Evaluation ...")
        ytest = []
        ypred = []
        hidden = kwargs.get("hidden", None)
        # make model work in evaluate state
        self.model.eval()
        with torch.no_grad():
            for data, labels in self.valid_dataset:
                if hidden is not None:
                    hidden = tuple([each.data for each in hidden])
                outputs = self.model(data, hidden=hidden)
                _, predt1 = torch.max(outputs[0], 1)
                _, predt2 = torch.max(outputs[1], 1)
                predicted = torch.stack([predt1, predt2], dim=1)
                ytest.append(labels)
                ypred.append(predicted)

        # make model work in train state
        self.model.train()
        ytest = torch.cat(ytest, 0)
        ypred = torch.cat(ypred, 0)

        # 1. copy the data from cuda to cpu using [.cpu] if the dataset is a tensor of cuda
        # 2. Convert a tensor into a numpy data using [.detach().numpy()]
        ytest = ytest.cpu().detach().numpy() if ytest.is_cuda else ytest.detach().numpy()
        ypred = ypred.cpu().detach().numpy() if ypred.is_cuda else ypred.detach().numpy()

        task_0_stats = None
        stats = None
        delta = 0.2
        for idx in range(ytest.shape[-1]):
            y_pred = ypred[:, idx]
            y_test = ytest[:, idx]
            stats = self.performance_evaluate(y_pred, y_test, idx)
            if idx == 0:
                task_0_stats = stats
            else:
                stats["accuracy"] += delta * task_0_stats["accuracy"]
                stats["f1"] += delta * task_0_stats["f1"]
                stats["recall"] += delta * task_0_stats["recall"]
                stats["precision"] += delta * task_0_stats["precision"]

        return stats, ytest, ypred

    def predict(self):
        self.logging.info("Predicting ...")
        hidden = None
        if self.config.classification == "lstm":
            hidden = self.model.init_hidden(self.config.batch_size)

        for (test_data, test_dim), test_data_name in zip(self.test_dataset, self.context.data_test):
            # ytest = []
            ypred = []
            save_path = join(self.context.predict_dir,
                             f"predict_{self.context.version}_{basename(test_data_name)}")
            # make model work in evaluate state
            self.model.eval()
            with torch.no_grad():
                for data, labels in test_data:
                    if hidden is not None:
                        hidden = tuple([each.data for each in hidden])

                    outputs = self.model(data, hidden=hidden)
                    _, predt1 = torch.max(outputs[0], 1)
                    _, predt2 = torch.max(outputs[1], 1)
                    predicted = torch.stack([predt1, predt2], dim=1)
                    # ytest.append(labels)
                    ypred.append(predicted)
            # make model work in train state
            self.model.train()
            # ytest = torch.cat(ytest, 0)
            ypred = torch.cat(ypred, 0)

            # 1. copy the data from cuda to cpu using [.cpu] if the dataset is a tensor of cuda
            # 2. Convert a tensor into a numpy data using [.detach().numpy()]
            # ytest = ytest.cpu().detach().numpy() if ytest.is_cuda else ytest.detach().numpy()
            ypred = ypred.cpu().detach().numpy() if ypred.is_cuda else ypred.detach().numpy()

            self.logging.info(f"Save predict test result: {save_path}")
            save_hdf5_data(save_path, ypred)

            # for idx in range(ytest.shape[-1]):
            #     y_pred = ypred[:, idx]
            #     y_test = ytest[:, idx]
            #     self.performance_evaluate(y_pred, y_test, idx)


    def save_model(self, model_path, **kwargs):
        epoch = kwargs.get("epoch", 0)
        loss = kwargs.get("loss", 0.0)

        torch.save({'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                    'optimizer_state_dict': self.optimizer.state_dict()}, model_path)

        self.logging.info(f'Epoch {epoch}, Checkpoint model with Loss {loss:.4f} is saved to {model_path}')

    def load_model(self, model_path, model=None, **kwargs):
        assert model is not None
        self.logging.info(f"Loading the model from checkpoint: {model_path}, to device {self.context.device}")

        if self.context.device == 'cpu':
            checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
        else:
            # checkpoint = torch.load(self.checkpoint_path, map_location={model_data['device']: self.args.device})
            checkpoint = torch.load(model_path, map_location='cuda:0')

        model.load_state_dict(checkpoint.get('model_state_dict'))

        if hasattr(self, 'optimizer') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict'))

        return checkpoint.get('epoch')