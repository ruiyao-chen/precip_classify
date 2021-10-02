from os.path import join
import scipy.io as sio
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy import savetxt
import matplotlib.pyplot as plt
from torch.nn import Module, Conv1d, Flatten
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


class CNNModel(Module):
    def __init__(self, ctx, input_feature_size):
        super(CNNModel, self).__init__()
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.feature_size = input_feature_size

        self.conv1d = Conv1d(1, 32, 3)
        self.input_dim = (self.feature_size - 3 + 1) * 32
        self.flatten = Flatten()
        self.l1 = torch.nn.Linear(self.input_dim, 64)
        self.l2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1d(x))  # activation function for hidden layer

        x = self.flatten(x)

        x = F.relu(self.l1(x))

        x = self.l2(x)  # linear output
        return x


class MLPModel(Module):
    def __init__(self, ctx, input_feature_size):
        super(MLPModel, self).__init__()
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.feature_size = input_feature_size
        self.n_hidden = 128

        self.hidden = torch.nn.Linear(self.feature_size, self.n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(self.n_hidden, 1)  # output layer

    def forward(self, x):
        x = x.squeeze(-1)
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x


class MyRunner:
    def __init__(self, ctx):
        super(MyRunner, self).__init__()
        self.feature_size = 1
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Loading the datasets and make them work in [self.device]
        self.train_dataset, self.valid_dataset, self.test_dataset = self.load_data()

        # build model and make it work using [self.device]
        # self.model = CNNModel(ctx, self.feature_size).to(self.device)
        self.model = CNNModel(ctx, self.feature_size).to(self.device)

        # The optimizer for training the model
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # The regression loss function to calcuate the loss
        self.loss_func = torch.nn.MSELoss()

        # The current training step count
        self._step = 0

    @property
    def get_current_step(self):
        return self._step

    def load_data(self):
        data_train_mat = join(self.context.project_dir, self.context.config.data_train)
        data_train_np = sio.loadmat(data_train_mat)
        # convert data into tensor and expand the dimension; torch.Size([29659, 16, 1])
        train_sample = Tensor(data_train_np['train_sample']).to(self.device).unsqueeze(-1)

        # convert data into tensor and expand the dimension; torch.Size([29659, 1])
        train_label = Tensor(data_train_np['train_label']).to(self.device)

        data_test_mat = join(self.context.project_dir, self.context.config.data_test)
        data_test_np = sio.loadmat(data_test_mat)
        # torch.Size([12711, 16, 1])
        test_sample = Tensor(data_test_np['test_sample']).to(self.device).unsqueeze(-1)

        # torch.Size([12711, 1])
        test_label = Tensor(data_test_np['test_label']).to(self.device)

        # Get input dataset feature size
        if train_sample.shape[1] == test_sample.shape[1]:
            self.feature_size = train_sample.shape[1]
        else:
            self.logging.error(f"The size feature of training[{train_sample.shape[1]}] and "
                               f"testing[{test_sample.shape[1]}] dataset does not match")

        # Build training iterated datasets for training process.
        train_ds = DataLoader(TensorDataset(train_sample, train_label),
                              batch_size=self.config.batch_size,
                              shuffle=False)

        valid_ds = DataLoader(TensorDataset(test_sample, test_label),
                              batch_size=self.config.batch_size,
                              shuffle=False)

        return train_ds, valid_ds, (test_sample, test_label)

    def train(self):
        for epoch in range(self.config.epochs):

            # x: torch.Size([32, 16, 1]), y: torch.Size([32, 1, 1])
            for x, y in iter(self.train_dataset):

                # clear previous training gradients
                self.optimizer.zero_grad()
                # Invoke the model on [x] to get the prediction values
                prediction = self.model(x)

                # Using loss function to calculate loss
                loss = self.loss_func(prediction, y)

                # backpropagation, compute gradients
                loss.backward()

                # apply gradients
                self.optimizer.step()
                self._step += 1

                # Evaluate the model every [self.config.valid_steps]
                if self.get_current_step % self.config.valid_steps == 0:
                    self.predict(epoch)

                # Print training information every [self.config.report_every] steps
                if self.get_current_step % self.config.report_every == 0:
                    self.logging.info(f"Epoch {epoch + 1}, Step {self.get_current_step}, "
                                      f"Train Loss: {loss.item():.4f}")

    def predict(self, epoch, saveOutput=False):
        self.logging.info("Predicting ...")
        xtest, ytest = self.test_dataset
        # make model work in evaluate state
        self.model.eval()

        ypred = self.model(xtest)

        # make model work in train state
        self.model.train()

        # 1. copy the data from cuda to cpu using [.cpu] if the dataset is a tensor of cuda
        # 2. Convert a tensor into a numpy data using [.detach().numpy()]
        ytest = ytest.cpu().detach().numpy() if ytest.is_cuda else ytest.detach().numpy()
        ypred = ypred.cpu().detach().numpy() if ypred.is_cuda else ypred.detach().numpy()

        # Model Accuracy, how often is the classifier correct?
        mse = mean_squared_error(ytest, ypred)
        r2 = r2_score(ytest, ypred)
        self.logging.info(f"Epoch {epoch} MSE Error: {mse:.4f}, R2 Score: {r2:.4f}")

        if saveOutput:
            save_path = join(self.context.output_dir, f"y_pred-{mse:.4f}-{r2:.4f}.csv")
            self.logging.info(f"Epoch {epoch} Save predicting result to {save_path}")
            savetxt(save_path, ypred, delimiter=",")

        self.generate_save_loss_image(ytest=ytest, ypred=ypred, epoch=epoch)

    def generate_save_loss_image(self, ytest, ypred, epoch):
        x_ax = range(len(ypred))
        plt.scatter(x_ax, ytest, s=5, color="blue", label="DPR")
        plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
        plt.legend()
        plt.savefig(join(self.context.output_dir, f"image_at_epoch_{epoch:04d}"))
        plt.show()

    def run(self):
        self.logging.info("Traing model ...")
        self.train()
        self.predict(epoch=self.config.epochs, saveOutput=True)