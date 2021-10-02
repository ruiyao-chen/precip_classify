from os.path import join as pjoin
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error


class RadomForest:
    def __init__(self, ctx):
        self.context = ctx
        self.config = ctx.config
        self.logging = ctx.logger
        self.model = self.build_model()
        self.train_sample, self.train_label, self.test_sample, self.test_label = self.load_data()

    def load_data(self):
        data_train_mat = pjoin(self.context.project_dir, self.config.data_train)
        data_train_np = sio.loadmat(data_train_mat)
        train_sample = data_train_np['train_sample']  # np_arrary 29659 * 16
        train_label = data_train_np['train_label']  # 29659 * 1

        data_test_mat = pjoin(self.context.project_dir, self.config.data_test)
        data_test_np = sio.loadmat(data_test_mat)
        test_sample = data_test_np['test_sample']  # np_arrary 12711 * 16
        test_label = data_test_np['test_label']  # 12711 * 1

        return train_sample, train_label, test_sample, test_label

    @staticmethod
    def build_model():
        # return RandomForestRegressor(n_estimators=1000, n_jobs=6)
        return RandomForestClassifier(n_estimators=1000)

    def train(self, train_sample, train_label):
        # Train the model using the training sets y_pred=clf.predict(X_test)
        self.model.fit(train_sample, train_label)

    def evaluate(self, epoch):
        pass

    def predict(self, test_sample=None):
        test_sample = test_sample if test_sample is not None else self.test_sample
        return self.model.predict(test_sample)

    def run(self):
        self.train()
        y_pred = self.predict(self.test_sample)

        # Model Accuracy, how often is the classifier correct?
        self.logging.info(f"MSE Error: {mean_squared_error(self.test_label, y_pred)}")
        self.logging.info(f"R2 Score: {r2_score(self.test_label, y_pred)}")
