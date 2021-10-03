import os
import random
import warnings
from os import makedirs
from os.path import dirname, abspath, join, exists, isfile, isdir
from atmos.model.classification import str2classification
import numpy as np
from datetime import date
from atmos.utils.logging import init_logger
from os import listdir
import torch
from os import path as osp
from atmos.utils.argument import get_main_argument

BASE_DIR = dirname(dirname(dirname(abspath(__file__))))


def init_rng(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def create_dir(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)


class Context:
    def __init__(self, config=None, logger=None):
        self.description = "Atmosphere project"
        # A dictionary of Config Parameters
        if config:
            self.config = config
        else:
            config, kwargs = get_main_argument(self.description)
            self.config = config

        self.task = self.config.mode
        self.version = self.config.version
        self.current_date = str(date.today())

        config.version = "v0" if config.version is None else config.version.lower()

        config_version = ".".join(osp.basename(config.config).split("_")[-1].split(".")[:-1])

        assert config.version == config_version, \
            f"The versions does not match, please double check: {config_version} --> {config.version}"

        if 'fc_config' in config and config.fc_config is not None:
            import ast
            config.fc_config = ast.literal_eval(config.fc_config)

        if self.config.device_id is not None and self.config.device == 'cuda':
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config.device_id)
            self.is_cuda = True if torch.cuda.is_available() else False
        else:
            self.is_cuda = False

        self.device = 'cuda' if self.is_cuda else 'cpu'

        self.project_dir = str(self.config.project_dir) if self.config.project_dir != "" else str(BASE_DIR)
        create_dir(self.project_dir)

        self.output_dir = os.path.join(self.project_dir, self.config.output_dir,
                                       self.config.classification, self.version)
        create_dir(self.output_dir)

        if isfile(self.config.project_log) and exists(self.config.project_log):
            self.project_log = self.config.project_log
        else:
            self.project_log = join(self.output_dir, 'logs', f'{self.current_date}_{self.task}_{self.version}.log')
            create_dir(os.path.dirname(self.project_log))

        # logger interface
        self.isDebug = self.config.debug == 'True'
        self.logger = init_logger(self.description, self.project_log, self.isDebug) if logger is None else logger
        self.logger.debug("The logger interface is initited ...")

        self.checkpoint_dir = os.path.join(self.output_dir, self.config.checkpoint_dir)
        create_dir(self.checkpoint_dir)

        if self.config.checkpoint is not None:
            self.checkpoint = os.path.join(self.checkpoint_dir, self.config.checkpoint)
        else:
            self.checkpoint = None

        self.evaluate_dir = os.path.join(self.output_dir, "evaluate")
        create_dir(self.evaluate_dir)

        self.predict_dir = os.path.join(self.output_dir, "predict")
        create_dir(self.predict_dir)

        init_rng(seed=0)
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id[0])
        warnings.filterwarnings('ignore')

        self.data_train = join(self.project_dir, self.config.data_train)
        self.data_valid = join(self.project_dir, self.config.data_valid)

        if self.config.data_test is not None:
            data_path = join(self.project_dir, self.config.data_test)

            if isfile(data_path):
                self.data_test = [data_path]
            elif isdir(data_path):
                self.data_test = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
        else:
            self.data_test = []

        self.data_test_bias = join(self.project_dir, self.config.data_test_bias) if self.config.data_test_bias else None

        if self.task == 'predict' and self.checkpoint is None:
            self.logger.info("Please set up checkpoint path")
            raise ValueError

        self.classification = self.config.classification
        self.regression = self.config.regression

        for key, val in vars(self.config).items():
            self.logger.info(f"[Config]: {key} => {val}")

    def get_model(self):
        model = str2classification[self.classification]
        self.logger.info(f"Using the model {model} ... ")
        return model
