import configargparse as cfargparse
from configargparse import ConfigFileParser, ConfigFileParserException
from atmos.utils.logging import logger
from collections import OrderedDict
import configargparse
from os import path as osp


class StoreLoggingLevelAction(configargparse.Action):
    """ Convert string to logging level """
    import logging
    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(
            option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


class DeprecateAction(configargparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.help is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise configargparse.ArgumentTypeError(msg)


class CustomizedYAMLConfigFileParser(ConfigFileParser):
    """Parses YAML config files. Depends on the PyYAML module.
    https://pypi.python.org/pypi/PyYAML
    """

    def get_syntax_description(self):
        msg = ("The config file uses YAML syntax and must represent a YAML "
               "'mapping' (for details, see http://learn.getgrav.org/advanced/yaml).")
        return msg

    @staticmethod
    def _load_yaml():
        """lazy-import PyYAML so that configargparse doesn't have to dependend
        on it unless this parser is used."""
        try:
            import yaml
        except ImportError:
            raise ConfigFileParserException("Could not import yaml. "
                                            "It can be installed by running 'pip install PyYAML'")
        return yaml

    def parse(self, stream):
        """Parses the keys and values from a config file."""
        yaml = self._load_yaml()

        try:
            parsed_obj = yaml.safe_load(stream)
        except Exception as e:
            raise ConfigFileParserException("Couldn't parse config file: %s" % e)

        if not isinstance(parsed_obj, dict):
            raise ConfigFileParserException("The config file doesn't appear to "
                                            "contain 'key: value' pairs (aka. a YAML mapping). "
                                            "yaml.load('%s') returned type '%s' instead of 'dict'."
                                            % (getattr(stream, 'name', 'stream'),
                                               type(parsed_obj).__name__))

        result = OrderedDict()
        for key, value in parsed_obj.items():
            if isinstance(value, list):
                result[key] = value
            else:
                result[key] = str(value)

        return result

    def serialize(self, items, default_flow_style=False):
        """Does the inverse of config parsing by taking parsed values and
        converting them back to a string representing config file contents.

        Args:
            :param default_flow_style: defines serialization format (see PyYAML docs)
            :param items:
        """

        # lazy-import so there's no dependency on yaml unless this class is used
        yaml = self._load_yaml()

        # it looks like ordering can't be preserved: http://pyyaml.org/ticket/29
        items = dict(items)
        return yaml.dump(items, default_flow_style=default_flow_style)


class ArgumentParser(cfargparse.ArgumentParser):
    def __init__(self, config_file_parser_class=CustomizedYAMLConfigFileParser,
                 formatter_class=cfargparse.ArgumentDefaultsHelpFormatter, **kwargs):
        super(ArgumentParser, self).__init__(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

    @classmethod
    def defaults(cls, model, *args):
        """Get default arguments added to a parser by all ``*args``."""
        dummy_parser = cls(model)
        for callback in args:
            callback(dummy_parser)
        defaults = dummy_parser.parse_known_args([])[0]
        return defaults

    # @classmethod
    # def update_model_opts(cls, model_opt):
    #     pass
    #
    # @classmethod
    # def validate_model_opts(cls, model_opt):
    #     pass
    #
    # @classmethod
    # def ckpt_model_opts(cls, model, ckpt_opt):
    #     # Load default opt values, then overwrite with the opts in
    #     # the checkpoint. That way, if there are new options added,
    #     # the defaults are used.
    #     opt = cls.defaults(model, model_opts)
    #     opt.__dict__.update(ckpt_opt.__dict__)
    #     return opt
    #
    # @classmethod
    # def validate_train_opts(cls, opt):
    #     assert opt.num_gpus == 1, "Currently we only support one GPU training"
    #
    #     assert (opt.save_data is None) != (opt.real_data is None), \
    #         "Only one of \"config.data\" and \"config.real_data\" should be provided."
    #
    #     if opt.with_generation:
    #         assert opt.real_data is not None and \
    #                opt.output is not None, "Need provide real " \
    #                                        "data and output data path for data generation"
    #
    # @classmethod
    # def validate_evaluate_opts(cls, opt):
    #     if opt.classify_tasks and opt.regression_tasks:
    #         raise AssertionError("Classification and Regression "
    #                              "evaluator cannot be enabled at the same time")
    #
    # @classmethod
    # def validate_preprocess_args(cls, opt):
    #     pass


def config_opts(parser):
    # Command Project Related Parameters
    group = parser.add_argument_group("General Configuration")
    group.add('--project_dir', type=str, required=False, default="")
    group.add('--project_log', type=str, required=False, default="")
    group.add('--output_dir', type=str, required=False, default="examples/output")
    group.add('--config', '-config', required=False, is_config_file_arg=True, help='config file path')
    group.add('--debug', type=str, default="False")
    group.add('--version', type=str, required=True, default="v1")


def model_opts(parser):
    group = parser.add_argument_group("Model")
    group.add('--classification', type=str, required=False,
              choices=["mlp", "selfattn",
                       "transformer", "random_forest", "gaussian_nb", "lstm", "resnet18"],
              default="mlp")
    group.add_argument('--multi_label', action='store_true')
    group.add('--regression', type=str, required=False, choices=["v1", "v2", "v3"], default=None)
    group.add('--early_stopping', '-early_stopping', type=int, default=10,
              help='Number of validation steps without improving.')

    group.add('--early_stopping_criteria', '-early_stopping_criteria',
              nargs="*", default=None, help='Criteria to use for early stopping.')

    # Transformer
    transformer_cmd = parser.add_argument_group('Transformer', 'configuration for Transformer')
    transformer_cmd.add('--layers_count', type=int, default=6)
    transformer_cmd.add('--d_model', type=int, default=128)
    transformer_cmd.add("--head", "-head", type=int, default=8,
                        help="The number of head in multiple attention head")
    transformer_cmd.add('--d_ff', type=int, default=1024)
    transformer_cmd.add('--dropout', '-dropout', type=float,
                        default=0.1, help="the ratio of Drop model")

    # LSTM
    lstm_cmd = parser.add_argument_group('LSTM', 'configuration for LSTM')
    lstm_cmd.add('--layers', type=int, default=2, help="The number of layers for LSTM")
    lstm_cmd.add('--layer_size', type=int, default=64, help="The dimension size of hidden layer")
    lstm_cmd.add('--n_residual_layers', type=int, default=2, help="")
    lstm_cmd.add('--n_highway_layers', type=int, default=2, help="")
    lstm_cmd.add('--bidir', type=bool, default=True, help="")

    # ResNet
    resnet_cmd = parser.add_argument_group('ResNet', 'configuration for ResNet')
    resnet_cmd.add('--resnet_layers', type=int, default=[2, 2, 2, 2], nargs="*", help="The number of layers for LSTM")
    resnet_cmd.add('--fc_config', type=str, default=None, help="The configure for FC layer")
    resnet_cmd.add('--resnet_block', type=str, default='Block',
                   choices=['Block', 'Bottleneck'], help="The configure for FC layer")
    resnet_cmd.add('--resnet_atts', type=str, default='None',
                   choices=['None', 'se'], help="The configure for FC layer")
    resnet_cmd.add('--resnet_local_atts', type=str, default='Conv1d',
                   choices=['Conv1d', 'CotLayer', 'CoXtLayer'], help="The configure for FC layer")


def optimizer_opts(parser):
    # Optimization
    group = parser.add_argument_group("Optimization")
    group.add_argument('--learning_rate', '-learning_rate', type=float, default=1.0,
                       help="Starting learning rate. "
                            "Recommended settings: sgd = 1, adagrad = 0.1, "
                            "adadelta = 1, adam = 0.001")
    group.add_argument('--adam_beta1', '-adam_beta1', type=float, default=0.9,
                       help="The beta1 parameter used by Adam. "
                            "Almost without exception a value of 0.9 is used in "
                            "the literature, seemingly giving good results, "
                            "so we would discourage changing this value from "
                            "the default without due consideration.")
    group.add_argument('--adam_beta2', '-adam_beta2', type=float, default=0.999,
                       help='The beta2 parameter used by Adam. '
                            'Typically a value of 0.999 is recommended, as this is '
                            'the value suggested by the original paper describing '
                            'Adam, and is also the value adopted in other frameworks '
                            'such as Tensorflow and Keras, i.e. see: '
                            'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
                            'Optimizer or https://keras.io/optimizers/ . '
                            'Whereas recently the paper "Attention is All You Need" '
                            'suggested a value of 0.98 for beta2, this parameter may '
                            'not work well for normal models / default '
                            'baselines.')

    # Loss
    group = parser.add_argument_group("Loss")

    group.add_argument('--loss_func', '-loss_func', type=str,
                       choices=["cross_entropy", "focal_loss", "multi_loss"], default="cross_entropy")

    group.add('--focal_loss', '-focal_loss', type=float, default=0.1,
              help="alpha values of Focal loss for imbalance class"
                   "For more detailed information, see: "
                   "https://arxiv.org/pdf/1708.02002.pdf")

    group.add('--focal_loss_learnable', '-focal_loss_learnable', type=bool, default=True,
              help="Alpha values of Focal loss for imbalance class is learnable or not"
                   "For more detailed information, see: "
                   "https://arxiv.org/pdf/1708.02002.pdf")

    group.add('--focal_loss_gamma', '-focal_loss_gamma', type=int, default=4,
              help="gamma values of Focal loss for imbalance class"
                   "For more detailed information, see: "
                   "https://arxiv.org/pdf/1708.02002.pdf")


def checkpoint_opts(parser):
    group = parser.add_argument_group("Checkpoint")
    group.add('--checkpoint', '-checkpoint', default=None, type=str,
              help="If training from a checkpoint then this is the "
                   "path to the pretrained model's state_dict.")

    group.add('--checkpoint_dir', type=str, required=False, default="checkpoints")
    group.add('--save_checkpoint_steps', '-save_checkpoint_steps',
              type=int, default=5000, help="""Save a checkpoint every X steps""")


def runtime_opts(parser):
    group = parser.add_argument_group('Runtime Backend')
    group.add('--device', '-device', type=str, choices=['cuda', 'cpu'], default='cuda')

    group.add('--device_id', '-device_id', type=int, default=0,
              help="Specify the GPU visible Device ID for training/testing task")

    group.add('--workers', type=int, required=False, default=2)


def dataset_opts(parser):
    # Data process
    group = parser.add_argument_group("Data")
    group.add('--data_train', type=str, required=True, default=None)
    group.add('--data_valid', type=str, required=True, default=None)
    group.add('--data_test', type=str, required=False, default=None)
    group.add('--data_test_bias', type=str, required=False, default=None)
    group.add('--batch_size', type=int, required=False, default=128)


def train_opts(parser):
    group = parser.add_argument_group("Train")
    group.add_argument('--epochs', type=int, required=False, default=100)
    group.add_argument('--train_steps', '-train_steps', type=int, default=100000,
                       help='Number of training steps')
    group.add_argument('--report_every', '-report_every', type=int, default=1000,
                       help="Print stats at this interval.")

    # Validate
    group = parser.add_argument_group("Validate")
    group.add_argument("--valid_steps", type=int, default=5000, help="Eval model every # steps")
    group.add_argument('--valid_batch_size', '-valid_batch_size', type=int, default=256,
                       help='Maximum batch size for validation')


def train_commands(parser):
    config_opts(parser)
    model_opts(parser)
    optimizer_opts(parser)
    checkpoint_opts(parser)
    runtime_opts(parser)
    dataset_opts(parser)
    train_opts(parser)


def test_commands(parser):
    config_opts(parser)
    model_opts(parser)
    optimizer_opts(parser)
    checkpoint_opts(parser)
    runtime_opts(parser)
    dataset_opts(parser)


def run_commands(parser):
    config_opts(parser)
    model_opts(parser)
    optimizer_opts(parser)
    checkpoint_opts(parser)
    runtime_opts(parser)
    dataset_opts(parser)
    train_opts(parser)


def get_main_argument(desc='Atmosphere'):
    parser = ArgumentParser(description=desc)

    mode = parser.add_subparsers(title='mode', dest='mode',
                                 help='Operation: '
                                      '[train] train model, '
                                      '[predict] evaluate model, '
                                      '[run] run train and then predict model')
    mode.required = True

    # train
    train_commands(mode.add_parser('train'))

    # test
    test_commands(mode.add_parser('predict'))

    # run
    run_commands(mode.add_parser('run'))

    args, kwargs = parser.parse_known_args()

    if args.multi_label:
        assert args.loss_func == 'multi_loss', "Please use multiple loss function for multiple task learning"

        assert args.classification == "resnet18", "Multi task learning only support ResNet Model"


    return args, kwargs
