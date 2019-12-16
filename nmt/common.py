# coding: utf-8
"""
Definition of global constants
"""

import logging
import inspect
from typing import Callable, Union, T

from nmt.config import Configuration

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

epsilon = 1e-6

configuration = Configuration()

def configured(parent_configuration: str = None):

    global configuration
    config_provider = configuration
    if parent_configuration is not None:
        path = parent_configuration.split('.')
        for part in path:
            config_provider = config_provider.ensure_submodule(part)

    def make_configured_function(f: Callable):

        param_names = []

        signature = inspect.signature(f)
        for param_name in signature.parameters:
            param_names.append(param_name)
            param_specs = signature.parameters[param_name]
            config_provider.ensure_param(
                param_name,
                None \
                if param_specs.default == inspect.Parameter.empty else \
                param_specs.default
            )

        def configured_f():
            params = { name: getattr(config_provider, name) for name in param_names }
            return f(**params)
        configured_f.__name__ = f.__name__

        return configured_f

    def make_configured_class(c: T):
        param_names = []

        signature = inspect.signature(c.__init__)
        first_param = True
        for param_name in signature.parameters:
            if first_param:
                first_param = False
                continue
            param_names.append(param_name)
            param_specs = signature.parameters[param_name]
            config_provider.ensure_param(
                param_name,
                None \
                if param_specs.default == inspect.Parameter.empty else \
                param_specs.default
            )

        class configured_c(c):
            def __init__(self):
                params = { name: getattr(config_provider, name) for name in param_names }
                c.__init__(self, **params)
        configured_c.__name__ = c.__name__

        return configured_c
        

    def make_configured(o: Union[Callable, object]):
        if inspect.isclass(o):
            return make_configured_class(o)
        else:
            return make_configured_function(o)

    return make_configured

logger: logging.Logger = None

@configured('model')
def make_logger(output_path: str = './model/'):
    global logger

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    fh = logging.FileHandler('{}/train.log'.format(output_path))
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

def get_logger():
    global logger
    return logger