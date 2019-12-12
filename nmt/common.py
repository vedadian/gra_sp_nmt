# coding: utf-8
"""
Definition of global constants
"""

import inspect
from typing import Callable, Union, T

from nmt.config import Configuration

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

configuration = Configuration()

def configurable(parent_configuration: str = None):

    global configuration
    config_provider = configuration
    if parent_configuration is not None:
        path = parent_configuration.split('.')
        for part in path:
            config_provider = config_provider.add_submodule(part)

    def make_configurable_function(f: Callable):

        param_names = []

        signature = inspect.signature(f)
        for param_name in signature.parameters:
            param_names.append(param_name)
            param_specs = signature.parameters[param_name]
            config_provider.register_param(
                param_name,
                None \
                if param_specs.default == inspect.Parameter.empty else \
                param_specs.default
            )

        def configured_f():
            params = { name: getattr(config_provider, name) for name in param_names }
            return f(**params)

        return configured_f

    def make_configurable_class(c: T):
        param_names = []

        signature = inspect.signature(c.__init__)
        first_param = True
        for param_name in signature.parameters:
            if first_param:
                first_param = False
                continue
            param_names.append(param_name)
            param_specs = signature.parameters[param_name]
            config_provider.register_param(
                param_name,
                None \
                if param_specs.default == inspect.Parameter.empty else \
                param_specs.default
            )

        class configured_c(c):
            def __init__(self):
                params = { name: getattr(config_provider, name) for name in param_names }
                super(configured_c, self).__init__(**params)

        return configured_c
        

    def make_configurable(o: Union[Callable, object]):
        if inspect.isclass(o):
            return make_configurable_class(o)
        else:
            return make_configurable_function(o)

    return make_configurable