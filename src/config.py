# coding: utf-8
"""
Configuration manager
"""

from typing import Any

class Configuration(object):

    def __init__(self, parent: Configuration = None, name: str = '__root__'):
        self.parent = parent
        self.name = name
        self.__submodules = {}
        self.__parameters = {}
    
    def fullname(self):
        
        path = []

        node = self
        while node is not None:
            path.append(node.name)
            node = node.parent
        
        return '::'.join(reversed(path))

    def add_submodule(self, name: str):
        if name in self.__submodules:
            raise Exception('{}: Submodule `{}` already exists!'.format(
                self.fullname(), name
            ))
        if name in self.__parameters:
            raise Exception('{}: Submodule name `{}` collides with a previously registered parameter.'.format(
                self.fullname(), name
            ))
        submodule = Configuration(self, name)
        self.__submodules[name] = submodule
        return submodule
    
    def register_param(self, name: str, default: Any = None):
        if name in self.__parameters:
            raise Exception('{}: Parameter `{}` already exists!'.format(
                self.fullname(), name
            ))
        if name in self.__submodules:
            raise Exception('{}: Parameter name `{}` collides with a previously registered submodule.'.format(
                self.fullname(), name
            ))
        self.__parameters[name] = { "default": default }
    
    def __setattr__(self, name: str, value: Any):
        if not name in self.__parameters:
            raise Exception('{}: Parameter `{}` is not registered.'.format(
                self.fullname(), name
            ))
        self.__parameters[name]["value"] = value
        return value
    
    def __getattr__(self, name: str):
        if not name in self.__parameters:
            raise Exception('{}: Parameter `{}` is not registered.'.format(
                self.fullname(), name
            ))
        return self.__parameters[name]["value"] \
               if "value" in self.__parameters[name] else \
               self.__parameters[name]["default"]
    
    def load(self, source: dict):
        for name in self.__parameters:
            if name in source:
                self.__parameters[name] = source[name]
        for name in self.__submodules:
            if name in source:
                if isinstance(source[name], dict):
                    self.__submodules[name].load(source[name])
                else:
                    raise Exception('{}: Source data for a submodule `{}` must be a `dict` not `{}`.'.format(
                        self.fullname(), name, type(source[dict]).__name__
                    ))
