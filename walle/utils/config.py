"""Read yaml config files with custom data types.
"""

import numpy as np
import ruamel.yaml


yaml = ruamel.yaml.YAML(typ='safe')

@yaml.register_class
class NumpyArray:
    yaml_tag = '!ndarray'
    @classmethod
    def from_yaml(cls, constructor, node):
        arr = constructor.construct_sequence(node, deep=True)
        return np.asarray(arr)


@yaml.register_class
class NumpyArrayRadians:
    yaml_tag = '!ndarrayrad'
    @classmethod
    def from_yaml(cls, constructor, node):
        arr = constructor.construct_sequence(node, deep=True)
        return np.radians(np.asarray(arr))


class YamlConfig:
    def __new__(cls, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f)
        return data