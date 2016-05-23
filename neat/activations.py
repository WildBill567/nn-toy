import numpy as np

# Adapted from
# https://github.com/CodeReclaimers/neat-python, accessed May 2016
# few changes - moved to numpy instead of math, added activation_types list

activation_types = ['sigmoid', 'tanh', 'sin', 'gauss', 'relu', 'identity', 'clamped',
                    'inv', 'log', 'exp', 'abs', 'hat', 'square', 'cube']


def sigmoid_activation(z):
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + np.exp(-z))


def tanh_activation(z):
    z = max(-60.0, min(60.0, z))
    return np.tanh(z)


def sin_activation(z):
    z = max(-60.0, min(60.0, z))
    return np.sin(z)


def gauss_activation(z):
    z = max(-60.0, min(60.0, z))
    return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


def relu_activation(z):
    return z if z > 0.0 else 0


def identity_activation(z):
    return z


def clamped_activation(z):
    return max(-1.0, min(1.0, z))


def inv_activation(z):
    if z == 0:
        return 0.0

    return 1.0 / z


def log_activation(z):
    z = max(1e-7, z)
    return np.log(z)


def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return np.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3


class InvalidActivationFunction(Exception):
    pass


class ActivationFunctionSet(object):
    def __init__(self):
        self.functions = {}

    def add(self, config_name, function):
        # TODO: Verify that the given function has the correct signature.
        self.functions[config_name] = function

    def get(self, config_name):
        f = self.functions.get(config_name)
        if f is None:
            raise InvalidActivationFunction("No such function: {0!r}".format(config_name))

        return f

    def is_valid(self, config_name):
        return config_name in self.functions
