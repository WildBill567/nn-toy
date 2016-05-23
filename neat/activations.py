# Copyright (C) 2016  William Langhoff WildBill567@users.noreply.github.com
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Adapted from:
# https://github.com/CodeReclaimers/neat-python, accessed May 2016
# Which is distributed with the following license:

#   Copyright (c) 2007-2011, cesar.gomes and mirrorballu2
#   Copyright (c) 2015, CodeReclaimers, LLC
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
#   following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution.
#
#   3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np

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
