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

import os
from configparser import ConfigParser

from neat.activations import activation_types

class Config:

    def __init__(self, filename=None):
        self.type_config = {}

        parameters = ConfigParser()
        if filename is not None:
            if not os.path.isfile(filename):
                raise Exception('No such config file: ' + os.path.abspath(filename))

            with open(filename) as f:
                parameters.read_file(f)

        # Phenotype parameters
        self.num_inputs = int(parameters.get('Phenotype', 'num_inputs', fallback=2))
        self.num_outputs = int(parameters.get('Phenotype', 'num_outputs', fallback=2))
        self.activation_functions = parameters.get('Phenotype', 'activation_functions', fallback="sigmoid relu").strip().split()

        # Population parameters
        self.pop_size = int(parameters.get('Population', 'pop_size', fallback=40))


        # Mutation parameters
        self.multiple_mutations = bool(parameters.getboolean('Mutation', 'multiple_mutations', fallback=False))
        self.prob_add_conn = float(parameters.get('Mutation', 'prob_add_conn', fallback=0.1))
        self.prob_add_node = float(parameters.get('Mutation', 'prob_add_node', fallback=0.05))
        self.prob_delete_conn = float(parameters.get('Mutation', 'prob_delete_conn', fallback=0.01))
        self.prob_delete_node = float(parameters.get('Mutation', 'prob_delete_node', fallback=0.01))
        self.prob_mutate_weight = float(parameters.get('Mutation', 'prob_mutate_weight', fallback=0.6))
        self.weight_mutation_power = float(parameters.get('Mutation', 'weight_mutation_power', fallback=1.2))
        self.prob_mutate_activation = float(parameters.get('Mutation', 'prob_mutate_activation', fallback=0.0))
        self.prob_toggle_link = float(parameters.get('Mutation', 'prob_toggle_link', fallback=0.02))

        # Speciation parameters
        self.compatibility_threshold = float(parameters.get('Speciation', 'compatibility_threshold', fallback=8.0))
        self.excess_coefficient = float(parameters.get('Speciation', 'excess_coefficient', fallback=1.0))
        self.disjoint_coefficient = float(parameters.get('Speciation', 'disjoint_coefficient', fallback=1.0))
        self.weight_coefficient = float(parameters.get('Speciation', 'weight_coefficient', fallback=0.4))

        self._sanity_check()

    def _sanity_check(self):
        for fn in self.activation_functions:
            assert fn in activation_types, "Invalid activation: %s" % fn
        assert self.num_inputs > 0
        assert self.num_outputs > 0
        assert self.pop_size > 0
        self._assert_is_probability(self.prob_add_conn)
        self._assert_is_probability(self.prob_add_node)
        self._assert_is_probability(self.prob_delete_conn)
        self._assert_is_probability(self.prob_delete_node)
        self._assert_is_probability(self.prob_mutate_activation)
        self._assert_is_probability(self.prob_mutate_weight)
        self._assert_is_probability(self.prob_toggle_link)

    @staticmethod
    def _assert_is_probability(val):
        assert 0.0 <= val <= 1.0 , "Probability cannot have value %.02f" % val