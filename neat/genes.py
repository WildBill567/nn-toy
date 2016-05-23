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

import numpy.random as random

from neat.activations import activation_types

# Adapted from
# https://github.com/CodeReclaimers/neat-python, accessed May 2016


class NodeGene:

    # counter starts at 1 and goes up
    # TODO find a better way
    counter = 1

    def __init__(self, *, activation='sigmoid', node_type='HIDDEN', idx=-1):
        """
        NodeGene - class for nodes in a NEAT network
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016

        @param activation: node's activation function
        @param node_type: node type, one of 'HIDDEN', 'INPUT', 'OUTPUT'
        @param idx: node's index
        """
        if idx == -1:
            self.idx = NodeGene.counter
            NodeGene.counter += 1
        elif idx >= 0:
            self.idx = idx
        else:
            raise ValueError("Node gene cannot have negative index")

        self.node_type = node_type
        assert self.node_type in ['INPUT', 'OUTPUT', 'HIDDEN', 'BIAS'], "Invalid node type"

        if self.node_type == 'INPUT':
            self.activation = 'identity'
        else:
            self.activation = activation
            assert self.activation in activation_types, "Invalid activation"


class LinkGene:
    innovation_counter = 0

    def __init__(self, src_node, sink_node, *, weight=None, innov=-1, enabled=True):
        """
        LinkGene - Class for gene for link between nodes
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016

        :param src_node: source
        :param sink_node: sink
        :param weight: weight
        :param innov: innovation number
        :param enabled: is link active?
        """
        if src_node == sink_node:
            raise ValueError("Links cannot be self-loops")
        self.src = src_node
        self.sink = sink_node
        self.enabled = enabled

        if weight is None:
            self.weight = random.random()
        else:
            self.weight = weight

        if innov == -1:
            self.innovation_number = LinkGene.innovation_counter
            LinkGene.innovation_counter += 1
        elif innov >= 0:
            self.innovation_number = innov
        else:
            raise ValueError("Link gene cannot have negative innovation number")
