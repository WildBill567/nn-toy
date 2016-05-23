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

import networkx as nx
import matplotlib.pyplot as plt

from neat import activation_functions

def find_feed_forward_layers(inputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016

    :param inputs: list of the network input nodes
    :param connections: list of (input, output) connections in the network.
    Returns a list of layers, with each layer consisting of a set of node identifiers.

    """

    # TODO: Detect and omit nodes whose output is ultimately never used.

    layers = []
    prev_nodes = set(inputs)
    prev_nodes.add(0)
    while 1:
        # Find candidate nodes for the next layer.  These nodes should connect
        # a node in S to a node not in S.
        candidate_set = set(b for (a, b) in connections if a in prev_nodes and b not in prev_nodes)
        # Keep only the nodes whose entire input set is contained in S.
        keeper_set = set()
        for n in candidate_set:
            if all(a in prev_nodes for (a, b) in connections if b == n):
                keeper_set.add(n)

        if not keeper_set:
            break

        layers.append(keeper_set)
        prev_nodes = prev_nodes.union(keeper_set)

    return layers


class FeedForwardPhenome:
    def __init__(self, genome):
        """
        FeedForwardPhenome - A feedforward network
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016

        :param genome: the genome to create the phenome
        """
        self.graph, node_lists = self._construct_graph(genome)
        self.input_nodes, self.hidden_nodes, self.output_nodes = node_lists
        self.links = [(g.src, g.sink) for g in genome.link_genes]
        self.node_evals = []

        layers = find_feed_forward_layers(self.input_nodes, self.links)
        used_nodes = set(self.input_nodes + self.output_nodes)
        for layer in layers:
            for node in layer:
                inputs = []
                # TODO: This could be more efficient.
                for cg in genome.link_genes:
                    if cg.sink == node and cg.enabled:
                        inputs.append((cg.src, cg.weight))
                        used_nodes.add(cg.src)

                used_nodes.add(node)
                ng = genome.get_node_by_index(node)
                activation_function = activation_functions.get(ng.activation)
                self.node_evals.append((node, activation_function, inputs))
        self.values = [0.0] * (1 + max(used_nodes))

    def serial_activate(self, inputs):
        """
        serial_activate - gives network output for an input
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016
        :param inputs: numerical input list
        :return: numerical output list
        """
        if len(self.input_nodes) != len(inputs):
            raise ValueError("Expected {0} inputs, got {1}".format(len(self.input_nodes), len(inputs)))
        self.values[0] = 1.0

        for idx, v in zip(self.input_nodes, inputs):
            self.values[idx] = v

        for node, func, links in self.node_evals:
            linear_activation = 0.0
            for idx, weight in links:
                linear_activation += self.values[idx] * weight
            self.values[node] = func(linear_activation)

        return [self.values[i] for i in self.output_nodes]

    def draw(self):
        """Draws the network with matplotlib"""
        pos = {0: (-1.5, 0)}
        for idx in range(len(self.input_nodes)):
            pos[idx+1] = (idx, 0)
        for idx, val in enumerate(self.output_nodes):
            pos[val] = (idx, 4)
        for idx, val in enumerate(self.hidden_nodes):
            pos[val] = (idx, 2)
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=self.input_nodes,
                               node_color='r')
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=self.output_nodes,
                               node_color='g')
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=[0],
                               node_color='k')

        nx.draw_networkx_edges(self.graph, pos)
        plt.show()

    @staticmethod
    def _construct_graph(genome):
        """Constructs the DiGraph"""
        graph = nx.DiGraph()
        graph.add_node(0, {'node_type': 'BIAS', 'val': 1})
        input_list = []
        output_list = []
        hidden_list = []

        for gene in genome.input_genes:
            graph.add_node(gene.idx)
            input_list.append(gene.idx)

        for gene in genome.output_genes:
            graph.add_node(gene.idx)
            output_list.append(gene.idx)

        for gene in genome.hidden_genes:
            graph.add_node(gene.idx)
            hidden_list.append(gene.idx)

        for gene in genome.link_genes:
            graph.add_edge(gene.src, gene.sink,
                           {'weight': gene.weight,
                            'enabled': gene.enabled})
        return graph, (input_list, hidden_list, output_list)


