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

from neat.config import Config
from .genes import NodeGene, LinkGene


class Genome:
    def __init__(self, config, *, node_genes=None, link_genes=None):
        """
        Genome for a NEAT network
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016

        :param config: Configuration
        :param node_genes: list of NodeGenes
        :param link_genes: list of LinkGenes
        """
        NodeGene.counter = 1
        self.fitness = 0
        self.link_genes = []
        self.input_genes = []
        self.hidden_genes = []
        self.output_genes = []

        self.config = config

        self._check_args(config.num_inputs,self.config.num_outputs, node_genes, link_genes)

        if node_genes is not None:
            self._parse_node_genes(node_genes)
            assert self.config.num_inputs == len(self.input_genes)
            assert self.config.num_outputs == len(self.output_genes)
        elif node_genes is None:
            self._random_genome(config.num_inputs,self.config.num_outputs)

        if link_genes is not None:
            for gene in link_genes:
                self.link_genes.append(gene)
        self.n_links = len(self.link_genes)

        assert not self._has_duplicate_links()
        assert not self._has_duplicate_node_indices(), "Genome has duplicate node indices"
        self._check_links_have_valid_nodes()

    # compatibility function
    def distance(self, other):
        """
        Distance between two genomes, used for speciation
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016

        :param other: the other genome
        :return: distance between the genomes
        """
        if len(self.link_genes) > len(other.link_genes):
            genome1 = self
            genome2 = other
        else:
            genome1 = other
            genome2 = self

        # Compute node gene differences.
        excess1 = 0
        excess2 = sum(1 for g2 in genome2.node_genes() if genome1.get_node_by_index(g2.idx) is None)
        activation_diff = 0
        num_common = 0
        for g1 in genome1.node_genes():
            g2 = genome2.get_node_by_index(g1.idx)
            if g2 is not None:
                num_common += 1
                if g1.activation != g2.activation:
                    activation_diff += 1
            else:
                excess1 += 1

        most_nodes = max(len(genome1.node_genes()),
                         len(genome2.node_genes()))
        distance = (self.config.excess_coefficient * float(excess1 + excess2) / most_nodes +
                    self.config.excess_coefficient * float(activation_diff) / most_nodes)

        # Compute connection gene differences.
        if genome1.link_genes:
            n_genes = len(genome1.link_genes)
            weight_diff = 0
            matching = 0
            disjoint = 0
            excess = 0

            max_innovation_genome2 = None
            if genome2.link_genes:
                max_innovation_genome2 = max([cg.innovation_number for cg in genome2.link_genes])

            for cg1 in genome1.link_genes:
                cg2 = genome2.get_link_by_indices(cg1.src, cg1.sink)
                if cg2 is not None:
                    # Homologous genes
                    weight_diff += abs(cg1.weight - cg2.weight)
                    matching += 1

                    if cg1.enabled != cg2.enabled:
                        weight_diff += 1.0
                else:
                    if max_innovation_genome2 is not None and cg1.innovation_number > max_innovation_genome2:
                        excess += 1
                    else:
                        disjoint += 1

            disjoint += len(genome2.link_genes) - matching

            distance += self.config.excess_coefficient * float(excess) / n_genes
            distance += self.config.disjoint_coefficient * float(disjoint) / n_genes
            if matching > 0:
                distance += self.config.weight_coefficient * (weight_diff / matching)

        return distance

    def size(self):
        """
        Complexity size: (n_hidden_nodes, enabled_links)
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016
        """
        hid = len(self.hidden_genes)
        enabled_links = sum([1 for gene in self.link_genes if gene.enabled])
        return hid, enabled_links

    def node_genes(self):
        """List of node genes"""
        return self.input_genes + self.hidden_genes + self.output_genes

    def get_link_by_indices(self, src, sink):
        """Returns a link gene from src to sink if it is in genome, else None"""
        for link in self.link_genes:
            if link.src == src and link.sink == sink:
                return link
        return None

    def get_node_by_index(self, idx):
        """Returns a node with index idx if it is in genome, else None"""
        assert idx > 0
        for node in self.input_genes:
            if node.idx == idx:
                return node
        for node in self.hidden_genes:
            if node.idx == idx:
                return node
        for node in self.output_genes:
            if node.idx == idx:
                return node
        return None

    def _random_genome(self, n_inputs, n_outputs):
        """Creates random, fully-connected genome with n_in and n_out"""
        for i in range(n_inputs):
            self.input_genes.append(NodeGene(node_type='INPUT'))
        for i in range(n_outputs):
            self.output_genes.append(NodeGene(node_type='OUTPUT'))
        for sink in self.output_genes:
            self.link_genes.append(LinkGene(0, sink.idx))
            for src in self.input_genes:
                self.link_genes.append(LinkGene(src.idx, sink.idx))

    def _parse_node_genes(self, node_genes):
        """Reads node genes into genome"""
        for gene in node_genes:
            if gene.node_type == 'INPUT':
                self.input_genes.append(gene)
            elif gene.node_type == 'HIDDEN':
                self.hidden_genes.append(gene)
            elif gene.node_type == 'OUTPUT':
                self.output_genes.append(gene)
            else:
                raise ValueError("genes has a node with invalid type")

    def _has_duplicate_node_indices(self):
        """Tests if a node is present twice"""
        node_genes = self.input_genes + self.hidden_genes + self.output_genes
        for i in range(len(node_genes)):
            for j in range(i + 1, len(node_genes)):
                if node_genes[i].idx == node_genes[j].idx:
                    return True
        return False

    def _has_duplicate_links(self):
        """Tests is a link is present twice"""
        links = [(a.src, a.sink) for a in self.link_genes]
        for i in range(len(links)):
            isrc, isnk = links[i]
            for j in range(i + 1, len(links)):
                jsrc, jsnk = links[j]
                if isrc == jsrc and isnk == jsnk:
                    return True
                elif isrc == jsnk and isnk == jsrc:
                    return True
                return False

    def _check_links_have_valid_nodes(self):
        """
        Checks if any links go to nodes which are not present, and
        ensures bias/inputs are never sinks
        """
        input_indices = [g.idx for g in self.input_genes]
        output_indices = [g.idx for g in self.output_genes]
        hidden_indices = [g.idx for g in self.hidden_genes]
        node_indices = [0] + input_indices + output_indices + hidden_indices

        for gene in self.link_genes:
            if gene.src not in node_indices:
                raise ValueError("Link connecting to missing node")
            if gene.sink not in node_indices:
                raise ValueError("Link connecting to missing node")
            if gene.sink in input_indices:
                raise ValueError("Link sink is an input")
            elif gene.sink == 0:
                raise ValueError("Link sink is bias")

    @staticmethod
    def _check_args(n_inputs, n_outputs, node_genes, link_genes):
        """Argument sanity check"""
        if node_genes is None and link_genes is not None:
            raise ValueError("Cannot pass links but not nodes to genome")
        if node_genes is None:
            if n_inputs is None or n_outputs is None:
                raise ValueError("If genes are not supplied, must have n_inputs and n_outputs")
        if n_inputs is not None:
            if n_inputs < 1:
                raise ValueError("Genome needs positive number of inputs")
        if n_outputs is not None:
            if n_outputs < 1:
                raise ValueError("Genome needs positive number of outputs")
