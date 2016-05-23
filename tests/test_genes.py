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

from unittest import TestCase

from neat.genes import NodeGene, LinkGene


class TestLinkGene(TestCase):

    def test_link_gene_requires_positive_innovation(self):
        try:
            link = LinkGene(1, 2, innov=-2)
            if link:
                self.fail("Negative innovation")
        except ValueError:
            pass

    def test_link_gene_increments_innovation_counter(self):
        link1 = LinkGene(1, 2)
        link2 = LinkGene(2, 3)
        link3 = LinkGene(1, 3)
        assert link1.innovation_number + 1 == link2.innovation_number
        assert link2.innovation_number + 1 == link3.innovation_number

    def test_link_gene_cannot_be_loop(self):
        try:
            link = LinkGene(1, 1)
            if link:
                self.fail("Created self loop")
        except ValueError:
            pass


class TestNodeGene(TestCase):

    def test_node_gene_requires_positive_index(self):
        try:
            gene = NodeGene(idx=-2)
            if gene:
                self.fail("Should have failed gene constructor for negative index")
        except ValueError:
            pass

    def test_node_gene_requires_valid_activation(self):
        try:
            gene = NodeGene(activation='invalid')
            if gene:
                self.fail("Should have failed assertion in gene constructor for invalid activation")
        except AssertionError:
            pass

    def test_node_gene_requires_valid_node_type(self):
        try:
            gene = NodeGene(node_type="invalid")
            if gene:
                self.fail("Should hvae failed assertion in gene constructor for invalid node type")
        except AssertionError:
            pass

    def test_node_gene_increments_counter(self):
        gene1 = NodeGene()
        gene2 = NodeGene()
        gene3 = NodeGene()
        assert gene1.idx + 1 == gene2.idx
        assert gene2.idx + 1 == gene3.idx

    def test_node_gene_has_no_activation_for_input(self):
        gene = NodeGene(activation='sigmoid', node_type='INPUT')
        assert gene.activation is 'identity', "Input node should have identity activation"
