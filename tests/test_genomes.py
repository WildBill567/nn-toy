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

from neat.config import Config
from neat.genes import NodeGene, LinkGene
from neat.genome import Genome


class TestGenome(TestCase):
    def setUp(self):
        self.config = Config('../tests/conf_tester.conf')
        NodeGene.counter = 1
        LinkGene.innovation_counter = 0
        self.test_node_genes = [NodeGene(node_type='INPUT',),
                                NodeGene(node_type='INPUT'),
                                NodeGene(node_type='OUTPUT'),
                                NodeGene(node_type='OUTPUT')]
        self.test_link_genes = [LinkGene(2, 3, weight=1), LinkGene(1, 3, weight=1)]
        self.genome = Genome(self.config, node_genes=self.test_node_genes, link_genes=self.test_link_genes)

    def test_simple_size(self):
        sz = self.genome.size()
        assert sz == (0, 2), "Size should be (0,2), is (%i, %i)" % (sz[0], sz[1])

    def test_genome_distance_to_self_is_zero(self):
        g2 = Genome(self.config, node_genes=self.test_node_genes, link_genes=self.test_link_genes)
        g1 = self.genome
        dist = g1.distance(g2)
        assert dist == 0, "Distance should be 0, got %.02f" % dist

    def test_randomly_created_genome(self):
        config = self.config
        config.num_outputs = 3
        config.num_inputs = 3
        genome = Genome(config)
        self.config = Config('../tests/conf_tester.conf')
        assert len(genome.output_genes) == 3, \
            "Genome should have 3 outputs, has %i" % len(genome.output_genes)
        assert len(genome.input_genes) == 3, \
            "Genome should have 3 inputs, has %i" % len(genome.input_genes)


    def test_genome_requires_inputs_as_1_to_n(self):
      for gene in self.genome.input_genes:
            assert gene.idx in (1, 2)

    def test_genome_doesnt_create_inputs_as_sinks(self):
        link_tester = [LinkGene(0, 1), LinkGene(3, 2), LinkGene(2, 4)]
        try:
            genome = Genome(self.config, node_genes=self.test_node_genes, link_genes=link_tester)
            self.fail("Genome should raise exception if input is sink")
        except ValueError:
            pass

    def test_genome_doesnt_create_duplicate_links(self):
        link_tester = [LinkGene(0, 3), LinkGene(1, 3), LinkGene(2, 4)]
        try:
            genome = Genome(self.config, node_genes=self.test_node_genes, link_genes=link_tester)
            self.fail("Genome should raise exception if duplicate links exist")
        except AssertionError:
            pass

    def test_genome_only_links_to_valid_nodes(self):
        link_tester = [LinkGene(0, 3), LinkGene(1, 3), LinkGene(2, 4), LinkGene(0, 5)]
        try:
            genome = Genome(self.config, node_genes=self.test_node_genes, link_genes=link_tester)
            self.fail("Genome should raise exception if link connects to non-existing node")
        except ValueError:
            pass

    def test_genome_creates_correct_number_of_links(self):
        assert self.genome.n_links == len(self.test_link_genes), "Genome did not create correct number of links"

    def test_genome_creates_correct_number_of_inputs(self):
        assert len(self.genome.input_genes) == self.config.num_inputs, \
            "Should have 2 inputs, has %i" % len(self.genome.input_genes)

    def test_genome_creates_correct_number_of_outputs(self):
        assert len(self.genome.output_genes) == self.config.num_outputs, "Should have 2 outputs, has %i" % \
                                                                         len(self.genome.output_genes)

    def test_genome_does_not_allow_duplicate_node_indices(self):
        print(len(self.test_node_genes))
        self.test_node_genes[0].idx = 1
        self.test_node_genes[1].idx = 1
        try:
            genome = Genome(self.config, node_genes=self.test_node_genes)
        except AssertionError:
            pass

    def test_genome_needs_right_number_input_genes(self):
        try:
            self.config.num_inputs = 3
            genome = Genome(self.config, node_genes=self.test_node_genes)
            self.config.num_inputs = 2
            self.fail("Should raise exception if n_inputs does not match number of input genes")
        except AssertionError:
            pass

    def test_genome_needs_right_number_output_genes(self):
        try:
            self.config.num_outputs = 3
            genome = Genome(self.config, node_genes=self.test_node_genes)
            self.config.num_outputs = 2
            self.fail("Should raise exception if n_outputs does not match number of output genes")
        except AssertionError:
            pass

