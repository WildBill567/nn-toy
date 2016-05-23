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
from nose.plugins.attrib import attr

from neat.config import Config
from neat.genes import NodeGene, LinkGene
from neat.genome import Genome
from neat.phenome import FeedForwardPhenome


class TestPhenome(TestCase):

    def setUp(self):
        self.config = Config('../tests/conf_tester.conf')
        NodeGene.counter = 1
        self.test_node_genes = [NodeGene(node_type='INPUT', activation='identity'),
                                NodeGene(node_type='INPUT', activation='identity'),
                                NodeGene(node_type='OUTPUT', activation='identity'),
                                NodeGene(node_type='OUTPUT', activation='identity')]
        self.test_link_genes = [LinkGene(1, 3, weight=1),
                                LinkGene(2, 3, weight=1),
                                LinkGene(1, 4, weight=1),
                                LinkGene(2, 4, weight=1),
                                LinkGene(0, 3, weight=1),
                                LinkGene(0, 4, weight=1)]
        self.genome = Genome(self.config, node_genes=self.test_node_genes, link_genes=self.test_link_genes, )
        self.phenome = FeedForwardPhenome(self.genome, self.config)

    def test_phenome_gives_correct_output_for_simple_net(self):
        outputs = self.phenome.serial_activate([1.0, 1.0])
        assert outputs == [3.0, 3.0], "Outpt should be [3, 3], got %s" % ("%.02f, %.02f" % (outputs[0], outputs[1]))

    def test_phenome_requires_correct_num_inputs(self):
        try:
            self.phenome.serial_activate([1.0, 1.0, 1.0])
            self.fail("Phenome should raise exception if number of inputs is wrong")
        except ValueError:
            pass

    def test_phenome_returns_correct_num_outputs(self):
        outputs = self.phenome.serial_activate([1.0, 1.0])
        assert len(outputs) == len(self.genome.output_genes)

    @attr('draw')
    def test_draw_premade_phenome(self):
        self.phenome.draw(testing=True)

    @attr('draw')
    def test_draw_random_phenome(self):
        self.config.num_inputs = 3
        self.config.num_outputs = 3
        gen = Genome(self.config)
        phenome = FeedForwardPhenome(genome=gen, config=self.config)
        self.config.num_inputs = 2
        self.config.num_outputs = 2
        phenome.draw(testing=True)
