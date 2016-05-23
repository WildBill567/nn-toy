from unittest import TestCase
import networkx as nx

from neat.genes import NodeGene, LinkGene
from neat.genome import Genome
from neat.feedforwardphenome import FeedForwardPhenome


class TestPhenome(TestCase):

    def setUp(self):
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
        self.genome = Genome(node_genes=self.test_node_genes, link_genes=self.test_link_genes)
        self.phenome = FeedForwardPhenome(self.genome)

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

    def test_draw_premade_phenome(self):
        self.phenome.draw()

    def test_draw_random_phenome(self):
        gen = Genome(n_inputs=3, n_outputs=3)
        phenome = FeedForwardPhenome(genome=gen)
        phenome.draw()
