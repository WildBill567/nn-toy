from unittest import TestCase
import networkx as nx

from neat.genes import NodeGene, LinkGene
from neat.genome import Genome
from neat.phenome import Phenome


class TestPhenome(TestCase):

    def setUp(self):
        NodeGene.counter = 1
        self.test_node_genes = [NodeGene(node_type='INPUT', activation='identity'),
                                NodeGene(node_type='INPUT', activation='identity'),
                                NodeGene(node_type='OUTPUT', activation='identity'),
                                NodeGene(node_type='OUTPUT', activation='identity')]
        self.test_link_genes = [LinkGene(0, 3), LinkGene(1, 3), LinkGene(1,4), LinkGene(2, 4)]
        self.genome = Genome(node_genes=self.test_node_genes, link_genes=self.test_link_genes)

    def test_phenome_requires_correct_num_inputs(self):
        phenome = Phenome(self.genome)
        try:
            phenome.serial_activate([1.0, 1.0, 1.0])
            self.fail("Phenome should raise exception if number of inputs is wrong")
        except ValueError:
            pass

    def test_draw(self):
        phenome = Phenome(self.genome)
        phenome.draw()
