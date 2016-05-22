from unittest import TestCase

from neat.genes import NodeGene, LinkGene
from neat.genome import Genome

class TestGenome(TestCase):

    def setUp(self):
        self.test_genes = [NodeGene(node_type='INPUT'), NodeGene(node_type='INPUT'), NodeGene(node_type='OUTPUT')]

    def test_genome_does_not_allow_duplicate_node_indices(self):
        print(len(self.test_genes))
        self.test_genes[0].idx = 1
        self.test_genes[1].idx = 1
        try:
            genome = Genome(node_genes=self.test_genes)
        except AssertionError:
            pass

    def test_genome_needs_right_number_input_genes(self):
        try:
            genome = Genome(n_inputs=1, n_outputs=2, node_genes=self.test_genes)
            self.fail("Should raise exception if n_inputs does not match number of input genes")
        except AssertionError:
            pass

    def test_genome_needs_right_number_output_genes(self):
        try:
            genome = Genome(n_inputs=2, n_outputs=1, node_genes=self.test_genes)
            self.fail("Should raise exception if n_outputs does not match number of output genes")
        except AssertionError:
            pass

    def test_genome_needs_genes_or_input_output_numbers(self):
        try:
            genome = Genome()
            self.fail("Should raise exception if no args supplied to genome")
        except ValueError:
            pass

    def test_genome_requires_inputs(self):
        try:
            genome = Genome(n_inputs = 0, n_outputs = 1)
            self.fail("Genome cannot have zero inputs")
        except ValueError:
            pass
        try:
            genome = Genome(n_inputs=-1, n_outputs=1)
            self.fail("Genome needs positive number of inputs")
        except ValueError:
            pass

    def test_genome_requires_positive_outputs(self):
        try:
            genome = Genome(n_inputs=1, n_outputs=0)
            self.fail("Genome cannot have zero outputs")
        except ValueError:
            pass
        try:
            genome = Genome(n_inputs=1, n_outputs=-1)
            self.fail("Genome needs positive number of outputs")
        except ValueError:
            pass



