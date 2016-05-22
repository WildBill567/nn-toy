from unittest import TestCase

from neat.genome import Genome

class TestGenome(TestCase):

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



