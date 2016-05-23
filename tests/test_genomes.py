from unittest import TestCase

from neat.genes import NodeGene, LinkGene
from neat.genome import Genome


class TestGenome(TestCase):
    def setUp(self):
        NodeGene.counter = 1
        LinkGene.innovation_counter = 0
        self.test_node_genes = [NodeGene(node_type='INPUT',),
                                NodeGene(node_type='INPUT'),
                                NodeGene(node_type='OUTPUT'),
                                NodeGene(node_type='OUTPUT')]
        self.test_link_genes = [LinkGene(2, 3, weight=1), LinkGene(1, 3, weight=1)]

    def test_simple_size(self):
        g = Genome(node_genes=self.test_node_genes, link_genes=self.test_link_genes)
        sz = g.size()
        assert sz == (0, 2), "Size should be (0,2), is (%i, %i)" % (sz[0], sz[1])

    def test_genome_distance_to_self_is_zero(self):
        g2 = Genome(node_genes=self.test_node_genes, link_genes=self.test_link_genes)
        g1 = Genome(node_genes=self.test_node_genes, link_genes=self.test_link_genes)
        dist = g1.distance(g2)
        assert dist == 0, "Distance should be 0, got %.02f" % dist

    def test_randomly_created_genome(self):
        genome = Genome(n_inputs=3, n_outputs=3)
        assert len(genome.output_genes) == 3, \
            "Genome should have 3 outputs, has %i" % len(genome.output_genes)
        assert len(genome.input_genes) == 3, \
            "Genome should have 3 inputs, has %i" % len(genome.input_genes)

    def test_genome_requires_inputs_as_1_to_n(self):
        genome = Genome(node_genes=self.test_node_genes, link_genes=self.test_link_genes)
        for gene in genome.input_genes:
            assert gene.idx in (1, 2)

    def test_genome_doesnt_create_inputs_as_sinks(self):
        link_tester = [LinkGene(0, 1), LinkGene(3, 2), LinkGene(2, 4)]
        try:
            genome = Genome(node_genes=self.test_node_genes, link_genes=link_tester)
            self.fail("Genome should raise exception if input is sink")
        except ValueError:
            pass

    def test_genome_doesnt_create_duplicate_links(self):
        link_tester = [LinkGene(0, 3), LinkGene(1, 3), LinkGene(2, 4)]
        try:
            genome = Genome(node_genes=self.test_node_genes, link_genes=link_tester)
            self.fail("Genome should raise exception if duplicate links exist")
        except AssertionError:
            pass

    def test_genome_only_links_to_valid_nodes(self):
        link_tester = [LinkGene(0, 3), LinkGene(1, 3), LinkGene(2, 4), LinkGene(0, 5)]
        try:
            genome = Genome(node_genes=self.test_node_genes, link_genes=link_tester)
            self.fail("Genome should raise exception if link connects to non-existing node")
        except ValueError:
            pass

    def test_genome_creates_correct_number_of_links(self):
        genome = Genome(node_genes=self.test_node_genes, link_genes=self.test_link_genes)
        assert genome.n_links == len(self.test_link_genes), "Genome did not create correct number of links"

    def test_genome_creates_correct_number_of_inputs(self):
        genome = Genome(node_genes=self.test_node_genes)
        assert len(genome.input_genes) == 2, "Should have 2 inputs, has %i" % len(genome.input_genes)

    def test_genome_creates_correct_number_of_outputs(self):
        genome = Genome(node_genes=self.test_node_genes)
        assert len(genome.output_genes) == 2, "Should have 2 outputs, has %i" % len(genome.output_genes)

    def test_genome_does_not_allow_duplicate_node_indices(self):
        print(len(self.test_node_genes))
        self.test_node_genes[0].idx = 1
        self.test_node_genes[1].idx = 1
        try:
            genome = Genome(node_genes=self.test_node_genes)
        except AssertionError:
            pass

    def test_genome_needs_right_number_input_genes(self):
        try:
            genome = Genome(n_inputs=1, n_outputs=2, node_genes=self.test_node_genes)
            self.fail("Should raise exception if n_inputs does not match number of input genes")
        except AssertionError:
            pass

    def test_genome_needs_right_number_output_genes(self):
        try:
            genome = Genome(n_inputs=2, n_outputs=1, node_genes=self.test_node_genes)
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
            genome = Genome(n_inputs=0, n_outputs=1)
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
