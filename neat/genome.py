from .genes import NodeGene, LinkGene


class Genome:
    def __init__(self, *, n_inputs=None, n_outputs=None, node_genes=None, link_genes=None):
        self.input_genes = []
        self.output_genes = []
        self.hidden_genes = []
        self.link_genes = []

        if node_genes is None:
            if n_inputs is None or n_outputs is None:
                raise ValueError("If genes are not supplied, must have n_inputs and n_outputs")
        if n_inputs is not None:
            if n_inputs < 1:
                raise ValueError("Genome needs positive number of inputs")
            else:
                self.n_inputs = n_inputs
        if n_outputs is not None:
            if n_outputs < 1:
                raise ValueError("Genome needs positive number of outputs")
            else:
                self.n_outputs = n_outputs

        if node_genes is not None:
            for gene in node_genes:
                if gene.node_type == 'INPUT':
                    self.input_genes.append(gene)
                elif gene.node_type == 'OUTPUT':
                    self.output_genes.append(gene)
                elif gene.node_type == 'HIDDEN':
                    self.hidden_genes.append(gene)
                else:
                    raise ValueError("genes has a node with invalid type")

            if n_inputs is None:
                self.n_inputs = len(self.input_genes)
            else:
                assert n_inputs == len(self.input_genes)
            if n_outputs is None:
                self.n_outputs = len(self.output_genes)
            else:
                assert n_outputs == len(self.output_genes)

            self.n_hidden = len(self.hidden_genes)
        else:
            for i in range(self.n_inputs):
                self.input_genes.append(NodeGene(node_type='INPUT'))
            for i in range(self.n_outputs):
                self.output_genes.append(NodeGene(node_type='OUTPUT'))

        assert not self._has_duplicate_node_indices(), "Genome has duplicate node indices"

        if link_genes is not None:
            for gene in link_genes:
                self.link_genes.append(gene)
            self.n_links = len(self.link_genes)

    def _has_duplicate_node_indices(self):
        node_genes = self.input_genes + self.hidden_genes + self.output_genes
        for i in range(len(node_genes)):
            for j in range(i+1, len(node_genes)):
                if node_genes[i].idx == node_genes[j].idx:
                    return True
        return False
