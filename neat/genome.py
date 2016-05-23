from .genes import NodeGene, LinkGene


class Genome:
    def __init__(self, *, n_inputs=None, n_outputs=None, node_genes=None, link_genes=None):
        self.link_genes = []
        self.input_genes = []
        self.hidden_genes = []
        self.output_genes = []

        if node_genes is None:
            if n_inputs is None or n_outputs is None:
                raise ValueError("If genes are not supplied, must have n_inputs and n_outputs")
        if n_inputs is not None:
            if n_inputs < 1:
                raise ValueError("Genome needs positive number of inputs")
        if n_outputs is not None:
            if n_outputs < 1:
                raise ValueError("Genome needs positive number of outputs")

        if node_genes is not None:
            self._parse_node_genes(node_genes)

            if n_inputs is not None:
                assert n_inputs == len(self.input_genes)
            if n_outputs is not None:
                assert n_outputs == len(self.output_genes)
        elif node_genes is None:
            for i in range(n_inputs):
                self.input_genes.append(NodeGene(node_type='INPUT'))
            for i in range(n_outputs):
                self.output_genes.append(NodeGene(node_type='OUTPUT'))

        if link_genes is not None:
            for gene in link_genes:
                self.link_genes.append(gene)
        self.n_links = len(self.link_genes)

        assert not self._has_duplicate_links()
        assert not self._has_duplicate_node_indices(), "Genome has duplicate node indices"
        self._check_links_have_valid_nodes()

    def get_node_by_index(self, idx):
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

    def _parse_node_genes(self, node_genes):
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
        node_genes = self.input_genes + self.hidden_genes + self.output_genes
        for i in range(len(node_genes)):
            for j in range(i+1, len(node_genes)):
                if node_genes[i].idx == node_genes[j].idx:
                    return True
        return False

    def _has_duplicate_links(self):
        links = [(a.src, a.sink) for a in self.link_genes]
        for i in range(len(links)):
            isrc, isnk = links[i]
            for j in range(i+1, len(links)):
                jsrc, jsnk = links[j]
                if isrc == jsrc and isnk == jsnk:
                    return True
                elif isrc == jsnk and isnk == jsrc:
                    return True
                return False

    def _check_links_have_valid_nodes(self):
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
