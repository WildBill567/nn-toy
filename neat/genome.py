from .genes import NodeGene, LinkGene

class Genome:

    def __init__(self, *, n_inputs, n_outputs):
        if n_inputs < 1:
            raise ValueError("Genome needs positive number of inputs")
        if n_outputs < 1:
            raise ValueError("Genome needs positive number of outputs")
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs