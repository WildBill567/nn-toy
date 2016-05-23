import numpy.random as random

from neat.activations import activation_types

# Adapted from
# https://github.com/CodeReclaimers/neat-python, accessed May 2016


class NodeGene:

    # counter starts at 1 and goes up
    # TODO find a better way
    counter = 1

    def __init__(self, *, activation='sigmoid', node_type='HIDDEN', idx=-1):
        """
        NodeGene - class for nodes in a NEAT network
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016

        @param activation: node's activation function
        @param node_type: node type, one of 'HIDDEN', 'INPUT', 'OUTPUT'
        @param idx: node's index
        """
        if idx == -1:
            self.idx = NodeGene.counter
            NodeGene.counter += 1
        elif idx >= 0:
            self.idx = idx
        else:
            raise ValueError("Node gene cannot have negative index")

        self.node_type = node_type
        assert self.node_type in ['INPUT', 'OUTPUT', 'HIDDEN', 'BIAS'], "Invalid node type"

        if self.node_type == 'INPUT':
            self.activation = 'identity'
        else:
            self.activation = activation
            assert self.activation in activation_types, "Invalid activation"


class LinkGene:
    innovation_counter = 0

    def __init__(self, src_node, sink_node, *, weight=None, innov=-1, enabled=True):
        """
        LinkGene - Class for gene for link between nodes
        Adapted from: https://github.com/CodeReclaimers/neat-python, accessed May 2016

        :param src_node: source
        :param sink_node: sink
        :param weight: weight
        :param innov: innovation number
        :param enabled: is link active?
        """
        if src_node == sink_node:
            raise ValueError("Links cannot be self-loops")
        self.src = src_node
        self.sink = sink_node
        self.enabled = enabled

        if weight is None:
            self.weight = random.random()
        else:
            self.weight = weight

        if innov == -1:
            self.innovation_number = LinkGene.innovation_counter
            LinkGene.innovation_counter += 1
        elif innov >= 0:
            self.innovation_number = innov
        else:
            raise ValueError("Link gene cannot have negative innovation number")
