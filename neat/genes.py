

class NodeGene:
    #Counter starts at zero, phenotype will always create a bias
    #node with index 0
    #TODO initialize this at program start
    counter = 1

    def __init__(self, *, activation='sigmoid', node_type='HIDDEN', idx=-1):
        if idx == -1:
            self.idx = NodeGene.counter
            NodeGene.counter += 1
        elif idx >= 0:
            self.idx = idx
        else:
            raise ValueError("Node gene cannot have negative index")

        self.node_type = node_type
        assert self.node_type in ['INPUT', 'OUTPUT', 'HIDDEN'], "Invalid node type"

        if self.node_type == 'INPUT':
            self.activation = None
        else:
            self.activation = activation
            assert self.activation in ['sigmoid', 'relu', 'tanh'], "Invalid activation"


class LinkGene:
    innovation_counter = 0

    def __init__(self, src_node, sink_node, *, weight=0.0, innov=-1, enabled=True):
        if src_node == sink_node:
            raise ValueError("Links cannot be self-loops")
        self.src = src_node
        self.sink = sink_node
        self.weight = weight
        self.enabled = enabled

        if innov == -1:
            self.innovation_number = LinkGene.innovation_counter
            LinkGene.innovation_counter += 1
        elif innov >= 0:
            self.innovation_number = innov
        else:
            raise ValueError("Link gene cannot have negative innovation number")
