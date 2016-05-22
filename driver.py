import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge(1, 2)
    nx.draw(G)
    plt.show()