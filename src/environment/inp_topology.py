import random
import networkx as nx
import matplotlib.pyplot as plt


def createGraph2(write_path, num_mec, num_dc, graph_seed, show_graph=True):
    """
        Creates a graph with a specified number of MEC (Mobile Edge Computing) and DC (Data Center) nodes.

        Parameters:
        - write_path (str): Path to save the generated graph file.
        - num_mec (int): Number of MEC nodes in the graph.
        - num_dc (int): Number of DC nodes in the graph.
        - graph_seed (int): Seed for random number generation to ensure reproducibility.
        - show_graph (bool, optional): Whether to display the graph using Matplotlib. Default is True.

        Returns:
        - G (networkx.Graph): Generated graph.
        """
    random.seed(graph_seed)

    total_cp = num_mec + num_dc
    list_of_cp = []  # list of cp numbers
    for cp_num in range(total_cp):
        list_of_cp.append(cp_num)  # create a list of cp numbers from zero to total_cp
    random.shuffle(list_of_cp)  # shuffle the list of cp numbers

    G = nx.Graph()

    counter = 0
    while counter < total_cp:
        if counter < num_mec:
            mec_num = list_of_cp[counter]  # get the mec_num from the list_of_cp
            #  (node_num(int), cpu(float), memory(GiB),    bandwidth=(Gbps), icd(ms),  icr(Gbps), type(string))
            G.add_node(mec_num, cpu=16.0, memory=400.0, bandwidth=25.0, icd=10.0, icr=25.0, type='MEC')  # Add MEC nodes
            counter += 1
        else:
            dc_num = list_of_cp[counter]  # get the dc_num from list of cps
            G.add_node(dc_num, cpu=64.0, memory=14000.0, bandwidth=25.0, icd=10.0, icr=25.0, type='DC')  # Add DC nodes
            counter += 1

    # Add edges with latency based on node types
    for i in range(len(list_of_cp)):
        for j in range(len(list_of_cp)):
            if list_of_cp[i] != list_of_cp[j] and list_of_cp[j] != list_of_cp[i]:
                if G.nodes[i]['type'] == 'MEC' and G.nodes[j]['type'] == 'MEC':
                    G.add_edge(list_of_cp[i], list_of_cp[j], latency=3.2)
                elif G.nodes[i]['type'] == 'MEC' and G.nodes[j]['type'] == 'DC':
                    G.add_edge(list_of_cp[i], list_of_cp[j], latency=3.5)
                elif G.nodes[i]['type'] == 'DC' and G.nodes[j]['type'] == 'MEC':
                    G.add_edge(list_of_cp[i], list_of_cp[j], latency=3.5)
                elif G.nodes[i]['type'] == 'DC' and G.nodes[j]['type'] == 'DC':
                    G.add_edge(list_of_cp[i], list_of_cp[j], latency=4.7)

    print("Graph is_directed: ", nx.is_directed(G))
    print("Graph is_connected: ", nx.is_connected(G))
    nx.draw(G, with_labels=True, font_weight='bold')

    nx.write_gpickle(G, write_path)
    nx.write_gml(G, write_path + ".gml")
    if show_graph:
        plt.show()
    return G


def createGraph(write_path, graph_seed, show_graph=True):
    """
        Creates a simple graph with predefined nodes, edges, and attributes.

        Parameters:
        - write_path (str): Path to save the generated graph file.
        - graph_seed (int): Seed for random number generation to ensure reproducibility.
        - show_graph (bool, optional): Whether to display the graph using Matplotlib. Default is True.

        Returns:
        - G (networkx.Graph): Generated graph.
        """

    random.seed(graph_seed)

    G = nx.Graph()
    G.add_node(0, cpu=64.0, memory=14000.0, bandwidth=25.0, icd=10.0, icr=25.0, type='MEC')
    G.add_node(1, cpu=64.0, memory=14000.0, bandwidth=25.0, icd=20.0, icr=25.0, type='DC')
    G.add_node(2, cpu=64.0, memory=14000.0, bandwidth=25.0, icd=30.0, icr=25.0, type='DC')
    G.add_node(3, cpu=64.0, memory=14000.0, bandwidth=25.0, icd=40.0, icr=25.0, type='DC')

    G.add_edge(0, 1, latency=5.0)
    G.add_edge(1, 2, latency=10.0)
    G.add_edge(2, 0, latency=15.0)
    G.add_edge(3, 0, latency=20.0)
    G.add_edge(3, 1, latency=25.0)
    G.add_edge(3, 2, latency=30.0)

    nx.draw(G, with_labels=True, font_weight='bold')

    nx.write_gpickle(G, write_path)
    if show_graph:
        plt.show()
    return G


if __name__ == '__main__':
    createGraph2("graph2.gpickle", 1, 3, show_graph=True)
