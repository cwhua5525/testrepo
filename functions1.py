# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 23:20:07 2025

@author: chenwei

Save the functions needed to for p-value approach
"""

import networkx as nx
import random
import time
import pickle

def relabel_graph_if_needed(G):
    """
    Relabels the nodes of a graph to 0, 1, 2, ..., n-1 if they are not already labeled as such.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        networkx.Graph: The relabelled graph (or the original if no relabeling was needed).
    """
    # Check if the node labels are already 0, 1, 2, ..., n-1
    expected_labels = set(range(len(G.nodes)))
    current_labels = set(G.nodes)
    
    if current_labels == expected_labels:
        print("Graph node labels are already 0, 1, 2, ..., n-1. No relabeling needed.")
        return G  # Return the original graph

    # Create the mapping to relabel nodes
    node_mapping = {old_name: new_index for new_index, old_name in enumerate(G.nodes)}

    # Relabel the nodes
    G_relabelled = nx.relabel_nodes(G, node_mapping)
    print("Graph node labels have been relabelled to 0, 1, 2, ..., n-1.")
    return G_relabelled

def load_graph(file_path):
    """
    Loads a graph from a file, compatible with .mtx and .edges formats.

    Args:
        file_path (str): Path to the graph file.

    Returns:
        networkx.Graph: The loaded graph.
    """
    G = nx.Graph()

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('%'):  # Skip comments
                continue
            # Determine the delimiter
            delimiter = ',' if ',' in line else ' '
            fields = line.strip().split(delimiter)

            # Extract node indices
            node1, node2 = map(int, fields[:2])

            # Check for weights (optional)
            if len(fields) > 2:
                weight = float(fields[2])
                G.add_edge(node1 - 1, node2 - 1, weight=weight)  # Adjust for 1-based indexing
            else:
                G.add_edge(node1 - 1, node2 - 1)
            
    G_relabelled = relabel_graph_if_needed(G)

    return G_relabelled

def display_graph_statistics(G, name="Graph"):
    """
    Computes and displays statistics for a given graph in a formatted, vertical manner.

    Args:
        G (networkx.Graph): The graph to analyze.
        name (str): Name of the graph for identification.
    """
    # Compute statistics
    stats = {
        "Graph Name": name,
        "Number of Nodes": G.number_of_nodes(),
        "Number of Edges": G.number_of_edges(),
        "Is Directed": G.is_directed(),
        "Graph Density": round(nx.density(G), 4),  # Round density to 4 digits
        "Number of Connected Components": nx.number_connected_components(G) if not G.is_directed() else "N/A",
    }

    # Display statistics
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


def snowball_sampling_with_infection_time(G, start_node):
    """
    Perform snowball sampling until all nodes in the graph are infected,
    recording the infection time (order of each node).

    Parameters:
    -----------
    G : nx.Graph
        The input network graph.
    start_node : int
        The starting node for the infection process.

    Returns:
    --------
    infection_order : List[int]
        A list where the index represents the infection time, and the value is the node infected at that time.
    """
    visited = set([start_node])
    frontier_edges = [(start_node, neighbor) for neighbor in G.neighbors(start_node)]
    infection_order = [start_node]

    while frontier_edges:
        edge = random.choice(frontier_edges)
        new_node = edge[1] if edge[0] in visited else edge[0]

        if new_node not in visited:
            visited.add(new_node)
            infection_order.append(new_node)
            new_edges = [(new_node, neighbor) for neighbor in G.neighbors(new_node) if neighbor not in visited]
            frontier_edges.extend(new_edges)

        # Remove edges where both nodes are already infected
        frontier_edges = [e for e in frontier_edges if e[0] not in visited or e[1] not in visited]

    return infection_order

def precompute_mc_set(G, m: int):
    """
    Precompute the MC-set, which contains |V| * 2m infection sequences,
    where each sequence is the order of node infections starting from a specific source node.

    Parameters:
    -----------
    G : nx.Graph
        The input network graph.
    m : int
        Number of Monte Carlo samples per source node.

    Returns:
    --------
    mc_set : Dict[int, List[List[int]]]
        A dictionary where keys are source nodes, and values are lists of infection sequences.
    """
    mc_set = {}

    for node in G.nodes:
        mc_set[node] = [snowball_sampling_with_infection_time(G, node) for _ in range(2 * m)]

    return mc_set

def fetch_samples_from_mc_set(mc_set, s: int, T: int):
    """
    Fetch precomputed samples from the MC-set for a given source node and infection size.

    Parameters:
    -----------
    mc_set : Dict[int, List[List[int]]]
        The precomputed MC-set containing infection sequences for all source nodes.
    s : int
        The source node for which samples are requested.
    T : int
        The infection size (number of nodes to include in the sequence).

    Returns:
    --------
    trimmed_samples : List[List[int]]
        A list of infection sequences trimmed to the first T infected nodes.
    """
    samples = mc_set[s]
    trimmed_samples = [seq[:T] for seq in samples]
    return trimmed_samples

# Save the MC-set to a file
def save_mc_set(mc_set, filename):
    """
    Save the precomputed MC-set to a file using pickle.

    Parameters:
    -----------
    mc_set : Dict[int, List[List[int]]]
        The precomputed MC-set containing infection sequences for all source nodes.
    filename : str
        The path to the file where the MC-set will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(mc_set, f)
    print(f"MC-set saved to {filename}")

# Load the MC-set from a file
def load_mc_set(filename):
    """
    Load a precomputed MC-set from a file using pickle.

    Parameters:
    -----------
    filename : str
        The path to the file where the MC-set is stored.

    Returns:
    --------
    mc_set : Dict[int, List[List[int]]]
        The loaded MC-set containing infection sequences for all source nodes.
    """
    with open(filename, 'rb') as f:
        mc_set = pickle.load(f)
    print(f"MC-set loaded from {filename}")
    return mc_set