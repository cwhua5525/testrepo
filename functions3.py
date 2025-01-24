import networkx as nx
import random
import time
import pickle
import numpy as np
from typing import List, Dict
from collections import Counter

def snowball_sampling_with_infection_time(G, start_node):
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

def snowball_sampling_with_infection_time_trim(G, start_node,T):
    visited = set([start_node])
    frontier_edges = [(start_node, neighbor) for neighbor in G.neighbors(start_node)]
    infection_order = [start_node]
    while frontier_edges and len(infection_order) < T:
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

# Fetch samples from MC-set
def fetch_samples_from_mc_set(mc_set, s: int, T: int):
    samples = mc_set[s]
    trimmed_samples = [seq[:T] for seq in samples]
    return trimmed_samples

# Calculate loss between observed and simulated infection sequences
def calculate_loss(y: List[int], z: List[int]) -> float:
    observed_set = set(y)
    simulated_set = set(z)
    return len(observed_set.symmetric_difference(simulated_set))

# Compute expected loss T_s(y) using Monte Carlo samples
def compute_t_s(y: List[int], simulated_samples: List[List[int]]) -> float:
    losses = [calculate_loss(y, z) for z in simulated_samples]
    return sum(losses) / len(losses)

# Calculate p-values for all infected nodes and rank the true source
def calculate_p_values_for_infected_nodes(
    observed_sequence: List[int], mc_set: Dict[int, List[List[int]]], T: int, m: int, true_source: int
) -> Dict[str, Dict[int, float]]:
    p_values = {}
    observed_set = set(observed_sequence)

    for s in observed_set:
        # Fetch simulated samples for this node
        simulated_samples = fetch_samples_from_mc_set(mc_set, s, T)

        # Split into first and second halves
        simulated_samples_first = simulated_samples[:m]
        simulated_samples_second = simulated_samples[m:]

        # Step 1: Compute T_s(y)
        T_s_y = compute_t_s(observed_sequence, simulated_samples_first)

        # Step 2: Compute T_s(zeta(Z_j)) for each sample in the second set
        T_s_z = [compute_t_s(z, simulated_samples_second) for z in simulated_samples_first]

        # Step 3: Calculate p-value
        p_value = sum(1 for t in T_s_z if t >= T_s_y) / len(T_s_z)

        # Store p-value for this node
        p_values[s] = p_value

    # Step 4: Calculate the rank of the true source based on p-values
    sorted_nodes = sorted(p_values, key=lambda node: p_values[node], reverse=True)
    true_source_rank = sorted_nodes.index(true_source) + 1  # Rank starts from 1

    return {"p_values": p_values, "true_source_rank": {true_source: true_source_rank}}

# Generate observed infection sequence
def generate_observed_sequence(G, source, T):
    full_infection_sequence = snowball_sampling_with_infection_time(G, source)
    return full_infection_sequence[:T]

# Main sampling and computation function
def sample_and_compute(G, mc_set, num_samples=1000, log_file="training_log.txt", results_file="results.pkl"):
    results = []
    mean_T = len(G) / 5  # Poisson mean based on graph size

    with open(log_file, "w") as log:
        log.write("Starting training...\n")

        for i in range(num_samples):
            # Step 1: Generate a random infection size T
            T = np.random.poisson(mean_T)
            if T <= 0:  # Ensure T is at least 1
                continue

            # Step 2: Select a random source node and generate an observed sequence
            source = random.choice(list(G.nodes))
            observed_sequence = generate_observed_sequence(G, source, T)
            

            # Step 3: Calculate p-values and rank of the true source
            result = calculate_p_values_for_infected_nodes(
                observed_sequence=observed_sequence,
                mc_set=mc_set,
                T=T,
                m=1000,  # Assuming 250 Monte Carlo samples
                true_source=source,
            )

            # Store the results
            results.append({
                "source": source,
                "infected_set": observed_sequence,
                "p_values": result["p_values"],
                "true_source_rank": result["true_source_rank"][source]
            })

            # Log progress every 100 samples
            if (i + 1) % 100 == 0:
                log.write(f"Processed {i + 1}/{num_samples} samples...\n")
                print(f"Processed {i + 1}/{num_samples} samples...")

        log.write("Training completed.\n")

    # Save results to file
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_file}")

    return results
