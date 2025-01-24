# -*- coding: utf-8 -*-
"""
importing functions
"""
#%% section 1: import functions 

from collections import Counter
import numpy as np
import networkx as nx
import random
import time
import pickle
import os
import pandas as pd  # Use this if your train_data is a CSV file
from functions1 import (
    load_graph,
    display_graph_statistics,
    snowball_sampling_with_infection_time,
    precompute_mc_set,
    fetch_samples_from_mc_set,
    save_mc_set,
    load_mc_set
)

from functions2 import (
    calculate_discrepancy,
    calculate_discrepancies_for_samples,
    calculate_loss,
    compute_t_s,
    calculate_expected_discrepancies_from_mc_set,
    calculate_expected_discrepancies_from_mc_set_optimized,
    calculate_expected_discrepancies_from_mc_set_optimized2
)

from functions3 import (
    snowball_sampling_with_infection_time,
    snowball_sampling_with_infection_time_trim,
    fetch_samples_from_mc_set,
    calculate_loss,
    calculate_p_values_for_infected_nodes,
    generate_observed_sequence,
    sample_and_compute
)

#%% section 2: load data
'''
load data
change the data set you want to load here
'''

edges_file_path ='dataset/ia-enron-only/ia-enron-only.mtx'

#edges_file_path ='dataset/ca-sandi_auths/ca-sandi_auths.mtx'

# Save the current working directory
current_dir = os.getcwd()

# Move to the root directory (parent of 'code')
root_dir = os.path.dirname(current_dir)

# Construct the path 
data_path = os.path.join(root_dir, edges_file_path)
G = load_graph(data_path)

display_graph_statistics(G , name = edges_file_path)

# Move back to the current directory
os.chdir(current_dir)

# Confirm the working directory is back to the original
print("Current directory:", os.getcwd())

dat_name = os.path.splitext(os.path.basename(data_path))[0]

#%% section 3.1: MC set generating
'''
Main processing
'''
start_time_mcset = time.time()
m = 100  # Number of samples per node

# Step 1: Precompute MC-set
print("Precomputing MC-set...")
mc_set = precompute_mc_set(G, m)
save_mc_set(mc_set, f'mc_set_{dat_name}_size_{m}.pkl')

#%% section 3.2: MC set display

# Step 2: Fetch samples for a given source and infection size
s = random.choice(list(G.nodes))  # Example source node
T = len(G) // 5  # Example infection size (20% of |V|)

trimmed_samples = fetch_samples_from_mc_set(mc_set, s, T)

# Display results
print(f"Trimmed samples for source node {s} with size {T}:")
for i, sample in enumerate(trimmed_samples[:5]):  # Display first 5 samples for brevity
    print(f"Sample {i + 1}: {sample}")


print(f"Execution Time for MC set: {time.time() - start_time_mcset:.2f} seconds")


#%% section 4 : Calculate discrepancy
start_time_discrepancies = time.time()

discrepancies = {}  # Initialize the dictionary to store discrepancies

for s in G.nodes:  # Iterate over source nodes 0 to 197
    #start_time = time.time()
    #print(s)
    discrepancies[s] = {}  # Initialize a nested dictionary for each source node
    for T in range(1, 76):  # Iterate over T = 1 to 75
        # Get discrepancies for the current s and T
        discrepancies[s][T] = calculate_expected_discrepancies_from_mc_set_optimized2(mc_set, s, T)
    #print(f"Execution Time: {time.time() - start_time:.2f} seconds")

# Save discrepancies to a file

discrepancies_filename = f"discrepancies_symmetric_difference_{dat_name}_size_{m}.pkl"

with open(discrepancies_filename, 'wb') as f:
    pickle.dump(discrepancies, f)
print(f"Discrepancies saved to {discrepancies_filename}")

print(f"Execution Time for discrepancies: {time.time() - start_time_discrepancies:.2f} seconds")




#%% section 5 : p-value calculation

##### For P values simple use : np.mean(np.array(discrepancies[s][T])>T(y)) , T(y) is observed discrepencies
print("start samples ....")
start_time = time.time()

results = []  # Initialize a list to store results

num_samples = 10*len(G)

Ts = [20,25,30,35,40]

for T in Ts:
    print("")
    print(f"Processing T = {T}...")
    print("")
    for t in range(num_samples):
        # Log progress every 1% of num_samples
        if (t + 1) % (num_samples // 50) == 0:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print(f"Progress: {(t + 1) / num_samples:.0%} ({t + 1}/{num_samples}), Elapsed time: {elapsed_time:.2f} seconds")

        node_list = list(G.nodes)
        true_source = node_list[t % len(node_list)]
    
        p_values = {}
        observed_sequence = generate_observed_sequence(G, true_source, T)
        observed_set = set(observed_sequence)
    
        for s in observed_set:
            # Fetch simulated samples for this node
            simulated_samples = fetch_samples_from_mc_set(mc_set, s, T)
            m = len(simulated_samples) // 2
    
            # Split into first and second halves
            simulated_samples_first = simulated_samples[:m]
            simulated_samples_second = simulated_samples[m:]
    
            # Step 1: Compute T_s(y)
            T_s_y = compute_t_s(observed_sequence, simulated_samples_first)
    
            # Step 2: Compute T_s(zeta(Z_j)) for each sample in the second set
            # This is precalculated and recorded in discrepancies[s][T]
    
            # Step 3: Calculate p-value
            p_value = np.mean(np.array(discrepancies[s][T]) > T_s_y)
    
            # Store p-value for this node
            p_values[s] = p_value
    
        # Step 4: Calculate the rank of the true source based on p-values
        sorted_nodes = sorted(p_values, key=lambda node: p_values[node], reverse=True)
        true_source_rank = sorted_nodes.index(true_source) + 1  # Rank starts from 1
    
        # Record the results
        results.append({
            "true_source": true_source,
            "T": T,
            "observed_sequence": observed_sequence,
            "p_values": p_values,
            "true_source_rank": true_source_rank
        })

print("Sampling completed!")

# Extract all true_source values from results
true_sources = [result["true_source"] for result in results]

# Count the occurrences of each true_source
source_counts = Counter(true_sources)

# Check if the distribution is balanced
min_count = min(source_counts.values())
max_count = max(source_counts.values())

if max_count == min_count:
    print("True sources are perfectly balanced!")
else:
    print(f"True sources are not perfectly balanced. Min: {min_count}, Max: {max_count}")
    
# Save results to a file
with open(f"results_symmetric_difference_{dat_name}_size_{m}.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"Results saved to results_symmetric_difference_{dat_name}_size_{m}.pkl")

print(f"Execution Time: {time.time() - start_time:.2f} seconds")