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

#edges_file_path ='dataset/ia-enron-only/ia-enron-only.mtx'

edges_file_path ='dataset/ca-sandi_auths/ca-sandi_auths.mtx'

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

