# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:00:15 2025

@author: cwhua
"""
#%% section 1 : load data
import math
import networkx as nx
import random
import time
import pickle
import numpy as np
from typing import List, Dict
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import poisson

m = 1000

dat_name = 'ia-enron-only'
#dat_name = 'ca-sandi_auths'
#dat_name = 'eco-florida'

discrepancies_filename = f"files/discrepancies_symmetric_difference_{dat_name}_size_{m}.pkl"
with open(discrepancies_filename, 'rb') as f:
    discrepancies = pickle.load(f)

results_filename = f"files/results_symmetric_difference_{dat_name}_size_{m}.pkl"  
with open(results_filename, "rb") as f:
    results = pickle.load(f)

true_sources = [result["true_source"] for result in results]
Ts = [result["T"] for result in results if result["true_source"] == 0]

Rankings = [result["true_source_rank"] for result in results if result["true_source"] == 0]


#%% 1 : Checking if the T is correctly poisson distributed
"""
1. Checking if the T is correctly poisson distributed
"""

# Generate Poisson probabilities for the range of T values
t_values = np.arange(min(Ts), max(Ts) + 1)
poisson_probs = poisson.pmf(t_values, len(discrepancies)//5) * len(Ts)  # Scale by the number of samples

# Create bins for each unique integer value in Ts
min_value = min(Ts)
max_value = max(Ts)
bins = range(min_value, max_value + 2)  # +2 to include the last value as a bin

# Plot the histogram
plt.hist(Ts, bins=bins, edgecolor='black', alpha=0.7, label="Observed Ts", density=False, align='left')

# Overlay Poisson distribution
plt.plot(t_values, poisson_probs, 'r-', label=f"Poisson (λ={len(discrepancies)//5})", linewidth=2)

# Add labels and legend
plt.title("Comparison of Observed T Distribution and Poisson Distribution")
plt.xlabel('Value')
plt.ylabel('Frequency')

# Adjust x-axis ticks to show only multiples of 10
start_tick = math.floor(min_value / 10) * 10  # Nearest lower multiple of 10
end_tick = math.ceil(max_value / 10) * 10    # Nearest upper multiple of 10
plt.xticks(range(start_tick, end_tick + 1, 10))

plt.legend()
plt.show()



#%% 2. Checking the true source rank of specific node
"""
2. Checking the true source rank of specific node
"""

Rankings = [result["true_source_rank"] for result in results if result["true_source"] == 16]
# Plot histogram of Rankings
plt.hist(Rankings, bins=range(1, max(Rankings) + 2), edgecolor='black', alpha=0.7, align='left')
plt.title("Histogram of True Source Rankings for true_source=0")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.xticks(range(1, max(Rankings) + 1))  # Ensure ticks match ranks
plt.show()

#%% 3. Checking the overall true source rank
"""
3. Checking the overall true source rank
"""

Overall_Rankings = [result["true_source_rank"] for result in results]

# Plot histogram of Rankings
plt.hist(Overall_Rankings, bins=range(1, max(Overall_Rankings) + 2), edgecolor='black', alpha=0.7, align='left')
plt.title("Histogram of overall True Source Rankings")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.xticks(range(1, max(Rankings) + 1))  # Ensure ticks match ranks
plt.show()


#%% 4. overall mean
"""
4. overall mean
"""

print('overall means ranks and its ratio')

print(np.mean(Overall_Rankings))
print(len(discrepancies)//5)
print(np.mean(Overall_Rankings)/(len(discrepancies)//5))


#%% 5.1. ratio on fixed node
"""
5.1 Ratio on fixed source
"""

print('rank ratios for one node')

Ratios = [result["true_source_rank"]/result["T"] for result in results if result["true_source"] == 3]
Ratios = np.array(Ratios)

plt.hist(Ratios, bins=20, edgecolor='black', alpha=0.7, align='left',density=True)
plt.title("Histogram of True Source Rankings Ratio(quantile) for true_source=3")
plt.xlabel("Ratio")
plt.ylabel("density")
plt.show()

print(np.percentile(Ratios, 95))
print(np.mean(Ratios < 0.5))

#%% 5.2. ratio
"""
5.2 overall ratio
"""

print('overall rank ratios')

Ratios_all = [result["true_source_rank"]/result["T"] for result in results]
Ratios_all = np.array(Ratios_all)

plt.hist(Ratios_all, bins=20, edgecolor='black', alpha=0.7, align='left',density=True)
plt.title("Histogram of overall True Source Rankings Ratio(quantile)")
plt.xlabel("Ratio")
plt.ylabel("density")
plt.show()

print(np.percentile(Ratios_all, 95))
print(np.mean(Ratios_all < 0.5))
