# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 00:23:32 2025

@author: chenwei
"""

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

def calculate_statistics(data, name="Data"):
    """
    Calculate basic statistics for the given data.

    Parameters:
        data (list or np.array): The dataset to analyze.
        name (str): Name of the dataset (for labeling purposes).

    Returns:
        dict: A dictionary containing the calculated statistics.
    """
    if len(data) == 0:
        raise ValueError("The dataset is empty.")

    stats = {
        "min": np.min(data),
        "max": np.max(data),
        "mean": round(np.mean(data), 4),
        "median": np.median(data),
        "std_dev": round(np.std(data), 4),
        "95% quantile": np.percentile(data, 95)
    }

    print(f"Basic statistics for {name}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return stats

m = 1000

#dat_name = 'karate'
#dat_name = 'eco-florida'
#dat_name = 'ca-sandi_auths'
dat_name = 'ia-enron-only'
#dat_name = 'insecta-ant-colony6-day11' # m = 500
#dat_name = 'soc-dolphins'




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

Rankings = [result["true_source_rank"] for result in results if result["true_source"] == 0]
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
 
print('rank ratios for one fixed node s')

truesourceforplot = 12
Ratios = [result["true_source_rank"]/result["T"] for result in results if result["true_source"] == truesourceforplot]
Ratios = np.array(Ratios)

plt.hist(Ratios, bins=20, edgecolor='black', alpha=0.7, align='left')
plt.title(f"Histogram of True Source Rankings Ratio(quantile) for true_source={truesourceforplot}")
plt.xlabel("Ratio")
plt.ylabel("density")
plt.show()

print(np.percentile(Ratios, 95))
print(np.mean(Ratios < 0.5))

#%% 5.1. ratio on fixed size
"""
5.1 Ratio on fixed source
"""

print('rank ratios for one fixed size T')
fixsize = int(np.mean([result["T"] for result in results]))
Ratios = [result["true_source_rank"]/result["T"] for result in results if result["T"] == fixsize]
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

plt.hist(Ratios_all, bins=20, edgecolor='black', alpha=0.7, align='left')
plt.title("Histogram of overall True Source Rankings Ratio(quantile)")
plt.xlabel("Ratio")
plt.ylabel("density")
plt.show()

print(np.percentile(Ratios_all, 95))
print(np.mean(Ratios_all < 0.5))

#%% 5.1.2 scatter plot ratio vs T

T_ALL = [result["T"] for result in results]
# Create a scatter plot
plt.scatter(Ratios_all, T_ALL, color='blue', label='Data Points')

# Add labels and title
plt.xlabel('ratios')
plt.ylabel('T')
plt.title('Scatter Plot Example')

# Show legend
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()

#%% 6 Basic statistics

calculate_statistics(Overall_Rankings, name="ranks")
calculate_statistics(Ratios_all, name="quantiles (= rank/infected size)")


#%% 7 parallel boxplots


# Example datasets (values between 0 and 1)
data1 = [result["true_source_rank"]/result["T"] for result in results if result["T"] == 20]
data2 = [result["true_source_rank"]/result["T"] for result in results if result["T"] == 25]
data3 = [result["true_source_rank"]/result["T"] for result in results if result["T"] == 30]
data4 = [result["true_source_rank"]/result["T"] for result in results if result["T"] == 35]

# Combine all datasets into a list
data = [data1, data2, data3, data4]

# Create a box plot
plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor='lightblue'))

# Customize the plot
plt.xticks(ticks=[1, 2, 3, 4], labels=['T = 20', 'T = 25', 'T = 30', 'T = 35'])
plt.title("Box Plots of ranks, fixed source s and fixed size T")
plt.ylabel("ranks/infexted size")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()