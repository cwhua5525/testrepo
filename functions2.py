# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 00:50:03 2025

@author: chenwei


"""

import networkx as nx
import random
import time
import pickle
from typing import Set, List
import numpy as np
from functions1 import fetch_samples_from_mc_set



def calculate_discrepancy(observed_sequence: List[int], simulated_sequence: List[int]) -> float:
    """
    Calculate the discrepancy between the observed and simulated infection sequences.

    Parameters:
    -----------
    observed_sequence : List[int]
        The observed infection sequence.
    simulated_sequence : List[int]
        A simulated infection sequence.

    Returns:
    --------
    discrepancy : float
        The size of the symmetric difference between the two infection sets.
    """
    observed_set = set(observed_sequence)
    simulated_set = set(simulated_sequence)
    return len(observed_set.symmetric_difference(simulated_set))

def calculate_discrepancies_for_samples(observed_sequence: List[int], simulated_samples: List[List[int]]) -> List[float]:
    """
    Calculate discrepancies between the observed infection sequence and multiple simulated samples.

    Parameters:
    -----------
    observed_sequence : List[int]
        The observed infection sequence.
    simulated_samples : List[List[int]]
        A list of simulated infection sequences.

    Returns:
    --------
    discrepancies : List[float]
        A list of discrepancies for each simulated sample.
    """
    discrepancies = [
        calculate_discrepancy(observed_sequence, simulated_sequence)
        for simulated_sequence in simulated_samples
    ]
    return discrepancies

def calculate_loss(y: List[int], z: List[int]) -> float:
    """
    Calculate the discrepancy (loss) between the observed infection sequence y and simulated path z.

    Parameters:
    -----------
    y : List[int]
        Observed infection sequence.
    z : List[int]
        Simulated infection sequence.

    Returns:
    --------
    loss : float
        The discrepancy value.
    """
    observed_set = set(y)
    simulated_set = set(z)
    return len(observed_set.symmetric_difference(simulated_set))

def compute_t_s(y: List[int], simulated_samples: List[List[int]]) -> float:
    """
    Compute T_s(y), the expected loss, using Monte Carlo samples.

    Parameters:
    -----------
    y : List[int]
        Observed infection sequence.
    simulated_samples : List[List[int]]
        Simulated diffusion paths.

    Returns:
    --------
    T_s_y : float
        The expected loss T_s(y).
    """
    losses = [calculate_loss(y, z) for z in simulated_samples]
    return sum(losses) / len(losses)

def compute_p_value(y: List[int], simulated_samples_first: List[List[int]], simulated_samples_second: List[List[int]]) -> float:
    """
    Compute the p-value for the hypothesis that the source is s.

    Parameters:
    -----------
    y : List[int]
        Observed infection sequence.
    simulated_samples_first : List[List[int]]
        First set of Monte Carlo samples used to compute T_s(y).
    simulated_samples_second : List[List[int]]
        Second set of Monte Carlo samples used to compute T_s(zeta(Z_j)).

    Returns:
    --------
    p_value : float
        The p-value for the hypothesis.
    """
    # Step 1: Compute T_s(y) using the first set of samples
    T_s_y = compute_t_s(y, simulated_samples_first)

    # Step 2: Compute T_s(zeta(Z_j)) for each sample in the second set
    T_s_z = [compute_t_s(z, simulated_samples_second) for z in simulated_samples_first]

    # Step 3: Calculate p-value as the fraction of T_s(zeta(Z_j)) >= T_s(y)
    p_value = sum(1 for t in T_s_z if t >= T_s_y) / len(T_s_z)
    return p_value

def calculate_expected_discrepancies_from_mc_set(mc_set, s: int, T: int):
    """
    Fetch samples from the MC-set and calculate expected discrepancies for j = m+1 to 2m.

    Parameters:
    -----------
    mc_set : Dict[int, List[List[int]]]
        Precomputed MC-set containing infection sequences for all source nodes.
    s : int
        The source node for which samples are requested.
    T : int
        The infection size (number of nodes to include in the sequence).

    Returns:
    --------
    expected_discrepancies : List[float]
        A list of expected discrepancies for each sample j in the range m+1 to 2m.
    """
    # Fetch samples for the given source node and infection size
    samples = fetch_samples_from_mc_set(mc_set, s, T)

    # Calculate m
    m = len(samples) // 2

    # Ensure we have exactly 2m samples
    if len(samples) != 2 * m:
        raise ValueError("Number of samples must be even and equal to 2m.")

    # Symmetric difference discrepancy function
    def symmetric_difference_discrepancy(seq1, seq2):
        set1, set2 = set(seq1), set(seq2)
        return len(set1.symmetric_difference(set2))

    # Initialize a list to store expected discrepancies
    expected_discrepancies = []

    # For each sample j in the range m+1 to 2m
    for j in range(m, 2 * m):
        total_discrepancy = 0

        # Calculate discrepancy with each sample i in the range 1 to m
        for i in range(m):
            total_discrepancy += symmetric_difference_discrepancy(samples[j], samples[i])

        # Compute the average discrepancy for sample j
        expected_discrepancies.append(total_discrepancy / m)

    return expected_discrepancies

def calculate_expected_discrepancies_from_mc_set_optimized(mc_set, s: int, T: int):
    """
    Optimized version to calculate expected discrepancies using NumPy.

    Parameters:
    -----------
    mc_set : Dict[int, List[List[int]]]
        Precomputed MC-set containing infection sequences for all source nodes.
    s : int
        The source node for which samples are requested.
    T : int
        The infection size (number of nodes to include in the sequence).

    Returns:
    --------
    expected_discrepancies : List[float]
        A list of expected discrepancies for each sample j in the range m+1 to 2m.
    """
    # Fetch samples for the given source node and infection size
    samples = fetch_samples_from_mc_set(mc_set, s, T)

    # Calculate m
    m = len(samples) // 2

    # Ensure we have exactly 2m samples
    if len(samples) != 2 * m:
        raise ValueError("Number of samples must be even and equal to 2m.")

    # Convert samples to a binary matrix
    binary_matrix = np.zeros((len(samples), 198), dtype=np.int32)
    for idx, seq in enumerate(samples):
        for node in seq:
            binary_matrix[idx, node] = 1  # Adjust to 0-based indexing

    # Split the binary matrix into two groups
    group1 = binary_matrix[:m]  # First m samples
    group2 = binary_matrix[m:]  # Last m samples

    # Debug: Print the binary matrices
    #print(group1)
    #print(group2)

    # Compute pairwise symmetric differences using XOR and sum
    xor_matrix = np.bitwise_xor(group1[:, None, :], group2[None, :, :])  # Shape: (m, m, T)
    discrepancy_matrix = np.sum(xor_matrix, axis=-1)  # Shape: (m, m)

    # Debug: Print the XOR matrix
    #print(xor_matrix)
    #print(discrepancy_matrix)

    # Calculate expected discrepancies for each sample in group2
    expected_discrepancies = np.mean(discrepancy_matrix, axis=0)  # Mean over group1

    return expected_discrepancies.tolist()

def calculate_expected_discrepancies_from_mc_set_optimized2(mc_set, s: int, T: int):
    """
    Optimized version to calculate expected discrepancies using NumPy with reduced memory usage.

    Parameters:
    -----------
    mc_set : Dict[int, List[List[int]]]
        Precomputed MC-set containing infection sequences for all source nodes.
    s : int
        The source node for which samples are requested.
    T : int
        The infection size (number of nodes to include in the sequence).

    Returns:
    --------
    expected_discrepancies : List[float]
        A list of expected discrepancies for each sample j in the range m+1 to 2m.
    """
    # Fetch samples for the given source node and infection size
    samples = fetch_samples_from_mc_set(mc_set, s, T)

    # Calculate m
    m = len(samples) // 2

    # Ensure we have exactly 2m samples
    if len(samples) != 2 * m:
        raise ValueError("Number of samples must be even and equal to 2m.")

    # Convert samples to a binary matrix using np.bool_ for memory efficiency
    binary_matrix = np.zeros((len(samples), len(mc_set[0][0])+10), dtype=np.bool_)  # len(mc_set[0][0]) assumed as node count
    for idx, seq in enumerate(samples):
        for node in seq:
            binary_matrix[idx, node] = True  # Adjust to 0-based indexing

    # Split the binary matrix into two groups
    group1 = binary_matrix[:m]  # First m samples
    group2 = binary_matrix[m:]  # Last m samples

    # Compute pairwise symmetric differences using XOR and sum
    xor_matrix = np.bitwise_xor(group1[:, None, :], group2[None, :, :])  # Shape: (m, m, T)
    discrepancy_matrix = np.sum(xor_matrix, axis=-1)  # Shape: (m, m)

    # Calculate expected discrepancies for each sample in group2
    expected_discrepancies = np.mean(discrepancy_matrix, axis=0)  # Mean over group1

    return expected_discrepancies.tolist()
