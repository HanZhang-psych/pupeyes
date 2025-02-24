# -*- coding:utf-8 -*-

"""
Area of Interest (AOI) Analysis Module

Author: Han Zhang
Email: hanzh@umich.edu

This module provides basic functions for analyzing eye tracking data in relation to Areas of Interest (AOIs).
"""

import numpy as np
import pandas as pd
from .utils import is_inside_parallel
import numba as nb
import warnings

def get_fixation_aoi(x, y, aois):
    """
    For each fixation point, get the Area of Interest (AOI) that contains it. If the point is outside all AOIs,
    return None.
    
    Parameters
    ----------
    x : float or numpy.ndarray
        X-coordinate(s) of fixation point(s)
    y : float or numpy.ndarray
        Y-coordinate(s) of fixation point(s)
    aois : dict or None
        Dictionary mapping AOI names to lists of vertex coordinates.
        Each vertex list should define a polygon as [(x1,y1), (x2,y2), ...].
    
    Returns
    -------
    str or list
        If input coordinates are scalars:
            Name of the AOI containing the point, or None if not in any AOI
        If input coordinates are arrays:
            List of AOI names for each point, with None for points outside all AOIs
            
    Notes
    -----
    If a point lies within multiple AOIs, it is assigned to the first AOI
    that contains it based on the iteration order of the aois dictionary.
    
    Examples
    --------
    >>> # Single point
    >>> aois = {
    ...     'face': [(0,0), (100,0), (100,100), (0,100)],
    ...     'text': [(150,0), (250,0), (250,50), (150,50)]
    ... }
    >>> get_fixation_aoi(50, 50, aois)
    'face'
    >>> get_fixation_aoi(300, 300, aois)
    None
    
    >>> # Multiple points
    >>> x = np.array([50, 200, 300])
    >>> y = np.array([50, 25, 300])
    >>> get_fixation_aoi(x, y, aois)
    ['face', 'text', None]
    """
    if aois is None:
        return None if np.isscalar(x) else [None] * len(x)
    
    # Convert input to arrays if they're scalars
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)
    points = np.column_stack((x_arr, y_arr))
    
    # Initialize results array
    results = [None] * len(points)
    
    # Check each AOI using parallel processing
    for aoi_name, vertices in aois.items():
        # Add first vertex to end to close the polygon
        vertices_array = np.array(vertices + [vertices[0]])
        # Use parallel processing to check all points against current AOI
        inside_mask = is_inside_parallel(points, vertices_array)
        # Update results for points inside this AOI
        for i, is_inside in enumerate(inside_mask):
            if is_inside and results[i] is None:  # Only update if not already assigned to an AOI
                results[i] = aoi_name
    
    # Return single result for scalar input, list for array input
    return results[0] if np.isscalar(x) else results

def compute_aoi_statistics(x, y, aois, durations=None):
    """
    Compute fixation statistics for each Area of Interest (AOI).
    
    Parameters
    ----------
    x : array-like
        Array of x-coordinates for fixation points
    y : array-like
        Array of y-coordinates for fixation points
    aois : dict
        Dictionary mapping AOI names to lists of vertex coordinates.
        Each vertex list should define a polygon as [(x1,y1), (x2,y2), ...].
    durations : array-like, optional
        Array of fixation durations corresponding to each (x,y) point. 
    
    Returns
    -------
    dict
        Dictionary containing statistics for each AOI and points outside AOIs:
        {
            'outside': {
                'count': int,  # number of fixations outside all AOIs
                'total_duration': float  # total duration of outside fixations
            },
            'aoi_name': {
                'count': int,  # number of fixations in this AOI
                'total_duration': float  # total duration in this AOI
            },
            ...
        }
        If durations is None, total_duration values will be 0.
        Returns empty dict if aois is empty.
        
    Notes
    -----
    If a fixation point lies within multiple AOIs, it is counted only in the
    first AOI that contains it based on the iteration order of the aois dictionary.
    
    Examples
    --------
    >>> aois = {
    ...     'face': [(0,0), (100,0), (100,100), (0,100)],
    ...     'text': [(150,0), (250,0), (250,50), (150,50)]
    ... }
    >>> x = np.array([50, 200, 300])  # points in face, text, outside
    >>> y = np.array([50, 25, 300])
    >>> durations = np.array([100, 150, 200])  # durations in milliseconds
    >>> stats = compute_aoi_statistics(x, y, aois, durations)
    >>> stats
    {
        'outside': {'count': 1, 'total_duration': 200.0},
        'face': {'count': 1, 'total_duration': 100.0},
        'text': {'count': 1, 'total_duration': 150.0}
    }
    """
    if not aois:
        return {}
    
    # Get AOI assignments for all points at once
    aoi_assignments = get_fixation_aoi(x, y, aois)
    
    # Convert string assignments to indices for Numba (-1 for outside)
    aoi_to_idx = {name: idx for idx, name in enumerate(aois.keys())}
    aoi_indices = np.array([aoi_to_idx[aoi] if aoi is not None else -1 for aoi in aoi_assignments])
    
    # making sure inputs are Numba compatible
    aoi_indices = np.asarray(aoi_indices)
    if durations is not None:
        durations = np.asarray(durations)

    # Compute statistics using Numba
    counts, total_durations = aoi_stats_parallel(aoi_indices, len(aois), durations)
    
    # Convert back to dictionary format
    stats = {'outside': {'count': counts[0], 'total_duration': total_durations[0]}}
    for aoi_name, idx in aoi_to_idx.items():
        stats[aoi_name] = {
            'count': counts[idx + 1],
            'total_duration': total_durations[idx + 1]
        }
    
    return stats

@nb.njit(parallel=True)
def aoi_stats_parallel(aoi_assignments, n_aois, durations=None):
    """
    Compute Area of Interest (AOI) statistics using parallel processing. This is a helper function for compute_aoi_statistics.
    
    Parameters
    ----------
    aoi_assignments : numpy.ndarray
        1D array of AOI indices where -1 represents outside any AOI, and indices
        0 to n_aois-1 represent specific AOIs. For example, np.array([-1, 0, 1, 2])
        means that the first fixation is outside any AOI, the second fixation is in AOI 0,
        the third fixation is in AOI 1, and the fourth fixation is in AOI 2.
    n_aois : int
        Number of AOIs (excluding outside). In the example above, n_aois would be 3.
    durations : numpy.ndarray, optional
        1D array of fixation durations corresponding to each assignment. 
        For example, np.array([100, 200, 300, 400]) means that the first fixation
        lasted 100 ms, the second fixation lasted 200 ms, the third fixation lasted
        300 ms, and the fourth fixation lasted 400 ms.
        
    Returns
    -------
    tuple
        A tuple containing:
        - counts : numpy.ndarray
            1D array of fixation counts for each AOI (including outside)
        - total_durations : numpy.ndarray
            1D array of total fixation durations for each AOI (including outside)
            Returns zeros if durations is None
            
    Notes
    -----
    The first element in the returned arrays corresponds to fixations outside
    any AOI, while subsequent elements correspond to AOIs indexed from 0 to n_aois-1.
    
    Examples
    --------
    >>> # Three AOIs (0,1,2) plus outside (-1)
    >>> assignments = np.array([-1, 0, 1, 2, 1, 0])
    >>> durations = np.array([100, 150, 200, 250, 300, 350])
    >>> counts, total_durations = aoi_stats_parallel(assignments, 3, durations)
    >>> counts
    array([1, 2, 2, 1])  # [outside, AOI0, AOI1, AOI2]
    >>> total_durations
    array([100., 500., 500., 250.])  # sums of durations for each AOI
    """
    counts = np.zeros(n_aois + 1, dtype=np.int64)  # +1 for outside
    total_durations = np.zeros(n_aois + 1)
    
    for i in nb.prange(len(aoi_assignments)):
        idx = aoi_assignments[i] + 1  # Shift by 1 to handle -1 index
        counts[idx] += 1
        if durations is not None:
            total_durations[idx] += durations[i]
    
    return counts, total_durations
