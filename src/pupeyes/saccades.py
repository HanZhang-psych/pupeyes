# -*- coding:utf-8 -*-

"""
Saccade Analysis Module

This module provides functions for analyzing saccadic eye movements recorded with Eyelink eye trackers.
Currently, this files only contains functions that are tailored for visual search tasks in which items are presented in a circular array (e.g., the additional singleton task).
"""

import numpy as np
import pandas as pd
from .utils import angular_distance

def saccade_aoi_annulus(data, 
                        item_coords, 
                        col_startx,
                        col_starty,
                        col_endx,
                        col_endy,
                        col_distractor_cond,
                        col_target_pos,
                        col_distractor_pos,
                        col_other_pos=None, 
                        screen_dims=(1600, 1200),
                        annulus_range=(50, 600),
                        item_range=None, 
                        start_range=None, 
                        fixation_mode=False):
    """
    Classify saccade endpoints or fixations based on their proximity to items within an annular region.
    The function assumes eyelink coordinates are used, where the origin is in the top-left corner. 
    You might need to convert your coordinates before using this function.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing saccade or fixation data
    item_coords : list or numpy.ndarray
        List of (x,y) coordinates for all possible item positions
    col_startx, col_starty : str
        Column names for saccade start coordinates
    col_endx, col_endy : str
        Column names for saccade end coordinates
    col_distractor_cond : str
        Column name for distractor condition ('P' for present, 'A' for absent)
    col_target_pos : str
        Column name for target position coordinates
    col_distractor_pos : str
        Column name for distractor position coordinates
    col_other_pos : list of str, optional
        Column names for other item position coordinates
    screen_dims : tuple, optional
        Screen dimensions (width, height) in pixels (default: (1600, 1200))
    annulus_range : tuple, optional
        Inner and outer radius of annulus in pixels (default: (50, 600))
    item_range : float, optional
        Maximum distance to consider a point as belonging to an item
    start_range : float, optional
        Maximum allowed distance from screen center for start position
    fixation_mode : bool, optional
        If True, only check end positions (default: False)

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with added columns:

            - curritem : str
                Item type ('Target', 'Singleton', 'Non-singleton', or NaN)
            - currloc : int
                Index of closest item position, based on the order provided in item_coords
            - flag : str
                Reason for invalid classification ('invalid_start_pos', 
                'invalid_end_pos', 'no_item_in_range', or NaN)

    Notes
    -----
    - If a saccade starts outside the annulus, it is classified as 'invalid_start_pos'.
    - If a saccade ends outside the annulus, it is classified as 'invalid_end_pos'.
    - If a saccade ends too far from any item, it is classified as 'no_item_in_range'.
    """
    # Initialize new columns
    data['curritem'] = pd.NA
    data['currloc'] = pd.NA
    data['flag'] = pd.NA

    # Screen center coordinates
    screen_center = np.array([screen_dims[0]/2, screen_dims[1]/2])
    annulus_range = np.array(annulus_range)

    # Convert positions to numpy arrays
    end_pos = np.column_stack((data[col_endx], data[col_endy]))
    
    if fixation_mode:
        # For fixations, only check end positions
        valid_start_mask = np.ones(len(data), dtype=bool)
    else:
        # For saccades, convert start positions and check validity
        start_pos = np.column_stack((data[col_startx], data[col_starty]))
        d2center0 = np.sqrt(np.sum((start_pos - screen_center)**2, axis=1))
        
        # Mark invalid start positions if start_range is provided
        if start_range is not None:
            invalid_start_mask = d2center0 > start_range
            data.loc[invalid_start_mask, 'flag'] = 'invalid_start_pos'
            valid_start_mask = ~invalid_start_mask
        else:
            # if no start_range is provided, all start positions are valid
            valid_start_mask = np.ones(len(data), dtype=bool)
    
    # Calculate distances to screen center for end positions
    d2center = np.sqrt(np.sum((end_pos - screen_center)**2, axis=1))
    
    # Mark invalid end positions
    invalid_end_mask = (d2center < annulus_range[0]) | (d2center > annulus_range[1])
    data.loc[invalid_end_mask & valid_start_mask, 'flag'] = 'invalid_end_pos'
    
    # For valid positions, find closest items
    valid_mask = ~invalid_end_mask & valid_start_mask
    valid_end_pos = end_pos[valid_mask]
    
    # Convert item_coords to numpy array if not already
    positions = np.array(item_coords)
    
    # Calculate distances to all possible positions for each valid end position
    # Using broadcasting to compute all distances at once
    distances = np.sqrt(np.sum((valid_end_pos[:, np.newaxis] - positions)**2, axis=2))
    closest_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    # If item_range is provided, mark positions too far from any item as invalid
    if item_range is not None:
        too_far_mask = min_distances > item_range
        # Create a new mask for indexing the original DataFrame
        full_too_far_mask = np.zeros_like(valid_mask)
        full_too_far_mask[valid_mask] = too_far_mask
        data.loc[full_too_far_mask, 'flag'] = 'no_item_in_range'
        # remove invalid positions from valid_mask
        valid_mask[valid_mask] = ~too_far_mask
        # remove invalid positions from closest_indices
        closest_indices = closest_indices[~too_far_mask]

    # Assign currloc for valid positions
    data.loc[valid_mask, 'currloc'] = closest_indices
    
    # Get closest positions
    closest_positions = positions[closest_indices]
    
    # Extract target and distractor positions as arrays
    target_pos = np.array(data.loc[valid_mask, col_target_pos].tolist())
    distractor_pos = np.array(data.loc[valid_mask, col_distractor_pos].tolist())
    
    # Create masks for target and distractor
    eps = 1e-6 # tolerance for matching positions
    target_mask = np.all(np.abs(closest_positions - target_pos) < eps, axis=1)
    distractor_mask = np.all(np.abs(closest_positions - distractor_pos) < eps, axis=1)
    
    # Create combined mask for other positions
    if col_other_pos:
        other_masks = []
        for other_col in col_other_pos:
            other_pos = np.array(data.loc[valid_mask, other_col].tolist())
            other_mask = np.all(np.abs(closest_positions - other_pos) < eps, axis=1)
            other_masks.append(other_mask)
        other_combined_mask = np.any(other_masks, axis=0)  # True if position matches any other item
    else:
        other_combined_mask = np.zeros(np.sum(valid_mask), dtype=bool)
    
    # Initialize curritem array with the correct length
    curritem = np.full(np.sum(valid_mask), pd.NA)
    
    # Assign item types based on masks
    curritem[target_mask] = 'Target'
    
    # Handle singleton vs non-singleton based on distractor condition
    distractor_cond = data.loc[valid_mask, col_distractor_cond].values
    singleton_mask = (distractor_cond == 'P') & distractor_mask
    
    if col_other_pos:
        nonsing_mask = ((distractor_cond == 'A') & distractor_mask) | other_combined_mask
    else:
        # If no other positions provided:
        # - In condition 'A' (absent): any non-target position is non-singleton
        # - In condition 'P' (present): any non-target, non-singleton position is non-singleton
        nonsing_mask = ((distractor_cond == 'A') & ~target_mask) | \
                      ((distractor_cond == 'P') & ~target_mask & ~singleton_mask)
    
    curritem[singleton_mask] = 'Singleton'
    curritem[nonsing_mask] = 'Non-singleton'
    
    # Assign curritem values back to DataFrame
    data.loc[valid_mask, 'curritem'] = curritem
    
    return data

def saccade_aoi_angular(sample_data,
                        data,
                        col_sample_timestamp,
                        col_x,
                        col_y,
                        col_saccade_start_time,
                        col_saccade_end_time,
                        col_target_pos,
                        col_distractor_pos,
                        col_distractor_cond,
                        col_other_pos,
                        item_coords,
                        use = None,
                        threshold=30):
    """
    Classify saccades based on their angular deviation towards potential target locations.
    Different from saccade_aoi_annulus(), this function uses the initial firing direction of a saccade
    to classify its destination. As a result, it also requires raw gaze position data.
    Make sure to use the same coordinate system for both sample_data and data.
    
    Parameters
    ----------
    sample_data : pandas.DataFrame
        Raw eye tracking samples containing gaze positions
    data : pandas.DataFrame
        Saccade data with start/end times
    col_sample_timestamp : str
        Column name for timestamps in sample_data
    col_x, col_y : str
        Column names for x and y coordinates in sample_data
    col_saccade_start_time, col_saccade_end_time : str
        Column names for saccade start and end times
    col_target_pos : str
        Column name for target position coordinates
    col_distractor_pos : str
        Column name for distractor position coordinates
    col_distractor_cond : str
        Column name for distractor condition ('P' for present, 'A' for absent)
    col_other_pos : list of str or None
        Column names for other item position coordinates
    item_coords : list or numpy.ndarray
        List of (x,y) coordinates for all possible item positions
    use : str or int, optional
        Point in the trajectory of a saccade to use for classification:
        - 'mid': midpoint (default)
        - 'one-third': one-third point
        - int: specific sample number
        - None: endpoint
    threshold : float, optional
        Maximum angular deviation (degrees) to consider a saccade as directed
        towards an item (default: 30)

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with added columns:

            - curritem : str
                Item type ('Target', 'Singleton', 'Non-singleton', or NaN)
            - flag : str
                Reason for invalid classification ('insufficient_samples',
                'big_angle', or NaN)

    Notes
    -----
    - If a saccade starts outside the annulus, it is classified as 'invalid_start_pos'.
    - If a saccade ends outside the annulus, it is classified as 'invalid_end_pos'.
    - If a saccade ends too far from any item, it is classified as 'no_item_in_range'.
    """
    # Initialize new columns
    data = data.copy()
    data['curritem'] = pd.NA
    data['flag'] = pd.NA
    
    # Convert item_coords to numpy array
    item_coords = np.array(item_coords)
    
    # Process each saccade
    for idx in data.index:
        # Get saccade samples
        saccade_start_time = data.loc[idx, col_saccade_start_time]
        saccade_end_time = data.loc[idx, col_saccade_end_time]
        s = sample_data[(sample_data[col_sample_timestamp] >= saccade_start_time) & 
                       (sample_data[col_sample_timestamp] <= saccade_end_time)]
        
        if len(s) < 2:  # Need at least start and end points
            data.loc[idx, 'flag'] = 'insufficient_samples'
            continue
            
        x_pos = s[col_x].values
        y_pos = s[col_y].values
        
        # Define points for deviation calculation
        start_point = np.array([x_pos[0], y_pos[0]])
        
        # Determine which point to use for end position
        if use == 'mid':
            n = int(np.round(len(x_pos)/2)) - 1
        elif use == 'one-third':
            n = int(np.round(len(x_pos)/3)) - 1
        elif isinstance(use, int):
            n = min(use - 1, len(x_pos) - 1)
        else:
            n = -1
        end_point = np.array([x_pos[n], y_pos[n]])
        
        # Compute angular distances to all item locations
        distances = []
        for p in item_coords:
            line1 = (start_point, p)
            line2 = (start_point, end_point)
            d = angular_distance(line1, line2)
            distances.append(d)
            
        # Find closest item
        obj_dist = np.min(distances)
        if obj_dist < threshold:
            obj_pos = item_coords[np.argmin(distances)]
            target_pos = np.array(data.loc[idx, col_target_pos])
            distractor_pos = np.array(data.loc[idx, col_distractor_pos])
            distractor_presence = data.loc[idx, col_distractor_cond]
            
            # Check if position matches target
            if np.all(obj_pos == target_pos):
                data.loc[idx, 'curritem'] = 'Target'
            elif distractor_presence == 'P' and np.all(obj_pos == distractor_pos):
                data.loc[idx, 'curritem'] = 'Singleton'
            else:
                # Check other positions if provided
                if col_other_pos:
                    is_other = False
                    for other_col in col_other_pos:
                        other_pos = np.array(data.loc[idx, other_col])
                        if np.all(obj_pos == other_pos):
                            is_other = True
                            break
                    # In condition 'A', distractor position is treated as other position
                    if distractor_presence == 'A' and np.all(obj_pos == distractor_pos):
                        is_other = True
                    if is_other:
                        data.loc[idx, 'curritem'] = 'Non-singleton'
                else:
                    # If no other positions provided:
                    # - In condition 'A': any non-target position is non-singleton
                    # - In condition 'P': any non-target, non-singleton position is non-singleton
                    data.loc[idx, 'curritem'] = 'Non-singleton'
        else:
            data.loc[idx, 'flag'] = 'big_angle'
            
    return data


def saccade_deviation(sample_data, 
                      data,
                      col_sample_timestamp,
                      col_x,
                      col_y,
                      col_saccade_start_time,
                      col_saccade_end_time,
                      find = 'mid'):
    """
    Compute the angular deviation of saccade trajectories from a straight path.
    
    This function measures how much a saccade's trajectory deviates from a straight line
    between its start and end points. The deviation is measured as the angle between
    two lines: one from start to end point, and another from start to a specified
    point along the trajectory. This function may be helpful for detecting curved saccades.
    Make sure to use the same coordinate system for both sample_data and data.

    Parameters
    ----------
    sample_data : pandas.DataFrame
        Raw eye tracking samples containing gaze positions
    data : pandas.DataFrame
        Saccade data with start/end times
    col_sample_timestamp : str
        Column name for timestamps in sample_data
    col_x, col_y : str
        Column names for x and y coordinates in sample_data
    col_saccade_start_time, col_saccade_end_time : str
        Column names for saccade start and end times
    find : str or int, optional
        Point in trajectory for curvature calculation:
        - 'mid': use midpoint (default)
        - 'one-third': use one-third point
        - 'max': find point of maximum deviation
        - int: use specific sample number
        - None: use endpoint

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with added columns:

            - deviation : float
                Angular deviation at specified point (degrees)
            - deviation_idx : int
                Sample index where deviation was computed
            - deviation_time : float
                Timestamp where deviation was computed

    Notes
    -----
    - If a saccade starts outside the annulus, it is classified as 'invalid_start_pos'.
    - If a saccade ends outside the annulus, it is classified as 'invalid_end_pos'.
    - If a saccade ends too far from any item, it is classified as 'no_item_in_range'.
    """
    # Initialize new columns
    data = data.copy()
    data['deviation'] = np.empty(len(data), dtype=float)
    data['deviation_idx'] = np.empty(len(data), dtype=int)
    data['deviation_time'] = np.empty(len(data), dtype=float)

    # Process each saccade
    for idx in data.index:
        # Get saccade samples
        saccade_start_time = data.loc[idx, col_saccade_start_time]
        saccade_end_time = data.loc[idx, col_saccade_end_time]
        s = sample_data[(sample_data[col_sample_timestamp] >= saccade_start_time) & 
                       (sample_data[col_sample_timestamp] <= saccade_end_time)]
        
        if len(s) < 2:  # Need at least start and end points
            continue
            
        x_pos = s[col_x].values
        y_pos = s[col_y].values
        timestamps = s[col_sample_timestamp].values

        # Get start and end points for all calculations
        start_point = (x_pos[0], y_pos[0])
        end_point = (x_pos[-1], y_pos[-1])
        line1 = (start_point, end_point)

        if find == 'max' and len(x_pos) > 2:  # Need at least 3 points to find max
            # Compute curvature at all points except start and end
            curvatures = []
            for i in range(1, len(x_pos)-1):
                point = (x_pos[i], y_pos[i])
                line2 = (start_point, point)
                curvatures.append(angular_distance(line1, line2))
            
            # Find maximum curvature and its index
            max_idx = np.argmax(curvatures) + 1  # +1 because we skipped start point
            data.loc[idx, 'deviation'] = curvatures[max_idx-1]
            data.loc[idx, 'deviation_idx'] = max_idx
            data.loc[idx, 'deviation_time'] = timestamps[max_idx]
        else:
            # Determine which point to use
            if find == 'mid':
                n = int(np.round(len(x_pos)/2)) - 1
            elif find == 'one-third':
                n = int(np.round(len(x_pos)/3)) - 1
            elif isinstance(find, int):
                n = min(find - 1, len(x_pos) - 1)
            else:
                n = -1

            # Compute curvature at specified point
            point = (x_pos[n], y_pos[n])
            line2 = (start_point, point)
            data.loc[idx, 'deviation'] = angular_distance(line1, line2)
            data.loc[idx, 'deviation_idx'] = n
            data.loc[idx, 'deviation_time'] = timestamps[n]

    return data