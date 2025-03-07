# -*- coding:utf-8 -*-

"""
Area of Interest (AOI) Analysis Module

This module provides basic functions for analyzing eye tracking data in relation to Areas of Interest (AOIs).
"""

import numpy as np
import pandas as pd
import warnings

try:
    import numba as nb
    HAS_NUMBA = True
    print("Numba is available. Using parallel processing for AOI assignment.")
except ImportError:
    HAS_NUMBA = False

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
        The last vertex should be the same as the first vertex to close the polygon.

    Returns
    -------
    str or list
        If input coordinates are scalars:

            - str
                Name of the AOI containing the point, or None if not in any AOI

        If input coordinates are arrays:

            - list
                List of AOI names for each point, with None for points outside all AOIs
            
    Notes
    -----
    If a point lies within multiple AOIs, it is assigned to the first AOI
    that contains it based on the iteration order of the aois dictionary.
    
    Examples
    --------
    >>> # Single point
    >>> aois = {
    ...     'face': [(0,0), (100,0), (100,100), (0,100), (0,0)],
    ...     'text': [(150,0), (250,0), (250,50), (150,50), (150,0)]
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
        # Check if the last vertex is the same as the first vertex
        if vertices[-1] != vertices[0]:
            # Add first vertex to end to close the polygon
            vertices_array = np.array(vertices + [vertices[0]])
            print(f"Closing polygon for {aoi_name}")
        else:
            vertices_array = np.array(vertices)
        # Use parallel processing to check all points against current AOI
        inside_mask = is_inside(points, vertices_array)
        # Update results for points inside this AOI
        for i, inside in enumerate(inside_mask):
            if inside and results[i] is None:  # Only update if not already assigned to an AOI
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

        - outside : dict
            - count : int
                Number of fixations outside all AOIs
            - total_duration : float
                Total duration of outside fixations

        - aoi_name : dict
            - count : int
                Number of fixations in this AOI
            - total_duration : float
                Total duration in this AOI

        If durations is None, total_duration values will be 0.
        Returns empty dict if aois is empty.
        
    Notes
    -----
    If a fixation point lies within multiple AOIs, it is counted only in the
    first AOI that contains it based on the iteration order of the aois dictionary.
    
    Examples
    --------
    >>> aois = {
    ...     'face': [(0,0), (100,0), (100,100), (0,100), (0,0)],
    ...     'text': [(150,0), (250,0), (250,50), (150,50), (150,0)]
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
    
    # Convert string assignments to indices (-1 for outside)
    aoi_to_idx = {name: idx for idx, name in enumerate(aois.keys())}
    aoi_indices = np.array([aoi_to_idx[aoi] if aoi is not None else -1 for aoi in aoi_assignments])
    
    # Initialize arrays for counts and durations
    n_aois = len(aois)
    counts = np.zeros(n_aois + 1, dtype=np.int64)  # +1 for outside
    total_durations = np.zeros(n_aois + 1)
    
    # Convert inputs to numpy arrays
    aoi_indices = np.asarray(aoi_indices)
    if durations is not None:
        durations = np.asarray(durations)
    
    # Compute statistics
    for i in range(len(aoi_indices)):
        idx = aoi_indices[i] + 1  # Shift by 1 to handle -1 index
        counts[idx] += 1
        if durations is not None:
            total_durations[idx] += durations[i]
    
    # Convert back to dictionary format
    stats = {'outside': {'count': counts[0], 'total_duration': total_durations[0]}}
    for aoi_name, idx in aoi_to_idx.items():
        stats[aoi_name] = {
            'count': counts[idx + 1],
            'total_duration': total_durations[idx + 1]
        }
    
    return stats

if HAS_NUMBA:
    @nb.njit  # Add Numba decorator to is_inside_singlepoint when Numba is available
    def is_inside_singlepoint(polygon, point):
        """
        Check if a point lies inside a polygon using ray-casting algorithm.
        
        Parameters
        ----------
        polygon : array-like
            List of (x,y) coordinates defining the polygon vertices. The last vertex
            should be the same as the first to close the polygon.
        point : tuple
            (x,y) coordinates of the point to check
            
        Returns
        -------
        int
            Result code indicating point position:

                - 0
                    Point is outside the polygon
                - 1
                    Point is inside the polygon
                - 2
                    Point lies exactly on the polygon's edge or vertex
            
        Notes
        -----
        Uses a ray-casting algorithm that counts the number of times a horizontal ray 
        from the point intersects with polygon edges.

        Examples
        --------
        >>> # Define a square
        >>> square = [(0,0), (100,0), (100,100), (0,100), (0,0)]
        >>> 
        >>> # Check points
        >>> is_inside_singlepoint(square, (50, 50))  # inside
        1
        >>> is_inside_singlepoint(square, (150, 150))  # outside
        0
        >>> is_inside_singlepoint(square, (0, 50))    # on edge
        2
        >>> is_inside_singlepoint(square, (0, 0))     # on vertex
        2
        """
        length = len(polygon)-1
        dy2 = point[1] - polygon[0][1]
        intersections = 0
        ii = 0
        jj = 1

        while ii < length:
            dy = dy2
            dy2 = point[1] - polygon[jj][1]

            # consider only lines which are not completely above/below/right from the point
            if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):
                # non-horizontal line
                if dy < 0 or dy2 < 0:
                    F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                    if point[0] > F:  # if line is left from the point - the ray moving towards left, will intersect it
                        intersections += 1
                    elif point[0] == F:  # point on line
                        return 2

                # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
                elif dy2 == 0 and (point[0] == polygon[jj][0] or 
                                (dy == 0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0]) <= 0)):
                    return 2

            ii = jj
            jj += 1

        return intersections & 1

    @nb.njit(parallel=True)
    def is_inside(points, polygon):
        """
        Check if multiple points lie inside a polygon using parallel processing.

        Parameters
        ----------
        points : numpy.ndarray
            Nx2 array of (x,y) coordinates to check
        polygon : numpy.ndarray
            Array of (x,y) coordinates defining the polygon vertices
            
        Returns
        -------
        numpy.ndarray
            Boolean array indicating whether each point is inside the polygon.
            True for points inside or on the polygon, False for points outside.

        Examples
        --------
        >>> # Define a square polygon
        >>> square = np.array([(0,0), (100,0), (100,100), (0,100), (0,0)])
        >>> 
        >>> # Check multiple points
        >>> points = np.array([
        ...     [50, 50],    # inside
        ...     [150, 150],  # outside
        ...     [0, 50],     # on edge
        ...     [0, 0]       # on vertex
        ... ])
        >>> is_inside(points, square)
        array([ True, False,  True,  True])
        """
        ln = len(points)
        D = np.empty(ln, dtype=nb.boolean) 
        for i in nb.prange(ln):
            D[i] = is_inside_singlepoint(polygon, points[i])
        return D

else:
    def is_inside_singlepoint(polygon, point):
        """
        Check if a point lies inside a polygon using ray-casting algorithm.
        
        Parameters
        ----------
        polygon : array-like
            List of (x,y) coordinates defining the polygon vertices. The last vertex
            should be the same as the first to close the polygon.
        point : tuple
            (x,y) coordinates of the point to check
            
        Returns
        -------
        int
            Result code indicating point position:

                - 0
                    Point is outside the polygon
                - 1
                    Point is inside the polygon
                - 2
                    Point lies exactly on the polygon's edge or vertex
            
        Notes
        -----
        Uses a ray-casting algorithm that counts the number of times a horizontal ray 
        from the point intersects with polygon edges.

        Examples
        --------
        >>> # Define a square
        >>> square = [(0,0), (100,0), (100,100), (0,100), (0,0)]
        >>> 
        >>> # Check points
        >>> is_inside_singlepoint(square, (50, 50))  # inside
        1
        >>> is_inside_singlepoint(square, (150, 150))  # outside
        0
        >>> is_inside_singlepoint(square, (0, 50))    # on edge
        2
        >>> is_inside_singlepoint(square, (0, 0))     # on vertex
        2
        """
        length = len(polygon)-1
        dy2 = point[1] - polygon[0][1]
        intersections = 0
        ii = 0
        jj = 1

        while ii < length:
            dy = dy2
            dy2 = point[1] - polygon[jj][1]

            # consider only lines which are not completely above/below/right from the point
            if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):
                # non-horizontal line
                if dy < 0 or dy2 < 0:
                    F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                    if point[0] > F:  # if line is left from the point - the ray moving towards left, will intersect it
                        intersections += 1
                    elif point[0] == F:  # point on line
                        return 2

                # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
                elif dy2 == 0 and (point[0] == polygon[jj][0] or 
                                (dy == 0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0]) <= 0)):
                    return 2

            ii = jj
            jj += 1

        return intersections & 1

    def is_inside(points, polygon):
        """
        Check if multiple points lie inside a polygon.
        
        Parameters
        ----------
        points : numpy.ndarray
            Nx2 array of (x,y) coordinates to check
        polygon : numpy.ndarray
            Array of (x,y) coordinates defining the polygon vertices
            
        Returns
        -------
        numpy.ndarray
            Boolean array indicating whether each point is inside the polygon

        Examples
        --------
        >>> # Define a square polygon
        >>> square = np.array([(0,0), (100,0), (100,100), (0,100), (0,0)])
        >>> 
        >>> # Check multiple points
        >>> points = np.array([
        ...     [50, 50],    # inside
        ...     [150, 150],  # outside
        ...     [0, 50],     # on edge
        ...     [0, 0]       # on vertex
        ... ])
        >>> is_inside(points, square)
        array([ True, False,  True,  True])
        """
        ln = len(points)
        D = np.empty(ln, dtype=bool)
        for i in range(ln):
            D[i] = is_inside_singlepoint(polygon, points[i])
        return D
