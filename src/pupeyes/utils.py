# -*- coding:utf-8 -*-

"""
Utility Functions Module

This module provides utility functions used across the pupeyes package, including:
- Coordinate system conversions between Eyelink and PsychoPy
- Point-in-polygon testing with parallel processing
- Signal filtering and data masking
- Geometric calculations for circular stimulus arrangements
and others.
"""

import math
import numpy as np
import cv2
import warnings
import pandas as pd
import scipy.signal as signal

def lowpass_filter(data, sampling_freq, cutoff_freq=4, order=3):
    """
    Apply a Butterworth lowpass filter to the input data.

    Uses scipy.signal to create and apply a Butterworth filter that removes high frequency 
    components above the cutoff frequency while preserving lower frequencies.

    Parameters
    ----------
    data : array-like
        Input signal to be filtered
    sampling_freq : float
        Sampling frequency of the input signal in Hz
    cutoff_freq : float, optional (default=4)
        Cutoff frequency of the filter in Hz. Frequencies above this will be attenuated.
    order : int, optional (default=3)
        Order of the Butterworth filter. Higher orders give sharper frequency cutoffs
        but may introduce more ringing artifacts.

    Returns
    -------
    numpy.ndarray
        Filtered version of the input signal with same shape as input

    Notes
    -----
    - Uses scipy.signal.butter() to design the filter coefficients
    - Applies zero-phase filtering using scipy.signal.filtfilt()
    - The filter is applied forward and backward to avoid phase shifts
    """
    b, a = signal.butter(N=order, Wn=cutoff_freq, btype='low', analog=False, fs=sampling_freq)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def make_mask(data, trials_to_mask, invert=False):
    """
    Create a boolean mask for filtering data based on specified trials.

    Parameters
    ----------
    data : pandas.DataFrame
        The main dataset to create a mask for
    trials_to_mask : pandas.DataFrame or dict
        Trials to use for creating the mask. Can be a DataFrame or a dictionary that can
        be converted to a DataFrame. Should have matching column names with `data`
    invert : bool, optional (default=False)
        If True, inverts the mask (changes True to False and vice versa)

    Returns
    -------
    pandas.Series
        Boolean mask series with same length as input data. True values indicate rows
        to keep, False values indicate rows to filter out

    Notes
    -----
    - If trials_to_mask is a dictionary, it will attempt to convert it to a DataFrame
    - Warns if resulting mask is all True or all False
    - Uses pandas merge with indicator to create the mask

    Examples
    --------
    >>> # Create sample dataset
    >>> data = pd.DataFrame({
    ...     'trial': [1, 2, 3, 4, 5],
    ...     'condition': ['A', 'B', 'A', 'B', 'C'],
    ...     'rt': [0.5, 0.6, 0.4, 0.7, 0.5]
    ... })
    >>> 
    >>> # Mask trials with condition 'A' using dictionary
    >>> to_mask = {'condition': 'A'}
    >>> mask = make_mask(data, to_mask)
    >>> data[mask]  # Shows only trials with conditions B and C
       trial condition   rt
    1     2         B  0.6
    3     4         B  0.7
    4     5         C  0.5
    >>> 
    >>> # Mask multiple trials using DataFrame
    >>> to_mask_df = pd.DataFrame({
    ...     'trial': [1, 3],
    ...     'condition': ['A', 'A']
    ... })
    >>> mask = make_mask(data, to_mask_df)
    >>> data[mask]  # Same result as above
       trial condition   rt
    1     2         B  0.6
    3     4         B  0.7
    4     5         C  0.5
    >>> 
    >>> # Keep only the masked trials using invert=True
    >>> mask = make_mask(data, to_mask_df, invert=True)
    >>> data[mask]  # Shows only trials with condition A
       trial condition   rt
    0     1         A  0.5
    2     3         A  0.4
    """
    # check if the joining data is a dataframe
    if not isinstance(trials_to_mask, pd.DataFrame):
        # try to convert to dataframe
        try:
            dtype = type(trials_to_mask)
            trials_to_mask = pd.DataFrame([trials_to_mask.values()], columns=trials_to_mask.keys())
        except:
            raise ValueError("Cannot convert trials_to_mask to DataFrame.")

    # mask data
    mask = ~data.merge(trials_to_mask, how='left', indicator=True)['_merge'].eq('both') 

    # check if there are any bad trials
    if mask.all() or (not mask.any()):
        warnings.warn("Mask contains all True or all False values.")

    # if invert is True, invert the mask
    if invert:
        mask = ~mask

    return mask


def convert_coordinates(coord, screen_dims=None, direction='to_el', psychopy_units='pix', round_to=2):
    """
    Convert coordinates between Eyelink and PsychoPy coordinate systems.
    For Eyelink, the origin is at the top-left corner of the screen.
    For PsychoPy, the origin is at the center of the screen.
    For more information on the psychopy coordinate system, see:
    https://psychopy.org/general/units.html

    Parameters
    ----------
    coord : array-like or str
        The coordinates to convert. Can be:
        - array-like: [x, y]
        - string: 'x,y' or '[x,y]' or '(x,y)'
    screen_dims : array-like, optional
        Screen dimensions [width, height] in pixels. Default is [1600, 1200].
    direction : {'to_el', 'to_psychopy'}, optional
        Conversion direction:
        - 'to_el': convert from PsychoPy to Eyelink coordinates
        - 'to_psychopy': convert from Eyelink to PsychoPy coordinates
        Default is 'to_el'.
    psychopy_units : {'pix', 'norm', 'height'}, optional
        PsychoPy units to convert from/to:
        - 'pix': pixels from center
        - 'norm': normalized units [-1, 1]
        - 'height': units relative to screen height
        Default is 'pix'.
    round_to : int or None, optional
        Number of decimal places to round coordinates to. Default is 2.
        If None, no rounding is performed.

    Returns
    -------
    numpy.ndarray
        Converted [x, y] coordinates

    Notes
    -----
    Coordinate system details:
    - Eyelink: origin at top-left, positive x right, positive y down
    - PsychoPy: origin at center, positive x right, positive y up

    Examples
    --------
    >>> # Convert screen center from PsychoPy to Eyelink coordinates
    >>> convert_coordinates([0, 0], screen_dims=[1600, 1200])
    array([800., 600.])  # half width, half height in Eyelink coordinates

    >>> # Convert back from Eyelink to PsychoPy coordinates
    >>> convert_coordinates([800, 600], direction='to_psychopy')
    array([0., 0.])  # back to center in PsychoPy coordinates

    >>> # Convert normalized coordinates (range -1 to 1)
    >>> convert_coordinates([0.5, 0.5], psychopy_units='norm')
    array([1200., 300.])  # scaled by screen dimensions

    >>> # Convert height units (relative to screen height)
    >>> convert_coordinates([0.5, 0.5], screen_dims=[1600, 1200], 
    ...                    psychopy_units='height')
    array([1400., 0.])  # 50% of screen height = 600 pixels

    >>> # Convert from string input
    >>> convert_coordinates("100,100")
    array([900., 500.])  # PsychoPy (100,100) to Eyelink coordinates

    Raises
    ------
    ValueError
        If direction is not 'to_el' or 'to_psychopy'
        If psychopy_units is not 'pix', 'norm', or 'height'
        If string coordinates cannot be parsed
    """
    # Set default screen dimensions once
    if screen_dims is None:
        screen_dims = np.array([1600, 1200])
    else:
        screen_dims = np.array(screen_dims)

    # Pre-compute half dimensions
    half_width = screen_dims[0]/2
    half_height = screen_dims[1]/2
    
    # Convert string coordinates to numpy array
    if isinstance(coord, str):
        try:
            coord = np.fromstring(coord.strip('[]()'), sep=',')
        except:
            raise ValueError("Could not convert string coordinates to array. Format should be 'x,y' or '[x,y]' or '(x,y)'")
    else:
        coord = np.asarray(coord)

    # Convert from PsychoPy units to pixels if needed
    if psychopy_units == 'norm':
        coord = coord * np.array([half_width, half_height])
    elif psychopy_units == 'height':
        coord = coord * screen_dims[1]
    elif psychopy_units != 'pix':
        raise ValueError("units must be 'pix', 'norm', or 'height'")
        
    if direction == 'to_el':
        # Convert from PsychoPy (center origin) to Eyelink (top-left origin)
        converted = np.array([
            coord[0] + half_width,
            half_height - coord[1]
        ])
    elif direction == 'to_psychopy':
        # Convert from Eyelink (top-left origin) to PsychoPy (center origin)
        converted = np.array([
            coord[0] - half_width,
            half_height - coord[1]
        ])
        
        # Convert back to original units if needed
        if psychopy_units == 'norm':
            converted /= np.array([half_width, half_height])
        elif psychopy_units == 'height':
            converted /= screen_dims[1]
    else:
        raise ValueError("direction must be either 'to_el' or 'to_psychopy'")

    if round_to is not None:
        converted = np.round(converted, round_to)
    return converted


def get_isoeccentric_positions(n_items, radius, offset_deg=0, coordinate_system='psychopy', screen_dims=None, round_to=2):
    """
    Get coordinates for items arranged in a circle around screen center.

    Parameters
    ----------
    n_items : int
        Number of items to position in circle
    radius : float
        Distance from screen center to each item
    offset_deg : float, optional
        Rotation offset in degrees from rightmost position (counterclockwise).
        Default is 0.
    coordinate_system : {'psychopy', 'eyelink'}, optional
        Output coordinate system:
        - 'psychopy': origin at center, positive y up
        - 'eyelink': origin at top-left, positive y down
        Default is 'psychopy'.
    screen_dims : list, optional
        Screen dimensions [width, height] in pixels. Only used if 
        coordinate_system is 'eyelink'. Default is [1600, 1200].
    round_to : int or None, optional
        Number of decimal places to round coordinates to. Default is 2.
        If None, no rounding is performed.

    Returns
    -------
    list
        List of (x,y) coordinate tuples for each item position, arranged
        counterclockwise starting from the rightmost position.

    Notes
    -----
    - Items are arranged counterclockwise at equal angular intervals
    - First item is placed at the rightmost position (0 degrees) plus any offset
    - Angular separation between items is 360°/n_items

    Examples
    --------
    >>> # Get 4 positions in PsychoPy coordinates (origin at center)
    >>> get_isoeccentric_positions(4, 100, round_to=0)
    [(100, 0), (0, 100), (-100, 0), (0, -100)]

    >>> # Get 4 positions with 45° offset
    >>> get_isoeccentric_positions(4, 100, offset_deg=45, round_to=0)
    [(71, 71), (-71, 71), (-71, -71), (71, -71)]

    >>> # Get positions in Eyelink coordinates (origin at top-left)
    >>> get_isoeccentric_positions(4, 100, coordinate_system='eyelink', round_to=0)
    [(900, 600), (800, 500), (700, 600), (800, 700)]
    """
    if screen_dims is None:
        screen_dims = [1600, 1200]
    
    # Get raw positions centered at origin
    positions = xy_circle(n_items, radius, phi0=offset_deg)
    
    if coordinate_system == 'eyelink':
        # Convert to eyelink coordinates (origin at top-left)
        positions = [(x + screen_dims[0]/2, screen_dims[1]/2 - y) for x,y in positions]
    
    if round_to is not None:
        positions = [(round(x, round_to), round(y, round_to)) for x,y in positions]
        
    return positions


def xy_circle(n, rho, phi0=0, pole=(0, 0)):
    """
    Generate points arranged in a circle. 
    from https://osdoc.cogsci.nl/3.3/manual/python/common/

    Parameters
    ----------
    n : int
        Number of points to generate
    rho : float
        Radius of the circle (distance from center)
    phi0 : float, optional
        Starting angle in degrees (counterclockwise from right). Default is 0.
    pole : tuple, optional
        Center point (x, y) coordinates. Default is (0, 0).

    Returns
    -------
    list
        List of (x, y) coordinate tuples for points arranged in a circle

    Notes
    -----
    Points are arranged counterclockwise starting from phi0. The angular
    separation between points is 360°/n.

    Examples
    --------
    >>> # Generate 4 points in a circle of radius 100
    >>> xy_circle(4, 100)
    [(100, 0), (0, 100), (-100, 0), (0, -100)]
    
    >>> # Generate 4 points with 45° offset
    >>> xy_circle(4, 100, phi0=45)
    [(70.71, 70.71), (-70.71, 70.71), (-70.71, -70.71), (70.71, -70.71)]
    """
    try:
        n = int(n)
        if n < 0:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError('n should be a non-negative integer in xy_circle()')
    try:
        phi0 = float(phi0)
    except (ValueError, TypeError):
        raise TypeError('phi0 should be numeric in xy_circle()')
    l = []
    for i in range(n):
        l.append(xy_from_polar(rho, phi0, pole=pole))
        phi0 += 360./n
    return l


def xy_from_polar(rho, phi, pole=(0, 0)):
    """
    Convert polar coordinates to Cartesian coordinates.
    from https://osdoc.cogsci.nl/3.3/manual/python/common/

    Parameters
    ----------
    rho : float
        Radial distance from origin (or pole)
    phi : float
        Angle in degrees (counterclockwise from right)
    pole : tuple, optional
        Origin point (x, y) coordinates. Default is (0, 0).

    Returns
    -------
    tuple
        (x, y) coordinates in Cartesian system

    Notes
    -----
    The angle phi is measured counterclockwise from the positive x-axis,
    following the mathematical convention.

    Examples
    --------
    >>> # Convert 45° angle at distance 100
    >>> xy_from_polar(100, 45)
    (70.71, 70.71)
    
    >>> # Convert with offset origin
    >>> xy_from_polar(100, 0, pole=(50, 50))
    (150, 50)
    """
    try:
        rho = float(rho)
    except:
        raise TypeError('rho should be numeric in xy_from_polar()')
    try:
        phi = float(phi)
    except:
        raise TypeError('phi should be numeric in xy_from_polar()')
    phi = math.radians(phi)
    ox, oy = parse_pole(pole)
    x = rho * math.cos(phi) + ox
    y = rho * math.sin(phi) + oy
    return x, y

def parse_pole(pole):
    """
    Parse and validate pole (origin) coordinates.
    from https://osdoc.cogsci.nl/3.3/manual/python/common/

    Parameters
    ----------
    pole : tuple or array-like
        (x, y) coordinates for the pole/origin point

    Returns
    -------
    tuple
        Validated (x, y) coordinates as floats

    Raises
    ------
    ValueError
        If pole is not a valid 2D coordinate pair

    Examples
    --------
    >>> parse_pole((1, 2))
    (1.0, 2.0)
    >>> parse_pole([1.5, 2.5])
    (1.5, 2.5)
    """
    try:
        ox = float(pole[0])
        oy = float(pole[1])
        assert(len(pole) == 2)
    except:
        raise ValueError('pole should be a tuple (or similar) of length '
                         'with two numeric values')
    return ox, oy


def angular_distance(line1, line2):
    """
    Calculate the angle between two lines in degrees.

    Parameters
    ----------
    line1 : tuple
        Tuple of two points ((x1,y1), (x2,y2)) defining the first line
    line2 : tuple
        Tuple of two points ((x1,y1), (x2,y2)) defining the second line

    Returns
    -------
    float
        Angle between the lines in degrees, always in range [0, 180]

    Examples
    --------
    >>> # Perpendicular lines
    >>> line1 = ((0,0), (1,0))  # horizontal line
    >>> line2 = ((0,0), (0,1))  # vertical line
    >>> angular_distance(line1, line2)
    90.0
    
    >>> # 45-degree angle
    >>> line1 = ((0,0), (1,0))
    >>> line2 = ((0,0), (1,1))
    >>> angular_distance(line1, line2)
    45.0
    """
    # Calculate the direction vectors of both lines
    direction_vector1 = np.array(line1[1]) - np.array(line1[0])
    direction_vector2 = np.array(line2[1]) - np.array(line2[0])

    # Calculate the dot product of the direction vectors
    dot_product = np.dot(direction_vector1, direction_vector2)

    # Calculate the magnitudes (norms) of the direction vectors
    norm1 = np.linalg.norm(direction_vector1)
    norm2 = np.linalg.norm(direction_vector2)

    # Calculate the cosine of the angle between the lines
    cosine_similarity = dot_product / (norm1 * norm2)

    # Calculate the angle in radians using arccosine
    angular_distance_radians = np.abs(np.arccos(cosine_similarity))

    # Convert the angle from radians to degrees
    angular_distance_degrees = np.degrees(angular_distance_radians)

    # Ensure the result is within 0-180 degrees
    if angular_distance_degrees > 180:
        angular_distance_degrees = 360 - angular_distance_degrees

    return angular_distance_degrees

    
def gaussian_2d(img, fc):
    """
    Apply a 2D Gaussian filter to an image.
    Python adaptation of https://github.com/cvzoya/saliency/blob/master/code_forMetrics/antonioGaussian.m
    
    Parameters
    ----------
    img : numpy.ndarray
        2D input image array
    fc : float
        Cut-off frequency (-6dB)

    Returns
    -------
    numpy.ndarray
        Filtered image with same shape as input

    Notes
    -----
    Python adaptation of the Gaussian filtering method from the saliency metrics
    toolbox [1]_. The filter is applied in the frequency domain using FFT.

    References
    ----------
    .. [1] Bylinskii, Z., Judd, T., Oliva, A., Torralba, A., & Durand, F. (2016).
           "What do different evaluation metrics tell us about saliency models?"
           arXiv preprint arXiv:1604.03605.

    Examples
    --------
    >>> # Create sample image with noise
    >>> img = np.random.randn(100, 100)
    >>> # Apply Gaussian filter
    >>> filtered = gaussian_2d(img, fc=10)
    """
    sn, sm = img.shape
    n = max(sn, sm)
    n = n + np.mod(n,2)
    n = int(2**np.ceil(np.log2(n)))
    # frequencies
    fx,fy = np.meshgrid(range(n),range(n))
    fx = fx-n/2
    fy = fy-n/2
    # convert cut of frequency into gaussian width
    s = fc/np.sqrt(np.log(2))
    # compute transfer function of gaussian filter
    gf = np.exp(-(fx**2+fy**2)/(s**2))
    gf = np.fft.fftshift(gf)
    # convolve (in Fourier domain) each color band:
    BF = np.zeros((n,n))
    BF[:,:] = np.real(np.fft.ifft2(np.fft.fft2(img[:,:], s=[n,n])*gf))
    # crop output to have same size than the input
    BF = BF[:sn,:sm]
    return BF

def mat2gray(img):
    """
    Scale image values to grayscale range [0, 1].

    Parameters
    ----------
    img : numpy.ndarray
        Input image array

    Returns
    -------
    numpy.ndarray
        Normalized image with values scaled to range [0, 1]

    Examples
    --------
    >>> # Create sample image
    >>> img = np.array([[0, 127, 255], [63, 191, 255]])
    >>> normalized = mat2gray(img)
    >>> normalized
    array([[0. , 0.5, 1. ],
           [0.25, 0.75, 1. ]])
    """
    img = np.double(img)
    out = np.zeros(img.shape, dtype=np.double)
    normalized = cv2.normalize(img, out, 1.0, 0.0, cv2.NORM_MINMAX)
    return normalized
