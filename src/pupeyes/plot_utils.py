"""
Plotting Utilities for Eye Movement Data

This module provides plotting functions for eye movement data visualization,
including heatmaps, scanpaths, and areas of interest (AOIs).
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.collections import LineCollection
import warnings
from .utils import gaussian_2d, mat2gray
import pandas as pd

def draw_heatmap(x, y, screen_dims, durations=None, fc=6, colormap='viridis', 
                 alpha=0.7, background_img=None, return_data=False):
    """
    Create a heatmap visualization of fixation density using 2D histogram and Gaussian smoothing.
    
    This function generates a heatmap by first creating a 2D histogram of fixation locations,
    then applying Gaussian smoothing to create a continuous representation of fixation density.
    The resulting heatmap can be overlaid on a background image if provided.
    
    Parameters
    ----------
    x : array-like
        X coordinates of fixations in screen coordinates (0 = left)
    y : array-like
        Y coordinates of fixations in screen coordinates (0 = top)
    screen_dims : tuple
        Screen dimensions in pixels (width, height). Used to set the histogram bins
        and plot boundaries.
    durations : array-like, optional
        Fixation durations for weighting the heatmap. If provided, longer fixations
        will contribute more to the density estimate.
    fc : float, default=6
        Cut off frequency (-6dB) for Gaussian smoothing. Higher values result in
        less smoothing.
    colormap : str, default='viridis'
        Matplotlib colormap to use for the heatmap visualization
    alpha : float, default=0.7
        Transparency of the heatmap overlay (0 = transparent, 1 = opaque)
    background_img : str, PIL.Image or numpy.ndarray, optional
        Background image to overlay heatmap on. Can be:
        - Path to an image file (str)
        - PIL Image object
        - Numpy array of image data
        Image will be resized to match screen_dims if necessary.
    return_data : bool, default=False
        If True, returns the raw heatmap array instead of plotting
    
    Returns
    -------
    tuple or numpy.ndarray
        If return_data is True:
            Returns the normalized heatmap array (shape: height x width)
        If return_data is False:
            Returns (figure, axes) tuple containing the plot
    
    Notes
    -----
    - The heatmap is generated using numpy.histogram2d and smoothed using a 
      Gaussian filter
    - The coordinate system uses screen coordinates where (0,0) is at the top-left
    - The heatmap values are normalized to the range [0,1]
    - When using a background image, the heatmap is overlaid with the specified
      alpha transparency
    """
    # Generate heatmap using histogram2d and gaussian smoothing
    heatmap = np.histogram2d(
        x=y,  # Note: x and y are swapped because histogram2d uses matrix coordinates
        y=x,
        bins=(screen_dims[1], screen_dims[0]),
        range=[[0, screen_dims[1]], [0, screen_dims[0]]],
        weights=durations
    )[0]
    
    # Apply Gaussian smoothing
    heatmap = gaussian_2d(heatmap, fc=fc)
    
    # Normalize to [0, 1]
    heatmap = mat2gray(heatmap)
    
    if return_data:
        return heatmap, None
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot background if provided
    if background_img is not None:
        if isinstance(background_img, str):
            img = Image.open(background_img)
            if img.size != screen_dims:
                print('Original size:', img.size, 'Resized size:', screen_dims)
                img = img.resize(screen_dims)
            background_img = np.asarray(img)
        elif isinstance(background_img, np.ndarray):
            background_img = background_img
        else:
            raise ValueError('Invalid background image type')
        ax.imshow(background_img, extent=[0, screen_dims[0], screen_dims[1], 0])
    
    # Plot heatmap
    im = ax.imshow(heatmap, extent=[0, screen_dims[0], screen_dims[1], 0],
                   cmap=colormap, alpha=alpha)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set labels
    ax.set_title('Fixation Density Heatmap')
    
    return fig, ax

def draw_scanpath(x, y, screen_dims, durations=None, dot_size_scale=3.0, line_width=1.0,
                 dot_cmap='viridis', line_cmap='coolwarm', dot_alpha=0.8, line_alpha=0.5,
                 background_img=None, show_labels=True, label_offset=(5, 5)):
    """
    Create a visualization of fixation sequence (scanpath) with numbered points and connecting lines.
    
    This function visualizes the sequence of fixations by plotting points at fixation locations
    and connecting them with lines to show the order. The points can be sized by fixation duration
    and colored using a colormap. The connecting lines use a different colormap to show sequence order.
    
    Parameters
    ----------
    x : array-like
        X coordinates of fixations in screen coordinates (0 = left)
    y : array-like
        Y coordinates of fixations in screen coordinates (0 = top)
    screen_dims : tuple
        Screen dimensions in pixels (width, height). Used to set plot boundaries.
    durations : array-like, optional
        Fixation durations in milliseconds. If provided, dot sizes will be scaled
        by the square root of duration.
    dot_size_scale : float, default=3.0
        Base size for dots if no duration data, or scaling factor for dot sizes 
        when durations are provided. Larger values = bigger dots.
    line_width : float, default=1.0
        Width of the lines connecting fixation points
    dot_cmap : str, default='viridis'
        Colormap for dots. If durations provided, represents duration.
        If no durations, all dots will be blue.
    line_cmap : str, default='coolwarm'
        Colormap for connecting lines to show sequence order.
        Earlier saccades are colored differently from later ones.
    dot_alpha : float, default=0.8
        Transparency of fixation dots (0 = transparent, 1 = opaque)
    line_alpha : float, default=0.5
        Transparency of connecting lines (0 = transparent, 1 = opaque)
    background_img : str, PIL.Image or numpy.ndarray, optional
        Background image to overlay scanpath on. Can be:
        - Path to an image file (str)
        - PIL Image object
        - Numpy array of image data
        Image will be resized to match screen_dims if necessary.
    show_labels : bool, default=True
        Whether to show numeric labels for fixation sequence order
    label_offset : tuple, default=(5, 5)
        (x, y) offset in pixels for the position of numeric labels relative
        to fixation points
    
    Returns
    -------
    tuple
        (figure, axes) tuple containing the plot
    
    Notes
    -----
    - The coordinate system uses screen coordinates where (0,0) is at the top-left
    - Dot sizes are scaled by sqrt(duration) if durations are provided
    - When using a background image, it is displayed with 40% opacity
    - Fixation sequence is numbered starting from 1
    - Lines between fixations show the saccade paths
    """
    # Create figure
    fig, ax = plt.subplots()
    
    # Plot background if provided
    if background_img is not None:
        if isinstance(background_img, str):
            img = Image.open(background_img)
            if img.size != screen_dims:
                print('Original size:', img.size, 'Resized size:', screen_dims)
                img = img.resize(screen_dims)
            background_img = np.asarray(img)
        elif isinstance(background_img, np.ndarray):
            background_img = background_img
        else:
            raise ValueError('Invalid background image type')
        ax.imshow(background_img, extent=[0, screen_dims[0], screen_dims[1], 0], alpha=0.4)
    
    # Handle dot sizes and colors based on duration availability
    if durations is not None:
        dot_sizes = np.sqrt(durations) * dot_size_scale
        norm_durations = (durations - durations.min()) / (durations.max() - durations.min())
        scatter = ax.scatter(x, y, s=dot_sizes, c=norm_durations, cmap=dot_cmap,
                           alpha=dot_alpha, zorder=2)
        plt.colorbar(scatter, ax=ax, orientation='vertical', label='Fixation Duration')
    else:
        # Use uniform size and color if no duration data
        scatter = ax.scatter(x, y, s=dot_size_scale*50, c='blue',
                           alpha=dot_alpha, zorder=2)
    
    # Create line segments for saccades
    points = np.column_stack((x, y))
    segments = np.column_stack((points[:-1], points[1:]))
    segments = segments.reshape(-1, 2, 2)
    
    # Create line collection with color gradient
    norm = plt.Normalize(0, len(segments))
    lc = LineCollection(segments, cmap=line_cmap, norm=norm, alpha=line_alpha,
                       linewidth=line_width)
    lc.set_array(np.arange(len(segments)))
    ax.add_collection(lc)
    
    # Add fixation order labels
    if show_labels:
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(str(i+1), (xi + label_offset[0], yi + label_offset[1]),
                       fontsize=8, ha='left', va='bottom')
    
    # Set axis limits and labels
    ax.set_xlim(0, screen_dims[0])
    ax.set_ylim(screen_dims[1], 0)  # Invert y-axis for screen coordinates
    ax.set_title('Scanpath')
    
    return fig, ax

def draw_aois(aois, screen_dims, x=None, y=None, background_img=None, alpha=0, colors=None, save=None):
    """
    Draw Areas of Interest (AOIs) and optionally plot fixation points within them.
    
    This function visualizes AOIs as polygons and can optionally show fixation points
    colored according to which AOI they fall within. AOIs are drawn as outlined polygons
    with optional fill color and can be overlaid on a background image.
    
    Parameters
    ----------
    aois : dict
        Dictionary mapping AOI names to lists of (x, y) vertex coordinates defining
        the AOI polygons. The last vertext should be the same as the first vertex 
        to close the polygon.
        Example: {'AOI1': [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]}
    screen_dims : tuple
        Screen dimensions in pixels (width, height). Used to set plot boundaries
        and maintain correct aspect ratio.
    x : array-like, optional
        X coordinates of fixation points in screen coordinates (0 = left).
        If provided along with y, points will be plotted and colored based on
        which AOI they fall within.
    y : array-like, optional
        Y coordinates of fixation points in screen coordinates (0 = top)
    background_img : str, PIL.Image or numpy.ndarray, optional
        Background image to overlay AOIs on. Can be:
        - Path to an image file (str)
        - PIL Image object
        - Numpy array of image data
        Image will be resized to match screen_dims if necessary.
    alpha : float, default=0
        Fill transparency for AOI polygons (0 = transparent, 1 = opaque).
        The outlines remain fully opaque regardless of this value.
    colors : dict, optional
        Dictionary mapping AOI names to colors for both the AOI polygons
        and their associated fixation points. If None, uses matplotlib's
        tab20 colormap to assign colors automatically.
    save : str, optional
        Path where the plot should be saved. If None, plot is not saved
        to disk.
    
    Returns
    -------
    tuple
        (figure, axes) tuple containing the plot
    
    Notes
    -----
    - The coordinate system uses screen coordinates where (0,0) is at the top-left
    - AOIs are drawn with solid outlines and optional transparent fill
    - When background_img is provided, it is displayed with 40% opacity
    - Fixation points outside any AOI are colored gray
    - A legend is automatically added showing AOI names
    - The plot maintains the correct aspect ratio based on screen dimensions
    """
    # Set figure size based on screen dimensions, maintaining aspect ratio
    aspect_ratio = screen_dims[1] / screen_dims[0]
    fig, ax = plt.subplots()
    ax.set_aspect(aspect_ratio)
    
    # Plot background if provided
    if background_img is not None:
        if isinstance(background_img, str):
            # read image as numpy array
            img = Image.open(background_img)
            if img.size != screen_dims:
                print('Original size:', img.size, 'Resized size:', screen_dims)
                img = img.resize(screen_dims)
            background_img = np.asarray(img)
        elif isinstance(background_img, np.ndarray):
            background_img = background_img
        else:
            raise ValueError('Invalid background image type')
        
        ax.imshow(background_img, extent=[0, screen_dims[0], screen_dims[1], 0], alpha=0.4)
    
    # Use default colormap if no colors provided
    if colors is None:
        cmap = plt.cm.get_cmap('tab20')
        colors = {name: cmap(i/len(aois)) for i, name in enumerate(aois.keys())}
    
    # Draw each AOI
    for aoi_name, vertices in aois.items():
        vertices = np.array(vertices)
        color = colors.get(aoi_name, 'blue')
        
        # Draw filled polygon with transparency
        ax.fill(vertices[:, 0], vertices[:, 1], alpha=alpha, color=color)
        # Draw outline
        ax.plot(np.append(vertices[:, 0], vertices[0, 0]),
               np.append(vertices[:, 1], vertices[0, 1]),
               color=color, linewidth=2, label=aoi_name)
    
    # Plot fixation points if provided
    if x is not None and y is not None:
        from .aoi import get_fixation_aoi
        # Convert to numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Get AOI for each fixation point
        point_aois = get_fixation_aoi(x, y, aois)
        
        # Plot points with different colors based on AOI membership
        for aoi_name in aois.keys():
            mask = np.array([p == aoi_name if p is not None else False for p in point_aois])
            if np.any(mask):
                ax.scatter(x[mask], y[mask], 
                         color=colors[aoi_name],
                         alpha=1)
        
        # Plot points not in any AOI
        mask = pd.isna(point_aois) | (point_aois == None)
        if np.any(mask):
            ax.scatter(x[mask], y[mask], 
                     color='gray',
                     alpha=1)
    
    # Set axis limits and labels
    ax.set_xlim(0, screen_dims[0])
    ax.set_ylim(screen_dims[1], 0)  # reverse y-axis for screen coordinates
    ax.set_title('Areas of Interest (AOIs)')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    
    return fig, ax 