"""
Plotting Utilities for Eye Movement Data

Author: Han Zhang
Email: hanzh@umich.edu

This module provides plotting functions for eye movement data visualization,
including heatmaps, scanpaths, and areas of interest (AOIs).
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.collections import LineCollection
import warnings
from .utils import gaussian_2d, mat2gray

def draw_heatmap(x, y, screen_dims, durations=None, fc=6, colormap='viridis', 
                 alpha=0.7, background_img=None, return_data=False):
    """
    Create a heatmap visualization of fixation density.
    
    Parameters
    ----------
    x : array-like
        X coordinates of fixations
    y : array-like
        Y coordinates of fixations
    screen_dims : tuple
        Screen dimensions in pixels (width, height)
    durations : array-like, optional
        Fixation durations for weighting
    fc : float, default=6
        Cut off frequency (-6dB) for Gaussian smoothing
    colormap : str, default='viridis'
        Matplotlib colormap to use
    alpha : float, default=0.7
        Transparency of the heatmap
    background_img : PIL.Image or numpy.ndarray, optional
        Background image to overlay heatmap on. If a string, it is assumed to be a path to an image file.
        If a numpy array, it is assumed to be an image array.
    return_data : bool, default=False
        If True, returns the heatmap array along with the figure
    
    Returns
    -------
    tuple or matplotlib.figure.Figure
        If return_data is True, returns (heatmap_array, (figure, axes))
        Otherwise, returns (figure, axes)
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
    Create a visualization of fixation sequence with numbered points and connecting lines.
    
    Parameters
    ----------
    x : array-like
        X coordinates of fixations
    y : array-like
        Y coordinates of fixations
    screen_dims : tuple
        Screen dimensions in pixels (width, height)
    durations : array-like, optional
        Fixation durations for dot sizing
    dot_size_scale : float, default=3.0
        Base size for dots if no duration data, or scaling factor for dot sizes with duration
    line_width : float, default=1.0
        Width of the connecting lines
    dot_cmap : str, default='viridis'
        Colormap for dots (representing duration if provided)
    line_cmap : str, default='coolwarm'
        Colormap for lines (representing sequence order)
    dot_alpha : float, default=0.8
        Transparency of dots
    line_alpha : float, default=0.5
        Transparency of lines
    background_img : PIL.Image or numpy.ndarray, optional
        Background image to overlay on the plot. If a string, it is assumed to be a path to an image file.
        If a numpy array, it is assumed to be an image array.
    show_labels : bool, default=True
        Whether to show numeric labels for fixation order
    label_offset : tuple, default=(5, 5)
        Offset for label positions in pixels
    
    Returns
    -------
    tuple
        (figure, axes) containing the plot
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

def draw_aois(aois, screen_dims, background_img=None, alpha=0.3, colors=None, save=None):
    """
    Draw AOIs on a plot, optionally with a background stimulus image.
    
    Parameters
    ----------
    aois : dict
        Dictionary mapping AOI names to lists of (x, y) vertex coordinates. 
        For example, {'AOI1': [(100, 100), (200, 100), (200, 200), (100, 200)]}
    screen_dims : tuple
        Screen dimensions in pixels (width, height)
    background_img : PIL.Image or numpy.ndarray, optional
        Background image to overlay AOIs on. If a string, it is assumed to be a path to an image file.
        If a numpy array, it is assumed to be an image array.
    alpha : float, optional
        Transparency of AOI fill colors (0-1)
    colors : dict, optional
        Dictionary mapping AOI names to colors. If None, uses default colors.
    save : str, optional
        Path to save the plot
    
    Returns
    -------
    tuple
        (figure, axes) containing the plot
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
        ax.fill(vertices[:, 0], vertices[:, 1], alpha=alpha, color=color, label=aoi_name)
        # Draw outline
        ax.plot(np.append(vertices[:, 0], vertices[0, 0]),
               np.append(vertices[:, 1], vertices[0, 1]),
               color=color, linewidth=2)
    
    # Set axis limits and labels
    ax.set_xlim(0, screen_dims[0])
    ax.set_ylim(screen_dims[1], 0)  # reverse y-axis for screen coordinates
    ax.set_title('Areas of Interest (AOIs)')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save)
    
    return fig, ax 