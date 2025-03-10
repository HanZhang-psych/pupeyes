# -*- coding:utf-8 -*-

"""
Default Settings Module

This module contains default visualization settings for matplotlib and plotly.
"""

## matplotlib default settings (rcParams)
default_mpl = {
    # figure settings
    'figure.figsize': (10, 5),
    'figure.dpi': 800, # resolution
    'figure.autolayout': True, # automatically adjust subplot parameters
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
     # axes settings
    'axes.grid': False,
    'axes.labelsize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
     # font settings
    'font.size': 16, # default font size
    'font.family': 'sans-serif',
    'font.weight': 'normal',
    # legend settings
    'legend.fontsize': 14,
    'legend.loc': 'best',
    'legend.frameon': False,
    # grid settings
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.color': 'black',
    'grid.alpha': 0.5,
    # line settings
    'lines.linewidth': 2, 
    'lines.color': 'black',
    'lines.linestyle': '-',
    'lines.marker': ''
}

# default layout for plotly 
default_plotly = {
    # figure settings
    'width': 1000,
    'height': 500,
    # title settings
    'title_text': 'title',
    'title_font_size': 20,
    'title_font_family': 'sans-serif',
    'title_font_weight': 'bold',
    'title_x': 0.5,
    'title_xanchor': 'center',
    # x-axis settings
    'xaxis_title': 'x',
    'xaxis_title_font_size': 20,
    'xaxis_title_font_family': 'Arial',
    'xaxis_title_font_weight': 'bold',
    'xaxis_tickfont_size': 16,
    'xaxis_tickformat': 'i',
    'xaxis_tickfont_family': 'Arial',
    'xaxis_showgrid': False,
    'xaxis_zeroline': False,
    'xaxis_showline': True, # spine
    'xaxis_linecolor': 'black', # spine color
    'xaxis_linewidth': 2, # spine width
    # y-axis settings
    'yaxis_title': 'y',
    'yaxis_title_font_size': 20,
    'yaxis_title_font_family': 'Arial',
    'yaxis_title_font_weight': 'bold',
    'yaxis_tickfont_size': 16,
    'yaxis_tickfont_family': 'Arial',
    'yaxis_showgrid': False,
    'yaxis_zeroline': False,
    'yaxis_showline': True, # spine
    'yaxis_linecolor': 'black', # spine color
    'yaxis_linewidth': 2, # spine width
    # legend settings
    'showlegend': True,
    'legend_font_size': 14,
    'legend_font_family': 'Arial',
    'legend_yanchor': 'middle',
    'legend_xanchor': 'left',
    'legend_x': 1.01,
    'legend_y': 0.5,
    # template settings
    'template': 'plotly_white'
}
