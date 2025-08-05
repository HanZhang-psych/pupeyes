#-*- coding:utf-8 -*-

# Core functionality
from .data.eyelink import EyelinkReader
from .data.tobii_titta import TobiiTittaReader
from .pupil import (PupilProcessor, convert_pupil)

# Utilities
from .utils import make_mask
from .plot_utils import (draw_aois, draw_scanpath, draw_heatmap)

# Applications
from .apps.fixation_viewer import FixationViewer
from .apps.aoi_drawer import AOIDrawer
from .apps.pupil_viewer import PupilViewer

__all__ = ['EyelinkReader', 'TobiiTittaReader', 'PupilProcessor', 'make_mask', 'convert_pupil', 
           'FixationViewer', 'AOIDrawer', 'PupilViewer', 
           'draw_aois', 'draw_scanpath', 'draw_heatmap']

# read version from installed package
from importlib.metadata import version
__version__ = version("pupeyes")