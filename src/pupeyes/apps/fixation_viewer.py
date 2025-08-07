"""
Interactive Eye Movement Visualization Module using Dash

This module provides an interactive web-based visualization tool for eye movement data,
including scanpath replay, heatmaps, areas of interest, and fixation sequence plots.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
from PIL import Image
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
from ..aoi import get_fixation_aoi, compute_aoi_statistics
from ..plot_utils import draw_heatmap

class FixationViewer:
    """An interactive web-based visualization tool for eye movement data.
    
    This class provides a Dash-based interface for visualizing eye movement data with 
    multiple visualization modes (scanpath, heatmap, AOI), interactive controls, and 
    data export capabilities.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Eye movement data with columns for timestamps, coordinates, etc.
    screen_dims : tuple, default=(1920, 1080)
        Screen dimensions in pixels (width, height)
    col_mapping : dict, optional
        Column name mapping for required fields:

            - trial_id : str or list
                Trial identifier column(s). Can be a single column name or a list of column names
                that together uniquely identify a trial (e.g., ['subject', 'block', 'trial'])
            - timestamp : str
                Timestamp column (optional)
            - x : str
                X coordinate
            - y : str
                Y coordinate
            - duration : str
                Fixation duration (optional)
            - stimuli : str
                Stimuli path/identifier

    stimuli_path : str, optional
        Base path for stimuli images
    animation_speed : int, default=500
        Animation playback speed in milliseconds
    dot_size : int, default=10
        Fixed size for fixation dots

    Attributes
    ----------
    data : pandas.DataFrame
        The eye movement data being visualized
    screen_dims : tuple
        The dimensions of the visualization canvas
    col_mapping : dict
        Mapping of required columns to data columns
    aois : dict
        Dictionary of Areas of Interest definitions
    app : dash.Dash
        The Dash application instance
    """
    
    def __init__(self, data=None, screen_dims=(1920, 1080), 
                 col_mapping=None, stimuli_path=None,
                 animation_speed=500, dot_size=10):
        """Initialize the visualizer.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Eye movement data with columns for timestamps, coordinates, etc.
        screen_dims : tuple, default=(1920, 1080)
            Screen dimensions in pixels (width, height)
        col_mapping : dict, optional
            Column name mapping for required fields:

                - trial_id : str or list
                    Trial identifier column(s). Can be a single column name or a list of column names
                    that together uniquely identify a trial (e.g., ['subject', 'block', 'trial'])
                - timestamp : str
                    Timestamp column (optional)
                - x : str
                    X coordinate
                - y : str
                    Y coordinate
                - duration : str
                    Fixation duration (optional)
                - stimuli : str
                    Stimuli path/identifier

        stimuli_path : str, optional
            Base path for stimuli images
        animation_speed : int, default=500
            Animation playback speed in milliseconds
        dot_size : int, default=50
            Fixed size for fixation dots
        """
        self.screen_dims = screen_dims
        self.stimuli_path = stimuli_path
        self.animation_speed = animation_speed
        self.dot_size = dot_size
        
        # Default column mapping
        self._default_col_mapping = {
            'trial_id': 'trial_id',  # Can be overridden with a list of columns
            'timestamp': None,       # Optional timestamp column
            'x': 'x',
            'y': 'y',
            'duration': None,        # Optional duration column
            'stimuli': 'stimuli'
        }
        
        # Update with user provided mapping
        self.col_mapping = self._default_col_mapping.copy()
        if col_mapping is not None:
            self.col_mapping.update(col_mapping)
            
        # Store data if provided and check for missing values
        self.data = None
        self._stimuli_cache = {}  # Cache for loaded stimuli images
        self.aois = None  # Will store AOI definitions
        
        if data is not None:
            self.set_data(data)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Create app layout
        self.app.layout = self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
    def set_data(self, data):
        """Set the eye movement data for visualization."""
        self.data = data.copy()
        self._validate_data()
        if hasattr(self, 'app'):
            # Update trial selector options
            self.app.layout = self._create_layout()
            
    def set_aois(self, aois):
        """Set Areas of Interest (AOIs) for visualization.
        
        Parameters
        ----------
        aois : dict
            Can be either:

            - A nested dictionary mapping stimulus IDs to AOI definitions.

            - A simple dictionary of AOIs that applies to all stimuli.

            where each AOI is defined by a list of (x,y) vertex coordinates. 
            The last point should be the same as the first point to close the polygon.
        
        Returns
        -------
        None

        Examples
        --------
        >>> # Define AOIs for each stimulus
        >>> aois = {
        ...     'stimulus1': {
        ...         'aoi1': [(x1,y1), (x2,y2), ..., (x1, y1)],
        ...         'aoi2': [(x1,y1), (x2,y2), ..., (x1, y1)]
        ...     },
        ...     'stimulus2': {
        ...         'aoi1': [(x1,y1), (x2,y2), ..., (x1, y1)],
        ...         'aoi2': [(x1,y1), (x2,y2), ..., (x1, y1)]
        ...     }
        ... }

        >>> # Define AOIs for all stimuli
        >>> aois = {
        ...     'aoi1': [(x1,y1), (x2,y2), ..., (x1, y1)],
        ...     'aoi2': [(x1,y1), (x2,y2), ..., (x1, y1)]
        ... }
        """
        if not isinstance(aois, dict):
            raise ValueError("AOIs must be provided as a dictionary")
            
        # Check if this is a simple (unnested) AOI dictionary
        is_simple = all(isinstance(v, (list, tuple)) for v in aois.values())
        
        if is_simple:
            # Validate the simple structure
            for aoi_name, vertices in aois.items():
                if not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in vertices):
                    raise ValueError(f"Invalid vertices for AOI {aoi_name}")
                    
            # Get unique stimulus IDs from the data if available
            stimulus_ids = set()
            if hasattr(self, 'data') and self.data is not None and self.col_mapping['stimuli'] in self.data.columns:
                stimulus_ids = set(self.data[self.col_mapping['stimuli']].unique())
            
            # If no data is set yet, create a default key
            if not stimulus_ids:
                stimulus_ids = {'default'}
                
            # Convert to nested structure
            self.aois = {stim_id: aois.copy() for stim_id in stimulus_ids}
            
            print('Assuming AOIs are the same for all stimuli.')
        else:
            # Validate the nested structure
            for stim_id, stim_aois in aois.items():
                if not isinstance(stim_aois, dict):
                    raise ValueError(f"AOIs for stimulus {stim_id} must be a dictionary")
                for aoi_name, vertices in stim_aois.items():
                    if not isinstance(vertices, (list, tuple)) or \
                       not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in vertices):
                        raise ValueError(f"Invalid vertices for AOI {aoi_name} in stimulus {stim_id}")
            
            self.aois = aois
        
    def _get_stimulus_aois(self, stim_id):
        """Get AOIs for a specific stimulus."""
        if not hasattr(self, 'aois') or self.aois is None:
            return None
            
        # Try to get stimulus-specific AOIs
        aois = self.aois.get(stim_id)
        
        # If not found and we have a 'default' key, use that
        if aois is None and 'default' in self.aois:
            aois = self.aois['default']
            
        return aois
        
    def _validate_data(self):
        """Validate the input data and check for missing values."""
        if self.data is None:
            return
            
        # Check for missing values in required columns
        required_cols = [self.col_mapping[col] for col in ['x', 'y']]
        if isinstance(self.col_mapping['trial_id'], (list, tuple)):
            required_cols.extend(self.col_mapping['trial_id'])
        else:
            required_cols.append(self.col_mapping['trial_id'])
                
        # Only check optional columns if they're specified
        if self.col_mapping['duration'] is not None:
            required_cols.append(self.col_mapping['duration'])
        else:
            print('No duration column specified. Fixation duration will not be displayed.')

        if self.col_mapping['timestamp'] is not None:
            required_cols.append(self.col_mapping['timestamp'])
        else:
            print('No timestamp column specified.')
                
        missing_mask = self.data[required_cols].isna().any(axis=1)
        trials_with_missing = self.data[missing_mask]
            
        if len(trials_with_missing) > 0:
            warnings.warn(f"Found {len(trials_with_missing)} rows with missing values")
            print("Trials with missing values:")
            print(trials_with_missing)
        
    def _get_unique_trials(self):
        """Get all unique trial identifiers from the data."""
        if self.data is None:
            return []
            
        trial_cols = self.col_mapping['trial_id']
        if not isinstance(trial_cols, (list, tuple)):
            trial_cols = [trial_cols]
            
        # Get unique combinations of trial identifier columns
        unique_trials = self.data[trial_cols].drop_duplicates()
        
        # Convert to list of tuples
        if len(trial_cols) == 1:
            return [(val,) for val in unique_trials[trial_cols[0]].values]
        else:
            return [tuple(row) for row in unique_trials.values]
            
    def _format_trial_label(self, trial_values):
        """Format trial identifier for display in plots."""
        trial_cols = self.col_mapping['trial_id']
        if not isinstance(trial_cols, (list, tuple)):
            trial_cols = [trial_cols]
            
        return ' | '.join(f"{col}: {val}" for col, val in zip(trial_cols, trial_values))
        
    def _get_trial_data(self, trial_values):
        """Get data for a specific trial."""
        if self.data is None:
            raise ValueError("No data has been set")
            
        trial_cols = self.col_mapping['trial_id']
        if not isinstance(trial_cols, (list, tuple)):
            trial_cols = [trial_cols]
            
        # Create mask for all trial identifier columns
        mask = pd.Series(True, index=self.data.index)
        for col, val in zip(trial_cols, trial_values):
            mask &= (self.data[col] == val)
            
        return self.data[mask].copy()
        
    def _get_stimuli_image(self, stimuli_id):
        """Get the stimuli image for a given identifier."""
        if stimuli_id not in self._stimuli_cache:
            if self.stimuli_path is None:
                warnings.warn("No stimuli path set")
                return None
            try:
                path = f"{self.stimuli_path}/{stimuli_id}"
                # Try to verify the image file is valid
                with Image.open(path) as img:
                    try:
                        img.verify()  # Verify it's a valid image
                    except Exception as e:
                        warnings.warn(f"Invalid image file {stimuli_id}: {str(e)}")
                        return None
                
                # If verification passed, load and resize the image
                img = Image.open(path)
                if img.size != self.screen_dims:
                    print('Original size:', img.size, 'Resized size:', self.screen_dims)
                    img = img.resize(self.screen_dims)
                self._stimuli_cache[stimuli_id] = img
            except (OSError, IOError) as e:
                warnings.warn(f"Failed to load stimuli {stimuli_id}: {str(e)}")
                return None
            except Exception as e:
                warnings.warn(f"Unexpected error loading stimuli {stimuli_id}: {str(e)}")
                return None
        return self._stimuli_cache.get(stimuli_id)
        
    def _format_trial_id(self, trial_id):
        """
        Format trial identifier for consistent handling.
        
        Parameters
        ----------
        trial_id : str, int, tuple, list, or dict
            Trial identifier. If using composite identifiers, can be:
            - tuple/list of values in order of trial_id columns
            - dict mapping column names to values
            - string representation of a tuple (will be evaluated)
            
        Returns
        -------
        tuple
            Values corresponding to trial_id columns
        """
        if isinstance(trial_id, str):
            # Convert string representation back to tuple
            return eval(trial_id)
            
        trial_cols = self.col_mapping['trial_id']
        if not isinstance(trial_cols, (list, tuple)):
            trial_cols = [trial_cols]
            
        if isinstance(trial_id, dict):
            # Convert dict to tuple in correct order
            return tuple(trial_id[col] for col in trial_cols)
        elif isinstance(trial_id, (list, tuple)):
            if len(trial_id) != len(trial_cols):
                raise ValueError(f"Trial ID should have {len(trial_cols)} values")
            return tuple(trial_id)
        else:
            # Single value
            if len(trial_cols) > 1:
                raise ValueError(f"Trial ID should have {len(trial_cols)} values")
            return (trial_id,)
        
    def _create_layout(self):
        """Create the Dash app layout."""
        # Calculate aspect ratio and height based on screen dimensions
        aspect_ratio = self.screen_dims[1] / self.screen_dims[0]
        plot_height = 600  # Base height in pixels
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Fixation Viewer", className="text-center mb-4")
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Control Panel", className="p-1 small"),
                        dbc.CardBody([
                            # Single row for all controls
                            dbc.Row([
                                # Trial Selection
                                dbc.Col([
                                    html.Label("Trial:", className="mb-0 small"),
                                    dcc.Dropdown(
                                        id='trial-selector',
                                        options=self._get_trial_options(),
                                        value=self._get_default_trial(),
                                        clearable=False,
                                        className="small"
                                    )
                                ], width=5),
                                
                                # Visualization Type Selection
                                dbc.Col([
                                    html.Label("View:", className="mb-0 small"),
                                    dcc.Dropdown(
                                        id='viz-selector',
                                        options=[
                                            {'label': 'Scanpath', 'value': 'scanpath'},
                                            {'label': 'Heatmap', 'value': 'heatmap'},
                                            {'label': 'AOI', 'value': 'aoi'}
                                        ],
                                        value='scanpath',
                                        clearable=False,
                                        className="small"
                                    )
                                ], width=3),
                                
                                # Display Options
                                dbc.Col([
                                    html.Label("Show:", className="mb-0 small"),
                                    dcc.Checklist(
                                        id='display-options',
                                        options=[
                                            {'label': ' BG', 'value': 'background'},
                                            {'label': ' AOI', 'value': 'aois'},
                                            {'label': ' Label', 'value': 'labels'}
                                        ],
                                        value=['background', 'aois', 'labels'],
                                        className="d-flex gap-2 small",
                                        inputClassName="me-1"
                                    )
                                ], width=2),
                                
                                # Export Button
                                dbc.Col([
                                    dbc.Button(
                                        "Export Trial Data",
                                        id="export-button",
                                        color="primary",
                                        size="sm",
                                        className="mt-3 py-0 px-2"
                                    ),
                                    dcc.Download(id="download-data")
                                ], width=2, className="text-end")
                            ])
                        ], className="p-2")
                    ], className="mb-2")
                ], width=12)
            ]),
            
            # Main Visualization Area
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            # Scanpath Plot
                            dcc.Graph(
                                id='scanpath-plot',
                                config={'displayModeBar': True},
                                style={'display': 'block', 'height': f'{plot_height}px'}
                            ),
                            # Heatmap Plot
                            dcc.Graph(
                                id='heatmap-plot',
                                config={'displayModeBar': True},
                                style={'display': 'none', 'height': f'{plot_height}px'}
                            ),
                            # AOI Plot
                            dcc.Graph(
                                id='aoi-plot',
                                config={'displayModeBar': True},
                                style={'display': 'none', 'height': f'{plot_height}px'}
                            )
                        ], className="p-0")  # Remove padding from card body
                    ])
                ], width=8),
                
                # Statistics Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Statistics"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='stats-table',
                                columns=[],
                                data=[],
                                style_table={'overflowX': 'auto', 'height': f'{plot_height}px'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '5px',
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                    'fontSize': '12px',
                                    'fontFamily': 'Arial'
                                },
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold',
                                    'fontSize': '13px',
                                    'fontFamily': 'Arial'
                                },
                                style_data={
                                    'whiteSpace': 'normal',
                                    'height': 'auto'
                                }
                            )
                        ], className="p-0")  # Remove padding from card body
                    ])
                ], width=4)
            ], className="g-0")  # Remove gutters between columns
        ], fluid=True)
        
    def _setup_callbacks(self):
        """Set up all callbacks."""
        self._setup_plot_callbacks()
        self._setup_export_callbacks()
        self._setup_display_callbacks()
        
    def _setup_plot_callbacks(self):
        """Set up callbacks for plot updates."""
        # Update scanpath plot
        @self.app.callback(
            Output('scanpath-plot', 'figure'),
            [Input('trial-selector', 'value'),
             Input('display-options', 'value')]
        )
        def update_scanpath(trial_id, display_options):
            return self._create_scanpath_figure(trial_id, display_options)
            
        # Update heatmap plot
        @self.app.callback(
            Output('heatmap-plot', 'figure'),
            [Input('trial-selector', 'value'),
             Input('display-options', 'value')]
        )
        def update_heatmap(trial_id, display_options):
            return self._create_heatmap_figure(trial_id, display_options)
            
        # Update AOI plot
        @self.app.callback(
            Output('aoi-plot', 'figure'),
            [Input('trial-selector', 'value'),
             Input('display-options', 'value')]
        )
        def update_aoi_plot(trial_id, display_options):
            return self._create_aoi_figure(trial_id, display_options)
            
        # Update statistics table
        @self.app.callback(
            [Output('stats-table', 'data'),
             Output('stats-table', 'columns')],
            [Input('trial-selector', 'value')]
        )
        def update_statistics(trial_id):
            stats = self._create_statistics_data(trial_id)
            if not stats:
                return [], []
            columns = [{'name': col, 'id': col} for col in ['Metric', 'Value']]
            return stats, columns
            
    def _setup_export_callbacks(self):
        """Set up callbacks for data export."""
        @self.app.callback(
            Output('download-data', 'data'),
            [Input('export-button', 'n_clicks')],
            [State('trial-selector', 'value')]
        )
        def export_data(n_clicks, trial_id):
            if n_clicks is None:
                return None
                
            try:
                # Convert string representation of trial ID back to tuple
                trial_values = self._format_trial_id(trial_id)
                
                # Get trial data
                trial_data = self._get_trial_data(trial_values)
                
                if len(trial_data) == 0:
                    print(f"No data found for trial {trial_id}")
                    return None
                    
                # Format filename using trial label
                filename = f"eye_movement_data_{self._format_trial_label(trial_values).replace(' | ', '_')}.csv"
                
                return dcc.send_data_frame(trial_data.to_csv, filename, index=False)
            except Exception as e:
                print(f"Error exporting data: {str(e)}")
                return None
            
    def _setup_display_callbacks(self):
        """Set up callbacks for display control."""
        @self.app.callback(
            [Output('scanpath-plot', 'style'),
             Output('heatmap-plot', 'style'),
             Output('aoi-plot', 'style')],
            [Input('viz-selector', 'value')]
        )
        def update_visible_plot(selected_viz):
            # Create style dicts for each plot
            styles = []
            for viz_type in ['scanpath', 'heatmap', 'aoi']:
                if viz_type == selected_viz:
                    styles.append({'display': 'block'})
                else:
                    styles.append({'display': 'none'})
            return styles
            
    def _get_trial_options(self):
        """Get options for trial selector dropdown."""
        if self.data is None:
            return []
        trial_ids = self._get_unique_trials()
        return [{'label': self._format_trial_label(t), 'value': str(t)} for t in trial_ids]
            
    def _get_default_trial(self):
        """Get the default trial ID for initial display."""
        if self.data is None:
            return None
        trial_ids = self._get_unique_trials()
        if trial_ids:
            return str(trial_ids[0])
        return None
        
    def _create_marker_dict(self, durations=None, duration_index=None):
        """Create a marker dictionary for scanpath visualization.
        
        Parameters
        ----------
        durations : array-like, optional
            Array of fixation durations
        duration_index : int, optional
            If provided, only use durations up to this index
        """
        marker_dict = dict(size=self.dot_size)
        if durations is not None:
            if duration_index is not None:
                dur_values = durations[:duration_index + 1]
            else:
                dur_values = [durations[0]]  # For initial state
                
            marker_dict.update(dict(
                color=dur_values,
                colorscale='Viridis',
                showscale=True,
                cmin=durations.min(),
                cmax=durations.max()
            ))
        else:
            marker_dict.update(dict(color='blue'))
        return marker_dict

    def _create_scatter_trace(self, x, y, marker_dict, display_options, index=None):
        """Create a scatter trace for scanpath visualization.
        
        Parameters
        ----------
        x, y : array-like
            Coordinates for the scatter points
        marker_dict : dict
            Marker styling dictionary
        display_options : list
            Display options for the visualization
        index : int, optional
            If provided, create trace for frame at this index
        """
        is_initial = index is None
        x_vals = [x[0]] if is_initial else x[:index + 1]
        y_vals = [y[0]] if is_initial else y[:index + 1]
        
        text = ['1'] if is_initial else [str(j+1) for j in range(index + 1)]
        text = text if 'labels' in display_options else None
        
        return go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text' if 'labels' in display_options else 'markers',
            marker=marker_dict,
            text=text,
            textposition="top center",
            name='Fixations',
            hovertemplate='<b>Fixation %{text}</b><br>' +
                        'X: %{x}<br>' +
                        'Y: %{y}<br>' +
                        ('Duration: %{marker.color:.0f} ms<br>' if 'color' in marker_dict else '') +
                        '<extra></extra>'
        )

    def _create_scanpath_figure(self, trial_id, display_options):
        """Create an interactive scanpath visualization.
        
        Parameters
        ----------
        trial_id : str or tuple
            Identifier for the trial to visualize
        display_options : list
            List of display options to enable:

                - background : bool
                    Show stimulus image
                - aois : bool
                    Show Areas of Interest
                - labels : bool
                    Show fixation number labels

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure with scanpath visualization including:

                - Animated fixation sequence
                - Duration-based color coding
                - Background image (if enabled)
                - Playback controls
                - Hover information

        Notes
        -----
        - Creates animation frames for sequential display
        - Includes play/pause controls and frame slider
        - Supports duration-based color coding of fixations
        - Maintains aspect ratio and proper axis scaling
        """
        if trial_id is None:
            return go.Figure()
            
        # Get trial data
        trial_values = self._format_trial_id(trial_id)
        trial_data = self._get_trial_data(trial_values)
        
        if len(trial_data) == 0:
            return go.Figure()
            
        # Extract coordinates and durations
        x = trial_data[self.col_mapping['x']].values
        y = trial_data[self.col_mapping['y']].values
        durations = trial_data[self.col_mapping['duration']].values if self.col_mapping['duration'] is not None else None
        
        # Create figure
        fig = go.Figure()
        
        # Add background image if requested
        if 'background' in display_options and self.col_mapping['stimuli'] in trial_data.columns:
            stim_id = trial_data[self.col_mapping['stimuli']].iloc[0]
            try:
                background_img = self._get_stimuli_image(stim_id)
                if background_img is not None:
                    fig.add_layout_image(
                        dict(
                            source=background_img,
                            xref="x",
                            yref="y",
                            x=0,
                            y=0,
                            sizex=self.screen_dims[0],
                            sizey=self.screen_dims[1],
                            sizing="stretch",
                            opacity=0.4,
                            layer="above"
                        )
                    )
            except:
                warnings.warn(f"Could not load stimuli for trial {self._format_trial_label(trial_values)}")

        # Create animation frames
        frames = []
        
        # Add initial state
        initial_marker_dict = self._create_marker_dict(durations)
        fig.add_trace(self._create_scatter_trace(x, y, initial_marker_dict, display_options))

        # Create frames
        for i in range(len(x)):
            marker_dict = self._create_marker_dict(durations, i)
            frames.append(go.Frame(
                data=[self._create_scatter_trace(x, y, marker_dict, display_options, i)],
                name=f'frame{i+1}'
            ))
            
        # Update layout with screen dimensions and animation controls
        fig = self._update_figure_layout(fig)
        
        # Add animation controls
        animation_controls = {
            'updatemenus': [{
                'buttons': [
                    dict(
                        args=[None, {'frame': {'duration': self.animation_speed, 'redraw': True},
                                   'mode': 'immediate',
                                   'transition': {'duration': 0}}],
                        label='▶️ Play',
                        method='animate'
                    ),
                    dict(
                        args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                     'mode': 'immediate',
                                     'transition': {'duration': 0}}],
                        label='⏸️ Pause',
                        method='animate'
                    )
                ],
                'type': 'buttons',
                'showactive': True,
                'x': 0,
                'y': -0.1,
                'xanchor': 'right',
                'yanchor': 'top'
            }],
            'sliders': [{
                'currentvalue': {'visible': True},
                'steps': [
                    dict(
                        args=[[f'frame{k+1}'], {'frame': {'duration': self.animation_speed, 'redraw': True},
                                               'mode': 'immediate',
                                               'transition': {'duration': 0}}],
                        label=str(k+1),
                        method='animate'
                    ) for k in range(len(frames))
                ],
                'x': 0.5,
                'y': -0.1,
                'xanchor': 'center',
                'yanchor': 'top',
                'len': 0.9,
                'pad': {'t': 0}
            }]
        }
        fig.update_layout(**animation_controls)
        
        # Add frames to figure
        fig.frames = frames
        
        return fig

    def _create_heatmap_figure(self, trial_id, display_options):
        """Create a heatmap visualization of fixation density.
        
        Parameters
        ----------
        trial_id : str or tuple
            Identifier for the trial to visualize
        display_options : list
            List of display options to enable:

                - background : bool
                    Show stimulus image

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure with heatmap visualization including:

                - Fixation density heatmap
                - Background image (if enabled)
                - Color scale
                - Hover information

        Notes
        -----
        - Uses Gaussian kernel density estimation
        - Supports adjustable background image opacity
        - Maintains aspect ratio and proper axis scaling
        - Includes interactive hover information
        """
        if trial_id is None:
            return go.Figure()
            
        # Get trial data
        trial_values = self._format_trial_id(trial_id)
        trial_data = self._get_trial_data(trial_values)
        
        if len(trial_data) == 0:
            return go.Figure()
            
        # Extract coordinates and durations
        x = trial_data[self.col_mapping['x']].values
        y = trial_data[self.col_mapping['y']].values
        durations = trial_data[self.col_mapping['duration']].values if self.col_mapping['duration'] is not None else None
        
        # Get background image if requested
        background_img = None
        if 'background' in display_options and self.col_mapping['stimuli'] in trial_data.columns:
            stim_id = trial_data[self.col_mapping['stimuli']].iloc[0]
            try:
                background_img = self._get_stimuli_image(stim_id)
            except:
                warnings.warn(f"Could not load stimuli for trial {self._format_trial_label(trial_values)}")
        
        # Generate heatmap data using the plotting utility
        heatmap, _ = draw_heatmap(x, y, screen_dims=self.screen_dims, durations=durations, background_img=background_img, return_data=True)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap,
                colorscale='Viridis',
                showscale=True,
                hoverongaps=False,
                hovertemplate='X: %{x:.0f}<br>' +
                            'Y: %{y:.0f}<br>' +
                            'Density: %{z:.3f}<br>' +
                            '<extra></extra>'
            )
        )
        
        # Add background image if available
        if background_img is not None:
            fig.add_layout_image(
                dict(
                    source=background_img,
                    xref="x",
                    yref="y",
                    x=0,
                    y=0,
                    sizex=self.screen_dims[0],
                    sizey=self.screen_dims[1],
                    sizing="stretch",
                    opacity=0.4,
                    layer="above"
                )
            )
                
        # Update layout
        fig = self._update_figure_layout(fig)
        return fig
        
    def _create_aoi_figure(self, trial_id, display_options):
        """Create an AOI visualization with fixation overlay.
        
        Parameters
        ----------
        trial_id : str or tuple
            Identifier for the trial to visualize
        display_options : list
            List of display options to enable:

                - background : bool
                    Show stimulus image
                - aois : bool
                    Show Areas of Interest
                - labels : bool
                    Show fixation number labels

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive figure with AOI visualization including:

                - AOI polygons with labels
                - Fixations colored by AOI
                - Background image (if enabled)
                - Hover information

        Notes
        -----
        - Groups fixations by AOI membership
        - Uses distinct colors for different AOIs
        - Supports interactive hover information
        - Maintains aspect ratio and proper axis scaling
        """
        if trial_id is None:
            return go.Figure()
            
        # Get trial data
        trial_values = self._format_trial_id(trial_id)
        trial_data = self._get_trial_data(trial_values)
        
        if len(trial_data) == 0:
            return go.Figure()
            
        # Extract coordinates and durations
        x = trial_data[self.col_mapping['x']].values
        y = trial_data[self.col_mapping['y']].values
        durations = trial_data[self.col_mapping['duration']].values if self.col_mapping['duration'] is not None else None
        
        # Get stimulus ID for AOI lookup
        stim_id = None
        if self.col_mapping['stimuli'] in trial_data.columns:
            stim_id = trial_data[self.col_mapping['stimuli']].iloc[0]
        
        # Create figure
        fig = go.Figure()
        
        # Add background image if requested
        if 'background' in display_options and stim_id is not None:
            try:
                background_img = self._get_stimuli_image(stim_id)
                if background_img is not None:
                    fig.add_layout_image(
                        dict(
                            source=background_img,
                            xref="x",
                            yref="y",
                            x=0,
                            y=0,
                            sizex=self.screen_dims[0],
                            sizey=self.screen_dims[1],
                            sizing="stretch",
                            opacity=0.4,
                            layer="above"
                        )
                    )
            except:
                warnings.warn(f"Could not load stimuli for trial {self._format_trial_label(trial_values)}")
                
        # Get stimulus-specific AOIs
        stimulus_aois = self._get_stimulus_aois(stim_id) if stim_id is not None else None
                
        # Add AOIs if defined and requested
        if stimulus_aois and 'aois' in display_options:
            # Use a colormap for AOIs
            aoi_colors = plt.cm.Set3(np.linspace(0, 1, len(stimulus_aois)))
            
            for (aoi_name, vertices), color in zip(stimulus_aois.items(), aoi_colors):
                vertices_array = np.array(vertices) 
                fig.add_trace(
                    go.Scatter(
                        x=vertices_array[:, 0],
                        y=vertices_array[:, 1],
                        fill="toself",
                        fillcolor=f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.8)',
                        line=dict(
                            color=f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},1)',
                            width=2
                        ),
                        name=aoi_name,
                        hoverinfo='name'
                    )
                )
        
        # Group fixations by AOI if AOIs are defined
        if stimulus_aois:
            # Initialize fixation groups
            aoi_fixations = {'outside': {'x': [], 'y': [], 'dur': [], 'text': []}}
            for aoi_name in stimulus_aois.keys():
                aoi_fixations[aoi_name] = {'x': [], 'y': [], 'dur': [], 'text': []}
            
            # Group fixations
            for i, (xi, yi) in enumerate(zip(x, y)):
                dur = durations[i] if durations is not None else None
                aoi = get_fixation_aoi(xi, yi, stimulus_aois)
                target_dict = aoi_fixations[aoi if aoi else 'outside']
                target_dict['x'].append(xi)
                target_dict['y'].append(yi)
                if dur is not None:
                    target_dict['dur'].append(dur)
                target_dict['text'].append(str(i+1))
            
            # Plot fixations for each AOI
            for aoi_name, fixations in aoi_fixations.items():
                if len(fixations['x']) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=fixations['x'],
                            y=fixations['y'],
                            mode='markers+text' if 'labels' in display_options else 'markers',
                            marker=dict(
                                size=self.dot_size,
                                color='grey' if aoi_name == 'outside' else None
                            ),
                            text=fixations['text'],
                            textposition="top center",
                            name=f'Fixations ({aoi_name})',
                            hovertemplate='<b>Fixation %{text}</b><br>' +
                                        'X: %{x}<br>' +
                                        'Y: %{y}<br>' +
                                        f'AOI: {aoi_name}<br>' +
                                        '<extra></extra>'
                        )
                    )
        else:
            # Plot all fixations without AOI grouping
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers+text' if 'labels' in display_options else 'markers',
                    marker=dict(
                        size=self.dot_size,
                        color='blue'
                    ),
                    text=[str(i+1) for i in range(len(x))],
                    textposition="top center",
                    name='Fixations',
                    hovertemplate='<b>Fixation %{text}</b><br>' +
                                'X: %{x}<br>' +
                                'Y: %{y}<br>' +
                                '<extra></extra>'
                )
            )
        
        # Update layout
        fig = self._update_figure_layout(fig)
        
        return fig
        
    def _create_statistics_data(self, trial_id):
        """Generate statistics for the current trial and AOIs.
        
        Parameters
        ----------
        trial_id : str or tuple
            Identifier for the trial to analyze

        Returns
        -------
        list of dict
            List of dictionaries containing statistics:

                - Basic metrics (number of fixations)
                - Duration statistics (if available)
                - AOI-specific metrics (if AOIs defined)

            Each dictionary has 'Metric' and 'Value' keys.

        Notes
        -----
        - Calculates basic fixation statistics
        - Includes duration-based metrics if available
        - Computes AOI-specific statistics if AOIs are defined
        - Handles missing data and edge cases
        """
        if trial_id is None:
            return []
            
        # Get trial data
        trial_values = self._format_trial_id(trial_id)
        trial_data = self._get_trial_data(trial_values)
        
        if len(trial_data) == 0:
            return []
            
        # Extract coordinates
        x = trial_data[self.col_mapping['x']].values
        y = trial_data[self.col_mapping['y']].values
        has_duration = self.col_mapping['duration'] is not None
        
        # Get stimulus ID for AOI lookup
        stim_id = None
        if self.col_mapping['stimuli'] in trial_data.columns:
            stim_id = trial_data[self.col_mapping['stimuli']].iloc[0]
        
        # Initialize statistics
        stats = []
        
        # Basic statistics
        stats.append({
            'Metric': 'Number of Fixations',
            'Value': len(x)
        })
        
        # Duration-related statistics
        if has_duration:
            durations = trial_data[self.col_mapping['duration']].values
            stats.extend([
                {
                    'Metric': 'Mean Fixation Duration',
                    'Value': f'{np.mean(durations):.2f}'
                },
                {
                    'Metric': 'Sum Fixation Duration',
                    'Value': f'{np.sum(durations):.2f}'
                },
                {
                    'Metric': 'Min Fixation Duration',
                    'Value': f'{np.min(durations):.2f}'
                },
                {
                    'Metric': 'Max Fixation Duration (ms)',
                    'Value': f'{np.max(durations):.2f}'
                }
            ])
        
        # Get stimulus-specific AOIs
        stimulus_aois = self._get_stimulus_aois(stim_id) if stim_id is not None else None
        
        # AOI statistics
        if stimulus_aois:
            aoi_stats = compute_aoi_statistics(x, y, stimulus_aois, durations if has_duration else None)
            for aoi_name, aoi_data in aoi_stats.items():
                # Add count statistics
                stats.append({
                    'Metric': f'{aoi_name} - Number of Fixations',
                    'Value': aoi_data['count']
                })
                # Add duration-related statistics if available
                if has_duration and aoi_data['count'] > 0:
                    # Total duration
                    if 'total_duration' in aoi_data:
                        stats.append({
                            'Metric': f'{aoi_name} - Sum Fixation Duration',
                            'Value': f'{aoi_data["total_duration"]:.2f}'
                        })
                elif has_duration:
                    # Add N/A for duration stats when count is 0
                    stats.extend([
                        {
                            'Metric': f'{aoi_name} - Sum Fixation Duration',
                            'Value': 'N/A'
                        }
                    ])
        
        return stats

    def _update_figure_layout(self, fig):
        """Update the layout of a figure to match screen dimensions."""
        fig.update_layout(
            autosize=False,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                range=[0, self.screen_dims[0]],
                showgrid=False,
                zeroline=False,
                constrain="domain"
            ),
            yaxis=dict(
                range=[self.screen_dims[1], 0],  # Invert y-axis
                showgrid=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,  # Maintain aspect ratio
                constrain="domain"
            )
        )
        return fig 

    def run(self, debug=False, port=8050, **kwargs):
        """Start the Dash server and run the fixation viewer application.

        This method initializes and starts the web server for the fixation viewer application.
        The application will be accessible through a web browser at the specified port.
        
        Parameters
        ----------
        debug : bool, default=False
            Whether to run the server in debug mode
        port : int, default=8050
            Port to run the server on
        **kwargs : dict
            Additional arguments to pass to dash.run_server()
            See Dash documentation for available options.

        Notes
        -----
        - The application will run until interrupted (Ctrl+C)
        - Access the interface at http://localhost:<port>
        - Debug mode provides additional error information
        - Default port (8050) can be changed if already in use
        """
        self.app.run(debug=debug, port=port, **kwargs) 