# -*- coding:utf-8 -*-

"""
Interactive Pupil Data Viewer

This module provides an interactive web application for visualizing pupil preprocessing steps.
It uses Dash and Plotly to create an interface where users can:
- Select individual trials
- View all preprocessing steps applied to pupil data
- Compare raw and processed pupil traces
"""

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class PupilViewer:
    """
    An interactive web-based visualization tool for pupil preprocessing data.
    
    This class provides a Dash-based interface for visualizing pupil data processing steps,
    allowing users to explore how different preprocessing operations affect the pupil signal.
    The interface supports trial selection, column selection for comparison, and interactive
    plotting with subplots for each processing step.

    Parameters
    ----------
    pupil_processor : PupilProcessor
        Instance of PupilProcessor containing the pupil data and processing history.
        This object should contain both raw and processed pupil data.
    hue : str, optional
        Column name to group data by for separate lines in the plot.
        Useful for visualizing different components of a single trial.
    columns : list of str, optional
        List of column names to plot. If not provided, all pupil columns
        from the PupilProcessor will be shown.

    Attributes
    ----------
    pupil_processor : PupilProcessor
        The PupilProcessor instance containing the data
    hue : str or None
        Column name used for plotting different components of a single trial
    columns : list
        List of column names being plotted
    app : dash.Dash
        The Dash application instance
    """

    def __init__(self, pupil_processor, hue=None, columns=None):
        """
        Initialize PupilViewer with a PupilProcessor instance.
        
        Parameters
        ----------
        pupil_processor : PupilProcessor
            Instance of PupilProcessor containing the pupil data
        hue : str, optional
            Column name to group data by for separate lines in the plot
        columns : list of str, optional
            List of column names to plot. Defaults to all pupil columns.
        """
        self.pupil_processor = pupil_processor
        self.hue = hue
        self.columns = columns if columns is not None else self.pupil_processor.all_pupil_cols
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Create app layout
        self.app.layout = self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
    def _create_layout(self):
        """Create the Dash app layout."""
        # Get trial options
        trial_options = []
        for _, trial in self.pupil_processor.trials.iterrows():
            label = ' | '.join([f"{k}: {v}" for k, v in trial.items()])
            value = {k: v for k, v in trial.items()}
            trial_options.append({'label': label, 'value': str(value)})
            
        # Get column options
        column_options = [{'label': col, 'value': col} for col in self.pupil_processor.all_pupil_cols]
            
        return dbc.Container([
            html.H1("Pupil Viewer", className="text-center my-4"),
            
            dbc.Row([
                # Trial Selection
                dbc.Col([
                    html.Label("Select Trial:", className="mb-2"),
                    dcc.Dropdown(
                        id='trial-selector',
                        options=trial_options,
                        value=str(trial_options[0]['value']),
                        clearable=False,
                        className="mb-4"
                    )
                ], width=12)
            ]),
            
            dbc.Row([
                # Column Selection
                dbc.Col([
                    html.Label("Select Columns to Plot:", className="mb-2"),
                    dcc.Dropdown(
                        id='column-selector',
                        options=column_options,
                        value=self.columns,
                        multi=True,
                        className="mb-4"
                    )
                ], width=12)
            ]),
            
            # Plot
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='pupil-plot', style={'height': '800px'})
                ], width=12)
            ])
        ], fluid=True)
        
    def _setup_callbacks(self):
        """Set up callbacks for plot updates."""
        @self.app.callback(
            Output('pupil-plot', 'figure'),
            [Input('trial-selector', 'value'),
             Input('column-selector', 'value')]
        )
        def update_plot(trial_str, selected_columns):
            # Convert string trial back to dict
            import ast
            trial = ast.literal_eval(trial_str)
            
            # Use selected columns or default to all columns
            plot_columns = selected_columns if selected_columns else self.columns
            
            # Plot parameters
            plot_params = {
                'layout': (len(plot_columns), 1),  # One row per preprocessing step
                'subplot_titles': plot_columns,
                'x_title': 'Time (ms)',
                'y_title': 'Pupil Size',
                'showlegend': True,
                'grid': False,
                'width': 1200,
                'height': 200 * len(plot_columns),
                'title_text': f"Pupil Preprocessing Steps - {' | '.join([f'{k}: {v}' for k, v in trial.items()])}"
            }
            
            # Create plot
            fig = self.pupil_processor._plot_trial_interactive(
                trial=trial,
                time_col=self.pupil_processor.time_col,
                pupil_col=plot_columns,
                hue=self.hue,
                plot_params=plot_params
            )
            
            return fig
            
    def run(self, port=8051, **kwargs):
        """
        Run the Dash server for the pupil data viewer.
        
        Parameters
        ----------
        port : int, default=8051
            Port number to run the server on.
            Make sure the port is available and not blocked by firewall.
        **kwargs : dict
            Additional keyword arguments passed to dash.run_server().
            See Dash documentation for available options.

        Notes
        -----
        - The application will run until interrupted (Ctrl+C)
        - Access the interface at http://localhost:<port>
        - Each preprocessing step is shown in a separate subplot
        - Interactive controls allow exploration of different trials and columns
        """
        self.app.run(port=port, **kwargs) 