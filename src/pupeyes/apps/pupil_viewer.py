# -*- coding:utf-8 -*-

"""
Interactive Pupil Data Viewer

Author: Han Zhang
Email: hanzh@umich.edu

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
    def __init__(self, pupil_processor):
        """
        Initialize PupilViewer with a PupilProcessor instance.
        
        Parameters
        ----------
        pupil_processor : PupilProcessor
            Instance of PupilProcessor containing the pupil data
        """
        self.p = pupil_processor
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Create app layout
        self.app.layout = self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
    def _create_layout(self):
        """Create the Dash app layout."""
        # Get trial options
        trial_options = []
        for _, trial in self.p.trials.iterrows():
            label = ' | '.join([f"{k}: {v}" for k, v in trial.items()])
            value = {k: v for k, v in trial.items()}
            trial_options.append({'label': label, 'value': str(value)})
            
        return dbc.Container([
            html.H1("Pupil Preprocessing Explorer", className="text-center my-4"),
            
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
            [Input('trial-selector', 'value')]
        )
        def update_plot(trial_str):
            # Convert string trial back to dict
            import ast
            trial = ast.literal_eval(trial_str)
            
            # Plot parameters
            plot_params = {
                'layout': (len(self.p.all_pupil_cols), 1),  # One row per preprocessing step
                'subplot_titles': self.p.all_pupil_cols,
                'x_title': 'Time (ms)',
                'y_title': 'Pupil Size',
                'showlegend': True,
                'grid': False,
                'width': 1200,
                'height': 200 * len(self.p.all_pupil_cols),
                'title_text': f"Pupil Preprocessing Steps - {' | '.join([f'{k}: {v}' for k, v in trial.items()])}"
            }
            
            # Create plot
            fig = self.p._plot_trial_interactive(
                trial=trial,
                x=self.p.time_col,
                y=self.p.all_pupil_cols,
                hue='routine',
                plot_params=plot_params
            )
            
            return fig
            
    def run_server(self, **kwargs):
        """Run the Dash server."""
        self.app.run_server(**kwargs) 