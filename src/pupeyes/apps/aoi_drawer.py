"""
Interactive AOI Drawing Tool using Dash

This module provides an interactive web-based tool for drawing Areas of Interest (AOIs)
that can be used with the EyeMovementVisualizer.
"""

import json
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from PIL import Image
import dash

class AOIDrawer:
    """
    An interactive web-based tool for drawing Areas of Interest (AOIs).

    This class provides a Dash-based web interface for drawing and managing Areas of Interest
    (AOIs) on stimulus images. It supports multiple drawing tools (freeform, rectangle, circle),
    editing capabilities, and export functionality.

    Parameters
    ----------
    screen_dims : tuple, default=(1920, 1080)
        Screen dimensions in pixels (width, height).
        Used to set the drawing canvas size and scale background images.
    stimuli : str or numpy.ndarray, optional
        Path to the stimulus image or a numpy array containing the image.
        Supports various image formats and both RGB and grayscale images.
    stimuli_name : str, optional
        Name of the stimulus image, used for display and as default save filename.
        If not provided, defaults to "AOIs".

    Attributes
    ----------
    aois : dict
        Dictionary storing AOI data, where keys are AOI names and values are lists
        of (x, y) coordinate tuples defining the AOI vertices.
    app : dash.Dash
        The Dash application instance.
    screen_dims : tuple
        The dimensions of the drawing canvas.
    """
    
    def __init__(self, screen_dims=(1920, 1080), stimuli=None, stimuli_name=None):
        """
        Initialize the AOI drawer.

        Parameters
        ----------
        screen_dims : tuple, default=(1920, 1080)
            Screen dimensions in pixels (width, height)
        stimuli : str or numpy.ndarray, optional
            Path to the stimulus image or a numpy array containing the image
        stimuli_name : str, optional
            Name of the stimulus image, used for display and as default save filename
        """
        self.screen_dims = screen_dims
        self.stimuli = stimuli
        self.stimuli_name = stimuli_name or "AOIs"
        self._stimuli_cache = None
        self._temp_shape = None  # Store temporary shape while waiting for name
        
        # Initialize Dash app
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Initialize AOI storage
        self.aois = {}
        
        # Create app layout
        self.app.layout = self._create_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
    def _get_stimuli_image(self):
        """Get the stimuli image."""
        if self._stimuli_cache is None and self.stimuli is not None:
            try:
                if isinstance(self.stimuli, str):
                    # Handle image file path
                    with Image.open(self.stimuli) as img:
                        try:
                            img.verify()
                        except Exception as e:
                            print(f"Invalid image file: {str(e)}")
                            return None
                    
                    img = Image.open(self.stimuli)
                    if img.size != self.screen_dims:
                        print('Original size:', img.size, 'Resized size:', self.screen_dims)
                        img = img.resize(self.screen_dims)
                    self._stimuli_cache = img
                    
                elif isinstance(self.stimuli, np.ndarray):
                    # Handle numpy array
                    array_shape = self.stimuli.shape[:2]  # Get height, width
                    # Ensure array is uint8 for proper image conversion
                    img_array = self.stimuli
                    if img_array.dtype != np.uint8:
                        img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1 else img_array.astype(np.uint8)
                    
                    # Convert to PIL Image based on array shape
                    if len(img_array.shape) == 2:  # Grayscale
                        img = Image.fromarray(img_array, mode='L')
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB
                        img = Image.fromarray(img_array, mode='RGB')
                    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                        img = Image.fromarray(img_array, mode='RGBA')
                    else:
                        print(f"Unsupported array shape: {img_array.shape}")
                        return None
                    
                    # Resize if necessary
                    if array_shape != (self.screen_dims[1], self.screen_dims[0]):
                        print('Original size:', array_shape[::-1], 'Resized size:', self.screen_dims)
                        img = img.resize(self.screen_dims)
                    
                    self._stimuli_cache = img
                else:
                    print("Invalid background image type. Must be a file path or numpy array.")
                    return None
                    
            except Exception as e:
                print(f"Failed to load stimuli: {str(e)}")
                return None
                
        return self._stimuli_cache
        
    def _create_layout(self):
        """Create the Dash app layout."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("AOI Drawer", className="text-center mb-2"),
                    html.H6(f"Stimulus: {self.stimuli_name}", className="text-center text-muted mb-2")
                ])
            ]),
            
            # Instructions Card
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Instructions", className="py-1"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li("Draw: Use the modebar to draw shapes (freeform, rectangle, or circle)"),
                                html.Li("Edit: Click on a shape's border to move it or adjust its vertices"),
                                html.Li("Erase: Click on a shape's border and select 'Erase Active Shape'"),
                            ], className="mb-0 small"),
                        ], className="p-2")
                    ], className="mb-2")
                ], width={"size": 8, "offset": 2})
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Controls", className="py-1"),
                        dbc.CardBody([
                            dbc.Row([
                                # Save and Load Buttons
                                dbc.Col([
                                    html.Div([
                                        dbc.Button(
                                            "Save AOIs",
                                            id="save-button",
                                            color="primary",
                                            size="sm",
                                            className="me-2"
                                        ),
                                        dcc.Download(id="download-aois"),
                                    ], className="d-flex justify-content-start")
                                ], width=6),
                                
                                # Background Opacity Control
                                dbc.Col([
                                    html.Div([
                                        html.Label("Opacity:", className="me-2 small", style={'whiteSpace': 'nowrap'}),
                                        html.Div([
                                            dcc.Slider(
                                                id='bg-opacity-slider',
                                                min=0,
                                                max=1,
                                                step=0.1,
                                                value=0.5,
                                                marks=None,
                                                tooltip={"placement": "bottom", "always_visible": True},
                                                className="mt-1"
                                            )
                                        ], style={'width': '150px'})
                                    ], className="d-flex align-items-center")
                                ], width=6)
                            ], className="g-1"),

                            # Line Color Control
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Label("Line Color:", className="me-2 small"),
                                        dcc.Dropdown(
                                            id='line-color-picker',
                                            options=[
                                                {'label': 'Black', 'value': 'black'},
                                                {'label': 'Red', 'value': 'red'},
                                                {'label': 'Blue', 'value': 'blue'},
                                                {'label': 'Green', 'value': 'green'},
                                                {'label': 'Yellow', 'value': 'yellow'},
                                                {'label': 'White', 'value': 'white'}
                                            ],
                                            value='black',
                                            clearable=False,
                                            style={'width': '150px'}
                                        )
                                    ], className="d-flex align-items-center mb-2")
                                ])
                            ]),
                            
                            # Current AOIs Table
                            html.Div([
                                html.H6("Current AOIs:", className="mb-1 mt-1 small"),
                                html.Div(id='aoi-list', className="small")
                            ])
                        ], className="p-2")
                    ], className="mb-2")
                ], width={"size": 8, "offset": 2})
            ]),
            
            # Drawing Area
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id='drawing-area',
                                config={
                                    'modeBarButtonsToAdd': [
                                        'drawclosedpath',
                                        'drawrect',
                                        'drawcircle',
                                        'eraseshape'
                                    ],
                                    'modeBarButtonsToRemove': [
                                        'autoScale2d',
                                        'pan2d',
                                        'zoom2d',
                                        'zoomIn2d',
                                        'zoomOut2d',
                                        'resetScale2d'
                                    ],
                                    'displaylogo': False
                                },
                                style={'height': '600px'}
                            )
                        ])
                    ])
                ])
            ]),
            
            # Modal for AOI naming
            dbc.Modal([
                dbc.ModalHeader("Name your AOI"),
                dbc.ModalBody([
                    dbc.Input(
                        id="aoi-name-input",
                        type="text",
                        placeholder="Enter AOI name"
                    ),
                    html.Div(
                        id="name-warning",
                        className="text-danger mt-2"
                    )
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Cancel",
                        id="modal-cancel",
                        className="me-2",
                        color="secondary"
                    ),
                    dbc.Button(
                        "Save",
                        id="modal-save",
                        color="primary"
                    )
                ])
            ], id="naming-modal", is_open=False)
        ], fluid=True)
        
    def _setup_callbacks(self):
        """Set up all callbacks."""
        
        @self.app.callback(
            [Output('drawing-area', 'figure'),
             Output('aoi-list', 'children'),
             Output('naming-modal', 'is_open'),
             Output('aoi-name-input', 'value'),
             Output('name-warning', 'children')],
            [Input('drawing-area', 'relayoutData'),
             Input('modal-save', 'n_clicks'),
             Input('modal-cancel', 'n_clicks'),
             Input('bg-opacity-slider', 'value'),
             Input('line-color-picker', 'value')],
            [State('drawing-area', 'figure'),
             State('aoi-name-input', 'value')]
        )
        def update_drawing_area(relayout_data, save_clicks, cancel_clicks, opacity, line_color, figure, aoi_name):
            ctx = callback_context
            if not ctx.triggered:
                return self._create_base_figure(), self._create_aoi_list(), False, '', ''
                
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Initialize figure if None
            if figure is None:
                figure = self._create_base_figure()
            
            # Handle line color change
            if trigger_id == 'line-color-picker':
                if 'layout' in figure:
                    # Update newshape defaults
                    if 'newshape' in figure['layout']:
                        figure['layout']['newshape']['line']['color'] = line_color
                    # Update existing shapes
                    if 'shapes' in figure['layout'] and figure['layout']['shapes']:
                        for shape in figure['layout']['shapes']:
                            if 'line' not in shape:
                                shape['line'] = {}
                            shape['line']['color'] = line_color
                return figure, self._create_aoi_list(), False, '', ''
            
            # Handle opacity change
            if trigger_id == 'bg-opacity-slider':
                if 'layout' in figure and 'images' in figure['layout']:
                    figure['layout']['images'][0]['opacity'] = opacity
                return figure, self._create_aoi_list(), False, '', ''
            
            # Handle shape changes (new shapes, deletions, or edits)
            if trigger_id == 'drawing-area' and relayout_data is not None:
                # Handle complete shape updates
                if 'shapes' in relayout_data:
                    current_shapes = relayout_data['shapes'] if relayout_data['shapes'] is not None else []
                    
                    # If shapes were deleted
                    if len(current_shapes) < len(self.aois):
                        # Create a mapping of shape coordinates to AOI names
                        shape_to_aoi = {}
                        for name, vertices in self.aois.items():
                            shape_hash = tuple(sorted((x, y) for x, y in vertices))
                            shape_to_aoi[shape_hash] = name
                        
                        # Check which shapes still exist
                        remaining_aois = {}
                        for shape in current_shapes:
                            vertices = self._shape_to_vertices(shape)
                            if vertices:
                                shape_hash = tuple(sorted((x, y) for x, y in vertices))
                                if shape_hash in shape_to_aoi:
                                    name = shape_to_aoi[shape_hash]
                                    remaining_aois[name] = vertices
                        
                        self.aois = remaining_aois
                        # Update annotations after deletion
                        figure = self._update_figure_annotations(figure)
                        return figure, self._create_aoi_list(), False, '', ''
                    
                    # Handle new shape
                    elif len(current_shapes) > len(self.aois):
                        self._temp_shape = current_shapes[-1]
                        return figure, self._create_aoi_list(), True, '', ''
                
                # Handle individual coordinate updates
                shape_updates = {}
                for key in relayout_data:
                    if key.startswith('shapes['):
                        # Extract shape index and property
                        parts = key.split('.')
                        idx = int(parts[0].split('[')[1].split(']')[0])
                        
                        # Get existing shape data
                        if idx not in shape_updates:
                            if 'layout' in figure and 'shapes' in figure['layout'] and idx < len(figure['layout']['shapes']):
                                shape_updates[idx] = figure['layout']['shapes'][idx].copy()
                            else:
                                shape_updates[idx] = {}
                        
                        # Update the specific coordinate
                        if len(parts) > 1:
                            prop = parts[1]
                            shape_updates[idx][prop] = relayout_data[key]
                            
                            # If this is a path edit, we need to handle it differently
                            if prop == 'path':
                                shape_updates[idx]['type'] = 'path'
                            elif any(prop.startswith(x) for x in ['x0', 'x1', 'y0', 'y1']):
                                # For rectangles and circles, make sure we preserve the type
                                if 'type' not in shape_updates[idx]:
                                    if 'layout' in figure and 'shapes' in figure['layout'] and idx < len(figure['layout']['shapes']):
                                        shape_updates[idx]['type'] = figure['layout']['shapes'][idx].get('type', 'rect')
                                    else:
                                        # Default to rect if we can't determine the type
                                        shape_updates[idx]['type'] = 'rect'
                
                # Apply updates to AOIs
                if shape_updates:
                    aoi_names = list(self.aois.keys())
                    for idx, shape_data in shape_updates.items():
                        if idx < len(aoi_names):
                            # For path shapes
                            if shape_data.get('type') == 'path' and 'path' in shape_data:
                                vertices = self._shape_to_vertices(shape_data)
                                if vertices:
                                    self.aois[aoi_names[idx]] = vertices
                            # For rect and circle shapes
                            elif all(k in shape_data for k in ['x0', 'x1', 'y0', 'y1']):
                                vertices = self._shape_to_vertices(shape_data)
                                if vertices:
                                    self.aois[aoi_names[idx]] = vertices
                    
                    # Update annotations after shape updates
                    figure = self._update_figure_annotations(figure)
                    return figure, self._create_aoi_list(), False, '', ''

            # Handle modal save
            elif trigger_id == 'modal-save' and self._temp_shape is not None and aoi_name:
                # Check for duplicate name
                if aoi_name in self.aois:
                    return figure, self._create_aoi_list(), True, aoi_name, f"An AOI named '{aoi_name}' already exists. Please choose a different name."
                
                vertices = self._shape_to_vertices(self._temp_shape)
                if vertices:
                    self.aois[aoi_name] = vertices
                    # Add annotation for the new AOI
                    if 'layout' in figure and 'shapes' in figure['layout']:
                        annotation = self._shape_to_annotation(figure['layout']['shapes'][-1], aoi_name)
                        if 'annotations' not in figure['layout']:
                            figure['layout']['annotations'] = []
                        figure['layout']['annotations'].append(annotation)
                self._temp_shape = None
                return figure, self._create_aoi_list(), False, '', ''
            
            # Handle modal cancel
            elif trigger_id == 'modal-cancel':
                # Remove the last shape when canceling
                if 'layout' in figure and 'shapes' in figure['layout']:
                    figure['layout']['shapes'] = figure['layout']['shapes'][:-1]
                self._temp_shape = None
                return figure, self._create_aoi_list(), False, '', ''
            
            return figure, self._create_aoi_list(), False, '', ''
            
        @self.app.callback(
            Output('download-aois', 'data'),
            Input('save-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def save_aois(n_clicks):
            if n_clicks is None:
                return None
                
            # Convert AOIs to a JSON-serializable format
            aois_json = {
                name: [list(vertex) for vertex in vertices]
                for name, vertices in self.aois.items()
            }
            
            return dict(
                content=json.dumps(aois_json, indent=2),
                filename=f'{self.stimuli_name}_aois.json'
            )
            
    def _create_base_figure(self):
        """Create the base figure for drawing."""
        fig = go.Figure()
        
        # Add background image if available
        if self.stimuli is not None:
            background_img = self._get_stimuli_image()
            if background_img is not None:
                # check if greyscale
                if background_img.mode != 'L':
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
                            opacity=0.5,  # Default opacity
                            layer="below"
                    )
                ) 
                else:
                    # For grayscale images, use heatmap to display intensity values
                    img_array = np.array(background_img)
                    fig.add_trace(
                        go.Heatmap(
                            z=img_array,
                            x=np.linspace(0, self.screen_dims[0], img_array.shape[1]),
                            y=np.linspace(0, self.screen_dims[1], img_array.shape[0]),
                            colorscale='gray',
                            showscale=False,
                            hoverongaps=False,
                            xaxis="x",
                            yaxis="y"
                        )
                    )

                # Update layout
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
                        scaleratio=1,
                        constrain="domain"
                    ),
                    dragmode='drawclosedpath',  # Default to closed path drawing
                    # Configure default shape properties
                    newshape=dict(
                        line=dict(
                            width=1,  # Thinner line width
                            color='black'  # Default color
                        ),
                        fillcolor='rgba(0,0,0,0)',  # Transparent fill
                        opacity=1
                    ),
                    # Apply same style to existing shapes
                    shapedefaults=dict(
                        line=dict(
                            width=1,  # Thinner line width
                            color='black'  # Default color
                        ),
                        fillcolor='rgba(0,0,0,0)',  # Transparent fill
                        opacity=1
                    )
                )
        
        return fig
        
    def _create_aoi_list(self):
        """Create the list of current AOIs."""
        if not self.aois:
            return html.P("No AOIs defined yet.", className="text-muted")
            
        return html.Ul([
            html.Li(
                f"{name} ({self._get_shape_type(vertices)} - {len(vertices)} vertices)"
            ) for name, vertices in self.aois.items()
        ], className="list-unstyled")
        
    def _get_shape_type(self, vertices):
        """Determine the shape type based on number of vertices."""
        num_vertices = len(vertices)
        if num_vertices == 4:
            return "Rectangle"
        elif num_vertices == 32:
            # Check if it's a circle or oval
            x_coords = [x for x, _ in vertices]
            y_coords = [y for _, y in vertices]
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            # If width and height are within 1% of each other, it's a circle
            if abs(width - height) / max(width, height) < 0.01:
                return "Circle"
            return "Oval"
        else:
            return "Free Form"
        
    def _shape_to_vertices(self, shape):
        """
        Convert a Plotly shape object to a list of vertices.

        This method extracts vertex coordinates from different types of Plotly shapes
        (path, rectangle, circle) and converts them to a consistent format.

        Parameters
        ----------
        shape : dict
            Plotly shape object containing shape type and coordinate information.

        Returns
        -------
        list of tuple or None
            List of (x, y) coordinate tuples defining the shape vertices.
            Returns None if shape type is not recognized or conversion fails.

        Notes
        -----
        - Handles three shape types:
            - path: Extracts vertices from SVG path string
            - rect: Converts rectangle coordinates to 4 vertices
            - circle: Approximates circle/oval with 32 vertices
        - For circles/ovals, uses evenly spaced points around the perimeter
        - All shapes are closed by adding the first vertex at the end
        """
        shape_type = shape.get('type', '')
        
        if shape_type == 'path':
            # Extract vertices from SVG path
            path = shape['path'].split('M')[1].split('Z')[0]
            vertices = [
                tuple(map(float, point.strip().split(',')))
                for point in path.split('L')
            ]
            # Close the path by adding the first vertex at the end
            if vertices:
                vertices.append(vertices[0])
            return vertices
            
        elif shape_type == 'rect':
            # Convert rectangle to vertices
            x0, y0 = shape['x0'], shape['y0']
            x1, y1 = shape['x1'], shape['y1']
            vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            # Close the rectangle by adding the first vertex at the end
            vertices.append(vertices[0])
            return vertices
            
        elif shape_type == 'circle':
            # Convert circle/oval to polygon approximation
            x0, y0 = shape['x0'], shape['y0']
            x1, y1 = shape['x1'], shape['y1']
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            radius_x = abs(x1 - x0) / 2
            radius_y = abs(y1 - y0) / 2
            
            # Create polygon approximation of oval
            num_points = 32  # Reduced from 99 to match the shape type check
            angles = np.linspace(0, 2*np.pi, num_points)
            vertices = [
                (center_x + radius_x * np.cos(angle),
                 center_y + radius_y * np.sin(angle))
                for angle in angles
            ]
            # Close the circle by adding the first vertex at the end
            if vertices:
                vertices.append(vertices[0])
            return vertices
            
        return None
        
    def _shape_to_annotation(self, shape, name):
        """Convert a shape to a Plotly annotation for the AOI name."""
        if shape.get('type') == 'path' and 'path' in shape:
            # For path shapes, use the first point as annotation position
            path = shape['path'].split('M')[1].split('Z')[0]
            first_point = path.split('L')[0]
            x, y = map(float, first_point.strip().split(','))
        else:
            # For rect and circle, use the top-left corner
            x = shape['x0']
            y = shape['y0']
        
        return dict(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=name,
            showarrow=False,
            font=dict(
                size=12,
                color="white"
            ),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )

    def _update_figure_annotations(self, figure):
        """Update figure annotations to match current AOIs."""
        if 'layout' not in figure:
            return figure
            
        # Clear existing annotations
        figure['layout']['annotations'] = []
        
        # Add annotation for each AOI
        if 'shapes' in figure['layout']:
            for idx, (name, _) in enumerate(self.aois.items()):
                if idx < len(figure['layout']['shapes']):
                    shape = figure['layout']['shapes'][idx]
                    annotation = self._shape_to_annotation(shape, name)
                    if 'annotations' not in figure['layout']:
                        figure['layout']['annotations'] = []
                    figure['layout']['annotations'].append(annotation)
        
        return figure
        
    def run(self, debug=False, port=8051, **kwargs):
        """
        Start the Dash server and run the AOI drawing application.

        This method initializes and starts the web server for the AOI drawing interface.
        The application will be accessible through a web browser at the specified port.

        Parameters
        ----------
        debug : bool, default=False
            Whether to run the server in debug mode
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
        - Debug mode provides additional error information
        - Default port (8051) can be changed if already in use
        """
        self.app.run(debug=debug, port=port, **kwargs)

