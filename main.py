import dash
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import base64
import io
import json
import os

# Initialize the Dash app with multi-page support
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder='assets',
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

server = app.server

# Global variables to store data
class DataStore:
    def __init__(self):
        self.df = None
        self.cleaned_df = None
        self.original_shape = (0, 0)
        self.missing_values_before = 0
        self.outliers_detected = 0

# Initialize data store
data_store = DataStore()

# Define the sidebar component
sidebar = html.Div(
    [
        html.H2("Statistical Dashboard", className="dashboard-title"),
        html.Hr(),
        html.Div(
            [
                html.Div(html.Button("Upload Data", id="btn-upload-data", className="button-36", n_clicks=0, style={'width': '220px'})),
                html.Div(html.Button("Data Cleaning", id="btn-data-cleaning", className="button-36", n_clicks=0, style={'width': '220px'})),
                html.Div(html.Button("Data Visualization", id="btn-data-visualization", className="button-36", n_clicks=0, style={'width': '220px'})),
                html.Div(html.Button("Statistical Tests", id="btn-statistical-tests", className="button-36", n_clicks=0, style={'width': '220px'})),
                html.Div(html.Button("Basic ML", id="btn-basic-ml", className="button-36", n_clicks=0, style={'width': '220px'})),
                html.Div(html.Button("User Guide", id="btn-user-guide", className="button-36", n_clicks=0, style={'width': '220px'})),
            ],
            className="nav-links"
        ),
    ],
    className="sidebar"
)

# Define the main layout
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="dataset-store", storage_type="memory"),
        dcc.Store(id="cleaned-dataset-store", storage_type="memory"),
        dcc.Store(id="cleaning-stats-store", storage_type="memory"),
        sidebar,
        html.Div(id="page-content", className="content"),
        html.Div(id="notification-container", className="notification-container")
    ],
    className="container"
)

# Import page layouts
from pages.upload_data import layout as upload_data_layout
from pages.data_cleaning import layout as data_cleaning_layout
from pages.data_visualization import layout as data_visualization_layout
from pages.statistical_tests import layout as statistical_tests_layout
from pages.basic_ml import layout as basic_ml_layout
from pages.user_guide import layout as user_guide_layout

# Callbacks for button clicks to change URL
@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    [
        Input("btn-upload-data", "n_clicks"),
        Input("btn-data-cleaning", "n_clicks"),
        Input("btn-data-visualization", "n_clicks"),
        Input("btn-statistical-tests", "n_clicks"),
        Input("btn-basic-ml", "n_clicks"),
        Input("btn-user-guide", "n_clicks")
    ],
    prevent_initial_call=True
)
def change_page(n1, n2, n3, n4, n5, n6):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "/"
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "btn-upload-data":
        return "/upload-data"
    elif button_id == "btn-data-cleaning":
        return "/data-cleaning"
    elif button_id == "btn-data-visualization":
        return "/data-visualization"
    elif button_id == "btn-statistical-tests":
        return "/statistical-tests"
    elif button_id == "btn-basic-ml":
        return "/basic-ml"
    elif button_id == "btn-user-guide":
        return "/user-guide"
    else:
        return "/"

# Callback to update page content based on URL
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/upload-data" or pathname == "/":
        return upload_data_layout()
    elif pathname == "/data-cleaning":
        return data_cleaning_layout()
    elif pathname == "/data-visualization":
        return data_visualization_layout()
    elif pathname == "/statistical-tests":
        return statistical_tests_layout()
    elif pathname == "/basic-ml":
        return basic_ml_layout()
    elif pathname == "/user-guide":
        return user_guide_layout()
    else:
        return "404 Page Not Found!"

# Import callbacks
from callbacks.upload_callbacks import register_upload_callbacks
from callbacks.data_cleaning_callbacks import register_data_cleaning_callbacks
from callbacks.notification_callbacks import register_notification_callbacks
from callbacks.data_visualization_callbacks import register_data_visualization_callbacks
from callbacks.statistical_tests_callbacks import register_statistical_tests_callbacks
from callbacks.basic_ml_callbacks import register_basic_ml_callbacks

# Register all callbacks
register_upload_callbacks(app)
register_data_cleaning_callbacks(app)
register_notification_callbacks(app)
register_data_visualization_callbacks(app)
register_statistical_tests_callbacks(app)
register_basic_ml_callbacks(app)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
