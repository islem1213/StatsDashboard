from dash.dependencies import Input, Output, State
from dash import html, dash_table, dcc
import pandas as pd
import base64
import io
import json
import plotly.express as px
import numpy as np
from dash.exceptions import PreventUpdate

def parse_contents(contents, filename):
    """Parse the contents of an uploaded file."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, f"Unsupported file type: {filename}"
        
        return df, None
    except Exception as e:
        return None, f"Error processing {filename}: {e}"

def get_data_summary(df):
    """Generate summary statistics for a DataFrame."""
    if df is None:
        return None
    
    # Calculate basic statistics
    summary = {}
    summary['shape'] = df.shape
    summary['columns'] = df.columns.tolist()
    summary['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Missing values statistics
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_stats = pd.DataFrame({
        'column': missing_data.index,
        'missing_count': missing_data.values,
        'missing_percent': missing_percent.values
    })
    missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values('missing_count', ascending=False)
    summary['missing_values'] = missing_stats.to_dict('records')
    summary['total_missing'] = df.isnull().sum().sum()
    
    return summary

def register_upload_callbacks(app):
    @app.callback(
        [
            Output('data-preview-container', 'children'),
            Output('data-summary-container', 'children'),
            Output('dataset-info', 'children'),
            Output('dataset-store', 'data'),
            Output('notification-container', 'children')
        ],
        [Input('upload-data', 'contents')],
        [State('upload-data', 'filename')]
    )
    def update_output(contents, filename):
        if contents is None:
            raise PreventUpdate
        
        # Process the uploaded file
        df, error = parse_contents(contents, filename)
        
        if error is not None:
            notification = html.Div(
                error,
                className="notification notification-error"
            )
            return None, None, None, None, notification
        
        # Store the data
        dataset_json = df.to_json(date_format='iso', orient='split')
        
        # Generate preview of the data
        preview = html.Div([
            html.H3(f"Preview of {filename}"),
            dash_table.DataTable(
                data=df.head(10).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'height': 'auto',
                    'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
                    'whiteSpace': 'normal',
                    'textAlign': 'left'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
        ])
        
        # Get summary of the data
        summary_data = get_data_summary(df)
        
        # Create summary display
        summary = html.Div([
            html.H3("Dataset Summary"),
            html.Div([
                html.P(f"Rows: {summary_data['shape'][0]}"),
                html.P(f"Columns: {summary_data['shape'][1]}"),
                html.P(f"Missing Values: {summary_data['total_missing']}")
            ]),
            
            html.H4("Missing Values"),
            dash_table.DataTable(
                data=summary_data['missing_values'],
                columns=[
                    {'name': 'Column', 'id': 'column'},
                    {'name': 'Missing Count', 'id': 'missing_count'},
                    {'name': 'Missing %', 'id': 'missing_percent', 'format': {'specifier': '.2f'}}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            ) if summary_data['missing_values'] else html.P("No missing values found.")
        ])
        
        # Create dataset info
        dataset_info = html.Div([
            html.H3("Dataset Information"),
            html.Div([
                html.P(f"Filename: {filename}"),
                html.P(f"Data Shape: {summary_data['shape'][0]} rows × {summary_data['shape'][1]} columns"),
                html.Button(
                    "Proceed to Data Cleaning", 
                    id="btn-proceed-cleaning", 
                    className="button-36", 
                    style={'width': '220px'}
                )
            ])
        ])
        
        # Success notification
        notification = html.Div(
            f"Dataset {filename} loaded successfully!",
            className="notification notification-success"
        )
        
        return preview, summary, dataset_info, dataset_json, notification
    
    # Callback to redirect to Data Cleaning page when Proceed button is clicked
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("btn-proceed-cleaning", "n_clicks"),
        prevent_initial_call=True
    )
    def proceed_to_cleaning(n_clicks):
        if n_clicks:
            return "/data-cleaning"
        raise PreventUpdate 