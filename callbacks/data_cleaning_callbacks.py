from dash.dependencies import Input, Output, State
from dash import html, dash_table, dcc, ALL
import pandas as pd
import plotly.express as px
import numpy as np
import json
from dash.exceptions import PreventUpdate

def register_data_cleaning_callbacks(app):
    
    # Check if data is loaded and show message if not
    @app.callback(
        [Output("no-data-message", "children"),
         Output("data-cleaning-content", "style")],
        Input("dataset-store", "data")
    )
    def check_data_loaded(data):
        if data is None:
            return html.Div([
                html.H3("No Dataset Loaded"),
                html.P("Please upload a dataset in the Upload Data section first."),
                html.Button(
                    "Go to Upload Data", 
                    id="btn-go-to-upload", 
                    className="button-36", 
                    style={'width': '220px', 'margin-top': '10px', 'margin-left': 'auto', 'margin-right': 'auto', 'display': 'block'}
                )
            ]), {'display': 'none'}
        else:
            return None, {'display': 'block'}
    
    # Go to upload data page
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Input("btn-go-to-upload", "n_clicks"),
        prevent_initial_call=True
    )
    def go_to_upload_data(n_clicks):
        if n_clicks:
            return "/upload-data"
        raise PreventUpdate
    
    # -------------------------------------------------
    # Missing Values Analysis Tab
    # -------------------------------------------------
    
    @app.callback(
        [Output("missing-values-table", "children"),
         Output("missing-values-chart", "children")],
        Input("dataset-store", "data"),
        State("cleaning-tabs", "value")
    )
    def update_missing_values_analysis(data, active_tab):
        if data is None or active_tab != "missing-values":
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # Calculate missing values
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_stats = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Values': missing_data.values,
            'Missing %': missing_percent.values
        })
        missing_stats = missing_stats.sort_values('Missing Values', ascending=False)
        
        # Filter to only show columns with missing values
        missing_stats_filtered = missing_stats[missing_stats['Missing Values'] > 0]
        
        if missing_stats_filtered.empty:
            table = html.Div("No missing values found in the dataset.")
        else:
            # Table component
            table = dash_table.DataTable(
                data=missing_stats_filtered.to_dict('records'),
                columns=[
                    {'name': 'Column', 'id': 'Column'},
                    {'name': 'Missing Values', 'id': 'Missing Values', 'type': 'numeric', 'format': {'specifier': ','}},
                    {'name': 'Missing %', 'id': 'Missing %', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'column_id': 'Missing %', 'filter_query': '{Missing %} > 50'},
                        'backgroundColor': '#FFC7CE',
                        'color': '#9C0006'
                    }
                ]
            )
        
        # Only include columns with missing values in the chart
        missing_stats_filtered = missing_stats[missing_stats['Missing Values'] > 0]
        
        if missing_stats_filtered.empty:
            chart = html.Div("No missing values to display in the chart.")
        else:
            # Chart component
            fig = px.bar(
                missing_stats_filtered,
                x='Column',
                y='Missing %',
                title='Percentage of Missing Values by Column',
                labels={'Missing %': 'Missing Values (%)'},
                color='Missing %',
                color_continuous_scale='reds',
                template='plotly_white'
            )
            fig.update_layout(
                xaxis={'categoryorder': 'total descending'},
                height=400,
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            chart = dcc.Graph(figure=fig)
        
        return table, chart
    
    # -------------------------------------------------
    # Handle Missing Values Tab
    # -------------------------------------------------
    
    # Generate column-specific imputation options
    @app.callback(
        Output("column-imputation-options", "children"),
        Input("dataset-store", "data"),
        State("cleaning-tabs", "value")
    )
    def update_column_imputation_options(data, active_tab):
        if data is None or active_tab != "handle-missing":
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # Get columns with missing values
        missing_columns = df.columns[df.isnull().any()].tolist()
        
        if not missing_columns:
            return html.Div(
                "No columns with missing values found.",
                className="column-options-empty"
            )
        
        # Create options for each column
        column_options = []
        
        for col in missing_columns:
            # Calculate percentage of missing values for this column
            missing_percent = (df[col].isnull().sum() / len(df)) * 100
            
            # Determine column type
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = "Numeric"
                options = [
                    {'label': 'Mean', 'value': 'mean'},
                    {'label': 'Median', 'value': 'median'},
                    {'label': 'Mode', 'value': 'mode'},
                    {'label': 'Fill with 0', 'value': 'zero'}
                ]
                default_value = 'mean'
            else:
                col_type = "Categorical"
                options = [
                    {'label': 'Mode (Most Frequent)', 'value': 'mode'},
                    {'label': 'Fill with "Unknown"', 'value': 'unknown'}
                ]
                default_value = 'mode'
            
            # Create option group for this column
            column_group = html.Div([
                html.Label(f"{col} ({col_type}) - {missing_percent:.2f}% missing:", 
                          style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                dcc.RadioItems(
                    id={'type': 'imputation-method', 'column': col},
                    options=options,
                    value=default_value,
                    labelStyle={'display': 'inline-block', 'marginRight': '12px', 'marginBottom': '5px', 'fontSize': '0.9rem'},
                    className="radio-group"
                ),
                html.Div(style={'margin': '5px 0', 'borderBottom': '1px dashed #eee'})
            ], className="column-option-group", style={'margin': '10px 0'})
            
            column_options.append(column_group)
        
        return column_options
    
    # Drop rows with missing values
    @app.callback(
        [Output("dataset-store", "data", allow_duplicate=True),
         Output("notification-container", "children", allow_duplicate=True)],
        Input("drop-rows-checkbox", "value"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def drop_missing_rows(value, data):
        if data is None or not value:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        original_rows = len(df)
        
        # Drop rows with missing values
        cleaned_df = df.dropna()
        rows_dropped = original_rows - len(cleaned_df)
        
        if rows_dropped == 0:
            notification = html.Div(
                "No rows were dropped as there are no rows with missing values.",
                className="notification notification-info"
            )
        else:
            notification = html.Div(
                f"Successfully dropped {rows_dropped} rows with missing values.",
                className="notification notification-success"
            )
        
        # Update the dataset store
        return cleaned_df.to_json(date_format='iso', orient='split'), notification
    
    # Drop columns with missing values above threshold
    @app.callback(
        [Output("dataset-store", "data", allow_duplicate=True),
         Output("notification-container", "children", allow_duplicate=True)],
        Input("btn-drop-columns", "n_clicks"),
        [State("dataset-store", "data"),
         State("drop-columns-threshold", "value")],
        prevent_initial_call=True
    )
    def drop_columns_above_threshold(n_clicks, data, threshold):
        if n_clicks is None or data is None:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        original_cols = len(df.columns)
        
        # Calculate missing percentage for each column
        missing_percent = df.isnull().mean() * 100
        
        # Get columns to drop
        cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
        
        if not cols_to_drop:
            notification = html.Div(
                f"No columns have missing values above {threshold}% threshold.",
                className="notification notification-info"
            )
            return data, notification
        
        # Drop the columns
        cleaned_df = df.drop(columns=cols_to_drop)
        
        notification = html.Div(
            f"Successfully dropped {len(cols_to_drop)} columns with more than {threshold}% missing values: {', '.join(cols_to_drop)}",
            className="notification notification-success"
        )
        
        # Update the dataset store
        return cleaned_df.to_json(date_format='iso', orient='split'), notification
    
    # Apply imputation
    @app.callback(
        [Output("dataset-store", "data", allow_duplicate=True),
         Output("notification-container", "children", allow_duplicate=True)],
        Input("btn-apply-imputation", "n_clicks"),
        [State("dataset-store", "data"),
         State("numeric-imputation-strategy", "value"),
         State("categorical-imputation-strategy", "value"),
         State({"type": "imputation-method", "column": ALL}, "value"),
         State({"type": "imputation-method", "column": ALL}, "id")],
        prevent_initial_call=True
    )
    def apply_imputation(n_clicks, data, numeric_strategy, categorical_strategy, column_strategies, column_ids):
        if n_clicks is None or data is None:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # Create copy to prevent modifying original
        cleaned_df = df.copy()
        
        # Create a dictionary of column-specific strategies
        column_specific = {}
        if column_ids and column_strategies:
            for col_id, strategy in zip(column_ids, column_strategies):
                column_specific[col_id['column']] = strategy
        
        # Apply imputation for each column
        imputed_columns = []
        
        for col in df.columns:
            if col in column_specific and df[col].isnull().any():
                strategy = column_specific[col]
                imputed_columns.append(col)
                
                # Apply the strategy
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
                    cleaned_df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
                    cleaned_df[col] = df[col].fillna(df[col].median())
                elif strategy == 'mode':
                    # Mode works for both numeric and categorical
                    most_frequent = df[col].mode()[0]
                    cleaned_df[col] = df[col].fillna(most_frequent)
                elif strategy == 'zero' and pd.api.types.is_numeric_dtype(df[col]):
                    cleaned_df[col] = df[col].fillna(0)
                elif strategy == 'unknown':
                    cleaned_df[col] = df[col].fillna("Unknown")
            
            # If not in column_specific, apply the default strategy
            elif df[col].isnull().any():
                imputed_columns.append(col)
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    if numeric_strategy == 'mean':
                        cleaned_df[col] = df[col].fillna(df[col].mean())
                    elif numeric_strategy == 'median':
                        cleaned_df[col] = df[col].fillna(df[col].median())
                    elif numeric_strategy == 'mode':
                        most_frequent = df[col].mode()[0]
                        cleaned_df[col] = df[col].fillna(most_frequent)
                else:
                    if categorical_strategy == 'mode':
                        most_frequent = df[col].mode()[0]
                        cleaned_df[col] = df[col].fillna(most_frequent)
                    elif categorical_strategy == 'unknown':
                        cleaned_df[col] = df[col].fillna("Unknown")
        
        if not imputed_columns:
            notification = html.Div(
                "No columns with missing values found to impute.",
                className="notification notification-info"
            )
        else:
            notification = html.Div(
                f"Successfully imputed missing values in {len(imputed_columns)} columns: {', '.join(imputed_columns)}",
                className="notification notification-success"
            )
        
        # Update the dataset store
        return cleaned_df.to_json(date_format='iso', orient='split'), notification
    
    # -------------------------------------------------
    # Outlier Detection and Handling Tab
    # -------------------------------------------------
    
    # Populate outlier columns dropdown
    @app.callback(
        Output("outlier-columns-dropdown", "options"),
        Input("dataset-store", "data"),
        State("cleaning-tabs", "value")
    )
    def update_outlier_columns_dropdown(data, active_tab):
        if data is None:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # Get only numeric columns
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        
        return [{'label': col, 'value': col} for col in numeric_columns]
    
    # Detect outliers
    @app.callback(
        [Output("outlier-results", "children"),
         Output("outlier-handling-container", "style")],
        Input("btn-detect-outliers", "n_clicks"),
        [State("dataset-store", "data"),
         State("outlier-columns-dropdown", "value"),
         State("iqr-factor", "value")],
        prevent_initial_call=True
    )
    def detect_outliers(n_clicks, data, selected_columns, iqr_factor):
        if n_clicks is None or data is None or not selected_columns:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # Create a mask for rows with outliers
        outlier_mask = pd.Series(False, index=df.index)
        outlier_summary = {}
        
        for col in selected_columns:
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            
            # Identify outliers in this column
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # Update the overall mask
            outlier_mask = outlier_mask | col_outliers
            
            # Store outlier count for this column
            outlier_count = col_outliers.sum()
            if outlier_count > 0:
                outlier_summary[col] = {
                    'count': int(outlier_count),
                    'percent': round(outlier_count / len(df) * 100, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
        
        # Get rows with outliers
        outlier_rows = df[outlier_mask]
        
        if outlier_rows.empty:
            return html.Div([
                html.H4("No outliers detected"),
                html.P(f"No outliers were found in the selected columns using IQR factor of {iqr_factor}.")
            ]), {'display': 'none'}
        
        # Create summary table data
        summary_data = [
            {
                'Column': col,
                'Outliers Count': info['count'],
                'Outliers %': info['percent'],
                'Lower Bound': info['lower_bound'],
                'Upper Bound': info['upper_bound']
            }
            for col, info in outlier_summary.items()
        ]
        
        # Create a preview of the outlier rows
        preview_rows = outlier_rows.head(10) if len(outlier_rows) > 10 else outlier_rows
        
        results = html.Div([
            html.H4(f"Outlier Detection Results (IQR Factor: {iqr_factor})"),
            html.P(f"Found {len(outlier_rows)} rows ({round(len(outlier_rows) / len(df) * 100, 2)}% of data) with outliers."),
            
            html.H5("Outlier Summary by Column"),
            dash_table.DataTable(
                data=summary_data,
                columns=[
                    {'name': 'Column', 'id': 'Column'},
                    {'name': 'Outlier Count', 'id': 'Outliers Count'},
                    {'name': 'Outlier %', 'id': 'Outliers %', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Lower Bound', 'id': 'Lower Bound', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Upper Bound', 'id': 'Upper Bound', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            ),
            
            html.H5("Preview of Rows with Outliers"),
            dash_table.DataTable(
                data=preview_rows.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in preview_rows.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': f'{{{col}}} < {info["lower_bound"]} || {{{col}}} > {info["upper_bound"]}',
                            'column_id': col
                        },
                        'backgroundColor': '#FFC7CE',
                        'color': '#9C0006'
                    }
                    for col, info in outlier_summary.items()
                ]
            ),
            
            # Store outlier mask in a hidden div
            html.Div(
                json.dumps(outlier_mask.tolist()),
                id='outlier-mask',
                style={'display': 'none'}
            )
        ])
        
        return results, {'display': 'block'}
    
    # Remove outliers
    @app.callback(
        [Output("dataset-store", "data", allow_duplicate=True),
         Output("notification-container", "children", allow_duplicate=True)],
        Input("btn-remove-outliers", "n_clicks"),
        [State("dataset-store", "data"),
         State("outlier-mask", "children")],
        prevent_initial_call=True
    )
    def remove_outliers(n_clicks, data, outlier_mask_json):
        if n_clicks is None or data is None or outlier_mask_json is None:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # Parse the outlier mask
        outlier_mask = json.loads(outlier_mask_json)
        
        # Remove rows with outliers
        cleaned_df = df[~pd.Series(outlier_mask, index=df.index)].copy()
        
        rows_removed = len(df) - len(cleaned_df)
        
        notification = html.Div(
            f"Successfully removed {rows_removed} rows with outliers.",
            className="notification notification-success"
        )
        
        # Store the number of outliers removed in a global variable or hidden div
        # for later use in the final summary
        
        # Update the dataset store
        return cleaned_df.to_json(date_format='iso', orient='split'), notification
    
    # -------------------------------------------------
    # Data Type Conversion Tab
    # -------------------------------------------------
    
    # Show current data types
    @app.callback(
        [Output("current-dtypes-table", "children"),
         Output("column-to-convert", "options")],
        Input("dataset-store", "data"),
        State("cleaning-tabs", "value")
    )
    def show_current_data_types(data, active_tab):
        if data is None:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # Prepare data types for display
        dtype_data = []
        for col in df.columns:
            dtype = df[col].dtype
            # Convert Python dtypes to user-friendly names
            if pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
                display_type = 'Numeric'
            elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                display_type = 'Categorical'
            elif pd.api.types.is_datetime64_dtype(df[col]):
                display_type = 'Date/Time'
            else:
                display_type = str(dtype)
                
            dtype_data.append({
                'Column': col,
                'Current Type': display_type
            })
        
        # Create table
        table = dash_table.DataTable(
            data=dtype_data,
            columns=[
                {'name': 'Column', 'id': 'Column'},
                {'name': 'Current Type', 'id': 'Current Type'}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
        
        # Options for column to convert dropdown
        options = [{'label': col, 'value': col} for col in df.columns]
        
        return table, options
    
    # Convert data type
    @app.callback(
        [Output("dataset-store", "data", allow_duplicate=True),
         Output("notification-container", "children", allow_duplicate=True)],
        Input("btn-convert-dtype", "n_clicks"),
        [State("dataset-store", "data"),
         State("column-to-convert", "value"),
         State("target-dtype", "value")],
        prevent_initial_call=True
    )
    def convert_data_type(n_clicks, data, column, target_dtype):
        if n_clicks is None or data is None or column is None:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # Create a copy to prevent modifying original
        converted_df = df.copy()
        
        try:
            if target_dtype == 'category':
                converted_df[column] = converted_df[column].astype('category')
                message = f"Converted '{column}' to categorical type."
            
            elif target_dtype == 'numeric':
                converted_df[column] = pd.to_numeric(converted_df[column], errors='coerce')
                message = f"Converted '{column}' to numeric type. Any non-numeric values were replaced with NaN."
            
            elif target_dtype == 'datetime':
                converted_df[column] = pd.to_datetime(converted_df[column], errors='coerce')
                message = f"Converted '{column}' to datetime type. Any invalid dates were replaced with NaN."
            
            notification = html.Div(
                message,
                className="notification notification-success"
            )
            
            # Update the dataset store
            return converted_df.to_json(date_format='iso', orient='split'), notification
            
        except Exception as e:
            notification = html.Div(
                f"Error converting '{column}': {str(e)}",
                className="notification notification-error"
            )
            return data, notification
    
    # -------------------------------------------------
    # Save Cleaned Data Tab
    # -------------------------------------------------
    
    # Generate cleaning summary
    @app.callback(
        Output("cleaning-summary", "children"),
        [Input("dataset-store", "data"),
         Input("cleaning-tabs", "value")]
    )
    def generate_cleaning_summary(data, active_tab):
        if data is None or active_tab != "save-data":
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        # For demonstration, we'll assume the original data is stored in an app global variable
        # In a real app, you would store this in dcc.Store or browser local storage
        try:
            original_df = pd.read_json(data, orient='split')  # Placeholder
            original_shape = (100, 10)  # Placeholder values
            original_missing = 50  # Placeholder
        except:
            original_shape = (0, 0)
            original_missing = 0
        
        current_shape = df.shape
        current_missing = df.isnull().sum().sum()
        
        # Calculate metrics
        rows_before = original_shape[0]
        rows_after = current_shape[0]
        rows_removed = rows_before - rows_after
        
        cols_before = original_shape[1]
        cols_after = current_shape[1]
        cols_removed = cols_before - cols_after
        
        missing_before = original_missing
        missing_after = current_missing
        missing_handled = missing_before - missing_after
        
        # Prepare summary
        summary = html.Div([
            html.H4("Data Cleaning Summary"),
            
            html.Div([
                html.Div([
                    html.H5("Rows"),
                    html.P(f"Original: {rows_before}"),
                    html.P(f"Current: {rows_after}"),
                    html.P(f"Removed: {rows_removed} ({round(rows_removed/rows_before*100, 2)}% reduction)")
                ], className="summary-column"),
                
                html.Div([
                    html.H5("Columns"),
                    html.P(f"Original: {cols_before}"),
                    html.P(f"Current: {cols_after}"),
                    html.P(f"Removed: {cols_removed} ({round(cols_removed/cols_before*100, 2)}% reduction)")
                ], className="summary-column"),
                
                html.Div([
                    html.H5("Missing Values"),
                    html.P(f"Original: {missing_before}"),
                    html.P(f"Current: {missing_after}"),
                    html.P(f"Handled: {missing_handled} ({round(missing_handled/missing_before*100, 2)}% reduction)")
                ], className="summary-column"),
                
                html.Div([
                    html.H5("Other Metrics"),
                    html.P(f"Outliers Removed: N/A"),  # Placeholder
                    html.P(f"Data Types Converted: N/A")  # Placeholder
                ], className="summary-column")
            ], className="summary-grid")
        ])
        
        return summary
    
    # Save cleaned dataset and offer download
    @app.callback(
        [Output("cleaned-dataset-store", "data"),
         Output("download-link-container", "children"),
         Output("notification-container", "children", allow_duplicate=True)],
        Input("btn-save-dataset", "n_clicks"),
        State("dataset-store", "data"),
        prevent_initial_call=True
    )
    def save_cleaned_dataset(n_clicks, data):
        if n_clicks is None or data is None:
            raise PreventUpdate
        
        # Store the cleaned dataset
        cleaned_dataset_json = data
        
        # Create download link
        download_link = html.Div([
            html.Button(
                "Download CSV", 
                id="btn-download-csv", 
                className="button-36", 
                style={'width': '220px', 'margin-top': '10px'}
            ),
            dcc.Download(id="download-cleaned-data")
        ])
        
        notification = html.Div(
            "Cleaned dataset saved successfully. You can now download it as CSV.",
            className="notification notification-success"
        )
        
        return cleaned_dataset_json, download_link, notification
    
    # Handle the download when clicking download button
    @app.callback(
        Output("download-cleaned-data", "data"),
        Input("btn-download-csv", "n_clicks"),
        State("cleaned-dataset-store", "data"),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, data):
        if n_clicks is None or data is None:
            raise PreventUpdate
        
        df = pd.read_json(data, orient='split')
        
        return dcc.send_data_frame(df.to_csv, "cleaned_dataset.csv", index=False) 