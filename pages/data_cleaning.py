from dash import html, dcc, dash_table
import plotly.express as px

def layout():
    return html.Div([
        html.H1("Data Cleaning"),
        html.P("Clean and preprocess your data before analysis."),
        
        # Check if data is loaded message
        html.Div(id="no-data-message", className="no-data-message"),
        
        # Data cleaning options
        html.Div([
            dcc.Tabs(id='cleaning-tabs', value='missing-values', children=[
                # Tab 1: Missing Values
                dcc.Tab(label='View Missing Values', value='missing-values', children=[
                    html.Div([
                        html.H3("Missing Values Analysis"),
                        html.P("View and analyze missing values in your dataset."),
                        
                        # Missing values table
                        html.Div(id="missing-values-table", className="data-table"),
                        
                        # Missing values chart
                        html.Div(id="missing-values-chart", className="chart-container")
                    ], className="tab-content")
                ]),
                
                # Tab 2: Handle Missing Values
                dcc.Tab(label='Handle Missing Values', value='handle-missing', children=[
                    html.Div([
                        html.H3("Handle Missing Values"),
                        html.P("Choose how to handle missing values in your dataset."),
                        
                        # Options for handling missing values
                        html.Div([
                            # HANDLE MISSING VALUES SECTION
                            html.H4("Handle Missing Values", className="section-title",
                                   style={
                                       'fontSize': '1.5rem',
                                       'fontFamily': '"Segoe UI", Arial, sans-serif',
                                       'fontWeight': 'bold',
                                       'color': '#2c3e50',
                                       'marginTop': '20px',
                                       'marginBottom': '15px',
                                       'paddingBottom': '8px',
                                       'borderBottom': '2px solid #eee'
                                   }),
                            
                            # Drop rows with missing values
                            html.Div([
                                html.Label("Drop rows with missing values:", 
                                           style={'fontWeight': 'bold'}),
                                dcc.Checklist(
                                    id='drop-rows-checkbox',
                                    options=[{'label': 'Drop rows with any missing values', 'value': 'drop_rows'}],
                                    value=[],
                                    className="custom-checklist"
                                )
                            ], className="option-section"),
                            
                            # Drop columns with too many missing values
                            html.Div([
                                html.Label("Drop columns with missing values above threshold:", 
                                           style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                                html.Div([
                                    dcc.Slider(
                                        id='drop-columns-threshold',
                                        min=0,
                                        max=100,
                                        step=5,
                                        value=50,
                                        marks={i: f'{i}%' for i in range(0, 101, 25)},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="compact-slider"
                                    )
                                ], style={'width': '80%', 'margin': '0 auto 15px auto'}),
                                html.Div([
                                    html.Button(
                                        "Drop Columns", 
                                        id="btn-drop-columns", 
                                        className="button-36", 
                                        style={'width': '220px', 'margin': '10px auto', 'display': 'block'}
                                    )
                                ], style={'textAlign': 'center'})
                            ], className="option-section"),
                            
                            html.Hr(style={'margin': '30px 0', 'borderTop': '1px solid #eee'}),
                            
                            # IMPUTE MISSING VALUES SECTION
                            html.H4("Impute Missing Values", className="section-title",
                                   style={
                                       'fontSize': '1.5rem',
                                       'fontFamily': '"Roboto", "Helvetica Neue", sans-serif',
                                       'fontWeight': 'bold',
                                       'color': '#2c3e50',
                                       'marginTop': '20px',
                                       'marginBottom': '15px',
                                       'paddingBottom': '8px',
                                       'borderBottom': '2px solid #eee'
                                   }),
                            
                            # Imputation strategy for numerical values
                            html.Div([
                                html.H5("Imputation Strategy for Numeric Columns:",
                                      style={'fontSize': '1.1rem', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                                dcc.RadioItems(
                                    id='numeric-imputation-strategy',
                                    options=[
                                        {'label': 'Mean', 'value': 'mean'},
                                        {'label': 'Median', 'value': 'median'},
                                        {'label': 'Mode', 'value': 'mode'}
                                    ],
                                    value='mean',
                                    labelStyle={'display': 'inline-block', 'marginRight': '15px', 'fontSize': '0.95rem'},
                                    className="radio-group"
                                )
                            ], className="option-section"),
                            
                            # Imputation for categorical values
                            html.Div([
                                html.H5("Imputation Strategy for Categorical Columns:",
                                      style={'fontSize': '1.1rem', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                                dcc.RadioItems(
                                    id='categorical-imputation-strategy',
                                    options=[
                                        {'label': 'Mode (Most Frequent)', 'value': 'mode'},
                                        {'label': 'Fill with "Unknown"', 'value': 'unknown'}
                                    ],
                                    value='mode',
                                    labelStyle={'display': 'inline-block', 'marginRight': '15px', 'fontSize': '0.95rem'},
                                    className="radio-group"
                                )
                            ], className="option-section"),
                            
                            # Column-specific imputation
                            html.Div([
                                html.H5("Column-specific Imputation:",
                                      style={'fontSize': '1.1rem', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                                html.P("Select columns and their imputation methods:", 
                                       style={'fontStyle': 'italic', 'fontSize': '0.9rem'}),
                                html.Div(id="column-imputation-options", className="column-options"),
                                html.Div([
                                    html.Button(
                                        "Apply Imputation", 
                                        id="btn-apply-imputation", 
                                        className="button-36", 
                                        style={'width': '220px', 'margin': '15px auto 5px auto', 'display': 'block'}
                                    )
                                ], style={'textAlign': 'center'})
                            ], className="option-section")
                        ], className="handle-missing-options")
                    ], className="tab-content")
                ]),
                
                # Tab 3: Outlier Detection and Handling
                dcc.Tab(label='Detect & Handle Outliers', value='outliers', children=[
                    html.Div([
                        html.H3("Outlier Detection and Handling"),
                        html.P("Detect and handle outliers in your dataset using the IQR method."),
                        
                        # Outlier detection options
                        html.Div([
                            html.Label("Select columns for outlier detection:"),
                            dcc.Dropdown(
                                id='outlier-columns-dropdown',
                                multi=True,
                                placeholder="Select columns to check for outliers"
                            ),
                            
                            html.Label("IQR Factor (higher is more stringent):"),
                            dcc.Slider(
                                id='iqr-factor',
                                min=1.0,
                                max=3.0,
                                step=0.1,
                                value=1.5,
                                marks={i: str(i) for i in [1.0, 1.5, 2.0, 2.5, 3.0]},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            
                            html.Button(
                                "Detect Outliers", 
                                id="btn-detect-outliers", 
                                className="button-36", 
                                style={'width': '220px', 'margin-top': '10px'}
                            )
                        ], className="outlier-options"),
                        
                        # Outlier results
                        html.Div(id="outlier-results", className="results-container"),
                        
                        # Outlier handling
                        html.Div([
                            html.H4("Handle Outliers"),
                            html.Button(
                                "Remove Outlier Rows", 
                                id="btn-remove-outliers", 
                                className="button-36", 
                                style={'width': '220px', 'margin-top': '10px'}
                            )
                        ], className="outlier-handling", id="outlier-handling-container")
                    ], className="tab-content")
                ]),
                
                # Tab 4: Data Type Conversion
                dcc.Tab(label='Data Type Conversion', value='data-types', children=[
                    html.Div([
                        html.H3("Data Type Conversion"),
                        html.P("Convert data types to ensure proper analysis."),
                        
                        # Current data types
                        html.Div([
                            html.H4("Current Data Types"),
                            html.Div(id="current-dtypes-table", className="data-table")
                        ], className="dtypes-section"),
                        
                        # Data type conversion
                        html.Div([
                            html.H4("Convert Data Types"),
                            html.P("Select a column and the target data type:"),
                            
                            dcc.Dropdown(
                                id='column-to-convert',
                                placeholder="Select column to convert"
                            ),
                            
                            dcc.RadioItems(
                                id='target-dtype',
                                options=[
                                    {'label': 'Convert to Categorical', 'value': 'category'},
                                    {'label': 'Convert to Numeric', 'value': 'numeric'},
                                    {'label': 'Convert to Datetime', 'value': 'datetime'}
                                ],
                                value='category',
                                labelStyle={'display': 'block', 'margin': '10px 0'}
                            ),
                            
                            html.Button(
                                "Apply Conversion", 
                                id="btn-convert-dtype", 
                                className="button-36", 
                                style={'width': '220px', 'margin-top': '10px'}
                            )
                        ], className="conversion-section")
                    ], className="tab-content")
                ]),
                
                # Tab 5: Save Cleaned Data
                dcc.Tab(label='Save Cleaned Data', value='save-data', children=[
                    html.Div([
                        html.H3("Save Cleaned Dataset"),
                        html.P("Save your cleaned dataset and view cleaning summary."),
                        
                        # Before/After summary
                        html.Div(id="cleaning-summary", className="summary-container"),
                        
                        # Save options
                        html.Div([
                            html.Button(
                                "Save Cleaned Dataset", 
                                id="btn-save-dataset", 
                                className="button-36", 
                                style={'width': '220px', 'margin-top': '10px'}
                            ),
                            
                            html.Div(id="download-link-container", className="download-container")
                        ], className="save-options")
                    ], className="tab-content")
                ])
            ], className="tabs-container")
        ], id="data-cleaning-content", className="cleaning-content")
    ], className="page-content")
