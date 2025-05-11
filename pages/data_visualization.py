from dash import html, dcc

def layout():
    return html.Div([
        html.H1("Data Visualization"),
        html.P("Create visualizations to understand your data."),
        
        # Exploratory Data Analysis Section
        html.Div([
            html.H3("Exploratory Data Analysis"),
            html.P("Get a quick overview of your dataset structure and statistics."),
            
            # EDA Options
            html.Div([
                html.Button(
                    "Show Basic Statistics", 
                    id="btn-show-stats", 
                    className="button-36", 
                    style={'width': '220px', 'margin-right': '10px'}
                ),
                html.Button(
                    "Show Correlation Matrix", 
                    id="btn-show-corr", 
                    className="button-36", 
                    style={'width': '220px'}
                ),
            ], className="eda-buttons"),
            
            # EDA Output
            html.Div(id="eda-output", className="eda-output")
        ], id="eda-section", className="visualization-options"),
        
        # Data visualization options
        html.Div([
            html.H3("Visualization Options"),
            html.P("Select visualization type and columns to visualize."),
            
            # Plot type selection
            html.Div([
                html.Label("Plot Type:"),
                dcc.Dropdown(
                    id="plot-type-dropdown",
                    options=[
                        {"label": "Scatter Plot", "value": "scatter"},
                        {"label": "Bar Chart", "value": "bar"},
                        {"label": "Line Chart", "value": "line"},
                        {"label": "Histogram", "value": "histogram"},
                        {"label": "Box Plot", "value": "box"},
                        {"label": "Heatmap", "value": "heatmap"}
                    ],
                    placeholder="Select a plot type"
                )
            ], className="selector-item"),
            
            # Dynamic column selectors will be populated here
            html.Div(id="column-selectors", className="column-selectors"),
            
            # Additional options for bar chart
            html.Div(id="additional-options", className="additional-options"),
            
            # Generate and export buttons
            html.Div([
                html.Button(
                    "Generate Plot", 
                    id="btn-generate-plot", 
                    className="button-36", 
                    style={'width': '220px', 'margin-top': '10px'}
                ),
                html.Button(
                    "Export Chart", 
                    id="btn-export-chart", 
                    className="button-36", 
                    disabled=True,
                    style={'width': '220px', 'margin-top': '10px', 'margin-left': '10px'}
                )
            ], className="visualization-buttons")
        ], id="data-visualization-options", className="visualization-options"),
        
        # Output area for visualizations
        html.Div(id="visualization-output", className="visualization-output"),
        
        # Hidden placeholders for all possible dynamic components
        # These ensure the components always exist in the DOM
        html.Div([
            # Axis selections
            dcc.Input(id="x-axis-dropdown", type="hidden", value=""),
            dcc.Input(id="y-axis-dropdown", type="hidden", value=""),
            dcc.Input(id="color-dropdown", type="hidden", value=""),
            
            # Regression options for scatter plots
            dcc.Input(id="regression-type", type="hidden", value="none"),
            
            # Visualization graph (for download/export functions)
            dcc.Store(id="visualization-graph", data=None),
            
            # Correlation output
            html.Div(id="correlation-output", style={"display": "none"}),
            
            # Download components
            dcc.Download(id="download-image")
        ], style={"display": "none"})
        
    ], className="page-content") 