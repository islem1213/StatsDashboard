from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from dash import html, dcc, dash_table
import dash

def register_data_visualization_callbacks(app):
    # EDA Callbacks
    @app.callback(
        Output("eda-output", "children"),
        [
            Input("btn-show-stats", "n_clicks"),
            Input("btn-show-corr", "n_clicks")
        ],
        [State("cleaned-dataset-store", "data"), State("dataset-store", "data")],
        prevent_initial_call=True
    )
    def show_eda(stats_clicks, corr_clicks, cleaned_dataset_json, dataset_json):
        # Load dataframe, prioritize cleaned data
        if cleaned_dataset_json is not None:
            df = pd.read_json(cleaned_dataset_json, orient='split')
        elif dataset_json is not None:
            df = pd.read_json(dataset_json, orient='split')
        else:
            return html.Div("Please upload a dataset first", style={"color": "red"})
            
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "btn-show-stats":
            # Show basic statistics with data type info
            stats_df = df.describe(include='all').reset_index()
            stats_df.rename(columns={'index': 'statistic'}, inplace=True)
            
            # Add column data types to the stats table
            type_row = {'statistic': 'data_type'}
            for col in df.columns:
                if col != 'statistic':
                    if pd.api.types.is_numeric_dtype(df[col]):
                        type_row[col] = 'Numeric'
                    elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                        type_row[col] = 'Categorical'
                    elif pd.api.types.is_datetime64_dtype(df[col]):
                        type_row[col] = 'Date/Time'
                    else:
                        type_row[col] = 'Other'
            
            # Add missing value info
            missing_row = {'statistic': 'missing_values'}
            missing_percent_row = {'statistic': 'missing_percent'}
            for col in df.columns:
                if col != 'statistic':
                    missing = df[col].isna().sum()
                    missing_row[col] = missing
                    missing_percent_row[col] = f"{(missing / len(df)) * 100:.2f}%"
            
            # Add unique value counts
            unique_row = {'statistic': 'unique_values'}
            for col in df.columns:
                if col != 'statistic':
                    unique_row[col] = df[col].nunique()
            
            # Insert the new rows at the top
            new_rows = [type_row, missing_row, missing_percent_row, unique_row]
            for i, row in enumerate(new_rows):
                stats_df.loc[-i-1] = row
            
            stats_df = stats_df.sort_index().reset_index(drop=True)
            
            return html.Div([
                html.H4("Data Overview and Statistics"),
                dash_table.DataTable(
                    data=stats_df.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in stats_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'height': 'auto',
                        'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
                        'whiteSpace': 'normal'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 0},
                            'backgroundColor': 'rgba(0, 149, 255, 0.15)',
                            'fontWeight': 'bold'
                        },
                        {
                            'if': {'row_index': 1},
                            'backgroundColor': 'rgba(255, 149, 0, 0.15)'
                        },
                        {
                            'if': {'row_index': 2},
                            'backgroundColor': 'rgba(255, 149, 0, 0.15)'
                        }
                    ]
                )
            ])
            
        elif button_id == "btn-show-corr":
            # Only include numeric columns for correlation
            numeric_df = df.select_dtypes(include=['number'])
            
            if numeric_df.shape[1] < 2:
                return html.Div("Not enough numeric columns for correlation analysis", style={"color": "red"})
            
            corr_matrix = numeric_df.corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Matrix"
            )
            
            return html.Div([
                html.H4("Correlation Matrix"),
                html.P("Shows relationships between numerical variables. Values close to 1 or -1 indicate strong correlations."),
                dcc.Graph(figure=fig)
            ])

    # Update column selectors based on plot type
    @app.callback(
        Output("column-selectors", "children"),
        [Input("plot-type-dropdown", "value")],
        [State("cleaned-dataset-store", "data"), State("dataset-store", "data")]
    )
    def update_column_selectors(plot_type, cleaned_dataset_json, dataset_json):
        if not plot_type:
            return []
            
        # Load the dataframe from JSON, prioritize cleaned data
        if cleaned_dataset_json is not None:
            df = pd.read_json(cleaned_dataset_json, orient='split')
        elif dataset_json is not None:
            df = pd.read_json(dataset_json, orient='split')
        else:
            return []
        
        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Create selectors based on plot type
        selectors = []
        
        if plot_type == "scatter":
            # X-Axis (numeric)
            selectors.append(
                html.Div([
                    html.Label("X-Axis (Numeric):"),
                    dcc.Dropdown(
                        id="x-axis-dropdown",
                        options=[{"label": col, "value": col} for col in numeric_cols],
                        placeholder="Select X-Axis"
                    )
                ], className="selector-item")
            )
            
            # Y-Axis (numeric)
            selectors.append(
                html.Div([
                    html.Label("Y-Axis (Numeric):"),
                    dcc.Dropdown(
                        id="y-axis-dropdown",
                        options=[{"label": col, "value": col} for col in numeric_cols],
                        placeholder="Select Y-Axis"
                    )
                ], className="selector-item")
            )
            
            # Color by (categorical or numeric)
            color_options = [{"label": col, "value": col} for col in categorical_cols + numeric_cols]
            selectors.append(
                html.Div([
                    html.Label("Color by (Optional):"),
                    dcc.Dropdown(
                        id="color-dropdown",
                        options=color_options,
                        placeholder="Select Color Variable"
                    )
                ], className="selector-item")
            )
            
        elif plot_type == "bar":
            # X-Axis (categorical)
            selectors.append(
                html.Div([
                    html.Label("X-Axis (Categories):"),
                    dcc.Dropdown(
                        id="x-axis-dropdown",
                        options=[{"label": col, "value": col} for col in categorical_cols],
                        placeholder="Select X-Axis"
                    )
                ], className="selector-item")
            )
            
            # Y-Axis (numeric)
            selectors.append(
                html.Div([
                    html.Label("Y-Axis (Numeric):"),
                    dcc.Dropdown(
                        id="y-axis-dropdown",
                        options=[{"label": col, "value": col} for col in numeric_cols],
                        placeholder="Select Y-Axis"
                    )
                ], className="selector-item")
            )
            
            # Color by (categorical)
            selectors.append(
                html.Div([
                    html.Label("Color by (Optional):"),
                    dcc.Dropdown(
                        id="color-dropdown",
                        options=[{"label": col, "value": col} for col in categorical_cols],
                        placeholder="Select Color Variable"
                    )
                ], className="selector-item")
            )
            
        elif plot_type == "line":
            # X-Axis (numeric or date)
            x_options = numeric_cols + date_cols
            selectors.append(
                html.Div([
                    html.Label("X-Axis (Numeric/Date):"),
                    dcc.Dropdown(
                        id="x-axis-dropdown",
                        options=[{"label": col, "value": col} for col in x_options],
                        placeholder="Select X-Axis"
                    )
                ], className="selector-item")
            )
            
            # Y-Axis (numeric)
            selectors.append(
                html.Div([
                    html.Label("Y-Axis (Numeric):"),
                    dcc.Dropdown(
                        id="y-axis-dropdown",
                        options=[{"label": col, "value": col} for col in numeric_cols],
                        placeholder="Select Y-Axis"
                    )
                ], className="selector-item")
            )
            
            # Color by (categorical)
            selectors.append(
                html.Div([
                    html.Label("Color by (Optional):"),
                    dcc.Dropdown(
                        id="color-dropdown",
                        options=[{"label": col, "value": col} for col in categorical_cols],
                        placeholder="Select Line Groups"
                    )
                ], className="selector-item")
            )
            
        elif plot_type == "histogram":
            # X-Axis (numeric)
            selectors.append(
                html.Div([
                    html.Label("Column (Numeric):"),
                    dcc.Dropdown(
                        id="x-axis-dropdown",
                        options=[{"label": col, "value": col} for col in numeric_cols],
                        placeholder="Select Column"
                    )
                ], className="selector-item")
            )
            
            # Hidden Y-Axis input (not used but needed for callbacks)
            selectors.append(
                html.Div([
                    dcc.Input(
                        id="y-axis-dropdown",
                        type="hidden",
                        value=""
                    )
                ], style={"display": "none"})
            )
            
            # Color by (categorical)
            selectors.append(
                html.Div([
                    html.Label("Color by (Optional):"),
                    dcc.Dropdown(
                        id="color-dropdown",
                        options=[{"label": col, "value": col} for col in categorical_cols],
                        placeholder="Select Groups"
                    )
                ], className="selector-item")
            )
            
        elif plot_type == "box":
            # X-Axis (categorical)
            selectors.append(
                html.Div([
                    html.Label("X-Axis (Categories):"),
                    dcc.Dropdown(
                        id="x-axis-dropdown",
                        options=[{"label": col, "value": col} for col in categorical_cols],
                        placeholder="Select X-Axis"
                    )
                ], className="selector-item")
            )
            
            # Y-Axis (numeric)
            selectors.append(
                html.Div([
                    html.Label("Y-Axis (Numeric):"),
                    dcc.Dropdown(
                        id="y-axis-dropdown",
                        options=[{"label": col, "value": col} for col in numeric_cols],
                        placeholder="Select Y-Axis"
                    )
                ], className="selector-item")
            )
            
            # Color by (categorical)
            selectors.append(
                html.Div([
                    html.Label("Color by (Optional):"),
                    dcc.Dropdown(
                        id="color-dropdown",
                        options=[{"label": col, "value": col} for col in categorical_cols],
                        placeholder="Select Color Variable"
                    )
                ], className="selector-item")
            )
            
        elif plot_type == "heatmap":
            # X-Axis (categorical)
            selectors.append(
                html.Div([
                    html.Label("X-Axis (Categories):"),
                    dcc.Dropdown(
                        id="x-axis-dropdown",
                        options=[{"label": col, "value": col} for col in categorical_cols],
                        placeholder="Select X-Axis"
                    )
                ], className="selector-item")
            )
            
            # Y-Axis (categorical)
            selectors.append(
                html.Div([
                    html.Label("Y-Axis (Categories):"),
                    dcc.Dropdown(
                        id="y-axis-dropdown",
                        options=[{"label": col, "value": col} for col in categorical_cols],
                        placeholder="Select Y-Axis"
                    )
                ], className="selector-item")
            )
            
            # Color by (numeric)
            selectors.append(
                html.Div([
                    html.Label("Value (Numeric):"),
                    dcc.Dropdown(
                        id="color-dropdown",
                        options=[{"label": col, "value": col} for col in numeric_cols],
                        placeholder="Select Value Variable"
                    )
                ], className="selector-item")
            )
            
        return selectors

    # Additional options for Bar Chart
    @app.callback(
        Output("additional-options", "children"),
        [Input("plot-type-dropdown", "value")],
        [State("cleaned-dataset-store", "data"), State("dataset-store", "data")]
    )
    def show_additional_options(plot_type, cleaned_dataset_json, dataset_json):
        if not plot_type:
            return []
            
        # Load dataframe, prioritize cleaned data
        if cleaned_dataset_json is not None:
            df = pd.read_json(cleaned_dataset_json, orient='split')
        elif dataset_json is not None:
            df = pd.read_json(dataset_json, orient='split')
        else:
            return []
        
        # For bar chart - add category filters
        if plot_type == "bar":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_cols:
                return []
                
            # Create filters for categorical columns
            filters = [
                html.H4("Category Filters"),
                html.P("Filter bar chart by selecting category values:")
            ]
            
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns to avoid clutter
                unique_values = df[col].dropna().unique().tolist()
                
                if len(unique_values) > 20:
                    # Limit to top 20 values by frequency if too many categories
                    top_values = df[col].value_counts().nlargest(20).index.tolist()
                    unique_values = top_values
                
                filter_options = [{"label": str(val), "value": str(val)} for val in unique_values]
                
                filters.append(
                    html.Div([
                        html.Label(f"Filter by {col}:"),
                        dcc.Dropdown(
                            id={"type": "category-filter", "column": col},
                            options=filter_options,
                            multi=True,
                            placeholder=f"Select {col} values"
                        )
                    ], className="selector-item")
                )
            
            return filters
        
        # For scatter plot - add correlation info
        elif plot_type == "scatter":
            return html.Div([
                html.Div(id="correlation-output", className="correlation-info"),
                html.Div([
                    html.Label("Show Regression Line:"),
                    dcc.RadioItems(
                        id="regression-type",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Linear", "value": "ols"},
                            {"label": "Lowess", "value": "lowess"}
                        ],
                        value="none",
                        labelStyle={"display": "inline-block", "marginRight": "10px"}
                    )
                ], className="selector-item")
            ])
            
        return []

    # Update correlation info when x and y axis are selected for scatter plot
    @app.callback(
        Output("correlation-output", "children"),
        [
            Input("x-axis-dropdown", "value"),
            Input("y-axis-dropdown", "value")
        ],
        [State("cleaned-dataset-store", "data"), State("dataset-store", "data")]
    )
    def update_correlation(x_column, y_column, cleaned_dataset_json, dataset_json):
        # Check if both x and y columns are provided
        if not x_column or not y_column:
            return []
            
        # Load dataframe, prioritize cleaned data
        try:
            if cleaned_dataset_json is not None:
                df = pd.read_json(cleaned_dataset_json, orient='split')
            elif dataset_json is not None:
                df = pd.read_json(dataset_json, orient='split')
            else:
                return []
                
            # Verify columns exist in dataframe
            if x_column not in df.columns or y_column not in df.columns:
                return []
            
            # Check if both columns are numeric
            if df[x_column].dtype.kind in 'ifc' and df[y_column].dtype.kind in 'ifc':
                correlation = df[x_column].corr(df[y_column])
                return html.Div([
                    html.H4("Correlation Analysis"),
                    html.P(f"Pearson correlation between {x_column} and {y_column}: {correlation:.4f}"),
                    html.P(f"Strength: {get_correlation_strength(correlation)}")
                ], style={"marginBottom": "15px"})
        except Exception as e:
            # Handle any exceptions that might occur
            return html.Div(f"Error calculating correlation: {str(e)}", style={"color": "red"})
        
        return []

    # Main visualization callback - updated for bar chart filters and scatter plot enhancements
    @app.callback(
        Output("visualization-output", "children"),
        [Input("btn-generate-plot", "n_clicks")],
        [
            State("cleaned-dataset-store", "data"),
            State("dataset-store", "data"),
            State("plot-type-dropdown", "value"),
            State("x-axis-dropdown", "value"),
            State("y-axis-dropdown", "value"),
            State("color-dropdown", "value"),
            State("regression-type", "value"),
            State({"type": "category-filter", "column": dash.ALL}, "value")
        ]
    )
    def generate_visualization(n_clicks, cleaned_dataset_json, dataset_json, plot_type, x_column, y_column, 
                              color_column, regression_type, category_filters):
        if n_clicks is None or not n_clicks:
            raise PreventUpdate
        
        # Check for required parameters
        if plot_type is None:
            return html.Div("Please select a plot type", className="error-message")
            
        # Set default values for components that might not exist yet
        if regression_type is None:
            regression_type = "none"
            
        # Initialize category_filters if it's None
        if category_filters is None:
            category_filters = []
        
        # Load the dataframe from JSON, prioritize cleaned data
        try:
            if cleaned_dataset_json is not None:
                df = pd.read_json(cleaned_dataset_json, orient='split')
            elif dataset_json is not None:
                df = pd.read_json(dataset_json, orient='split')
            else:
                return html.Div("Please upload data first", className="error-message")
                
            # Check if columns exist in dataframe
            if x_column and x_column not in df.columns:
                return html.Div(f"Column '{x_column}' not found in the dataset", className="error-message")
                
            if y_column and y_column not in df.columns and plot_type != "histogram":
                return html.Div(f"Column '{y_column}' not found in the dataset", className="error-message")
                
            if color_column and color_column not in df.columns:
                return html.Div(f"Column '{color_column}' not found in the dataset", className="error-message")
        except Exception as e:
            return html.Div(f"Error loading data: {str(e)}", className="error-message")
        
        # Apply category filters for bar chart
        if plot_type == "bar" and category_filters:
            try:
                ctx = dash.callback_context
                filter_ids = ctx.states_list[6]
                
                for i, filter_value in enumerate(category_filters):
                    if filter_value:
                        column = filter_ids[i]['id']['column']
                        if column in df.columns:  # Make sure column exists
                            df = df[df[column].isin(filter_value)]
            except Exception as e:
                return html.Div(f"Error applying filters: {str(e)}", className="error-message")
        
        # Check if required columns are selected
        if not x_column:
            return html.Div("Please select an X-axis column", style={"color": "red"})
            
        if plot_type not in ["histogram"] and not y_column:
            return html.Div("Please select a Y-axis column", style={"color": "red"})
            
        # Create the visualization based on the selected plot type
        try:
            # Scatter Plot
            if plot_type == "scatter":
                fig = px.scatter(
                    df, 
                    x=x_column, 
                    y=y_column,
                    color=color_column,
                    title=f"{y_column} vs {x_column}",
                    labels={x_column: x_column, y_column: y_column},
                    hover_data=df.columns[:5],  # Show first 5 columns in hover data
                    trendline=None if regression_type == "none" else regression_type
                )
                
            # Bar Chart
            elif plot_type == "bar":
                # Group by x_column and calculate mean of y_column
                if df[x_column].nunique() > 30:
                    # If too many categories, show a warning
                    return html.Div([
                        html.Div(f"Too many categories ({df[x_column].nunique()}) in {x_column}. Please select a column with fewer categories or apply filters.", 
                                style={"color": "red", "margin-bottom": "10px"}),
                        html.Div("Consider using a histogram for columns with many unique values.")
                    ])
                
                if color_column:
                    fig = px.bar(
                        df, 
                        x=x_column, 
                        y=y_column,
                        color=color_column,
                        title=f"{y_column} by {x_column}",
                        labels={x_column: x_column, y_column: y_column},
                        barmode="group"
                    )
                else:
                    # Calculate aggregation
                    agg_df = df.groupby(x_column)[y_column].mean().reset_index()
                    fig = px.bar(
                        agg_df, 
                        x=x_column, 
                        y=y_column,
                        title=f"Average {y_column} by {x_column}",
                        labels={x_column: x_column, y_column: f"Avg. {y_column}"}
                    )
                
            # Line Chart
            elif plot_type == "line":
                if color_column:
                    fig = px.line(
                        df, 
                        x=x_column, 
                        y=y_column,
                        color=color_column,
                        title=f"{y_column} vs {x_column}",
                        labels={x_column: x_column, y_column: y_column}
                    )
                else:
                    fig = px.line(
                        df, 
                        x=x_column, 
                        y=y_column,
                        title=f"{y_column} vs {x_column}",
                        labels={x_column: x_column, y_column: y_column}
                    )
                
            # Histogram
            elif plot_type == "histogram":
                fig = px.histogram(
                    df, 
                    x=x_column,
                    color=color_column,
                    title=f"Distribution of {x_column}",
                    labels={x_column: x_column},
                    marginal="box"  # Add a box plot at the margin
                )
                
            # Box Plot
            elif plot_type == "box":
                fig = px.box(
                    df, 
                    x=x_column, 
                    y=y_column,
                    color=color_column,
                    title=f"{y_column} by {x_column}",
                    labels={x_column: x_column, y_column: y_column},
                    points="outliers"  # Only show outlier points
                )
                
            # Heatmap
            elif plot_type == "heatmap":
                # Create a cross-tabulation of the data
                heatmap_data = pd.crosstab(
                    index=df[y_column], 
                    columns=df[x_column], 
                    values=df[color_column], 
                    aggfunc='mean'
                ).fillna(0)
                
                fig = px.imshow(
                    heatmap_data,
                    title=f"Heatmap of {color_column} by {x_column} and {y_column}",
                    labels=dict(x=x_column, y=y_column, color=color_column),
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r'
                )
            
            # Update layout for all plots
            fig.update_layout(
                template="plotly_white",
                title_x=0.5,
                xaxis_title=x_column,
                legend_title_text="Legend"
            )
            
            if plot_type != "histogram":
                fig.update_layout(yaxis_title=y_column)
            
            # Add a download section for the visualization
            download_section = html.Div([
                html.H4("Export Options"),
                html.Button("Download as PNG", id="btn-download-png", className="button-36", style={'margin-right': '10px'}),
                html.Button("Download as SVG", id="btn-download-svg", className="button-36", style={'margin-right': '10px'}),
                html.Button("Download as HTML", id="btn-download-html", className="button-36"),
                dcc.Download(id="download-image")
            ], className="download-section", style={'marginTop': '20px'})
            
            return html.Div([
                dcc.Graph(
                    id="visualization-graph",
                    figure=fig,
                    style={'height': '600px'}
                ),
                download_section
            ])
            
        except Exception as e:
            return html.Div([
                html.Div(f"Error generating visualization: {str(e)}", style={"color": "red"}),
                html.Div("Please check your column selections and try again.")
            ])

    # Toggle export button based on whether a graph is present
    @app.callback(
        Output("btn-export-chart", "disabled"),
        [Input("visualization-graph", "figure")]
    )
    def toggle_export_button(figure):
        # Enable export button only when a graph is generated
        return figure is None

    # Export chart options
    @app.callback(
        Output("visualization-output", "children", allow_duplicate=True),
        [Input("btn-export-chart", "n_clicks")],
        [State("visualization-graph", "figure"),
         State("visualization-output", "children")],
        prevent_initial_call=True
    )
    def export_chart(n_clicks, figure, current_children):
        if n_clicks is None or not n_clicks or figure is None or current_children is None:
            raise PreventUpdate
            
        # Create a modal with export options
        modal = html.Div([
            html.Div([
                html.H3("Export Visualization"),
                html.Button("X", id="btn-close-modal", className="close-button"),
                html.Hr(),
                html.Div([
                    html.H4("Export Format"),
                    html.Button("PNG Image", id="btn-download-png", className="button-36", style={'margin-right': '10px'}),
                    html.Button("SVG Vector", id="btn-download-svg", className="button-36", style={'margin-right': '10px'}),
                    html.Button("HTML (Interactive)", id="btn-download-html", className="button-36")
                ]),
                dcc.Download(id="download-image")
            ], className="modal-content")
        ], className="modal-overlay", id="export-modal")
        
        # Append the modal to the current children
        if isinstance(current_children, list):
            current_children.append(modal)
        else:
            current_children = [current_children, modal]
            
        return current_children

    # Download chart as image
    @app.callback(
        Output("download-image", "data"),
        [
            Input("btn-download-png", "n_clicks"),
            Input("btn-download-svg", "n_clicks"),
            Input("btn-download-html", "n_clicks")
        ],
        [State("visualization-graph", "figure")],
        prevent_initial_call=True
    )
    def download_chart(png_clicks, svg_clicks, html_clicks, figure):
        # Handle missing clicks or figure
        if all(click is None for click in [png_clicks, svg_clicks, html_clicks]) or figure is None:
            raise PreventUpdate
            
        try:
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
                
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            # Handle downloads in different formats
            if button_id == "btn-download-png":
                return dict(
                    content=figure,
                    filename="visualization.png",
                    type="plotly",
                    mime_type="image/png"
                )
            elif button_id == "btn-download-svg":
                return dict(
                    content=figure,
                    filename="visualization.svg",
                    type="plotly",
                    mime_type="image/svg+xml"
                )
            elif button_id == "btn-download-html":
                return dict(
                    content=figure,
                    filename="visualization.html",
                    type="plotly",
                    mime_type="text/html"
                )
        except Exception as e:
            # Return a simple text file with the error message if something goes wrong
            return dict(
                content=f"Error downloading chart: {str(e)}",
                filename="error.txt",
                type="text/plain"
            )

# Helper function to describe correlation strength
def get_correlation_strength(corr):
    corr_abs = abs(corr)
    if corr_abs < 0.1:
        return "Negligible correlation"
    elif corr_abs < 0.3:
        return "Weak correlation"
    elif corr_abs < 0.5:
        return "Moderate correlation"
    elif corr_abs < 0.7:
        return "Strong correlation"
    elif corr_abs < 0.9:
        return "Very strong correlation"
    else:
        return "Near perfect correlation" 