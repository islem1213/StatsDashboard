from dash import html, dcc

def layout():
    return html.Div([
        html.H1("Basic Machine Learning"),
        html.P("Apply machine learning algorithms to make predictions and discover patterns in your data."),
        
        # ML options
        html.Div([
            html.H3("Machine Learning Model Selection"),
            html.P("Select a model type and configure its parameters."),
            
            # Model type selection
            html.Div([
                html.Label("Select ML Model:"),
                dcc.Dropdown(
                    id="model-type-dropdown",
                    options=[
                        {"label": "Linear Regression", "value": "linear_regression"},
                        {"label": "K-Nearest Neighbors", "value": "knn"},
                        {"label": "Random Forest", "value": "random_forest"}
                    ],
                    placeholder="Select a machine learning model"
                )
            ], className="selector-item"),
            
            # Task type selection (classification or regression)
            html.Div([
                html.Label("Task Type:"),
                dcc.RadioItems(
                    id="task-type-radio",
                    options=[
                        {"label": "Classification", "value": "classification"},
                        {"label": "Regression", "value": "regression"}
                    ],
                    value="classification",
                    labelStyle={"display": "inline-block", "marginRight": "20px"}
                )
            ], className="selector-item"),
            
            # Model configuration section
            html.Div(id="model-config-section", className="model-config-section"),
            
            # Data setup
            html.Div([
                html.H4("Data Configuration"),
                
                # Feature selection
                html.Div([
                    html.Label("Select Features:"),
                    dcc.Dropdown(
                        id="feature-selector",
                        multi=True,
                        placeholder="Select columns to use as features"
                    )
                ], className="selector-item"),
                
                # Target selection
                html.Div([
                    html.Label("Select Target:"),
                    dcc.Dropdown(
                        id="target-selector",
                        placeholder="Select column to predict"
                    )
                ], className="selector-item"),
                
                # Test size slider
                html.Div([
                    html.Label("Test Set Size (%)"),
                    dcc.Slider(
                        id="test-size-slider",
                        min=10,
                        max=50,
                        step=5,
                        value=25,
                        marks={10: '10%', 20: '20%', 30: '30%', 40: '40%', 50: '50%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="slider-container")
            ], className="data-config-section"),
            
            # Train and predict buttons
            html.Div([
                html.Button(
                    "Train Model", 
                    id="btn-train-model", 
                    className="button-36", 
                    style={'width': '220px', 'margin-right': '10px'}
                ),
                html.Button(
                    "Download Model", 
                    id="btn-download-model", 
                    className="button-36", 
                    disabled=True,
                    style={'width': '220px'}
                )
            ], className="button-container")
        ], className="ml-options-container"),
        
        # Model results section
        html.Div([
            html.H3("Model Results"),
            html.Div(id="model-results-container", className="results-container")
        ], className="ml-results-section"),
        
        # Store for trained model info
        dcc.Store(id="trained-model-store"),
        
        # Download component
        dcc.Download(id="download-model-pickle"),
        
        # Hidden placeholders for all possible model parameters to ensure they exist in the DOM
        html.Div([
            # Linear/Logistic Regression parameters
            dcc.RadioItems(id="fit-intercept-radio", options=[
                {"label": "Yes", "value": "yes"},
                {"label": "No", "value": "no"}
            ], value="yes"),
            
            dcc.Slider(id="logreg-c-slider", min=-3, max=3, step=0.5, value=0),
            dcc.Slider(id="logreg-iter-slider", min=100, max=1000, step=100, value=100),
            
            # KNN parameters
            dcc.Slider(id="knn-n-neighbors-slider", min=1, max=20, step=1, value=5),
            dcc.RadioItems(id="knn-weights-radio", options=[
                {"label": "Uniform", "value": "uniform"},
                {"label": "Distance", "value": "distance"}
            ], value="uniform"),
            
            # Random Forest parameters
            dcc.Slider(id="rf-n-estimators-slider", min=10, max=200, step=10, value=100),
            dcc.Slider(id="rf-max-depth-slider", min=2, max=20, step=1, value=None),
            dcc.Slider(id="rf-min-samples-split-slider", min=2, max=10, step=1, value=2)
        ], style={"display": "none"})
    ], className="page-content") 