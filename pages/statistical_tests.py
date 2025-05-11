from dash import html, dcc

def layout():
    return html.Div([
        html.H1("Statistical Tests"),
        html.P("Perform statistical tests on your data to derive insights and make informed decisions."),
        
        # Statistical tests options
        html.Div([
            html.H3("Statistical Test Options"),
            html.P("Select a test and configure its parameters to analyze your data."),
            
            # Test type selection
            html.Div([
                html.Label("Select Statistical Test:"),
                dcc.Dropdown(
                    id="test-type-dropdown",
                    options=[
                        {"label": "Linear Regression (t-test for significance)", "value": "linear_regression"},
                        {"label": "Chi-square Test (for categorical independence)", "value": "chi_square"},
                        {"label": "ANOVA Test (comparing means across groups)", "value": "anova"},
                        {"label": "Correlation Analysis (Pearson & Spearman)", "value": "correlation"}
                    ],
                    placeholder="Select a statistical test"
                )
            ], className="selector-item"),
            
            # Dynamic parameter selectors
            html.Div(id="test-parameter-selectors", className="parameter-selectors"),
            
            # Run test button
            html.Div([
                html.Button(
                    "Run Statistical Test", 
                    id="btn-run-test", 
                    className="button-36", 
                    style={'width': '220px', 'margin-top': '20px'}
                )
            ], className="test-button-container")
        ], className="test-options-container"),
        
        # Test results section
        html.Div([
            html.H3("Test Results"),
            html.Div(id="test-results-container", className="results-container")
        ], className="test-results-section"),
        
        # Hypothesis testing explainer
        html.Div([
            html.H3("Understanding Statistical Testing", className="explainer-title"),
            html.Div([
                html.H4("Key Concepts:"),
                html.Ul([
                    html.Li([
                        html.Strong("Null Hypothesis (H₀): "), 
                        "The assumption that there is no effect or no relationship between variables."
                    ]),
                    html.Li([
                        html.Strong("Alternative Hypothesis (H₁): "), 
                        "The hypothesis that there is an effect or relationship between variables."
                    ]),
                    html.Li([
                        html.Strong("p-value: "), 
                        "The probability of obtaining results as extreme as the observed results, assuming the null hypothesis is true. A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis."
                    ]),
                    html.Li([
                        html.Strong("Statistical Significance: "), 
                        "A result is considered statistically significant when the p-value is less than the significance level (α), typically 0.05."
                    ])
                ])
            ], className="explainer-content")
        ], className="test-explainer"),
        
        # Hidden placeholders for all possible dynamic components
        # These ensure the components always exist in the DOM
        html.Div([
            # Common components
            dcc.Slider(
                id="significance-level-slider",
                min=0.01,
                max=0.1,
                step=0.01,
                value=0.05,
                marks={0.01: '0.01', 0.05: '0.05', 0.1: '0.1'}
            ),
            
            # Linear regression params
            dcc.Dropdown(id="dependent-var-dropdown", options=[], value=None),
            dcc.Dropdown(id="independent-var-dropdown", options=[], value=None, multi=True),
            
            # Chi-square params
            dcc.Dropdown(id="chi-var1-dropdown", options=[], value=None),
            dcc.Dropdown(id="chi-var2-dropdown", options=[], value=None),
            
            # ANOVA params
            dcc.Dropdown(id="anova-numeric-dropdown", options=[], value=None),
            dcc.Dropdown(id="anova-group-dropdown", options=[], value=None),
            
            # Correlation params
            dcc.Dropdown(id="corr-var1-dropdown", options=[], value=None),
            dcc.Dropdown(id="corr-var2-dropdown", options=[], value=None),
            dcc.RadioItems(id="correlation-method", options=[
                {"label": "Pearson", "value": "pearson"},
                {"label": "Spearman", "value": "spearman"}
            ], value="pearson")
        ], style={"display": "none"})
        
    ], className="page-content") 