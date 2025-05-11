from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from dash import html, dcc, dash_table
import dash
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

def register_statistical_tests_callbacks(app):
    # Update parameter selectors based on test type
    @app.callback(
        Output("test-parameter-selectors", "children"),
        [Input("test-type-dropdown", "value")],
        [State("cleaned-dataset-store", "data"), State("dataset-store", "data")]
    )
    def update_test_parameters(test_type, cleaned_dataset_json, dataset_json):
        if not test_type:
            raise PreventUpdate
            
        # Load dataframe, prioritize cleaned data if available
        if cleaned_dataset_json is not None:
            df = pd.read_json(cleaned_dataset_json, orient='split')
        elif dataset_json is not None:
            df = pd.read_json(dataset_json, orient='split')
        else:
            raise PreventUpdate
        
        # Get column lists by type
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()
        
        # Create options lists
        numeric_options = [{"label": col, "value": col} for col in numeric_cols]
        categorical_options = [{"label": col, "value": col} for col in categorical_cols]
        all_options = [{"label": col, "value": col} for col in all_cols]
        
        # Common significance level slider for all test types
        significance_level_component = html.Div([
            html.Label("Significance Level (α):"),
            dcc.Slider(
                id="significance-level-slider",
                min=0.01,
                max=0.1,
                step=0.01,
                value=0.05,
                marks={0.01: '0.01', 0.05: '0.05', 0.1: '0.1'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], className="selector-item")
        
        # Different parameters based on test type
        if test_type == "linear_regression":
            return html.Div([
                html.H4("Linear Regression Parameters"),
                html.P("Test the significance of linear relationships between variables."),
                
                html.Div([
                    html.Label("Dependent Variable (Y):"),
                    dcc.Dropdown(
                        id="dependent-var-dropdown",
                        options=numeric_options,
                        placeholder="Select the dependent variable"
                    )
                ], className="selector-item"),
                
                html.Div([
                    html.Label("Independent Variable(s) (X):"),
                    dcc.Dropdown(
                        id="independent-var-dropdown",
                        options=numeric_options,
                        multi=True,
                        placeholder="Select one or more independent variables"
                    )
                ], className="selector-item"),
                
                significance_level_component
            ])
            
        elif test_type == "chi_square":
            return html.Div([
                html.H4("Chi-Square Test Parameters"),
                html.P("Test the independence between categorical variables."),
                
                html.Div([
                    html.Label("First Categorical Variable:"),
                    dcc.Dropdown(
                        id="chi-var1-dropdown",
                        options=categorical_options,
                        placeholder="Select first categorical variable"
                    )
                ], className="selector-item"),
                
                html.Div([
                    html.Label("Second Categorical Variable:"),
                    dcc.Dropdown(
                        id="chi-var2-dropdown",
                        options=categorical_options,
                        placeholder="Select second categorical variable"
                    )
                ], className="selector-item"),
                
                significance_level_component
            ])
            
        elif test_type == "anova":
            return html.Div([
                html.H4("ANOVA Test Parameters"),
                html.P("Compare means across multiple groups."),
                
                html.Div([
                    html.Label("Numeric Variable to Compare:"),
                    dcc.Dropdown(
                        id="anova-numeric-dropdown",
                        options=numeric_options,
                        placeholder="Select numeric variable"
                    )
                ], className="selector-item"),
                
                html.Div([
                    html.Label("Grouping Variable:"),
                    dcc.Dropdown(
                        id="anova-group-dropdown",
                        options=categorical_options,
                        placeholder="Select categorical grouping variable"
                    )
                ], className="selector-item"),
                
                significance_level_component
            ])
            
        elif test_type == "correlation":
            return html.Div([
                html.H4("Correlation Analysis Parameters"),
                html.P("Measure the strength and direction of relationships between variables."),
                
                html.Div([
                    html.Label("First Variable:"),
                    dcc.Dropdown(
                        id="corr-var1-dropdown",
                        options=numeric_options,
                        placeholder="Select first variable"
                    )
                ], className="selector-item"),
                
                html.Div([
                    html.Label("Second Variable:"),
                    dcc.Dropdown(
                        id="corr-var2-dropdown",
                        options=numeric_options,
                        placeholder="Select second variable"
                    )
                ], className="selector-item"),
                
                html.Div([
                    html.Label("Correlation Method:"),
                    dcc.RadioItems(
                        id="correlation-method",
                        options=[
                            {"label": "Pearson (linear relationships)", "value": "pearson"},
                            {"label": "Spearman (monotonic relationships)", "value": "spearman"}
                        ],
                        value="pearson",
                        labelStyle={"display": "block", "marginBottom": "5px"}
                    )
                ], className="selector-item"),
                
                significance_level_component
            ])
        
        return html.Div([
            html.P("Please select a statistical test"),
            # Including the significance slider even when no test is selected
            # to ensure it's always present in the DOM
            significance_level_component
        ])

    # Run test and display results
    @app.callback(
        Output("test-results-container", "children"),
        [Input("btn-run-test", "n_clicks")],
        [
            State("cleaned-dataset-store", "data"),
            State("dataset-store", "data"),
            State("test-type-dropdown", "value"),
            # Optional States with default values
            State("significance-level-slider", "value"),
            # Linear regression params
            State("dependent-var-dropdown", "value"),
            State("independent-var-dropdown", "value"),
            # Chi-square params
            State("chi-var1-dropdown", "value"),
            State("chi-var2-dropdown", "value"),
            # ANOVA params
            State("anova-numeric-dropdown", "value"),
            State("anova-group-dropdown", "value"),
            # Correlation params
            State("corr-var1-dropdown", "value"),
            State("corr-var2-dropdown", "value"),
            State("correlation-method", "value")
        ]
    )
    def run_statistical_test(n_clicks, cleaned_dataset_json, dataset_json, test_type, significance_level,
                            dependent_var, independent_vars, 
                            chi_var1, chi_var2,
                            anova_numeric, anova_group,
                            corr_var1, corr_var2, corr_method):
        # Validate inputs
        if n_clicks is None or not n_clicks or test_type is None:
            raise PreventUpdate
            
        # Set default significance level if missing
        if significance_level is None:
            significance_level = 0.05  # Default value
            
        # Make sure the callback doesn't fail when components don't exist yet
        # For example, when switching between test types
        # Set default values for any parameters that might be missing
        if dependent_var is None: dependent_var = ""
        if independent_vars is None: independent_vars = []
        # Convert empty list to proper empty list (in case it comes as None)
        if isinstance(independent_vars, (str, type(None))): independent_vars = []
        if chi_var1 is None: chi_var1 = ""
        if chi_var2 is None: chi_var2 = ""
        if anova_numeric is None: anova_numeric = ""
        if anova_group is None: anova_group = ""
        if corr_var1 is None: corr_var1 = ""
        if corr_var2 is None: corr_var2 = ""
        if corr_method is None: corr_method = "pearson"  # Default correlation method
        
        # Load dataframe, prioritize cleaned data if available
        if cleaned_dataset_json is not None:
            df = pd.read_json(cleaned_dataset_json, orient='split')
        elif dataset_json is not None:
            df = pd.read_json(dataset_json, orient='split')
        else:
            return html.Div("Please upload and clean data first", className="error-message")
        
        # Run appropriate test
        if test_type == "linear_regression":
            if not dependent_var or not independent_vars or len(independent_vars) == 0:
                return html.Div("Please select dependent and independent variables", className="error-message")
                
            # Create formula for statsmodels
            formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
            
            try:
                # Validate data for regression
                if len(df.dropna(subset=[dependent_var] + independent_vars)) < 3:
                    return html.Div("Not enough data points for regression (after removing missing values)", className="error-message")
                
                # Check for multicollinearity if multiple predictors
                if len(independent_vars) > 1:
                    X = df[independent_vars].dropna()
                    if len(X) > 0:
                        # Check correlation between predictors
                        corr_matrix = X.corr().abs()
                        high_corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                if corr_matrix.iloc[i, j] > 0.8:  # 0.8 is a common threshold
                                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
                # Fit the model
                model = ols(formula, data=df).fit()
                
                # Check if model fit is valid
                if np.isnan(model.rsquared) or np.isnan(model.f_pvalue):
                    return html.Div([
                        html.H4("Linear Regression Error", style={"color": "red"}),
                        html.P("The model could not be properly fit. This may be due to:"),
                        html.Ul([
                            html.Li("Perfect multicollinearity between predictors"),
                            html.Li("Not enough variation in the data"),
                            html.Li("Complete separation in logistic regression")
                        ])
                    ], className="error-message")
                
                # Get results
                summary = model.summary()
                
                # Extract key information
                r_squared = model.rsquared
                adj_r_squared = model.rsquared_adj
                f_stat = model.fvalue
                f_pvalue = model.f_pvalue
                params = model.params
                pvalues = model.pvalues
                
                # Create coefficient table
                coef_data = []
                for var in params.index:
                    coef_data.append({
                        'Variable': var,
                        'Coefficient': f"{params[var]:.4f}",
                        'p-value': f"{pvalues[var]:.4f}",
                        'Significant': "Yes" if pvalues[var] < significance_level else "No"
                    })
                
                coef_df = pd.DataFrame(coef_data)
                
                # Create scatter plot of actual vs. predicted values
                predicted = model.predict(df[independent_vars])
                fig = px.scatter(x=df[dependent_var], y=predicted, 
                                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                title="Actual vs. Predicted Values")
                fig.add_trace(
                    go.Scatter(x=[df[dependent_var].min(), df[dependent_var].max()], 
                              y=[df[dependent_var].min(), df[dependent_var].max()],
                              mode='lines', line=dict(color='red', dash='dash'),
                              name='Perfect Prediction')
                )
                
                # Create result component
                return html.Div([
                    html.H4("Linear Regression Results", style={"color": "#2980b9"}),
                    
                    html.Div([
                        html.Div([
                            html.H5("Model Quality:"),
                            html.P(f"R-squared: {r_squared:.4f}"),
                            html.P(f"Adjusted R-squared: {adj_r_squared:.4f}"),
                            html.P(f"F-statistic: {f_stat:.4f}"),
                            html.P(f"F-statistic p-value: {f_pvalue:.4f}"),
                            html.P(f"Model is significant overall: {'Yes' if f_pvalue < significance_level else 'No'}", 
                                  style={"fontWeight": "bold", "color": "green" if f_pvalue < significance_level else "red"})
                        ], className="model-metrics"),
                        
                        html.Div([
                            html.H5("Coefficients:"),
                            dash_table.DataTable(
                                data=coef_df.to_dict('records'),
                                columns=[{'name': col, 'id': col} for col in coef_df.columns],
                                style_cell={'textAlign': 'left', 'padding': '5px'},
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'filter_query': '{Significant} = "Yes"'},
                                        'backgroundColor': 'rgba(0, 255, 0, 0.1)'
                                    },
                                    {
                                        'if': {'filter_query': '{Significant} = "No"'},
                                        'backgroundColor': 'rgba(255, 0, 0, 0.1)'
                                    }
                                ]
                            )
                        ], className="coefficients-table")
                    ], className="results-grid"),
                    
                    html.Div([
                        html.H5("Actual vs Predicted Values:"),
                        dcc.Graph(figure=fig)
                    ], className="scatter-plot"),
                    
                    html.Div([
                        html.H5("Interpretation:"),
                        html.P([
                            "This linear regression model explains ", 
                            html.Strong(f"{r_squared*100:.1f}%"), 
                            " of the variance in ", 
                            html.Strong(dependent_var), 
                            ". The model is ", 
                            html.Strong("statistically significant" if f_pvalue < significance_level else "not statistically significant"),
                            " overall."
                        ]),
                        html.P([
                            "The coefficients indicate how much ", 
                            html.Strong(dependent_var), 
                            " changes when each independent variable increases by one unit, holding all other variables constant."
                        ])
                    ], className="interpretation")
                ])
                
            except Exception as e:
                return html.Div([
                    html.H4("Error Running Linear Regression", style={"color": "red"}),
                    html.P(f"An error occurred: {str(e)}")
                ])
                
        elif test_type == "chi_square":
            if not chi_var1 or not chi_var2:
                return html.Div("Please select two categorical variables", className="error-message")
                
            try:
                # Create contingency table
                contingency_table = pd.crosstab(df[chi_var1], df[chi_var2])
                
                # Check if table has at least 2 rows and 2 columns
                if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                    return html.Div([
                        html.H4("Chi-Square Error", style={"color": "red"}),
                        html.P("Chi-Square test requires at least 2 categories per variable."),
                        html.P(f"Your data has {contingency_table.shape[0]} categories for {chi_var1} and {contingency_table.shape[1]} for {chi_var2}.")
                    ], className="error-message")
                
                # Run chi-square test
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                
                # Check validity of the test
                expected_df = pd.DataFrame(
                    expected, 
                    index=contingency_table.index, 
                    columns=contingency_table.columns
                )
                
                # Check if any expected frequency is less than 5 (common rule of thumb)
                low_expected = (expected_df < 5).values.sum()
                low_expected_pct = low_expected / (expected_df.shape[0] * expected_df.shape[1])
                warning_msg = None
                if low_expected > 0:
                    if low_expected_pct > 0.2:  # More than 20% of cells have expected freq < 5
                        warning_msg = f"Warning: {low_expected} cells ({low_expected_pct:.1%}) have expected frequencies less than 5. Chi-square results may be unreliable."
                    
                # Check for extremely low expected frequencies (< 1)
                very_low_expected = (expected_df < 1).values.sum()
                if very_low_expected > 0:
                    warning_msg = f"Warning: {very_low_expected} cells have expected frequencies less than 1. Consider using Fisher's exact test instead."
                    
                # Check if results are valid
                if np.isnan(chi2) or np.isnan(p_value):
                    return html.Div("Chi-Square test could not be computed. Check your data for complete separation or zero counts.", className="error-message")
                    
                # Create heatmap of observed frequencies
                fig_observed = px.imshow(
                    contingency_table,
                    text_auto=True,
                    title=f"Observed Frequencies: {chi_var1} vs {chi_var2}",
                    labels=dict(x=chi_var2, y=chi_var1, color="Count")
                )
                
                # Create heatmap of expected frequencies
                fig_expected = px.imshow(
                    expected_df,
                    text_auto='.1f',
                    title=f"Expected Frequencies: {chi_var1} vs {chi_var2}",
                    labels=dict(x=chi_var2, y=chi_var1, color="Count")
                )
                
                return html.Div([
                    html.H4("Chi-Square Test Results", style={"color": "#2980b9"}),
                    
                    html.Div([
                        html.H5("Test Statistics:"),
                        html.P(f"Chi-Square Statistic: {chi2:.4f}"),
                        html.P(f"Degrees of Freedom: {dof}"),
                        html.P(f"p-value: {p_value:.4f}"),
                        html.P([
                            "Result: Variables are ", 
                            html.Strong(
                                "dependent" if p_value < significance_level else "independent", 
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            f" at α = {significance_level}"
                        ]),
                        html.P(warning_msg, style={"color": "orange", "fontWeight": "bold"}) if warning_msg else html.Div()
                    ], className="test-statistics"),
                    
                    html.Div([
                        html.Div([
                            html.H5("Observed Frequencies:"),
                            dcc.Graph(figure=fig_observed)
                        ], className="observed-plot"),
                        
                        html.Div([
                            html.H5("Expected Frequencies:"),
                            dcc.Graph(figure=fig_expected)
                        ], className="expected-plot")
                    ], className="plot-grid"),
                    
                    html.Div([
                        html.H5("Interpretation:"),
                        html.P([
                            "The chi-square test examines whether there is a significant association between ", 
                            html.Strong(chi_var1), 
                            " and ", 
                            html.Strong(chi_var2), 
                            "."
                        ]),
                        html.P([
                            "A p-value of ", 
                            html.Strong(f"{p_value:.4f}"), 
                            " means we ", 
                            html.Strong(
                                "reject" if p_value < significance_level else "fail to reject",
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            " the null hypothesis that the variables are independent."
                        ]),
                        html.P([
                            "In other words, there ", 
                            html.Strong(
                                "is" if p_value < significance_level else "is not",
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            " a statistically significant relationship between these variables."
                        ])
                    ], className="interpretation")
                ])
                
            except Exception as e:
                return html.Div([
                    html.H4("Error Running Chi-Square Test", style={"color": "red"}),
                    html.P(f"An error occurred: {str(e)}")
                ])
                
        elif test_type == "anova":
            if not anova_numeric or not anova_group:
                return html.Div("Please select numeric and grouping variables", className="error-message")
                
            try:
                # Group data
                groups = []
                labels = []
                
                for name, group in df.groupby(anova_group):
                    if len(group) > 0:
                        values = group[anova_numeric].dropna()
                        if len(values) > 0:
                            groups.append(values)
                            labels.append(str(name))
                
                if len(groups) < 2:
                    return html.Div("ANOVA requires at least 2 groups with valid data", className="error-message")
                
                # Check if all groups have at least 1 value
                if any(len(group) < 1 for group in groups):
                    return html.Div("Each group must have at least 1 valid data point for ANOVA", className="error-message")
                
                # Check if any group has no variance (all values are the same)
                if any(group.std() == 0 for group in groups if len(group) > 1):
                    return html.Div("Warning: Some groups have zero variance (all values are identical)", className="warning-message")
                    
                # Run ANOVA
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    # Check if results are valid
                    if np.isnan(f_stat) or np.isnan(p_value):
                        return html.Div("ANOVA could not be computed. Check that your groups have sufficient variation.", className="error-message")
                except Exception as e:
                    return html.Div([
                        html.H4("Error Running ANOVA Test", style={"color": "red"}),
                        html.P(f"An error occurred: {str(e)}"),
                        html.P("This could be due to insufficient data or zero variance within groups.")
                    ], className="error-message")
                
                # Create box plot
                fig = px.box(
                    df, 
                    x=anova_group, 
                    y=anova_numeric,
                    title=f"Distribution of {anova_numeric} by {anova_group}",
                    points="all"
                )
                
                # Calculate group statistics
                group_stats = df.groupby(anova_group)[anova_numeric].agg(['mean', 'std', 'count']).reset_index()
                group_stats.columns = [anova_group, 'Mean', 'Std Dev', 'Count']
                
                return html.Div([
                    html.H4("ANOVA Test Results", style={"color": "#2980b9"}),
                    
                    html.Div([
                        html.H5("Test Statistics:"),
                        html.P(f"F-statistic: {f_stat:.4f}"),
                        html.P(f"p-value: {p_value:.4f}"),
                        html.P([
                            "Result: Group means are ", 
                            html.Strong(
                                "significantly different" if p_value < significance_level else "not significantly different", 
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            f" at α = {significance_level}"
                        ])
                    ], className="test-statistics"),
                    
                    html.Div([
                        html.H5("Group Statistics:"),
                        dash_table.DataTable(
                            data=group_stats.to_dict('records'),
                            columns=[{'name': col, 'id': col} for col in group_stats.columns],
                            style_cell={'textAlign': 'left', 'padding': '5px'},
                            style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                            }
                        )
                    ], className="group-stats"),
                    
                    html.Div([
                        html.H5("Box Plot of Groups:"),
                        dcc.Graph(figure=fig)
                    ], className="box-plot"),
                    
                    html.Div([
                        html.H5("Interpretation:"),
                        html.P([
                            "The ANOVA test examines whether the means of ", 
                            html.Strong(anova_numeric), 
                            " differ significantly across the groups defined by ", 
                            html.Strong(anova_group), 
                            "."
                        ]),
                        html.P([
                            "A p-value of ", 
                            html.Strong(f"{p_value:.4f}"), 
                            " means we ", 
                            html.Strong(
                                "reject" if p_value < significance_level else "fail to reject",
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            " the null hypothesis that all group means are equal."
                        ]),
                        html.P([
                            "In other words, there ", 
                            html.Strong(
                                "are" if p_value < significance_level else "are not",
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            " statistically significant differences between the groups."
                        ])
                    ], className="interpretation")
                ])
                
            except Exception as e:
                return html.Div([
                    html.H4("Error Running ANOVA Test", style={"color": "red"}),
                    html.P(f"An error occurred: {str(e)}")
                ])
                
        elif test_type == "correlation":
            if not corr_var1 or not corr_var2:
                return html.Div("Please select two variables for correlation analysis", className="error-message")
                
            try:
                # Get data
                x = df[corr_var1].dropna()
                y = df[corr_var2].dropna()
                
                # Merge to handle missing values
                valid_data = pd.DataFrame({
                    'x': x,
                    'y': y
                }).dropna()
                
                if len(valid_data) < 3:  # Need at least 3 points for meaningful correlation
                    return html.Div("Not enough valid data points for correlation (minimum 3 required)", className="error-message")
                
                # Check for constant values (which would cause a division by zero in correlation)
                if valid_data['x'].std() == 0 or valid_data['y'].std() == 0:
                    return html.Div("Cannot compute correlation: One of the variables has constant values", className="error-message")
                
                # Calculate correlation
                try:
                    if corr_method == "pearson":
                        corr, p_value = stats.pearsonr(valid_data['x'], valid_data['y'])
                        method_name = "Pearson"
                    else:  # spearman
                        corr, p_value = stats.spearmanr(valid_data['x'], valid_data['y'])
                        method_name = "Spearman"
                except Exception as e:
                    return html.Div([
                        html.H4("Error Computing Correlation", style={"color": "red"}),
                        html.P(f"An error occurred: {str(e)}"),
                        html.P("This often happens when there's not enough variation in the data.")
                    ], className="error-message")
                
                # Create scatter plot
                fig = px.scatter(
                    valid_data, 
                    x='x', 
                    y='y',
                    trendline="ols" if corr_method == "pearson" else None,
                    labels={'x': corr_var1, 'y': corr_var2},
                    title=f"Scatter Plot: {corr_var2} vs {corr_var1}"
                )
                
                # Correlation strength description
                if abs(corr) < 0.3:
                    strength = "weak"
                    color = "orange"
                elif abs(corr) < 0.7:
                    strength = "moderate"
                    color = "blue"
                else:
                    strength = "strong"
                    color = "green"
                
                direction = "positive" if corr > 0 else "negative"
                
                return html.Div([
                    html.H4("Correlation Analysis Results", style={"color": "#2980b9"}),
                    
                    html.Div([
                        html.H5("Correlation Statistics:"),
                        html.P(f"{method_name} Correlation Coefficient: {corr:.4f}"),
                        html.P(f"p-value: {p_value:.4f}"),
                        html.P([
                            "Result: Correlation is ", 
                            html.Strong(
                                "statistically significant" if p_value < significance_level else "not statistically significant", 
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            f" at α = {significance_level}"
                        ])
                    ], className="correlation-stats"),
                    
                    html.Div([
                        html.H5("Scatter Plot:"),
                        dcc.Graph(figure=fig)
                    ], className="scatter-plot"),
                    
                    html.Div([
                        html.H5("Interpretation:"),
                        html.P([
                            "The ", 
                            html.Strong(f"{method_name} correlation"),
                            " between ", 
                            html.Strong(corr_var1), 
                            " and ", 
                            html.Strong(corr_var2), 
                            " is ", 
                            html.Strong(f"{corr:.4f}"),
                            ", indicating a ",
                            html.Strong(f"{strength} {direction}", style={"color": color}),
                            " relationship."
                        ]),
                        html.P([
                            "A p-value of ", 
                            html.Strong(f"{p_value:.4f}"), 
                            " means we ", 
                            html.Strong(
                                "reject" if p_value < significance_level else "fail to reject",
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            " the null hypothesis that there is no correlation between the variables."
                        ]),
                        html.P([
                            "In other words, the correlation ", 
                            html.Strong(
                                "is" if p_value < significance_level else "is not",
                                style={"color": "green" if p_value < significance_level else "red"}
                            ),
                            " statistically significant."
                        ])
                    ], className="interpretation")
                ])
                
            except Exception as e:
                return html.Div([
                    html.H4("Error Running Correlation Analysis", style={"color": "red"}),
                    html.P(f"An error occurred: {str(e)}")
                ])
        
        return html.Div("Please select a valid statistical test") 