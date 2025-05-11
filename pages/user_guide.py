from dash import html, dcc

def layout():
    return html.Div([
        html.Div([
            html.H1("User Guide", className="guide-title"),
            html.P("Learn how to make the most of this statistical dashboard.", className="guide-intro"),
            
            
            # Table of Contents
            html.Div([
                html.H3("Quick Navigation", className="toc-title"),
                html.Div([
                    html.A("1. Getting Started", href="#getting-started", className="toc-item"),
                    html.A("2. Upload Data", href="#upload-data", className="toc-item"),
                    html.A("3. Data Cleaning", href="#data-cleaning", className="toc-item"),
                    html.A("4. Data Visualization", href="#data-visualization", className="toc-item"),
                    html.A("5. Statistical Tests", href="#statistical-tests", className="toc-item"),
                    html.A("6. Basic Machine Learning", href="#basic-ml", className="toc-item"),
                    html.A("7. Tips & Best Practices", href="#tips", className="toc-item")
                ], className="toc-grid")
            ], className="table-of-contents"),
            
            # Getting Started Section
            html.Div([
                html.H2("1. Getting Started", id="getting-started", className="section-title"),
                html.P("This dashboard provides a complete workflow for data analysis:", className="section-intro"),
                html.Div([
                    html.Div([
                        html.Div("1", className="step-number"),
                        html.Div("Upload your dataset", className="step-text")
                    ], className="process-step"),
                    html.Div([
                        html.Div("2", className="step-number"),
                        html.Div("Clean and preprocess your data", className="step-text")
                    ], className="process-step"),
                    html.Div([
                        html.Div("3", className="step-number"),
                        html.Div("Create visualizations", className="step-text")
                    ], className="process-step"),
                    html.Div([
                        html.Div("4", className="step-number"),
                        html.Div("Perform statistical tests", className="step-text")
                    ], className="process-step"),
                    html.Div([
                        html.Div("5", className="step-number"),
                        html.Div("Build and evaluate machine learning models", className="step-text")
                    ], className="process-step"),
                ], className="process-steps"),
                html.P([
                    "Navigate through the application using the sidebar menu. Each section builds on the previous one, ",
                    "so it's recommended to follow the workflow in order."
                ], className="guide-text"),
            ], className="guide-section"),
            
            # Upload Data Section
            html.Div([
                html.H2("2. Upload Data", id="upload-data", className="section-title"),
                html.P("The first step is to upload your dataset:", className="section-intro"),
                html.Div([
                    html.Div([
                        html.Strong("Select File Format", className="feature-title"), 
                        html.P("Choose CSV or Excel format from the dropdown.", className="feature-text")
                    ], className="feature-box"),
                    html.Div([
                        html.Strong("Upload File", className="feature-title"), 
                        html.P("Click 'Select Files' and choose your data file.", className="feature-text")
                    ], className="feature-box"),
                    html.Div([
                        html.Strong("Configure Options", className="feature-title"), 
                        html.P("Set delimiter (for CSV), sheet name (for Excel), and whether the first row contains headers.", className="feature-text")
                    ], className="feature-box"),
                    html.Div([
                        html.Strong("Preview Data", className="feature-title"), 
                        html.P("Review the data preview to ensure correct importing.", className="feature-text")
                    ], className="feature-box"),
                ], className="feature-grid"),
                html.H4("Supported Formats", className="subsection-title"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-file-csv", style={"fontSize": "24px", "color": "#3498db"}),
                        html.P("CSV (Comma Separated Values)", className="format-text")
                    ], className="format-item"),
                    html.Div([
                        html.I(className="fas fa-file-excel", style={"fontSize": "24px", "color": "#27ae60"}),
                        html.P("Excel (.xlsx, .xls)", className="format-text")
                    ], className="format-item"),
                ], className="format-container"),
                html.P([
                    "After uploading, your data will be stored in the application's memory and available ",
                    "for all subsequent steps."
                ], className="guide-text"),
            ], className="guide-section"),
            
            # Data Cleaning Section
            html.Div([
                html.H2("3. Data Cleaning", id="data-cleaning", className="section-title"),
                html.P("Clean and preprocess your data to prepare it for analysis:", className="section-intro"),
                html.H4("Available Cleaning Operations", className="subsection-title"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-eraser", style={"color": "#e74c3c", "fontSize": "24px"}),
                        html.H5("Missing Values", className="card-title"),
                        html.P("View a summary of missing values and handle them using different strategies (drop rows/columns, replace with mean/median/mode/custom value).", className="card-text")
                    ], className="info-card"),
                    html.Div([
                        html.I(className="fas fa-filter", style={"color": "#f39c12", "fontSize": "24px"}),
                        html.H5("Outliers", className="card-title"),
                        html.P("Detect and handle outliers using the Interquartile Range (IQR) method for numerical columns.", className="card-text")
                    ], className="info-card"),
                    html.Div([
                        html.I(className="fas fa-exchange-alt", style={"color": "#3498db", "fontSize": "24px"}),
                        html.H5("Data Type Conversion", className="card-title"),
                        html.P("Convert columns between numerical, categorical, and datetime types.", className="card-text")
                    ], className="info-card"),
                ], className="info-card-container"),
                html.H4("Steps to Clean Data", className="subsection-title"),
                html.Ol([
                    html.Li("Select the cleaning operation from the dropdown", className="steps-li"),
                    html.Li("Configure the specific parameters for the selected operation", className="steps-li"),
                    html.Li("Apply the changes", className="steps-li"),
                    html.Li("Review the cleaning summary to see what changes were made", className="steps-li"),
                    html.Li("Optionally, export the cleaned dataset", className="steps-li")
                ], className="numbered-steps"),
                html.P([
                    "Each cleaning step will update the dataset for subsequent operations. ",
                    "You can perform multiple cleaning operations in sequence."
                ], className="guide-text")
            ], className="guide-section"),
            
            # Data Visualization Section
            html.Div([
                html.H2("4. Data Visualization", id="data-visualization", className="section-title"),
                html.P("Create insightful visualizations to understand your data:", className="section-intro"),
                html.H4("Exploratory Data Analysis (EDA)", className="subsection-title"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-bar", style={"color": "#3498db", "fontSize": "24px"}),
                        html.Strong("Basic Statistics", className="eda-title"), 
                        html.P("View summary statistics, data types, missing values, and unique value counts.", className="eda-text")
                    ], className="eda-item"),
                    html.Div([
                        html.I(className="fas fa-th", style={"color": "#9b59b6", "fontSize": "24px"}),
                        html.Strong("Correlation Matrix", className="eda-title"), 
                        html.P("Visualize correlations between numerical variables.", className="eda-text")
                    ], className="eda-item"),
                ], className="eda-container"),
                html.H4("Visualization Types", className="subsection-title"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-braille", style={"fontSize": "24px"}),
                        html.Strong("Scatter Plot", className="vis-title"),
                        html.P("Visualize relationships between two numerical variables, with optional trend lines and correlation analysis.", className="vis-text")
                    ], className="vis-card"),
                    html.Div([
                        html.I(className="fas fa-chart-bar", style={"fontSize": "24px"}),
                        html.Strong("Bar Chart", className="vis-title"),
                        html.P("Compare categories with interactive filtering capabilities.", className="vis-text")
                    ], className="vis-card"),
                    html.Div([
                        html.I(className="fas fa-chart-line", style={"fontSize": "24px"}),
                        html.Strong("Line Chart", className="vis-title"),
                        html.P("Show trends over a continuous variable.", className="vis-text")
                    ], className="vis-card"),
                    html.Div([
                        html.I(className="fas fa-signal", style={"fontSize": "24px"}),
                        html.Strong("Histogram", className="vis-title"),
                        html.P("View the distribution of a numerical variable.", className="vis-text")
                    ], className="vis-card"),
                    html.Div([
                        html.I(className="fas fa-box", style={"fontSize": "24px"}),
                        html.Strong("Box Plot", className="vis-title"),
                        html.P("Compare distributions across categories and identify outliers.", className="vis-text")
                    ], className="vis-card"),
                    html.Div([
                        html.I(className="fas fa-th", style={"fontSize": "24px"}),
                        html.Strong("Heatmap", className="vis-title"),
                        html.P("Visualize relationships between categorical variables or correlations.", className="vis-text")
                    ], className="vis-card"),
                ], className="vis-card-grid"),
                html.H4("Creating Visualizations", className="subsection-title"),
                html.Ol([
                    html.Li("Select a visualization type from the dropdown"),
                    html.Li("Choose the appropriate columns for the selected visualization"),
                    html.Li("Configure additional options (color mappings, regression lines for scatter plots, etc.)"),
                    html.Li("Click 'Generate Plot' to create the visualization"),
                    html.Li("Use the export options to download the chart in different formats")
                ], className="numbered-steps"),
                html.Div([
                    html.I(className="fas fa-info-circle", style={"color": "#3498db", "fontSize": "18px", "marginRight": "10px"}),
                    html.P("Interactive visualizations allow you to hover over data points, zoom in/out, and pan around the chart.")
                ], className="info-note")
            ], className="guide-section"),
            
            # Statistical Tests Section
            html.Div([
                html.H2("5. Statistical Tests", id="statistical-tests", className="section-title"),
                html.P("Perform statistical analyses to test hypotheses about your data:", className="section-intro"),
                html.H4("Available Tests", className="subsection-title"),
                html.Div([
                    html.Div([
                        html.H5("Linear Regression", className="test-title"),
                        html.P("Test the significance of linear relationships between variables. Assesses how well one or more independent variables predict a dependent variable.", className="test-desc")
                    ], className="test-box"),
                    html.Div([
                        html.H5("Chi-Square Test", className="test-title"),
                        html.P("Test independence between categorical variables. Determines if there's a significant association between two categorical variables.", className="test-desc")
                    ], className="test-box"),
                    html.Div([
                        html.H5("ANOVA Test", className="test-title"),
                        html.P("Compare means across multiple groups. Evaluates if there are significant differences in means across different categories.", className="test-desc")
                    ], className="test-box"),
                    html.Div([
                        html.H5("Correlation Analysis", className="test-title"),
                        html.P("Measure the strength and direction of relationships between variables using Pearson (linear) or Spearman (monotonic) methods.", className="test-desc")
                    ], className="test-box"),
                ], className="test-grid"),
                html.H4("Running a Statistical Test", className="subsection-title"),
                html.Ol([
                    html.Li("Select the test type from the dropdown"),
                    html.Li("Configure the test parameters (variables, significance level)"),
                    html.Li("Click 'Run Statistical Test'"),
                    html.Li("Review the results, including test statistics, p-values, and visualizations"),
                    html.Li("Interpret the findings using the provided explanations")
                ], className="numbered-steps"),
                html.P([
                    "Each test includes a detailed interpretation section to help you understand the results ",
                    "and their implications for your data."
                ], className="guide-text"),
                html.H4("Understanding Statistical Significance", className="subsection-title"),
                html.Div([
                    html.P([
                        "A result is typically considered statistically significant when the p-value is less than ",
                        "the chosen significance level (α), usually 0.05. This indicates strong evidence against ",
                        "the null hypothesis."
                    ], className="guide-text"),
                    html.Div([
                        html.P("p < 0.05", className="sig-level"),
                        html.P("Significant", className="sig-text")
                    ], className="sig-box sig-yes"),
                    html.Div([
                        html.P("p ≥ 0.05", className="sig-level"),
                        html.P("Not Significant", className="sig-text")
                    ], className="sig-box sig-no"),
                ], className="significance-container")
            ], className="guide-section"),
            
            # Basic ML Section
            html.Div([
                html.H2("6. Basic Machine Learning", id="basic-ml", className="section-title"),
                html.P("Apply machine learning algorithms to make predictions or classify data:", className="section-intro"),
                html.H4("Available Models", className="subsection-title"),
                html.Div([
                    html.Div([
                        html.Strong("Linear Regression", className="model-title"), 
                        html.P("For predicting numerical values. Simple and interpretable model for regression tasks.", className="model-desc")
                    ], className="model-card"),
                    html.Div([
                        html.Strong("Logistic Regression", className="model-title"), 
                        html.P("For classification tasks. Extension of linear regression for categorical outcomes.", className="model-desc")
                    ], className="model-card"),
                    html.Div([
                        html.Strong("K-Nearest Neighbors", className="model-title"), 
                        html.P("For classification or regression. Makes predictions based on similar training examples.", className="model-desc")
                    ], className="model-card"),
                    html.Div([
                        html.Strong("Random Forest", className="model-title"), 
                        html.P("For classification or regression. Ensemble method that builds multiple decision trees and merges their predictions.", className="model-desc")
                    ], className="model-card"),
                ], className="model-grid"),
                html.H4("Building a Model", className="subsection-title"),
                html.Ol([
                    html.Li("Select the model type from the dropdown"),
                    html.Li("Choose the task type (classification or regression)"),
                    html.Li("Configure model-specific parameters"),
                    html.Li("Select features (input variables) and target (output variable)"),
                    html.Li("Set the test set size for evaluation"),
                    html.Li("Click 'Train Model'"),
                    html.Li("Review performance metrics and visualizations"),
                    html.Li("Optionally, download the trained model for future use")
                ], className="numbered-steps"),
                html.H4("Understanding Model Performance", className="subsection-title"),
                html.Div([
                    html.Div([
                        html.H5("For Regression Models:", className="metrics-title"),
                        html.Ul([
                            html.Li([
                                html.Strong("R² Score: "), 
                                "The proportion of variance explained by the model (higher is better)."
                            ]),
                            html.Li([
                                html.Strong("Root Mean Squared Error (RMSE): "), 
                                "Average prediction error in the same units as the target variable (lower is better)."
                            ]),
                            html.Li([
                                html.Strong("Mean Absolute Error (MAE): "), 
                                "Average absolute prediction error (lower is better)."
                            ])
                        ], className="metrics-list")
                    ], className="metrics-box"),
                    html.Div([
                        html.H5("For Classification Models:", className="metrics-title"),
                        html.Ul([
                            html.Li([
                                html.Strong("Accuracy: "), 
                                "Proportion of correct predictions."
                            ]),
                            html.Li([
                                html.Strong("Precision: "), 
                                "Proportion of positive identifications that were actually correct."
                            ]),
                            html.Li([
                                html.Strong("Recall: "), 
                                "Proportion of actual positives that were correctly identified."
                            ]),
                            html.Li([
                                html.Strong("F1 Score: "), 
                                "Harmonic mean of precision and recall."
                            ])
                        ], className="metrics-list")
                    ], className="metrics-box"),
                ], className="metrics-container")
            ], className="guide-section"),
            
            # Tips Section
            html.Div([
                html.H2("7. Tips & Best Practices", id="tips", className="section-title"),
                html.P("Optimize your data analysis workflow with these key recommendations:", className="section-intro"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-check-circle", style={"color": "#27ae60", "fontSize": "24px"}),
                        html.Div([
                            html.Strong("Data Quality", className="tip-title"), 
                            html.P("Always clean your data thoroughly before analysis. Missing values and outliers can significantly impact results.", className="tip-text")
                        ], className="tip-content")
                    ], className="tip-item"),
                    html.Div([
                        html.I(className="fas fa-chart-pie", style={"color": "#3498db", "fontSize": "24px"}),
                        html.Div([
                            html.Strong("Exploratory Analysis", className="tip-title"), 
                            html.P("Start with exploratory visualizations to understand your data before jumping into statistical tests or machine learning.", className="tip-text")
                        ], className="tip-content")
                    ], className="tip-item"),
                    html.Div([
                        html.I(className="fas fa-percentage", style={"color": "#e74c3c", "fontSize": "24px"}),
                        html.Div([
                            html.Strong("Statistical Significance", className="tip-title"), 
                            html.P("Remember that statistical significance (p < 0.05) doesn't always imply practical significance.", className="tip-text")
                        ], className="tip-content")
                    ], className="tip-item"),
                    html.Div([
                        html.I(className="fas fa-table", style={"color": "#f39c12", "fontSize": "24px"}),
                        html.Div([
                            html.Strong("Feature Selection", className="tip-title"), 
                            html.P("For machine learning, choose features carefully. More features aren't always better.", className="tip-text")
                        ], className="tip-content")
                    ], className="tip-item"),
                    html.Div([
                        html.I(className="fas fa-tasks", style={"color": "#9b59b6", "fontSize": "24px"}),
                        html.Div([
                            html.Strong("Model Evaluation", className="tip-title"), 
                            html.P("Always evaluate models on test data not used during training to get a realistic assessment of performance.", className="tip-text")
                        ], className="tip-content")
                    ], className="tip-item"),
                    html.Div([
                        html.I(className="fas fa-random", style={"color": "#16a085", "fontSize": "24px"}),
                        html.Div([
                            html.Strong("Test Multiple Models", className="tip-title"), 
                            html.P("Try different algorithms and compare their performance for your specific dataset and task.", className="tip-text")
                        ], className="tip-content")
                    ], className="tip-item"),
                    html.Div([
                        html.I(className="fas fa-search", style={"color": "#2c3e50", "fontSize": "24px"}),
                        html.Div([
                            html.Strong("Interpret Results Carefully", className="tip-title"), 
                            html.P("Consider the context of your data and the limitations of the methods you're using.", className="tip-text")
                        ], className="tip-content")
                    ], className="tip-item"),
                ], className="tips-grid")
            ], className="guide-section"),
        ], className="guide-container")
    ], className="page-content")
