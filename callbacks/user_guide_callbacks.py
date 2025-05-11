from dash.dependencies import Input, Output, State
import base64
import io
import json
from dash import html, dcc

def register_user_guide_callbacks(app):
    @app.callback(
        Output("download-guide-pdf", "data"),
        [Input("btn-download-guide", "n_clicks"),
         Input("btn-download-guide-bottom", "n_clicks")],
        prevent_initial_call=True
    )
    def download_guide(n_clicks_top, n_clicks_bottom):
        if (n_clicks_top is None or n_clicks_top == 0) and (n_clicks_bottom is None or n_clicks_bottom == 0):
            return None
        
        # HTML file with improved styling
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Statistical Dashboard User Guide</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Roboto:wght@300;400&display=swap');
                
                body {
                    font-family: 'Roboto', sans-serif;
                    line-height: 1.8;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                    background-color: #f9f9f9;
                }
                
                .container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                }
                
                h1 {
                    font-family: 'Poppins', sans-serif;
                    color: #2c3e50;
                    text-align: center;
                    font-weight: 700;
                    font-size: 28px;
                    margin-bottom: 30px;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #3498db;
                }
                
                h2 {
                    font-family: 'Poppins', sans-serif;
                    color: #3498db;
                    margin-top: 30px;
                    font-weight: 600;
                    font-size: 22px;
                    padding-bottom: 8px;
                    border-bottom: 1px solid #eee;
                }
                
                h3 {
                    font-family: 'Poppins', sans-serif;
                    color: #2980b9;
                    font-weight: 600;
                    font-size: 18px;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }
                
                .section {
                    margin-bottom: 30px;
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }
                
                ul, ol {
                    margin-left: 30px;
                    margin-bottom: 20px;
                }
                
                li {
                    margin-bottom: 10px;
                }
                
                .emphasis {
                    font-weight: bold;
                    color: #2c3e50;
                }
                
                .intro-text {
                    font-size: 16px;
                    text-align: center;
                    margin-bottom: 30px;
                    color: #555;
                }
                
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    font-size: 14px;
                    color: #7f8c8d;
                    border-top: 1px solid #eee;
                    padding-top: 20px;
                }
                
                /* Print styles */
                @media print {
                    body {
                        background-color: white;
                    }
                    
                    .container {
                        box-shadow: none;
                        padding: 0;
                    }
                    
                    .section {
                        page-break-inside: avoid;
                        border-left: none;
                        padding: 10px 0;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Statistical Dashboard User Guide</h1>
                <p class="intro-text">Learn how to make the most of this statistical dashboard with our comprehensive user guide.</p>
                
                <div class="section">
                    <h2>1. Getting Started</h2>
                    <p>This dashboard provides a complete workflow for data analysis:</p>
                    <ol>
                        <li>Upload your dataset</li>
                        <li>Clean and preprocess your data</li>
                        <li>Create visualizations</li>
                        <li>Perform statistical tests</li>
                        <li>Build and evaluate machine learning models</li>
                    </ol>
                    <p>Navigate through the application using the sidebar menu. Each section builds on the previous one, so it's recommended to follow the workflow in order.</p>
                </div>
                
                <div class="section">
                    <h2>2. Upload Data</h2>
                    <p>The first step is to upload your dataset:</p>
                    <ol>
                        <li><span class="emphasis">Select File Format:</span> Choose CSV or Excel format from the dropdown.</li>
                        <li><span class="emphasis">Upload File:</span> Click 'Select Files' and choose your data file.</li>
                        <li><span class="emphasis">Configure Options:</span> Set delimiter (for CSV), sheet name (for Excel), and whether the first row contains headers.</li>
                        <li><span class="emphasis">Preview Data:</span> Review the data preview to ensure correct importing.</li>
                    </ol>
                    
                    <h3>Supported Formats:</h3>
                    <ul>
                        <li>CSV (Comma Separated Values)</li>
                        <li>Excel (.xlsx, .xls)</li>
                    </ul>
                    <p>After uploading, your data will be stored in the application's memory and available for all subsequent steps.</p>
                </div>
                
                <div class="section">
                    <h2>3. Data Cleaning</h2>
                    <p>Clean and preprocess your data to prepare it for analysis:</p>
                    
                    <h3>Available Cleaning Operations:</h3>
                    <ul>
                        <li><span class="emphasis">Missing Values:</span> View a summary of missing values and handle them using different strategies (drop rows/columns, replace with mean/median/mode/custom value).</li>
                        <li><span class="emphasis">Outliers:</span> Detect and handle outliers using the Interquartile Range (IQR) method for numerical columns.</li>
                        <li><span class="emphasis">Data Type Conversion:</span> Convert columns between numerical, categorical, and datetime types.</li>
                    </ul>
                    
                    <h3>Steps to Clean Data:</h3>
                    <ol>
                        <li>Select the cleaning operation from the dropdown</li>
                        <li>Configure the specific parameters for the selected operation</li>
                        <li>Apply the changes</li>
                        <li>Review the cleaning summary to see what changes were made</li>
                        <li>Optionally, export the cleaned dataset</li>
                    </ol>
                    <p>Each cleaning step will update the dataset for subsequent operations. You can perform multiple cleaning operations in sequence.</p>
                </div>
                
                <div class="section">
                    <h2>4. Data Visualization</h2>
                    <p>Create insightful visualizations to understand your data:</p>
                    
                    <h3>Exploratory Data Analysis (EDA):</h3>
                    <ul>
                        <li><span class="emphasis">Basic Statistics:</span> View summary statistics, data types, missing values, and unique value counts.</li>
                        <li><span class="emphasis">Correlation Matrix:</span> Visualize correlations between numerical variables.</li>
                    </ul>
                    
                    <h3>Visualization Types:</h3>
                    <ul>
                        <li><span class="emphasis">Scatter Plot:</span> Visualize relationships between two numerical variables, with optional trend lines and correlation analysis.</li>
                        <li><span class="emphasis">Bar Chart:</span> Compare categories with interactive filtering capabilities.</li>
                        <li><span class="emphasis">Line Chart:</span> Show trends over a continuous variable.</li>
                        <li><span class="emphasis">Histogram:</span> View the distribution of a numerical variable.</li>
                        <li><span class="emphasis">Box Plot:</span> Compare distributions across categories and identify outliers.</li>
                        <li><span class="emphasis">Heatmap:</span> Visualize relationships between categorical variables or correlations.</li>
                    </ul>
                    
                    <h3>Creating Visualizations:</h3>
                    <ol>
                        <li>Select a visualization type from the dropdown</li>
                        <li>Choose the appropriate columns for the selected visualization</li>
                        <li>Configure additional options (color mappings, regression lines for scatter plots, etc.)</li>
                        <li>Click 'Generate Plot' to create the visualization</li>
                        <li>Use the export options to download the chart in different formats</li>
                    </ol>
                </div>
                
                <div class="section">
                    <h2>5. Statistical Tests</h2>
                    <p>Perform statistical analyses to test hypotheses about your data:</p>
                    
                    <h3>Available Tests:</h3>
                    <ul>
                        <li><span class="emphasis">Linear Regression:</span> Test the significance of linear relationships between variables.</li>
                        <li><span class="emphasis">Chi-Square Test:</span> Test independence between categorical variables.</li>
                        <li><span class="emphasis">ANOVA Test:</span> Compare means across multiple groups.</li>
                        <li><span class="emphasis">Correlation Analysis:</span> Measure the strength and direction of relationships between variables.</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>6. Basic Machine Learning</h2>
                    <p>Apply machine learning algorithms to make predictions or classify data:</p>
                    
                    <h3>Available Models:</h3>
                    <ul>
                        <li><span class="emphasis">Linear Regression:</span> For predicting numerical values.</li>
                        <li><span class="emphasis">Logistic Regression:</span> For classification tasks.</li>
                        <li><span class="emphasis">K-Nearest Neighbors:</span> For classification or regression.</li>
                        <li><span class="emphasis">Random Forest:</span> For classification or regression.</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>7. Tips & Best Practices</h2>
                    <ul>
                        <li><span class="emphasis">Data Quality:</span> Always clean your data thoroughly before analysis.</li>
                        <li><span class="emphasis">Exploratory Analysis:</span> Start with visualizations to understand your data before jumping into tests or ML.</li>
                        <li><span class="emphasis">Statistical Significance:</span> Remember that statistical significance doesn't always imply practical significance.</li>
                        <li><span class="emphasis">Feature Selection:</span> For machine learning, choose features carefully.</li>
                        <li><span class="emphasis">Model Evaluation:</span> Always evaluate models on test data not used during training.</li>
                    </ul>
                </div>
                
                <div class="footer">
                    <p>Statistical Dashboard &copy; 2023 | For questions and support, please contact the development team</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Convert the HTML content to bytes and return it for download
        return {
            "content": html_content,
            "filename": "statistical_dashboard_user_guide.html",
            "type": "text/html",
        }
        
    # Also add a notification when the guide is downloaded
    @app.callback(
        Output("notification-container", "children"),
        [Input("btn-download-guide", "n_clicks"),
         Input("btn-download-guide-bottom", "n_clicks")],
        prevent_initial_call=True
    )
    def show_download_notification(n_clicks_top, n_clicks_bottom):
        if (n_clicks_top is None or n_clicks_top == 0) and (n_clicks_bottom is None or n_clicks_bottom == 0):
            return None
        
        return html.Div(
            "User guide downloaded successfully!",
            className="notification success-notification",
            id="download-notification"
        ) 