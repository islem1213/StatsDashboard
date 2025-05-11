from dash import html, dcc

def layout():
    return html.Div([
        html.H1("Upload Data"),
        html.P("Upload your dataset for analysis."),
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Select Files', className="button-34"),
                className="upload-component",
                multiple=False
            ),
        ], className="upload-section"),
        
        # Data preview section
        html.Div([
            html.Div(id="data-preview-container", className="preview-container"),
            html.Div(id="data-summary-container", className="summary-container")
        ], className="data-preview-section"),
        
        # Dataset information
        html.Div([
            html.Div(id="dataset-info", className="dataset-info")
        ], className="info-section")
    ], className="page-content")
