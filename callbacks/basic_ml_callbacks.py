from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import pickle
import base64
import io
from dash import html, dcc, dash_table
import dash

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

def register_basic_ml_callbacks(app):
    # Update feature and target selector dropdown options
    @app.callback(
        [
            Output("feature-selector", "options"),
            Output("target-selector", "options")
        ],
        [Input("cleaned-dataset-store", "data"), Input("dataset-store", "data")]
    )
    def update_column_options(cleaned_dataset_json, dataset_json):
        # Prioritize cleaned data if available
        if cleaned_dataset_json is not None:
            df = pd.read_json(cleaned_dataset_json, orient='split')
        elif dataset_json is not None:
            df = pd.read_json(dataset_json, orient='split')
        else:
            raise PreventUpdate
            
        columns = [{"label": col, "value": col} for col in df.columns]
        
        return columns, columns

    # Show model-specific configuration options
    @app.callback(
        Output("model-config-section", "children"),
        [
            Input("model-type-dropdown", "value"),
            Input("task-type-radio", "value")
        ]
    )
    def update_model_config(model_type, task_type):
        if not model_type or not task_type:
            return []
        
        # Different parameters based on model type and task
        if model_type == "linear_regression":
            if task_type == "regression":
                return html.Div([
                    html.H4("Linear Regression Configuration"),
                    html.P("A simple model that fits a linear relationship between features and target."),
                    
                    html.Div([
                        html.Label("Fit Intercept:"),
                        dcc.RadioItems(
                            id="fit-intercept-radio",
                            options=[
                                {"label": "Yes", "value": "yes"},
                                {"label": "No", "value": "no"}
                            ],
                            value="yes",
                            labelStyle={"display": "inline-block", "marginRight": "20px"}
                        )
                    ], className="config-item")
                ])
            else:  # classification
                return html.Div([
                    html.H4("Logistic Regression Configuration"),
                    html.P("A linear model for classification tasks."),
                    
                    html.Div([
                        html.Label("Regularization Strength (C):"),
                        dcc.Slider(
                            id="logreg-c-slider",
                            min=-3,
                            max=3,
                            step=0.5,
                            value=0,
                            marks={i: f"{10**i:.3f}" for i in range(-3, 4)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="config-item"),
                    
                    html.Div([
                        html.Label("Maximum Iterations:"),
                        dcc.Slider(
                            id="logreg-iter-slider",
                            min=100,
                            max=1000,
                            step=100,
                            value=100,
                            marks={i: str(i) for i in range(100, 1001, 100)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="config-item")
                ])
                
        elif model_type == "knn":
            return html.Div([
                html.H4(f"K-Nearest Neighbors Configuration ({task_type.title()})"),
                html.P("A non-parametric method that uses the k nearest samples for prediction."),
                
                html.Div([
                    html.Label("Number of Neighbors (K):"),
                    dcc.Slider(
                        id="knn-n-neighbors-slider",
                        min=1,
                        max=20,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(1, 21, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="config-item"),
                
                html.Div([
                    html.Label("Weight Function:"),
                    dcc.RadioItems(
                        id="knn-weights-radio",
                        options=[
                            {"label": "Uniform (all points weighted equally)", "value": "uniform"},
                            {"label": "Distance (closer neighbors have greater influence)", "value": "distance"}
                        ],
                        value="uniform",
                        labelStyle={"display": "block", "marginBottom": "5px"}
                    )
                ], className="config-item")
            ])
            
        elif model_type == "random_forest":
            return html.Div([
                html.H4(f"Random Forest Configuration ({task_type.title()})"),
                html.P("An ensemble of decision trees that improves accuracy and controls overfitting."),
                
                html.Div([
                    html.Label("Number of Trees:"),
                    dcc.Slider(
                        id="rf-n-estimators-slider",
                        min=10,
                        max=200,
                        step=10,
                        value=100,
                        marks={i: str(i) for i in range(10, 201, 30)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="config-item"),
                
                html.Div([
                    html.Label("Maximum Depth:"),
                    dcc.Slider(
                        id="rf-max-depth-slider",
                        min=2,
                        max=20,
                        step=1,
                        value=None,
                        marks={i: str(i) for i in range(2, 21, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="config-item"),
                
                html.Div([
                    html.Label("Minimum Samples for Split:"),
                    dcc.Slider(
                        id="rf-min-samples-split-slider",
                        min=2,
                        max=10,
                        step=1,
                        value=2,
                        marks={i: str(i) for i in range(2, 11, 1)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="config-item")
            ])
            
        return html.Div("Please select a model type")
    
    # Train model and display results
    @app.callback(
        [
            Output("model-results-container", "children"),
            Output("trained-model-store", "data"),
            Output("btn-download-model", "disabled")
        ],
        [Input("btn-train-model", "n_clicks")],
        [
            State("cleaned-dataset-store", "data"),
            State("dataset-store", "data"),
            State("model-type-dropdown", "value"),
            State("task-type-radio", "value"),
            State("feature-selector", "value"),
            State("target-selector", "value"),
            State("test-size-slider", "value"),
            # Linear/Logistic Regression params
            State("fit-intercept-radio", "value"),
            State("logreg-c-slider", "value"),
            State("logreg-iter-slider", "value"),
            # KNN params
            State("knn-n-neighbors-slider", "value"),
            State("knn-weights-radio", "value"),
            # Random Forest params
            State("rf-n-estimators-slider", "value"),
            State("rf-max-depth-slider", "value"),
            State("rf-min-samples-split-slider", "value")
        ]
    )
    def train_model(n_clicks, cleaned_dataset_json, dataset_json, model_type, task_type, features, target, 
                   test_size, fit_intercept, logreg_c, logreg_iter, knn_n_neighbors, 
                   knn_weights, rf_n_estimators, rf_max_depth, rf_min_samples_split):
        # Validate inputs
        if n_clicks is None or not n_clicks:
            raise PreventUpdate
            
        if not model_type or not task_type or not features or not target:
            return html.Div("Please select model type, task type, features, and target", className="error-message"), None, True
            
        if target in features:
            features.remove(target)
            
        if not features:
            return html.Div("Please select at least one feature different from target", className="error-message"), None, True
            
        # Load dataframe, prioritize cleaned data if available
        if cleaned_dataset_json is not None:
            df = pd.read_json(cleaned_dataset_json, orient='split')
        elif dataset_json is not None:
            df = pd.read_json(dataset_json, orient='split')
        else:
            return html.Div("Please upload and clean data first", className="error-message"), None, True
        
        # Prepare data
        X = df[features].copy()
        y = df[target].copy()
        
        # Check for sufficient data
        if len(X) < 10:  # Arbitrary minimum
            return html.Div("Not enough data points for training a model. Need at least 10 rows.", className="error-message"), None, True
        
        # Check if target column is appropriate for the task type
        if task_type == "classification":
            # For classification, check if target has reasonable number of classes
            n_classes = len(y.unique())
            if n_classes < 2:
                return html.Div("Classification requires at least 2 classes in target variable", className="error-message"), None, True
            if n_classes > 20:  # Arbitrary high number that might indicate a non-categorical target
                return html.Div(f"Target variable has {n_classes} unique values. For classification, target should be categorical with fewer classes.", className="error-message"), None, True
        elif task_type == "regression":
            # For regression, check if target is numeric
            if not pd.api.types.is_numeric_dtype(y):
                return html.Div("Regression requires a numeric target variable", className="error-message"), None, True
        
        # Handle missing values (simple imputation)
        missing_values_before = X.isna().sum().sum() + y.isna().sum()
        if missing_values_before > 0:
            for col in X.columns:
                if X[col].dtype.kind in 'ifc':  # numeric
                    X[col] = X[col].fillna(X[col].mean())
                else:  # categorical
                    X[col] = X[col].fillna(X[col].mode()[0])
            
            # Remove rows with missing target values
            valid_mask = ~y.isna()
            if valid_mask.sum() < 10:  # Too few valid samples after removing rows with missing target
                return html.Div("Too many missing values in target column. Cannot train a model.", className="error-message"), None, True
            
            X = X[valid_mask]
            y = y[valid_mask]
        
        # For classification, encode target if needed
        label_encoder = None
        if task_type == "classification" and y.dtype == "object":
            try:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
            except Exception as e:
                return html.Div([
                    html.H4("Error Encoding Target", style={"color": "red"}),
                    html.P(f"Failed to encode target variable: {str(e)}"),
                    html.P("Make sure your target variable is properly formatted for classification.")
                ], className="error-message"), None, True
        
        # Convert categorical features to one-hot encoding
        try:
            # Check if there are categorical columns
            cat_columns = X.select_dtypes(include=['object', 'category']).columns
            if len(cat_columns) > 0:
                # Check for high cardinality categorical features
                for col in cat_columns:
                    if len(X[col].unique()) > 100:  # Arbitrary threshold for high cardinality
                        return html.Div([
                            html.H4("Warning: High Cardinality Feature", style={"color": "orange"}),
                            html.P(f"Feature '{col}' has {len(X[col].unique())} unique values, which may cause issues with one-hot encoding."),
                            html.P("Consider using a different encoding method or dropping this feature.")
                        ], className="warning-message"), None, True
                
            # Apply one-hot encoding
            X = pd.get_dummies(X, drop_first=True)
            
            # Check if one-hot encoding resulted in too many features
            if X.shape[1] > 1000:  # Arbitrary threshold for too many features
                return html.Div([
                    html.H4("Error: Too Many Features", style={"color": "red"}),
                    html.P(f"One-hot encoding resulted in {X.shape[1]} features, which is too many for efficient model training."),
                    html.P("Consider feature selection or dimensionality reduction techniques.")
                ], className="error-message"), None, True
                
        except Exception as e:
            return html.Div([
                html.H4("Error in Feature Encoding", style={"color": "red"}),
                html.P(f"Failed to encode categorical features: {str(e)}"),
                html.P("Check your data for unusual values or try with different features.")
            ], className="error-message"), None, True
        
        # Split data
        try:
            test_size_fraction = test_size / 100  # Convert percentage to fraction
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_fraction, random_state=42
            )
            
            # Check if we have enough samples in each set
            if len(X_train) < 5 or len(X_test) < 5:
                return html.Div("Not enough samples after train-test split. Try with more data or a different split ratio.", className="error-message"), None, True
                
            # For classification, check if all classes are represented in both sets
            if task_type == "classification":
                train_classes = np.unique(y_train)
                test_classes = np.unique(y_test)
                if len(train_classes) < len(np.unique(y)) or len(test_classes) < len(np.unique(y)):
                    return html.Div([
                        html.H4("Warning: Imbalanced Split", style={"color": "orange"}),
                        html.P("Some classes are not represented in both training and test sets."),
                        html.P("Try using stratified sampling or a different random seed.")
                    ], className="warning-message"), None, True
                    
        except Exception as e:
            return html.Div([
                html.H4("Error Splitting Data", style={"color": "red"}),
                html.P(f"Failed to split data: {str(e)}")
            ], className="error-message"), None, True
        
        # Feature scaling for certain algorithms
        scaler = None
        try:
            if model_type in ["linear_regression", "knn"]:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
        except Exception as e:
            return html.Div([
                html.H4("Error in Feature Scaling", style={"color": "red"}),
                html.P(f"Failed to scale features: {str(e)}"),
                html.P("This may be due to non-numeric features or extreme outliers.")
            ], className="error-message"), None, True
        
        # Initialize and train model
        model = None
        try:
            # Check if there are enough samples for the model type
            if model_type == "random_forest" and len(X_train) < 20:
                return html.Div("Random Forest models typically need more training samples. Try with more data or a simpler model.", className="error-message"), None, True
                
            # Initialize model based on type and task
            if model_type == "linear_regression":
                if task_type == "regression":
                    model = LinearRegression(fit_intercept=(fit_intercept == "yes"))
                else:  # classification
                    model = LogisticRegression(
                        C=10**logreg_c, 
                        max_iter=int(logreg_iter),
                        random_state=42,
                        solver='liblinear' if X_train.shape[1] > X_train.shape[0] else 'lbfgs'  # Choose solver based on data dimensions
                    )
            elif model_type == "knn":
                if task_type == "regression":
                    model = KNeighborsRegressor(
                        n_neighbors=min(int(knn_n_neighbors), len(X_train)),  # Ensure n_neighbors doesn't exceed sample size
                        weights=knn_weights
                    )
                else:  # classification
                    model = KNeighborsClassifier(
                        n_neighbors=min(int(knn_n_neighbors), len(X_train)),  # Ensure n_neighbors doesn't exceed sample size
                        weights=knn_weights
                    )
            elif model_type == "random_forest":
                if task_type == "regression":
                    model = RandomForestRegressor(
                        n_estimators=int(rf_n_estimators),
                        max_depth=int(rf_max_depth) if rf_max_depth else None,
                        min_samples_split=int(rf_min_samples_split),
                        random_state=42
                    )
                else:  # classification
                    model = RandomForestClassifier(
                        n_estimators=int(rf_n_estimators),
                        max_depth=int(rf_max_depth) if rf_max_depth else None,
                        min_samples_split=int(rf_min_samples_split),
                        random_state=42,
                        class_weight='balanced' if len(np.unique(y_train)) <= 10 else None  # Use class weights for small number of classes
                    )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Check predictions
            if np.isnan(y_pred).any():
                return html.Div("Model produced NaN predictions. Try different features or model configuration.", className="error-message"), None, True
                
            # Evaluate model based on task type
            if task_type == "regression":
                results = evaluate_regression(y_test, y_pred)
            else:  # classification
                results = evaluate_classification(y_test, y_pred, label_encoder)
                
            # Store model and preprocessing info
            model_data = {
                "model_type": model_type,
                "task_type": task_type,
                "features": features,
                "target": target,
                "label_encoder": label_encoder.classes_.tolist() if label_encoder else None,
                "model_pickle": pickle_model_to_base64(model),
                "scaler_pickle": pickle_model_to_base64(scaler) if scaler else None,
                "feature_names": X.columns.tolist() if isinstance(X, pd.DataFrame) else None,
                "test_size": test_size,
                "metrics": results["metrics"]
            }
            
            return results["display"], model_data, False
                
        except Exception as e:
            return html.Div([
                html.H4("Error Training Model", style={"color": "red"}),
                html.P(f"An error occurred: {str(e)}"),
                html.P("Try with different features, model settings, or check your data for issues.")
            ], className="error-message"), None, True
            
    # Download trained model
    @app.callback(
        Output("download-model-pickle", "data"),
        [Input("btn-download-model", "n_clicks")],
        [State("trained-model-store", "data")],
        prevent_initial_call=True
    )
    def download_model(n_clicks, model_data):
        if n_clicks is None or not n_clicks or model_data is None:
            raise PreventUpdate
            
        # Create a dictionary with all model info
        download_data = {
            "model_info": {k: v for k, v in model_data.items() if k != "model_pickle" and k != "scaler_pickle"},
            "model": model_data["model_pickle"],
            "scaler": model_data["scaler_pickle"]
        }
        
        # Convert to string for download
        model_json = json.dumps(download_data)
        
        return dict(
            content=model_json,
            filename=f"{model_data['model_type']}_{model_data['task_type']}_model.json",
            type="text/json"
        )

# Helper Functions
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Store metrics
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    # Create scatter plot of actual vs. predicted values
    fig = px.scatter(
        x=y_true, 
        y=y_pred,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title="Actual vs. Predicted Values"
    )
    fig.add_trace(
        go.Scatter(
            x=[min(y_true), max(y_true)], 
            y=[min(y_true), max(y_true)],
            mode='lines', 
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    # Create residual plot
    residuals = y_true - y_pred
    fig_residuals = px.scatter(
        x=y_pred, 
        y=residuals,
        labels={'x': 'Predicted Values', 'y': 'Residuals'},
        title="Residual Plot"
    )
    fig_residuals.add_hline(y=0, line_color="red", line_dash="dash")
    
    # Build display component
    display = html.Div([
        html.H4("Regression Model Results", style={"color": "#2980b9"}),
        
        html.Div([
            html.H5("Performance Metrics:"),
            html.Div([
                html.Div([
                    html.P(f"Mean Squared Error (MSE): {mse:.4f}"),
                    html.P(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                ], className="metrics-column"),
                html.Div([
                    html.P(f"Mean Absolute Error (MAE): {mae:.4f}"),
                    html.P(f"R² Score: {r2:.4f}")
                ], className="metrics-column")
            ], className="metrics-grid")
        ], className="model-metrics"),
        
        html.Div([
            html.Div([
                html.H5("Actual vs Predicted Values:"),
                dcc.Graph(figure=fig)
            ], className="plot-column"),
            html.Div([
                html.H5("Residual Plot:"),
                dcc.Graph(figure=fig_residuals)
            ], className="plot-column")
        ], className="plot-grid"),
        
        html.Div([
            html.H5("Interpretation:"),
            html.P([
                "The model explains ", 
                html.Strong(f"{r2*100:.1f}%"), 
                " of the variance in the target variable."
            ]),
            html.P([
                "The Root Mean Squared Error (RMSE) is ",
                html.Strong(f"{rmse:.4f}"),
                ", which represents the average prediction error in the same units as the target variable."
            ]),
            html.P("A good model should have a high R² and low RMSE/MAE values.")
        ], className="interpretation")
    ])
    
    return {"display": display, "metrics": metrics}

def evaluate_classification(y_true, y_pred, label_encoder=None):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multi-class, use weighted average
    try:
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    except:
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Store metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # If label encoder exists, use class names
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]
    
    # Create heatmap of confusion matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix"
    )
    
    # Classification report as table
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_data = []
    
    # Format report for display
    for idx, row in report_df.iterrows():
        if idx not in ['accuracy', 'macro avg', 'weighted avg']:
            class_name = class_names[int(idx)] if label_encoder and idx.isdigit() else idx
            report_data.append({
                "Class": class_name,
                "Precision": f"{row['precision']:.4f}",
                "Recall": f"{row['recall']:.4f}",
                "F1-Score": f"{row['f1-score']:.4f}",
                "Support": int(row['support'])
            })
    
    report_data.append({
        "Class": "Average (weighted)",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1-Score": f"{f1:.4f}",
        "Support": int(report_df.loc['weighted avg', 'support'])
    })
    
    report_table_df = pd.DataFrame(report_data)
    
    # Build display component
    display = html.Div([
        html.H4("Classification Model Results", style={"color": "#2980b9"}),
        
        html.Div([
            html.H5("Performance Metrics:"),
            html.Div([
                html.Div([
                    html.P(f"Accuracy: {accuracy:.4f}"),
                    html.P(f"Precision: {precision:.4f}")
                ], className="metrics-column"),
                html.Div([
                    html.P(f"Recall: {recall:.4f}"),
                    html.P(f"F1 Score: {f1:.4f}")
                ], className="metrics-column")
            ], className="metrics-grid")
        ], className="model-metrics"),
        
        html.Div([
            html.H5("Confusion Matrix:"),
            dcc.Graph(figure=fig_cm)
        ], className="confusion-matrix"),
        
        html.Div([
            html.H5("Classification Report:"),
            dash_table.DataTable(
                data=report_table_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in report_table_df.columns],
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
        ], className="classification-report"),
        
        html.Div([
            html.H5("Interpretation:"),
            html.P([
                "The model achieved an accuracy of ", 
                html.Strong(f"{accuracy*100:.1f}%"), 
                ", correctly classifying this percentage of samples in the test set."
            ]),
            html.P([
                "Precision (", html.Strong(f"{precision:.4f}"), "): The proportion of positive identifications that were actually correct."
            ]),
            html.P([
                "Recall (", html.Strong(f"{recall:.4f}"), "): The proportion of actual positives that were correctly identified."
            ]),
            html.P([
                "F1 Score (", html.Strong(f"{f1:.4f}"), "): The harmonic mean of precision and recall."
            ])
        ], className="interpretation")
    ])
    
    return {"display": display, "metrics": metrics}

def pickle_model_to_base64(model):
    if model is None:
        return None
        
    # Pickle the model
    model_pickle = pickle.dumps(model)
    
    # Convert to base64 string
    return base64.b64encode(model_pickle).decode('utf-8') 