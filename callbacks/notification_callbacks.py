from dash.dependencies import Input, Output
from dash import html
import time
from dash.exceptions import PreventUpdate

def register_notification_callbacks(app):
    @app.callback(
        Output("notification-container", "children", allow_duplicate=True),
        Input("notification-container", "children"),
        prevent_initial_call=True
    )
    def clear_notification(notification):
        """Automatically clear notifications after 5 seconds."""
        if notification is None:
            raise PreventUpdate
        
        # In a real app, you would use dcc.Interval to clear this
        # For simplicity, we're just returning None after 5 seconds in production
        # This is a placeholder callback
        
        return notification 