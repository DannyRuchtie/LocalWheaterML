import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Added for more control if needed, though px might suffice
import numpy as np
import json # Added for json.JSONDecodeError handling in callback

# FastAPI server URL
API_URL = "http://127.0.0.1:8008/forecast"

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Weather Forecast Dashboard"

# App layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Drenthe Weather Forecast", className="text-center my-4"), width=12)),
    dbc.Row([
        dbc.Col(dcc.Graph(id='temp-graph'), width=12, md=4),
        dbc.Col(dcc.Graph(id='rhum-graph'), width=12, md=4),
        dbc.Col(dcc.Graph(id='wdir-graph'), width=12, md=4), # Wind direction graph as 3rd item
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='wspd-graph'), width=12, md=4),
        dbc.Col(dcc.Graph(id='prcp-graph'), width=12, md=4),
        # Optional: dbc.Col(width=12, md=4) # Add an empty column if you want to ensure the row structure is always 3-wide for alignment
    ]),
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # in milliseconds
        n_intervals=0
    )
], fluid=True)

# Helper function to convert degrees to cardinal directions
def degrees_to_cardinal(d):
    if pd.isna(d):
        return "N/A"
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    ix = int(round(d / (360. / len(dirs))))
    return dirs[ix % len(dirs)]

# Callback to update graphs
@app.callback(
    [Output('temp-graph', 'figure'),
     Output('rhum-graph', 'figure'),
     Output('wspd-graph', 'figure'),
     Output('prcp-graph', 'figure'),
     Output('wdir-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n_intervals):
    empty_fig_layout_spec = {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False}, "annotations": [{"xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20}}]}}
    
    def create_error_fig(message):
        spec = dict(empty_fig_layout_spec) # Create a copy
        spec["layout"]["annotations"][0]["text"] = message
        return spec

    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        forecast_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        err_fig = create_error_fig("No data: API error")
        return err_fig, err_fig, err_fig, err_fig, err_fig
    except json.JSONDecodeError:
        print("Error decoding JSON from API")
        err_fig = create_error_fig("Error decoding API data")
        return err_fig, err_fig, err_fig, err_fig, err_fig

    if not forecast_data or isinstance(forecast_data, dict) and forecast_data.get("message") == "Forecast data is currently empty or not available.":
        print("Forecast data is empty or not available from API.")
        err_fig = create_error_fig("Forecast data not available")
        return err_fig, err_fig, err_fig, err_fig, err_fig

    try:
        df = pd.DataFrame.from_dict(forecast_data, orient='index')
        if df.empty: # Check if DataFrame is empty after from_dict before accessing index
             raise ValueError("DataFrame is empty after from_dict, before to_datetime conversion.")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if df.empty:
            raise ValueError("DataFrame is empty after processing API data.")

        if 'wdir_sin' in df.columns and 'wdir_cos' in df.columns:
            df['wdir_deg'] = (np.degrees(np.arctan2(df['wdir_sin'], df['wdir_cos'])) + 360) % 360
        else:
            df['wdir_deg'] = np.nan
        
        df['wdir_cardinal'] = df['wdir_deg'].apply(degrees_to_cardinal)

        fig_temp = px.line(df, x=df.index, y='temp', title='Temperature Forecast (°C)', markers=True)
        fig_rhum = px.line(df, x=df.index, y='rhum', title='Relative Humidity Forecast (%)', markers=True)
        fig_wspd = px.line(df, x=df.index, y='wspd', title='Wind Speed Forecast (km/h)', markers=True)
        fig_prcp = px.bar(df, x=df.index, y='prcp', title='Precipitation Forecast (mm)')
        
        # New Wind Direction Plot (2D scatter)
        # Using wdir_sin for x (East-West) and wdir_cos for y (North-South) based on re-evaluation
        # This makes: North (0 deg) -> (sin=0, cos=1) -> (x=0, y=1) -> Top of plot
        # East (90 deg) -> (sin=1, cos=0) -> (x=1, y=0) -> Right of plot
        df_reset_for_wdir = df.reset_index().rename(columns={'index': 'time'})
        fig_wdir = px.scatter(
            df_reset_for_wdir, 
            x='wdir_sin',  # East-West component
            y='wdir_cos',  # North-South component
            title='Wind Direction Forecast',
            custom_data=['time', 'wdir_deg', 'wdir_cardinal', 'wspd'] # time, degrees, cardinal, speed
        )
        fig_wdir.update_traces(
            mode='lines+markers',
            marker=dict(size=10, symbol='arrow', angle=df_reset_for_wdir['wdir_deg'] * -1), # angle for marker
            line=dict(width=1),
            hovertemplate=(
                "<b>Time</b>: %{customdata[0]|%Y-%m-%d %H:%M}<br>" +
                "Sin(E/W): %{x:.2f}, Cos(N/S): %{y:.2f}<br>" +
                "Direction: %{customdata[1]:.0f}° (%{customdata[2]})<br>" +
                "Speed: %{customdata[3]:.1f} km/h" +
                "<extra></extra>"
            )
        )
        fig_wdir.update_layout(
            xaxis_title='Sine Component (Eastward if positive)',
            yaxis_title='Cosine Component (Northward if positive)',
            xaxis=dict(range=[-1.3, 1.3], showgrid=True, zeroline=True, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-1.3, 1.3], showgrid=True, zeroline=True),
            showlegend=False,
            annotations=[
                dict(x=0, y=1.1, text="N (0°)", showarrow=False, font=dict(size=12)),
                dict(x=1.1, y=0, text="E (90°)", showarrow=False, font=dict(size=12)),
                dict(x=0, y=-1.1, text="S (180°)", showarrow=False, font=dict(size=12)),
                dict(x=-1.1, y=0, text="W (270°)", showarrow=False, font=dict(size=12)),
            ]
        )
        
        return fig_temp, fig_rhum, fig_wspd, fig_prcp, fig_wdir
        
    except Exception as e:
        print(f"Error processing data or creating figures: {e}")
        err_fig = create_error_fig(f"Error: {str(e)[:100]}")
        return err_fig, err_fig, err_fig, err_fig, err_fig

# To run this Dash application:
# 1. Ensure your FastAPI server (serve.py) is running on http://127.0.0.1:8000
# 2. Ensure you are in the weather-ml directory.
# 3. Activate your virtual environment: source .venv/bin/activate
# 4. Run this script: python src/dashboard.py
#    This will typically start the Dash server on http://127.0.0.1:8050

if __name__ == '__main__':
    app.run(debug=True, port=8050) 