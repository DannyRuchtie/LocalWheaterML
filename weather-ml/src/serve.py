import json
import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Weather Forecast API",
    description="Provides 24-hour weather forecasts.",
    version="0.1.0"
)

# Configure CORS (Cross-Origin Resource Sharing)
# Allows requests from any origin during development. 
# For production, you might want to restrict this to specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET"], # Allows only GET requests for this simple API
    allow_headers=["*"],   # Allows all headers
)

# Define the path to the forecast JSON file
# Assumes serve.py is in the src/ directory, and forecast_24h.json is also in src/
FORECAST_FILE = pathlib.Path(__file__).parent / "forecast_24h.json"

@app.get("/forecast")
async def get_forecast():
    """
    Retrieves the latest 24-hour weather forecast.
    The forecast is read from the `forecast_24h.json` file, which is updated by the daily batch job.
    """
    if not FORECAST_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Forecast file not found at {FORECAST_FILE}")
    
    try:
        with open(FORECAST_FILE, 'r') as f:
            forecast_data = json.load(f)
        if not forecast_data: # Check if the JSON file is empty (e.g. {} or [])
            # Return a specific message or an empty dict, depending on desired behavior
            return {"message": "Forecast data is currently empty or not available.", "data": {}}
        return forecast_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding forecast JSON data.")
    except Exception as e:
        # Catch any other unexpected errors during file reading
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run this application:
# 1. Ensure you are in the weather-ml/src directory.
# 2. Activate your virtual environment: source ../.venv/bin/activate (if .venv is in weather-ml/)
# 3. Run uvicorn: uvicorn serve:app --reload
#    This will typically start the server on http://127.0.0.1:8000
#    You can then access the API at http://127.0.0.1:8000/forecast
#    And the auto-generated docs at http://127.0.0.1:8000/docs or /redoc

if __name__ == "__main__":
    # This allows running the app with `python serve.py` for development
    # However, uvicorn is the recommended way to run FastAPI in production/development.
    print("Starting FastAPI server with Uvicorn. Access at http://127.0.0.1:8008/forecast")
    print("API docs available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8008) 