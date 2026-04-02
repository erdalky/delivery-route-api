import os
import requests
import urllib.parse
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from openai import OpenAI
from fastapi.responses import HTMLResponse

# Initialize FastAPI
app = FastAPI(title="Delivery Driver Route Assistant")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class LocationItem(BaseModel):
    address: str
    place_id: str

class RouteRequest(BaseModel):
    locations: List[LocationItem]
    forced_indices: Optional[List[int]] = []
    times: Optional[List[str]] = []

# API credentials
GOOGLE_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Fetch traffic data from Google
def get_google_distance_matrix(location_items: List[LocationItem], departure_timestamp: float):
    ids_str = "|".join([f"place_id:{item.place_id}" for item in location_items])
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        'units': 'metric',
        'origins': ids_str,
        'destinations': ids_str,
        'key': GOOGLE_API_KEY,
        'departure_time': int(departure_timestamp),
        'traffic_model': 'best_guess'
    }
    try:
        response = requests.get(url, params=params).json()
        return response if response.get('status') == 'OK' else None
    except:
        return None

# Extract durations from Google response
def parse_matrix(data):
    matrix = []
    for row in data['rows']:
        row_list = []
        for element in row['elements']:
            if element.get('status') != 'OK':
                row_list.append(9999999) # High cost for invalid routes
            else:
                duration = element.get('duration_in_traffic', element.get('duration'))
                row_list.append(int(duration['value']) if duration else 0)
        matrix.append(row_list)
    return matrix

# Solve Traveling Salesperson Problem (TSP)
def solve_route(matrix, num_locations, forced_indices):
    try:
        manager = pywrapcp.RoutingIndexManager(num_locations, 1, [0], [num_locations - 1])
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
            
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.time_limit.seconds = 3
        
        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            ordered_indices = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                ordered_indices.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            ordered_indices.append(manager.IndexToNode(index))
            return solution.ObjectiveValue(), ordered_indices
        return None, None
    except:
        return None, None

# Generate AI tactical advice
def get_ai_delivery_insights(route_addresses: List[str], duration_mins: int, start_time: str) -> str:
    prompt = f"""
        You are a professional Delivery Dispatcher. 
        Analyze this route: {", ".join(route_addresses)}
        Total Drive Time: {duration_mins} mins. Start: {start_time}.
        STRICT: 1. Plain text delays only. 2. No bold/asterisks. 3. 2-3 sentences road assessment at end.
    """
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.6)
        return response.choices[0].message.content
    except:
        return "Tactical briefing unavailable."

# Optimization endpoint
@app.post("/optimize")
def optimize_route(request: RouteRequest):
    if len(request.locations) < 2:
        raise HTTPException(status_code=400, detail="Minimum 2 locations required.")

    # Setup timestamps for traffic analysis
    if not request.times:
        test_timestamps = [("Live Traffic", datetime.now().timestamp())]
    else:
        tomorrow = (datetime.now() + timedelta(days=1))
        test_timestamps = []
        for t_str in request.times:
            try:
                h, m = map(int, t_str.split(":"))
                dt = tomorrow.replace(hour=h, minute=m, second=0)
                test_timestamps.append((dt.strftime('%H:%M'), dt.timestamp()))
            except: continue

    # Evaluate route across requested time slots
    results = []
    for label, ts in test_timestamps:
        raw_data = get_google_distance_matrix(request.locations, ts)
        if raw_data:
            matrix = parse_matrix(raw_data)
            cost, indices = solve_route(matrix, len(request.locations), request.forced_indices)
            if cost is not None:
                results.append({'time': label, 'duration': cost, 'indices': indices})

    if not results:
        raise HTTPException(status_code=404, detail="No valid route found.")

    # Select most efficient time and sequence
    best = min(results, key=lambda x: x['duration'])
    final_ordered_locations = [request.locations[i].address for i in best['indices']]
    duration_mins = best['duration'] // 60
    ai_insights = get_ai_delivery_insights(final_ordered_locations, duration_mins, best['time'])

    return {
        "status": "success",
        "suggested_start_time": best['time'],
        "estimated_duration_mins": duration_mins,
        "suggested_route": final_ordered_locations,
        "ai_insights": ai_insights
    }