import os
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LocationItem(BaseModel):
    address: str
    place_id: str

class RouteRequest(BaseModel):
    locations: List[LocationItem]

GOOGLE_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')

def get_distance_matrix(location_items, departure_ts):
    """Fetch travel times from Google Maps API"""
    ids = "|".join([f"place_id:{item.place_id}" for item in location_items])
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        'origins': ids, 'destinations': ids,
        'key': GOOGLE_API_KEY, 'departure_time': int(departure_ts)
    }
    res = requests.get(url, params=params).json()
    return res if res.get('status') == 'OK' else None

def solve_tsp(matrix, num_locs):
    """Solve Traveling Salesperson Problem using OR-Tools"""
    manager = pywrapcp.RoutingIndexManager(num_locs, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def dist_callback(from_idx, to_idx):
        return matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
        
    transit_idx = routing.RegisterTransitCallback(dist_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    
    solution = routing.SolveWithParameters(params)
    if solution:
        path = []
        idx = routing.Start(0)
        while not routing.IsEnd(idx):
            path.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        path.append(manager.IndexToNode(idx))
        return solution.ObjectiveValue(), path
    return None, None

@app.post("/optimize")
def optimize(request: RouteRequest):
    if len(request.locations) < 2:
        raise HTTPException(status_code=400, detail="Add more stops")

    data = get_distance_matrix(request.locations, datetime.now().timestamp())
    if not data:
        raise HTTPException(status_code=500, detail="Maps API error")

    # Build cost matrix from durations
    matrix = []
    for row in data['rows']:
        matrix.append([el.get('duration', {}).get('value', 999999) for el in row['elements']])

    cost, indices = solve_tsp(matrix, len(request.locations))
    
    return {
        "duration_mins": cost // 60,
        "route": [request.locations[i].address for i in indices]
    }