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

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
# --- API KEYS ---
GOOGLE_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
# Replace with your actual OpenAI Key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

app = FastAPI(title="Delivery Driver Route Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    locations: List[str]
    forced_indices: Optional[List[int]] = []
    times: Optional[List[str]] = []


client = OpenAI(api_key=OPENAI_API_KEY)


# --- GOOGLE PLACES DATA ENRICHMENT ---
def get_place_details(address: str) -> Dict:
    """Fetches formatted address and location types (e.g., school, establishment)"""
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        'input': address,
        'inputtype': 'textquery',
        'fields': 'formatted_address,types',
        'key': GOOGLE_API_KEY
    }
    try:
        response = requests.get(url, params=params).json()
        if response.get('status') == 'OK' and response.get('candidates'):
            candidate = response['candidates'][0]
            return {
                "address": candidate['formatted_address'],
                "types": candidate.get('types', [])
            }
        return {"address": address, "types": []}
    except:
        return {"address": address, "types": []}


def get_google_distance_matrix(locations: List[str], departure_timestamp: float):
    locations_str = "|".join(locations)
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        'units': 'metric', 'origins': locations_str, 'destinations': locations_str,
        'key': GOOGLE_API_KEY, 'departure_time': int(departure_timestamp), 'traffic_model': 'best_guess'
    }
    try:
        response = requests.get(url, params=params).json()
        return response if response.get('status') == 'OK' else None
    except:
        return None


def parse_matrix(data):
    matrix = []
    for row in data['rows']:
        row_list = []
        for element in row['elements']:
            if element.get('status') != 'OK':
                row_list.append(9999999)
            else:
                duration = element.get('duration_in_traffic', element.get('duration'))
                row_list.append(int(duration['value']) if duration else 0)
        matrix.append(row_list)
    return matrix


def solve_route(matrix, num_locations, forced_indices):
    try:
        manager = pywrapcp.RoutingIndexManager(num_locations, 1, [0], [num_locations - 1])
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        current_idx = routing.Start(0)
        for node in forced_indices:
            if node == 0 or node == num_locations - 1: continue
            next_idx = manager.NodeToIndex(node)
            routing.NextVar(current_idx).SetValue(next_idx)
            current_idx = next_idx

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
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


# --- TACTICAL DELIVERY AI INSIGHTS ---
def get_ai_delivery_insights(route_details: List[Dict], duration_mins: int, start_time: str) -> str:
    """Generates delivery-specific tactical advice using location context"""
    print(f"\n[AI] Generating tactical briefing for {len(route_details)} stops...")

    route_summary = ""
    for i, loc in enumerate(route_details):
        type_info = ", ".join(loc['types'][:3]) if loc['types'] else "Unknown"
        route_summary += f"Stop {i + 1}: {loc['address']} (Type: {type_info})\n"

    prompt = f"""
        You are a professional Delivery Dispatcher. 
        Analyze this route for a delivery driver:
        {route_summary}

        Total Drive Time: {duration_mins} mins. Start: {start_time}.

        STRICT INSTRUCTIONS:
        1. For each stop, provide ONLY a plain text time delay prediction (e.g., Stop 5: +10 mins for university parking).
        2. Do NOT use any bold text, asterisks (*), or special formatting characters.
        3. AFTER the list of stops, add exactly 2-3 sentences of general road assessment and a specific break suggestion based on the total duration ({duration_mins} mins).
        4. Keep the entire response extremely clean and easy to read on a mobile screen.
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Recommended for better local reasoning
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400
        )
        print("[AI] Tactical briefing generated successfully.")
        return response.choices[0].message.content
    except Exception as e:
        print(f"🚨 OPENAI ERROR: {e}")
        return "Tactical briefing unavailable. Please proceed with caution."


# --- MAIN API ENDPOINT ---
@app.post("/optimize")
def optimize_route(request: RouteRequest):
    if len(request.locations) < 2:
        raise HTTPException(status_code=400, detail="Minimum 2 locations required.")

    print("\n[Step 1] Verifying addresses and fetching location types...")
    # Enrichment: Get details for all locations
    verified_data = [get_place_details(loc) for loc in request.locations]
    verified_addresses = [d['address'] for d in verified_data]

    # Traffic timestamps
    if not request.times:
        test_timestamps = [("Live Traffic", datetime.now().timestamp())]
    else:
        tomorrow = (datetime.now() + timedelta(days=1))
        test_timestamps = []
        for t_str in request.times:
            try:
                h, m = map(int, t_str.split(":"))
                dt = tomorrow.replace(hour=h, minute=m, second=0, microsecond=0)
                test_timestamps.append((dt.strftime('%H:%M'), dt.timestamp()))
            except:
                continue

    print("[Step 2] Calculating optimal sequence...")
    results = []
    for label, ts in test_timestamps:
        raw_data = get_google_distance_matrix(verified_addresses, ts)
        if raw_data:
            matrix = parse_matrix(raw_data)
            cost, indices = solve_route(matrix, len(verified_addresses), request.forced_indices)
            if cost is not None:
                results.append({'time': label, 'duration': cost, 'indices': indices})

    if not results:
        raise HTTPException(status_code=404, detail="No valid route found.")

    best = min(results, key=lambda x: x['duration'])

    # Re-order the enriched data based on the solved indices
    final_ordered_details = [verified_data[i] for i in best['indices']]
    final_route_addresses = [d['address'] for d in final_ordered_details]
    duration_mins = best['duration'] // 60

    # Get AI Insights
    ai_insights = get_ai_delivery_insights(final_ordered_details, duration_mins, best['time'])

    # Prepare Google Maps Link
    origin = urllib.parse.quote(final_route_addresses[0])
    destination = urllib.parse.quote(final_route_addresses[-1])
    waypoints = urllib.parse.quote("|".join(final_route_addresses[1:-1]))
    url = f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination}&waypoints={waypoints}&travelmode=driving"

    print("[Step 3] Success. Sending payload to frontend.\n")
    return {
        "status": "success",
        "suggested_start_time": best['time'],
        "estimated_duration_mins": duration_mins,
        "suggested_route": final_route_addresses,
        "google_maps_url": url,
        "ai_insights": ai_insights
    }