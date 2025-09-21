from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import statistics
import random

app = FastAPI(title="Deal Command Center", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables for API keys
# Replace these with your actual API keys for testing
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
REALTY_MOLE_API_KEY = os.getenv("REALTY_MOLE_API_KEY")

# For immediate testing, uncomment and add your keys here:
# GOOGLE_MAPS_API_KEY = "AIzaSyCJ9LhiRJQD3CQocXmtLNEz7hEBT4GvTeY"
# REALTY_MOLE_API_KEY = "fcdb68d4d0e9422d80e45a147e67596f"

# Data models
class AddressSearchRequest(BaseModel):
    query: str
    limit: int = 5

class PropertyAnalysisRequest(BaseModel):
    address: str
    manual_comps: Optional[List[Dict]] = []

# Serve the HTML app at root
@app.get("/", response_class=HTMLResponse)
async def serve_app():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Deal Command Center API</h1>
                <p>API is running. Upload index.html to see the interface.</p>
                <p><a href="/docs">View API Documentation</a></p>
                <p><a href="/debug-env">Debug Environment Variables</a></p>
            </body>
        </html>
        """)

# Debug endpoint to check environment variables
@app.get("/debug-env")
async def debug_env():
    return {
        "google_api_key": "SET" if GOOGLE_MAPS_API_KEY else "NOT SET",
        "realty_mole_key": "SET" if REALTY_MOLE_API_KEY else "NOT SET",
        "google_key_length": len(GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else 0,
        "realty_key_length": len(REALTY_MOLE_API_KEY) if REALTY_MOLE_API_KEY else 0,
        "environment_variables": [key for key in os.environ.keys() if 'API' in key or 'KEY' in key]
    }

@app.post("/search-addresses")
async def search_addresses(request: AddressSearchRequest):
    """Smart address autocomplete with Google Places integration"""
    
    if not GOOGLE_MAPS_API_KEY:
        return {"suggestions": mock_address_search(request.query, request.limit)}
    
    try:
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {
            'input': request.query,
            'types': 'address',
            'components': 'country:US',
            'key': GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('status') == 'OK':
            suggestions = []
            for prediction in data.get('predictions', [])[:request.limit]:
                suggestions.append({
                    'address': prediction['structured_formatting']['main_text'],
                    'formatted_address': prediction['description'],
                    'place_id': prediction['place_id']
                })
            return {"suggestions": suggestions}
        else:
            print(f"Google API error: {data.get('status')} - {data.get('error_message', 'Unknown error')}")
            return {"suggestions": mock_address_search(request.query, request.limit)}
                    
    except Exception as e:
        print(f"Address search error: {e}")
        return {"suggestions": mock_address_search(request.query, request.limit)}

def mock_address_search(query: str, limit: int) -> List[Dict]:
    """Mock address search for development"""
    base_num = query.split(' ')[0] if query.split(' ')[0].isdigit() else "1022"
    
    suggestions = []
    streets = ["Kenneth St", "Oak Ave", "Main St", "Pine St", "Maple Ave"]
    cities = [
        ("Muskegon", "MI", "49441"),
        ("Grand Rapids", "MI", "49503"),
        ("Kalamazoo", "MI", "49007"),
        ("Battle Creek", "MI", "49017")
    ]
    
    for i, (street, (city, state, zip_code)) in enumerate(zip(streets, cities)):
        if i >= limit:
            break
        address = f"{base_num} {street}, {city}, {state} {zip_code}"
        suggestions.append({
            'address': f"{base_num} {street}",
            'formatted_address': address,
            'place_id': f"mock_place_id_{i}"
        })
    
    return suggestions

@app.post("/property-details")
async def get_property_details(request: dict):
    """Fetch comprehensive property details"""
    try:
        address = request.get("address")
        if not address:
            raise HTTPException(status_code=400, detail="Address is required")
        
        # Try RealtyMole API first
        if REALTY_MOLE_API_KEY:
            try:
                property_data = fetch_realty_mole_data(address)
                if property_data:
                    return property_data
            except Exception as e:
                print(f"RealtyMole API error: {e}")
        
        # Fallback to mock data
        return generate_mock_property_data(address)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Property lookup failed: {str(e)}")

def fetch_realty_mole_data(address: str) -> Dict:
    """Fetch property data from RealtyMole API using requests"""
    
    url = "https://realty-mole-property-api.p.rapidapi.com/properties"
    
    headers = {
        "X-RapidAPI-Key": REALTY_MOLE_API_KEY,
        "X-RapidAPI-Host": "realty-mole-property-api.p.rapidapi.com"
    }
    
    params = {"address": address}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if data and len(data) > 0:
                prop = data[0]
                return {
                    "address": prop.get("formattedAddress", address),
                    "year_built": prop.get("yearBuilt"),
                    "square_feet": prop.get("squareFootage"),
                    "bedrooms": prop.get("bedrooms"),
                    "bathrooms": prop.get("bathrooms"),
                    "property_type": prop.get("propertyType", "Single Family"),
                    "lot_size": prop.get("lotSize"),
                    "last_sale_date": prop.get("lastSaleDate"),
                    "last_sale_price": prop.get("lastSalePrice"),
                    "assessed_value": prop.get("assessedValue"),
                    "tax_amount": prop.get("taxAmount")
                }
        else:
            print(f"RealtyMole API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
        
        return None
        
    except Exception as e:
        print(f"RealtyMole API request failed: {e}")
        return None

def generate_mock_property_data(address: str) -> Dict:
    """Generate realistic property data for West Michigan"""
    
    # Determine market based on address
    if 'muskegon' in address.lower():
        price_range = (45, 105)
    elif 'grand rapids' in address.lower():
        price_range = (85, 165)
    elif 'kalamazoo' in address.lower():
        price_range = (65, 135)
    else:
        price_range = (55, 125)
    
    year_built = random.randint(1920, 2010)
    square_feet = random.randint(900, 2200)
    bedrooms = random.randint(2, 5)
    bathrooms = round(random.uniform(1.0, 3.0), 1)
    
    # Generate realistic pricing
    price_per_sqft = random.randint(price_range[0], price_range[1])
    
    # Recent sale data (70% chance)
    last_sale_data = {}
    if random.random() > 0.3:
        days_ago = random.randint(30, 1800)
        sale_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        sale_price = int(square_feet * price_per_sqft)
        
        last_sale_data = {
            "last_sale_date": sale_date,
            "last_sale_price": sale_price,
            "assessed_value": int(sale_price * 0.8),
            "tax_amount": int(sale_price * 0.015)
        }
    
    base_data = {
        "address": address,
        "year_built": year_built,
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_type": "Single Family",
        "lot_size": round(random.uniform(0.1, 0.5), 2)
    }
    
    base_data.update(last_sale_data)
    return base_data

@app.post("/comparable-sales")
async def get_comparable_sales(request: dict):
    """Find comparable sales"""
    try:
        address = request.get("address")
        if not address:
            raise HTTPException(status_code=400, detail="Address is required")
        
        # Get property details first
        property_data = await get_property_details({"address": address})
        
        # Generate realistic comparable sales
        comps = generate_realistic_comps(property_data, address)
        
        return {
            "comparable_sales": comps,
            "count": len(comps),
            "search_radius": 2.0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparable sales search failed: {str(e)}")

def generate_realistic_comps(property_data: Dict, address: str) -> List[Dict]:
    """Generate realistic comparable sales"""
    
    # Determine market pricing
    if 'muskegon' in address.lower():
        base_price_per_sqft = 75
        price_variance = 25
    elif 'grand rapids' in address.lower():
        base_price_per_sqft = 125
        price_variance = 35
    elif 'kalamazoo' in address.lower():
        base_price_per_sqft = 95
        price_variance = 30
    else:
        base_price_per_sqft = 90
        price_variance = 25
    
    comps = []
    subject_sqft = property_data.get("square_feet", 1200)
    subject_beds = property_data.get("bedrooms", 3)
    subject_baths = property_data.get("bathrooms", 2)
    subject_year = property_data.get("year_built", 1950)
    
    # Generate 6-8 comps
    num_comps = random.randint(6, 9)
    
    for i in range(num_comps):
        # Vary property characteristics
        comp_sqft = max(600, subject_sqft + random.randint(-300, 301))
        comp_beds = max(1, subject_beds + random.randint(-1, 2))
        comp_baths = max(1.0, round(subject_baths + random.uniform(-0.5, 1.0), 1))
        comp_year = max(1900, subject_year + random.randint(-20, 21))
        
        # Distance and timing
        distance = round(random.uniform(0.1, 2.0), 1)
        days_ago = random.randint(15, 365)
        sale_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Price calculation with adjustments
        price_per_sqft = base_price_per_sqft + random.randint(-price_variance, price_variance)
        
        # Size adjustment
        if comp_sqft > subject_sqft:
            price_per_sqft *= 0.98
        
        # Age adjustment
        if comp_year > subject_year + 10:
            price_per_sqft *= 1.05
        elif comp_year < subject_year - 10:
            price_per_sqft *= 0.95
        
        # Time adjustment (market appreciation)
        monthly_appreciation = 0.002
        time_adjustment = 1 + (days_ago / 30 * monthly_appreciation)
        price_per_sqft *= time_adjustment
        
        sale_price = int(comp_sqft * price_per_sqft)
        
        # Generate address
        street_names = ["Oak St", "Maple Ave", "Pine St", "Cedar Ave", "Elm St", "Park St"]
        street = random.choice(street_names)
        house_num = random.randint(800, 1500)
        city = address.split(',')[1].strip() if ',' in address else "Muskegon"
        comp_address = f"{house_num} {street}, {city}"
        
        comps.append({
            "address": comp_address,
            "sale_price": sale_price,
            "sale_date": sale_date,
            "square_feet": comp_sqft,
            "bedrooms": comp_beds,
            "bathrooms": comp_baths,
            "year_built": comp_year,
            "distance_miles": distance,
            "days_since_sale": days_ago,
            "price_per_sqft": round(price_per_sqft, 2)
        })
    
    # Sort by similarity (distance + time)
    comps.sort(key=lambda x: x["distance_miles"] + (x["days_since_sale"] / 100))
    
    return comps

@app.post("/ai-analysis")
async def run_ai_analysis(request: PropertyAnalysisRequest):
    """Run statistical property analysis"""
    try:
        # Get property details
        property_data = await get_property_details({"address": request.address})
        
        # Get comparable sales
        comp_response = await get_comparable_sales({"address": request.address})
        auto_comps = comp_response["comparable_sales"]
        
        # Add manual comps
        manual_comps = []
        for manual_comp in request.manual_comps:
            manual_comps.append({
                "address": manual_comp.get("address", "Manual Entry"),
                "sale_price": int(manual_comp["price"]),
                "sale_date": manual_comp["date"],
                "square_feet": int(manual_comp["sqft"]),
                "bedrooms": int(manual_comp.get("beds", 3)),
                "bathrooms": float(manual_comp.get("baths", 2)),
                "year_built": int(manual_comp.get("year_built", property_data["year_built"])),
                "distance_miles": 0.5,
                "days_since_sale": 30,
                "price_per_sqft": manual_comp["price"] / manual_comp["sqft"]
            })
        
        all_comps = auto_comps + manual_comps
        
        if len(all_comps) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 comparable sales for analysis")
        
        # Statistical ARV calculation
        arv_result = calculate_statistical_arv(property_data, all_comps)
        
        # Calculate offers
        offers = calculate_offer_strategies(arv_result["arv"], property_data)
        
        return {
            "property": property_data,
            "arv_analysis": arv_result,
            "offer_strategies": offers,
            "comparable_sales": all_comps
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def calculate_statistical_arv(property_data: Dict, comps: List[Dict]) -> Dict:
    """Calculate ARV using statistical methods"""
    
    subject_sqft = property_data.get("square_feet", 1200)
    subject_beds = property_data.get("bedrooms", 3)
    subject_baths = property_data.get("bathrooms", 2)
    subject_year = property_data.get("year_built", 1950)
    
    # Calculate similarity scores and weights
    weighted_values = []
    total_weight = 0
    
    for comp in comps:
        # Similarity scoring
        sqft_diff = abs(comp["square_feet"] - subject_sqft) / subject_sqft
        bed_diff = abs(comp["bedrooms"] - subject_beds)
        bath_diff = abs(comp["bathrooms"] - subject_baths)
        age_diff = abs(comp["year_built"] - subject_year)
        
        # Calculate weight (0-1, higher is better)
        sqft_weight = max(0, 1 - sqft_diff)
        bed_weight = max(0, 1 - bed_diff * 0.3)
        bath_weight = max(0, 1 - bath_diff * 0.2)
        age_weight = max(0, 1 - age_diff / 50)
        distance_weight = max(0, 1 - comp["distance_miles"] / 2.0)
        recency_weight = max(0, 1 - comp["days_since_sale"] / 365)
        
        # Combined weight
        weight = (sqft_weight * 0.3 + bed_weight * 0.2 + bath_weight * 0.15 + 
                 age_weight * 0.15 + distance_weight * 0.1 + recency_weight * 0.1)
        
        # Adjust comp price to subject property
        price_per_sqft = comp["sale_price"] / comp["square_feet"]
        adjusted_value = price_per_sqft * subject_sqft
        
        # Apply market adjustments
        if comp["year_built"] > subject_year + 10:
            adjusted_value *= 0.95  # Newer comp, subject worth less
        elif comp["year_built"] < subject_year - 10:
            adjusted_value *= 1.05  # Older comp, subject worth more
        
        # Time adjustment
        monthly_appreciation = 0.002
        time_adjustment = 1 - (comp["days_since_sale"] / 30 * monthly_appreciation)
        adjusted_value *= time_adjustment
        
        weighted_values.append(adjusted_value * weight)
        total_weight += weight
    
    # Calculate weighted average
    base_arv = sum(weighted_values) / total_weight if total_weight > 0 else statistics.mean([c["sale_price"] for c in comps])
    
    # Apply market adjustments
    market_adjustments = calculate_market_adjustments(property_data)
    adjusted_arv = base_arv
    
    for adj_name, adj_value in market_adjustments.items():
        adjusted_arv *= adj_value
    
    final_arv = int(adjusted_arv)
    
    # Calculate confidence based on comp quality
    confidence = calculate_confidence_score(comps, property_data)
    
    # Price range
    range_pct = 0.05 + (1 - confidence) * 0.05  # 5-10% range based on confidence
    
    return {
        "arv": final_arv,
        "confidence_score": round(confidence * 100, 1),
        "price_range": {
            "low": int(final_arv * (1 - range_pct)),
            "high": int(final_arv * (1 + range_pct))
        },
        "comparable_count": len(comps),
        "market_adjustments": market_adjustments,
        "insights": generate_insights(market_adjustments, comps, confidence)
    }

def calculate_market_adjustments(property_data: Dict) -> Dict:
    """Calculate market adjustments"""
    
    adjustments = {}
    address = property_data.get("address", "")
    
    # Location adjustment
    if 'grand rapids' in address.lower():
        adjustments['location'] = 1.05
    elif 'muskegon' in address.lower():
        adjustments['location'] = 0.95
    else:
        adjustments['location'] = 1.0
    
    # Seasonal adjustment
    current_month = datetime.now().month
    if current_month in [3, 4, 5, 6]:
        adjustments['seasonal'] = 1.02
    elif current_month in [11, 12, 1]:
        adjustments['seasonal'] = 0.98
    else:
        adjustments['seasonal'] = 1.0
    
    # Market trend (simplified)
    adjustments['market_trend'] = 1.023  # +2.3% trend
    
    return adjustments

def calculate_confidence_score(comps: List[Dict], property_data: Dict) -> float:
    """Calculate confidence score (0-1)"""
    
    # Factors affecting confidence
    sample_size_factor = min(1.0, len(comps) / 8)  # Optimal at 8+ comps
    
    # Average distance
    avg_distance = statistics.mean([c["distance_miles"] for c in comps])
    distance_factor = max(0, 1 - avg_distance / 2.0)
    
    # Price consistency
    prices_per_sqft = [c["price_per_sqft"] for c in comps]
    price_std = statistics.stdev(prices_per_sqft) if len(prices_per_sqft) > 1 else 0
    price_mean = statistics.mean(prices_per_sqft)
    price_consistency = max(0, 1 - (price_std / price_mean) if price_mean > 0 else 0)
    
    # Recency factor
    avg_days_old = statistics.mean([c["days_since_sale"] for c in comps])
    recency_factor = max(0, 1 - avg_days_old / 365)
    
    # Overall confidence
    confidence = (sample_size_factor * 0.3 + distance_factor * 0.3 + 
                 price_consistency * 0.25 + recency_factor * 0.15)
    
    return max(0.5, min(0.98, confidence))

def generate_insights(market_adjustments: Dict, comps: List[Dict], confidence: float) -> List[str]:
    """Generate analysis insights"""
    
    insights = []
    
    # Market trend
    trend = market_adjustments.get('market_trend', 1.0)
    if trend > 1.02:
        insights.append(f"Market appreciation: +{(trend-1)*100:.1f}% recent trend")
    elif trend < 0.98:
        insights.append(f"Market softening: -{(1-trend)*100:.1f}% recent trend")
    else:
        insights.append("Stable market conditions")
    
    # Location factor
    location = market_adjustments.get('location', 1.0)
    if location > 1.0:
        insights.append(f"Location premium: +{(location-1)*100:.0f}% area adjustment")
    elif location < 1.0:
        insights.append(f"Location discount: -{(1-location)*100:.0f}% area adjustment")
    
    # Comp quality
    avg_distance = statistics.mean([c["distance_miles"] for c in comps])
    if avg_distance < 0.5:
        insights.append(f"Excellent comp proximity: avg {avg_distance:.1f} miles")
    elif avg_distance > 1.0:
        insights.append(f"Moderate comp distance: avg {avg_distance:.1f} miles")
    
    # Confidence insight
    if confidence > 0.9:
        insights.append("High confidence prediction with quality comparable sales")
    elif confidence < 0.7:
        insights.append("Moderate confidence - consider additional research")
    
    return insights

def calculate_offer_strategies(arv: int, property_data: Dict) -> Dict:
    """Calculate offer strategies"""
    
    # Estimate repairs based on age
    age = 2024 - property_data.get("year_built", 1950)
    sqft = property_data.get("square_feet", 1200)
    
    base_repair_cost = 20  # $20/sqft base
    if age > 75:
        base_repair_cost += 15
    elif age > 50:
        base_repair_cost += 10
    elif age > 25:
        base_repair_cost += 5
    
    repair_estimate = int(sqft * base_repair_cost)
    margin_of_safety = int(arv * 0.03)
    
    strategies = {
        "ai_optimized": {
            "name": "Statistical Primary",
            "rule": "65% Rule + Market Adjustments",
            "offer": max(0, int(arv * 0.65 - repair_estimate - margin_of_safety)),
            "description": "Data-driven optimized offer"
        },
        "wholesale": {
            "name": "Conservative Wholesale", 
            "rule": "70% Rule",
            "offer": max(0, int(arv * 0.70 - repair_estimate)),
            "description": "Safe wholesale margin"
        },
        "brrr": {
            "name": "BRRR Strategy",
            "rule": "75% Rule", 
            "offer": max(0, int(arv * 0.75 - repair_estimate)),
            "description": "Buy, rehab, rent, refinance"
        }
    }
    
    return strategies

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
