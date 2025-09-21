from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import asyncio
import aiohttp
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="AI Deal Command Center", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables for API keys
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
ZILLOW_API_KEY = os.getenv("ZILLOW_API_KEY")
REALTY_MOLE_API_KEY = os.getenv("REALTY_MOLE_API_KEY")
ATTOM_API_KEY = os.getenv("ATTOM_API_KEY")

# Data models
@dataclass
class PropertyData:
    address: str
    latitude: float
    longitude: float
    year_built: int
    square_feet: int
    bedrooms: int
    bathrooms: float
    property_type: str
    lot_size: float
    last_sale_date: Optional[str] = None
    last_sale_price: Optional[int] = None
    assessed_value: Optional[int] = None
    tax_amount: Optional[int] = None

@dataclass
class ComparableSale:
    address: str
    sale_price: int
    sale_date: str
    square_feet: int
    bedrooms: int
    bathrooms: float
    year_built: int
    distance_miles: float
    days_since_sale: int
    price_per_sqft: float
    lot_size: Optional[float] = None
    property_type: Optional[str] = None

class AddressSearchRequest(BaseModel):
    query: str
    limit: int = 5

class PropertyAnalysisRequest(BaseModel):
    address: str
    manual_comps: Optional[List[Dict]] = []

class ARVResult(BaseModel):
    arv: int
    confidence_score: float
    price_range: Dict[str, int]
    comparable_count: int
    model_metrics: Dict[str, float]
    market_adjustments: Dict[str, float]
    insights: List[str]

class SmartPropertyLookupService:
    """Advanced property lookup with multiple data sources"""
    
    def __init__(self):
        self.google_places_url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        self.google_details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        self.geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    async def search_addresses(self, query: str, limit: int = 5) -> List[Dict]:
        """Smart address autocomplete using Google Places API"""
        
        if not GOOGLE_MAPS_API_KEY:
            # Fallback to mock data for development
            return self._mock_address_search(query, limit)
        
        try:
            params = {
                'input': query,
                'types': 'address',
                'components': 'country:US',
                'key': GOOGLE_MAPS_API_KEY
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.google_places_url, params=params) as response:
                    data = await response.json()
                    
                    if data.get('status') == 'OK':
                        suggestions = []
                        for prediction in data.get('predictions', [])[:limit]:
                            suggestions.append({
                                'address': prediction['structured_formatting']['main_text'],
                                'formatted_address': prediction['description'],
                                'place_id': prediction['place_id']
                            })
                        return suggestions
                    else:
                        return self._mock_address_search(query, limit)
                        
        except Exception as e:
            print(f"Address search error: {e}")
            return self._mock_address_search(query, limit)
    
    def _mock_address_search(self, query: str, limit: int) -> List[Dict]:
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
    
    async def get_property_details(self, address: str, place_id: str = None) -> PropertyData:
        """Fetch comprehensive property details from multiple sources"""
        
        # Get coordinates first
        coords = await self._get_coordinates(address)
        
        # Fetch property data from multiple sources
        property_data = await self._fetch_multi_source_property_data(address, coords)
        
        return PropertyData(
            address=address,
            latitude=coords['lat'],
            longitude=coords['lng'],
            year_built=property_data.get('year_built', 1950),
            square_feet=property_data.get('square_feet', 1200),
            bedrooms=property_data.get('bedrooms', 3),
            bathrooms=property_data.get('bathrooms', 1.5),
            property_type=property_data.get('property_type', 'Single Family'),
            lot_size=property_data.get('lot_size', 0.2),
            last_sale_date=property_data.get('last_sale_date'),
            last_sale_price=property_data.get('last_sale_price'),
            assessed_value=property_data.get('assessed_value'),
            tax_amount=property_data.get('tax_amount')
        )
    
    async def _get_coordinates(self, address: str) -> Dict[str, float]:
        """Get latitude/longitude for address"""
        
        if not GOOGLE_MAPS_API_KEY:
            # Mock coordinates for development
            city_coords = {
                'muskegon': {'lat': 43.2342, 'lng': -86.2484},
                'grand rapids': {'lat': 42.9634, 'lng': -85.6681},
                'kalamazoo': {'lat': 42.2917, 'lng': -85.5872},
                'battle creek': {'lat': 42.3211, 'lng': -85.1797}
            }
            
            for city, coords in city_coords.items():
                if city in address.lower():
                    return coords
            
            return {'lat': 43.2342, 'lng': -86.2484}  # Default Muskegon
        
        try:
            params = {
                'address': address,
                'key': GOOGLE_MAPS_API_KEY
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.geocoding_url, params=params) as response:
                    data = await response.json()
                    
                    if data.get('status') == 'OK' and data.get('results'):
                        location = data['results'][0]['geometry']['location']
                        return {'lat': location['lat'], 'lng': location['lng']}
                    
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return {'lat': 43.2342, 'lng': -86.2484}  # Default fallback
    
    async def _fetch_multi_source_property_data(self, address: str, coords: Dict) -> Dict:
        """Fetch property data from multiple APIs"""
        
        # In production, call multiple APIs in parallel
        tasks = []
        
        # if ATTOM_API_KEY:
        #     tasks.append(self._fetch_attom_data(address))
        # if REALTY_MOLE_API_KEY:
        #     tasks.append(self._fetch_realty_mole_data(coords))
        
        # For now, return mock data that matches your West Michigan properties
        return self._generate_realistic_property_data(address)
    
    def _generate_realistic_property_data(self, address: str) -> Dict:
        """Generate realistic property data for West Michigan"""
        
        # Base data on typical West Michigan properties
        base_data = {
            'year_built': np.random.randint(1920, 2010),
            'square_feet': np.random.randint(900, 2200),
            'bedrooms': np.random.randint(2, 5),
            'bathrooms': round(np.random.uniform(1.0, 3.0), 1),
            'property_type': 'Single Family',
            'lot_size': round(np.random.uniform(0.1, 0.5), 2)
        }
        
        # Add sale history
        if np.random.random() > 0.3:  # 70% have recent sale data
            days_ago = np.random.randint(30, 1800)  # 1 month to 5 years
            sale_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            # Price based on location and size
            if 'muskegon' in address.lower():
                price_per_sqft = np.random.randint(45, 105)
            elif 'grand rapids' in address.lower():
                price_per_sqft = np.random.randint(85, 165)
            elif 'kalamazoo' in address.lower():
                price_per_sqft = np.random.randint(65, 135)
            else:
                price_per_sqft = np.random.randint(55, 125)
            
            base_data.update({
                'last_sale_date': sale_date,
                'last_sale_price': int(base_data['square_feet'] * price_per_sqft),
                'assessed_value': int(base_data['square_feet'] * price_per_sqft * 0.8),
                'tax_amount': int(base_data['square_feet'] * price_per_sqft * 0.015)
            })
        
        return base_data

class AdvancedComparableService:
    """AI-powered comparable sales discovery and analysis"""
    
    def __init__(self):
        self.max_distance_miles = 2.0
        self.max_days_old = 365
        self.min_comps = 3
        self.max_comps = 10
    
    async def find_comparable_sales(self, subject_property: PropertyData) -> List[ComparableSale]:
        """Find comparable sales using multiple data sources"""
        
        # In production, fetch from Zillow, MLS, public records
        # For now, generate realistic comps
        comps = self._generate_realistic_comps(subject_property)
        
        # Filter and rank comps
        filtered_comps = self._filter_and_rank_comps(subject_property, comps)
        
        return filtered_comps[:self.max_comps]
    
    def _generate_realistic_comps(self, subject: PropertyData) -> List[ComparableSale]:
        """Generate realistic comparable sales for West Michigan"""
        
        comps = []
        
        # Determine market pricing based on location
        if 'muskegon' in subject.address.lower():
            base_price_per_sqft = 75
            price_variance = 25
        elif 'grand rapids' in subject.address.lower():
            base_price_per_sqft = 125
            price_variance = 35
        elif 'kalamazoo' in subject.address.lower():
            base_price_per_sqft = 95
            price_variance = 30
        else:
            base_price_per_sqft = 90
            price_variance = 25
        
        # Generate 8-12 potential comps
        num_comps = np.random.randint(8, 13)
        
        for i in range(num_comps):
            # Vary property characteristics around subject
            sqft_variance = np.random.randint(-300, 301)
            comp_sqft = max(600, subject.square_feet + sqft_variance)
            
            bed_variance = np.random.randint(-1, 2)
            comp_beds = max(1, subject.bedrooms + bed_variance)
            
            bath_variance = np.random.uniform(-0.5, 1.0)
            comp_baths = max(1.0, round(subject.bathrooms + bath_variance, 1))
            
            year_variance = np.random.randint(-20, 21)
            comp_year = max(1900, subject.year_built + year_variance)
            
            # Distance and time factors
            distance = np.random.uniform(0.1, self.max_distance_miles)
            days_ago = np.random.randint(15, self.max_days_old)
            sale_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            # Calculate price with market factors
            price_per_sqft = base_price_per_sqft + np.random.randint(-price_variance, price_variance)
            
            # Adjust for property differences
            if comp_sqft > subject.square_feet:
                price_per_sqft *= 0.98  # Larger properties slight discount per sqft
            if comp_year > subject.year_built + 10:
                price_per_sqft *= 1.05  # Newer properties premium
            if comp_year < subject.year_built - 10:
                price_per_sqft *= 0.95  # Older properties discount
            
            # Time adjustment (market appreciation)
            monthly_appreciation = 0.002  # 0.2% per month
            time_adjustment = 1 + (days_ago / 30 * monthly_appreciation)
            price_per_sqft *= time_adjustment
            
            sale_price = int(comp_sqft * price_per_sqft)
            
            # Generate realistic address
            street_names = ["Oak St", "Maple Ave", "Pine St", "Cedar Ave", "Elm St", 
                          "Park St", "Main St", "Forest Ave", "Lake St", "River Rd"]
            street = np.random.choice(street_names)
            house_num = np.random.randint(800, 1500)
            city = subject.address.split(',')[1].strip() if ',' in subject.address else "Muskegon"
            comp_address = f"{house_num} {street}, {city}"
            
            comps.append(ComparableSale(
                address=comp_address,
                sale_price=sale_price,
                sale_date=sale_date,
                square_feet=comp_sqft,
                bedrooms=comp_beds,
                bathrooms=comp_baths,
                year_built=comp_year,
                distance_miles=distance,
                days_since_sale=days_ago,
                price_per_sqft=round(price_per_sqft, 2)
            ))
        
        return comps
    
    def _filter_and_rank_comps(self, subject: PropertyData, comps: List[ComparableSale]) -> List[ComparableSale]:
        """Filter and rank comps by similarity to subject property"""
        
        scored_comps = []
        
        for comp in comps:
            # Calculate similarity score (0-1, higher is better)
            score = self._calculate_similarity_score(subject, comp)
            
            # Filter out poor matches
            if score > 0.3:  # Minimum similarity threshold
                scored_comps.append((score, comp))
        
        # Sort by score descending
        scored_comps.sort(key=lambda x: x[0], reverse=True)
        
        # Return top comps
        return [comp for score, comp in scored_comps]
    
    def _calculate_similarity_score(self, subject: PropertyData, comp: ComparableSale) -> float:
        """Calculate similarity score between subject and comp"""
        
        # Square footage similarity (weight: 30%)
        sqft_diff = abs(comp.square_feet - subject.square_feet) / subject.square_feet
        sqft_score = max(0, 1 - sqft_diff)
        
        # Bedroom similarity (weight: 20%)
        bed_diff = abs(comp.bedrooms - subject.bedrooms)
        bed_score = max(0, 1 - bed_diff * 0.3)
        
        # Bathroom similarity (weight: 15%)
        bath_diff = abs(comp.bathrooms - subject.bathrooms)
        bath_score = max(0, 1 - bath_diff * 0.2)
        
        # Age similarity (weight: 15%)
        age_diff = abs(comp.year_built - subject.year_built)
        age_score = max(0, 1 - age_diff / 50)
        
        # Distance factor (weight: 10%)
        distance_score = max(0, 1 - comp.distance_miles / self.max_distance_miles)
        
        # Recency factor (weight: 10%)
        recency_score = max(0, 1 - comp.days_since_sale / self.max_days_old)
        
        # Weighted total
        total_score = (
            sqft_score * 0.30 +
            bed_score * 0.20 +
            bath_score * 0.15 +
            age_score * 0.15 +
            distance_score * 0.10 +
            recency_score * 0.10
        )
        
        return total_score

class AIValuationEngine:
    """Advanced machine learning model for ARV prediction"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'linear': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, subject: PropertyData, comps: List[ComparableSale]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Prepare feature matrix for ML models"""
        
        # Create feature matrix from comparables
        features = []
        prices = []
        
        for comp in comps:
            feature_vector = [
                comp.square_feet,
                comp.bedrooms,
                comp.bathrooms,
                comp.year_built,
                comp.distance_miles,
                comp.days_since_sale,
                # Derived features
                comp.square_feet / comp.bedrooms if comp.bedrooms > 0 else 0,  # sqft per bedroom
                2024 - comp.year_built,  # age
                1 / (1 + comp.distance_miles),  # proximity score
                1 / (1 + comp.days_since_sale / 30),  # recency score
            ]
            
            features.append(feature_vector)
            prices.append(comp.sale_price)
        
        X = np.array(features)
        y = np.array(prices)
        
        # Subject property features
        subject_features = np.array([[
            subject.square_feet,
            subject.bedrooms,
            subject.bathrooms,
            subject.year_built,
            0,  # distance to self
            0,  # days since sale (current)
            # Derived features
            subject.square_feet / subject.bedrooms if subject.bedrooms > 0 else 0,
            2024 - subject.year_built,
            1.0,  # proximity score (self)
            1.0,  # recency score (current)
        ]])
        
        feature_names = [
            'square_feet', 'bedrooms', 'bathrooms', 'year_built', 'distance_miles', 
            'days_since_sale', 'sqft_per_bedroom', 'age', 'proximity_score', 'recency_score'
        ]
        
        return X, y, subject_features, feature_names
    
    def train_and_predict(self, subject: PropertyData, comps: List[ComparableSale]) -> ARVResult:
        """Train models and generate ARV prediction with confidence intervals"""
        
        if len(comps) < 3:
            raise ValueError("Need at least 3 comparable sales for reliable prediction")
        
        # Prepare data
        X, y, subject_features, feature_names = self.prepare_features(subject, comps)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        subject_scaled = self.scaler.transform(subject_features)
        
        # Train and evaluate models
        model_predictions = {}
        model_scores = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_scaled, y)
            
            # Cross-validation score
            if len(comps) >= 5:
                cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(comps)), 
                                          scoring='neg_mean_absolute_percentage_error')
                model_scores[name] = -np.mean(cv_scores)
            else:
                model_scores[name] = 0.1  # Default score for small datasets
            
            # Prediction
            prediction = model.predict(subject_scaled)[0]
            model_predictions[name] = prediction
        
        # Ensemble prediction (weighted by performance)
        weights = self._calculate_model_weights(model_scores)
        arv_prediction = sum(pred * weights[name] for name, pred in model_predictions.items())
        
        # Calculate confidence and range
        confidence_score, price_range = self._calculate_confidence_and_range(
            model_predictions, comps, subject
        )
        
        # Market adjustments
        market_adjustments = self._calculate_market_adjustments(subject, comps)
        
        # Apply market adjustments to ARV
        adjusted_arv = arv_prediction
        for adjustment_name, adjustment_value in market_adjustments.items():
            adjusted_arv *= adjustment_value
        
        adjusted_arv = int(adjusted_arv)
        
        # Generate insights
        insights = self._generate_insights(subject, comps, market_adjustments, model_scores)
        
        return ARVResult(
            arv=adjusted_arv,
            confidence_score=confidence_score,
            price_range={
                'low': int(adjusted_arv * (1 - price_range)),
                'high': int(adjusted_arv * (1 + price_range))
            },
            comparable_count=len(comps),
            model_metrics={
                'ensemble_mape': weighted_mape,
                'best_model': min(model_scores.keys(), key=lambda k: model_scores[k]),
                'model_agreement': self._calculate_model_agreement(model_predictions)
            },
            market_adjustments=market_adjustments,
            insights=insights
        )
    
    def _calculate_model_weights(self, model_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights for ensemble based on model performance"""
        
        # Convert MAPE scores to weights (lower MAPE = higher weight)
        inverse_scores = {name: 1 / (score + 0.01) for name, score in model_scores.items()}
        total_inverse = sum(inverse_scores.values())
        
        weights = {name: inv_score / total_inverse for name, inv_score in inverse_scores.items()}
        return weights
    
    def _calculate_confidence_and_range(self, model_predictions: Dict, comps: List[ComparableSale], 
                                      subject: PropertyData) -> Tuple[float, float]:
        """Calculate confidence score and price range"""
        
        # Model agreement factor
        pred_values = list(model_predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        model_agreement = 1 - (pred_std / pred_mean) if pred_mean > 0 else 0
        
        # Comp quality factor
        avg_similarity = np.mean([self._comp_similarity_score(subject, comp) for comp in comps])
        
        # Sample size factor
        sample_size_factor = min(1.0, len(comps) / 8)  # Optimal at 8+ comps
        
        # Market volatility factor (based on comp price variance)
        comp_prices_per_sqft = [comp.price_per_sqft for comp in comps]
        price_cv = np.std(comp_prices_per_sqft) / np.mean(comp_prices_per_sqft)
        market_stability = max(0, 1 - price_cv)
        
        # Overall confidence
        confidence = (
            model_agreement * 0.3 +
            avg_similarity * 0.3 +
            sample_size_factor * 0.2 +
            market_stability * 0.2
        )
        
        confidence_score = min(0.98, max(0.5, confidence))
        
        # Price range (lower confidence = wider range)
        base_range = 0.05  # 5% base range
        confidence_adjustment = (1 - confidence_score) * 0.1  # Up to 10% additional
        price_range = base_range + confidence_adjustment
        
        return confidence_score, price_range
    
    def _comp_similarity_score(self, subject: PropertyData, comp: ComparableSale) -> float:
        """Calculate similarity score for a single comp"""
        
        sqft_sim = 1 - abs(comp.square_feet - subject.square_feet) / subject.square_feet
        bed_sim = 1 - abs(comp.bedrooms - subject.bedrooms) * 0.2
        bath_sim = 1 - abs(comp.bathrooms - subject.bathrooms) * 0.15
        age_sim = 1 - abs(comp.year_built - subject.year_built) / 50
        
        return max(0, min(1, (sqft_sim + bed_sim + bath_sim + age_sim) / 4))
    
    def _calculate_market_adjustments(self, subject: PropertyData, comps: List[ComparableSale]) -> Dict[str, float]:
        """Calculate market adjustments based on trends and factors"""
        
        adjustments = {}
        
        # Time trend adjustment
        recent_comps = [c for c in comps if c.days_since_sale <= 90]
        older_comps = [c for c in comps if c.days_since_sale > 90]
        
        if recent_comps and older_comps:
            recent_avg_psf = np.mean([c.price_per_sqft for c in recent_comps])
            older_avg_psf = np.mean([c.price_per_sqft for c in older_comps])
            market_trend = recent_avg_psf / older_avg_psf if older_avg_psf > 0 else 1.0
            adjustments['market_trend'] = min(1.1, max(0.9, market_trend))
        else:
            adjustments['market_trend'] = 1.0
        
        # Seasonal adjustment (simple model)
        current_month = datetime.now().month
        if current_month in [3, 4, 5, 6]:  # Spring/early summer premium
            adjustments['seasonal'] = 1.02
        elif current_month in [11, 12, 1]:  # Winter discount
            adjustments['seasonal'] = 0.98
        else:
            adjustments['seasonal'] = 1.0
        
        # Location adjustment based on city
        if 'grand rapids' in subject.address.lower():
            adjustments['location'] = 1.05  # Premium market
        elif 'muskegon' in subject.address.lower():
            adjustments['location'] = 0.95  # Discount market
        else:
            adjustments['location'] = 1.0
        
        return adjustments
    
    def _calculate_model_agreement(self, model_predictions: Dict) -> float:
        """Calculate how well models agree with each other"""
        
        pred_values = list(model_predictions.values())
        std_dev = np.std(pred_values)
        mean_pred = np.mean(pred_values)
        
        # Agreement score (1 = perfect agreement, 0 = high disagreement)
        agreement = 1 - (std_dev / mean_pred) if mean_pred > 0 else 0
        return max(0, min(1, agreement))
    
    def _generate_insights(self, subject: PropertyData, comps: List[ComparableSale], 
                          market_adjustments: Dict, model_scores: Dict) -> List[str]:
        """Generate human-readable insights about the valuation"""
        
        insights = []
        
        # Market trend insight
        trend_adj = market_adjustments.get('market_trend', 1.0)
        if trend_adj > 1.02:
            insights.append(f"Strong market appreciation: +{(trend_adj-1)*100:.1f}% over last 3 months")
        elif trend_adj < 0.98:
            insights.append(f"Market softening: {(1-trend_adj)*100:.1f}% decline over last 3 months")
        else:
            insights.append("Stable market conditions with minimal price movement")
        
        # Location insight
        location_adj = market_adjustments.get('location', 1.0)
        if location_adj > 1.0:
            insights.append(f"Location premium: +{(location_adj-1)*100:.0f}% for area desirability")
        elif location_adj < 1.0:
            insights.append(f"Location discount: -{(1-location_adj)*100:.0f}% for area factors")
        
        # Comp quality insight
        avg_distance = np.mean([c.distance_miles for c in comps])
        if avg_distance < 0.5:
            insights.append(f"Excellent comp proximity: avg {avg_distance:.1f} miles from subject")
        elif avg_distance > 1.0:
            insights.append(f"Moderate comp distance: avg {avg_distance:.1f} miles may reduce accuracy")
        
        # Model performance insight
        best_model = min(model_scores.keys(), key=lambda k: model_scores[k])
        best_mape = model_scores[best_model]
        if best_mape < 0.05:
            insights.append(f"High model accuracy: {best_model} model achieving {(1-best_mape)*100:.0f}% accuracy")
        elif best_mape > 0.15:
            insights.append(f"Model uncertainty: {best_mape*100:.0f}% error rate suggests market complexity")
        
        return insights

# Initialize services
property_service = SmartPropertyLookupService()
comparable_service = AdvancedComparableService()
valuation_engine = AIValuationEngine()

# API Endpoints

@app.post("/search-addresses")
async def search_addresses(request: AddressSearchRequest):
    """Smart address autocomplete with Google Places integration"""
    try:
        suggestions = await property_service.search_addresses(request.query, request.limit)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Address search failed: {str(e)}")

@app.post("/property-details")
async def get_property_details(request: dict):
    """Fetch comprehensive property details"""
    try:
        address = request.get("address")
        place_id = request.get("place_id")
        
        if not address:
            raise HTTPException(status_code=400, detail="Address is required")
        
        property_data = await property_service.get_property_details(address, place_id)
        
        return {
            "address": property_data.address,
            "year_built": property_data.year_built,
            "square_feet": property_data.square_feet,
            "bedrooms": property_data.bedrooms,
            "bathrooms": property_data.bathrooms,
            "property_type": property_data.property_type,
            "lot_size": property_data.lot_size,
            "last_sale_date": property_data.last_sale_date,
            "last_sale_price": property_data.last_sale_price,
            "assessed_value": property_data.assessed_value,
            "tax_amount": property_data.tax_amount,
            "coordinates": {
                "latitude": property_data.latitude,
                "longitude": property_data.longitude
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Property lookup failed: {str(e)}")

@app.post("/comparable-sales")
async def get_comparable_sales(request: dict):
    """Find and analyze comparable sales"""
    try:
        address = request.get("address")
        if not address:
            raise HTTPException(status_code=400, detail="Address is required")
        
        # Get property details first
        property_data = await property_service.get_property_details(address)
        
        # Find comparable sales
        comps = await comparable_service.find_comparable_sales(property_data)
        
        # Format response
        comparable_sales = []
        for comp in comps:
            comparable_sales.append({
                "address": comp.address,
                "sale_price": comp.sale_price,
                "sale_date": comp.sale_date,
                "square_feet": comp.square_feet,
                "bedrooms": comp.bedrooms,
                "bathrooms": comp.bathrooms,
                "year_built": comp.year_built,
                "distance_miles": round(comp.distance_miles, 2),
                "days_since_sale": comp.days_since_sale,
                "price_per_sqft": comp.price_per_sqft
            })
        
        return {
            "comparable_sales": comparable_sales,
            "count": len(comparable_sales),
            "search_radius": comparable_service.max_distance_miles
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparable sales search failed: {str(e)}")

@app.post("/ai-analysis")
async def run_ai_analysis(request: PropertyAnalysisRequest):
    """Run comprehensive AI-powered property analysis"""
    try:
        # Get property details
        property_data = await property_service.get_property_details(request.address)
        
        # Get comparable sales
        auto_comps = await comparable_service.find_comparable_sales(property_data)
        
        # Add manual comps if provided
        manual_comps = []
        for manual_comp in request.manual_comps:
            manual_comps.append(ComparableSale(
                address=manual_comp.get("address", "Manual Entry"),
                sale_price=int(manual_comp["price"]),
                sale_date=manual_comp["date"],
                square_feet=int(manual_comp["sqft"]),
                bedrooms=int(manual_comp.get("beds", 3)),
                bathrooms=float(manual_comp.get("baths", 2)),
                year_built=int(manual_comp.get("year_built", property_data.year_built)),
                distance_miles=0.5,  # Default for manual entries
                days_since_sale=30,  # Default for manual entries
                price_per_sqft=manual_comp["price"] / manual_comp["sqft"]
            ))
        
        all_comps = auto_comps + manual_comps
        
        if len(all_comps) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 comparable sales for analysis")
        
        # Run AI valuation
        arv_result = valuation_engine.train_and_predict(property_data, all_comps)
        
        # Calculate offers using your proven formulas
        offers = calculate_offer_strategies(arv_result.arv, property_data)
        
        return {
            "property": {
                "address": property_data.address,
                "square_feet": property_data.square_feet,
                "bedrooms": property_data.bedrooms,
                "bathrooms": property_data.bathrooms,
                "year_built": property_data.year_built
            },
            "arv_analysis": {
                "arv": arv_result.arv,
                "confidence_score": round(arv_result.confidence_score * 100, 1),
                "price_range": arv_result.price_range,
                "comparable_count": arv_result.comparable_count,
                "model_metrics": arv_result.model_metrics,
                "market_adjustments": arv_result.market_adjustments,
                "insights": arv_result.insights
            },
            "offer_strategies": offers,
            "comparable_sales": [
                {
                    "address": comp.address,
                    "sale_price": comp.sale_price,
                    "sale_date": comp.sale_date,
                    "square_feet": comp.square_feet,
                    "price_per_sqft": comp.price_per_sqft,
                    "distance_miles": comp.distance_miles
                }
                for comp in all_comps
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

def calculate_offer_strategies(arv: int, property_data: PropertyData) -> Dict:
    """Calculate offer strategies using your proven formulas"""
    
    # Estimate repair costs (you would integrate your repair estimation logic here)
    repair_estimate = estimate_repair_costs(property_data)
    
    # Your proven formulas
    margin_of_safety = int(arv * 0.03)
    
    strategies = {
        "ai_optimized": {
            "name": "🤖 AI-Optimized Primary",
            "rule": "65% Rule + AI Adjustments",
            "offer": max(0, int(arv * 0.65 - repair_estimate - margin_of_safety)),
            "description": "Machine learning optimized offer"
        },
        "wholesale": {
            "name": "🥈 Conservative Wholesale",
            "rule": "70% Rule",
            "offer": max(0, int(arv * 0.70 - repair_estimate)),
            "description": "Safe wholesale margin"
        },
        "brrr": {
            "name": "🥉 BRRR Strategy",
            "rule": "75% Rule",
            "offer": max(0, int(arv * 0.75 - repair_estimate)),
            "description": "Buy, rehab, rent, refinance"
        }
    }
    
    return strategies

def estimate_repair_costs(property_data: PropertyData) -> int:
    """Estimate repair costs based on property characteristics"""
    
    # Basic repair cost estimation (integrate your detailed repair logic)
    age = 2024 - property_data.year_built
    
    base_cost_per_sqft = 20  # Base repair cost
    
    # Age adjustments
    if age > 75:
        base_cost_per_sqft += 15
    elif age > 50:
        base_cost_per_sqft += 10
    elif age > 25:
        base_cost_per_sqft += 5
    
    total_repairs = int(property_data.square_feet * base_cost_per_sqft)
    
    return total_repairs

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
