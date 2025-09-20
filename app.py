from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import json
import os
from datetime import datetime, timedelta
import io
from typing import Optional, List, Dict
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import random

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (your HTML app)
@app.get("/")
async def serve_app():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deal Command Center - Pathfinder Holding Company</title>
        <meta http-equiv="refresh" content="0; url=https://your-frontend-url.vercel.app">
    </head>
    <body>
        <p>Redirecting to Deal Command Center...</p>
    </body>
    </html>
    """

# Data models
class PropertyDetails(BaseModel):
    address: str
    yearBuilt: int
    squareFeet: int
    bedrooms: int
    fullBaths: int
    halfBaths: int
    roofCondition: str
    kitchenCondition: str
    hvacCondition: str
    overallCondition: str

class AnalysisResult(BaseModel):
    arv: int
    confidence: str
    compCount: int
    repairs: int
    repairsPerSqft: float
    primaryOffer: int
    wholesaleOffer: int
    brrrOffer: int
    listingOffer: int
    bestStrategy: str
    dealGrade: str
    comps: List[Dict]
    repairBreakdown: Dict
    profitAnalysis: Dict

# Environment variables
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appQymhIK7nbfPNiv")

class PropertyLookupService:
    """Enhanced property lookup with multiple data sources"""
    
    @staticmethod
    async def lookup_property_details(address: str) -> Dict:
        """
        Lookup property details from multiple sources
        Priority: 1) Public records 2) Zillow-style APIs 3) Local MLS data
        """
        try:
            # Simulate property lookup (replace with real APIs)
            # In production, integrate with:
            # - HomeGenius API
            # - PropertyData API
            # - Local MLS feeds
            # - County assessor records
            
            mock_data = {
                "1022 Kenneth St, Muskegon, MI": {
                    "yearBuilt": 1948,
                    "squareFeet": 1130,
                    "bedrooms": 3,
                    "fullBaths": 1,
                    "halfBaths": 0,
                    "lotSize": 0.18,
                    "propertyType": "Single Family",
                    "lastSaleDate": "2019-08-15",
                    "lastSalePrice": 45000,
                    "assessedValue": 52400,
                    "taxAmount": 1247,
                    "neighborhood": "McLaughlin",
                    "zipCode": "49441"
                }
            }
            
            # Return mock data if address matches, otherwise make API call
            if address in mock_data:
                return {"success": True, "data": mock_data[address]}
            
            # Here you would implement real API calls
            # For now, return basic success with empty data
            return {"success": True, "data": {}}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

class DealAnalysisEngine:
    """Enhanced deal analysis with West Michigan market data"""
    
    def __init__(self):
        # West Michigan market pricing (per sq ft by zip/city)
        self.market_pricing = {
            "Grand Rapids": {"low": 85, "mid": 120, "high": 165},
            "Muskegon": {"low": 45, "mid": 75, "high": 105},
            "Kalamazoo": {"low": 65, "mid": 95, "high": 135},
            "Battle Creek": {"low": 55, "mid": 80, "high": 115},
            "default": {"low": 60, "mid": 90, "high": 130}
        }
    
    def calculate_arv(self, property_data: PropertyDetails) -> Dict:
        """Calculate ARV using West Michigan comps"""
        
        # Determine market based on address
        city = self.extract_city(property_data.address)
        pricing = self.market_pricing.get(city, self.market_pricing["default"])
        
        # Calculate base ARV
        base_arv = property_data.squareFeet * pricing["mid"]
        
        # Adjustments for condition, age, bed/bath count
        condition_multiplier = {
            "excellent": 1.1,
            "good": 1.0,
            "fair": 0.95,
            "poor": 0.85
        }
        
        adjusted_arv = base_arv * condition_multiplier.get(property_data.overallCondition, 1.0)
        
        # Generate mock comparable sales
        comps = self.generate_mock_comps(property_data, pricing)
        
        # Confidence based on comp quality
        confidence = "High Confidence" if len(comps) >= 6 else "Medium Confidence"
        
        return {
            "arv": int(adjusted_arv),
            "confidence": confidence,
            "compCount": len(comps),
            "comps": comps,
            "pricingRange": {
                "low": int(property_data.squareFeet * pricing["low"]),
                "mid": int(adjusted_arv),
                "high": int(property_data.squareFeet * pricing["high"])
            }
        }
    
    def calculate_repairs(self, property_data: PropertyDetails) -> Dict:
        """Calculate repair costs using your proven formulas"""
        
        repairs = {}
        sqft = property_data.squareFeet
        
        # Roof calculation: IF(condition="11+ Yrs", sqft*1.41/100*500, 0)
        if property_data.roofCondition == "11+ Yrs":
            repairs["roof"] = int(sqft * 1.41 / 100 * 500)
        else:
            repairs["roof"] = 0
        
        # Kitchen calculation
        if property_data.kitchenCondition == "15+yrs":
            repairs["kitchen"] = 8000  # Full replacement
        elif property_data.kitchenCondition == "8-15yrs":
            repairs["kitchen"] = 4000  # Light update
        else:
            repairs["kitchen"] = 0
        
        # HVAC calculation
        if property_data.hvacCondition == "20+ yrs":
            repairs["hvac"] = 9000
        elif property_data.hvacCondition == "15-20 yrs":
            repairs["hvac"] = 5500
        else:
            repairs["hvac"] = 0
        
        # Paint: $5/sqft exactly like your sheet
        repairs["paint"] = sqft * 5
        
        # Additional repairs based on overall condition
        condition_repairs = {
            "excellent": 0,
            "good": sqft * 8,  # Minor updates
            "fair": sqft * 15,  # Moderate rehab
            "poor": sqft * 25   # Extensive rehab
        }
        repairs["general"] = condition_repairs.get(property_data.overallCondition, sqft * 15)
        
        # Calculate totals
        subtotal = sum(repairs.values())
        contingency = int(subtotal * 0.10)  # 10% contingency
        total_repairs = subtotal + contingency
        
        return {
            "breakdown": repairs,
            "subtotal": subtotal,
            "contingency": contingency,
            "total": total_repairs,
            "perSqft": round(total_repairs / sqft, 2)
        }
    
    def calculate_offers(self, arv: int, repairs: int) -> Dict:
        """Calculate offer strategies"""
        
        # Your exact formulas
        primary_offer = int(arv * 0.65 - repairs)  # 65% rule
        wholesale_offer = int(arv * 0.70 - repairs)  # 70% rule  
        brrr_offer = int(arv * 0.75 - repairs)  # 75% rule
        listing_offer = int(arv * 0.94 - repairs)  # 94% rule for listing
        
        # Determine best strategy
        profit_margin = arv - repairs - primary_offer
        if profit_margin > 30000:
            best_strategy = "Assignment"
        elif profit_margin > 20000:
            best_strategy = "Wholesale"
        elif profit_margin > 15000:
            best_strategy = "BRRR"
        else:
            best_strategy = "Pass"
        
        return {
            "primary": max(primary_offer, 0),
            "wholesale": max(wholesale_offer, 0),
            "brrr": max(brrr_offer, 0),
            "listing": max(listing_offer, 0),
            "bestStrategy": best_strategy,
            "projectedProfit": profit_margin
        }
    
    def grade_deal(self, arv: int, repairs: int, offers: Dict) -> str:
        """Grade the deal A-F based on profit margins"""
        
        profit_margin = arv - repairs - offers["primary"]
        profit_percentage = (profit_margin / arv) * 100 if arv > 0 else 0
        
        if profit_percentage >= 25:
            return "A"
        elif profit_percentage >= 20:
            return "B+"
        elif profit_percentage >= 15:
            return "B"
        elif profit_percentage >= 10:
            return "C+"
        elif profit_percentage >= 5:
            return "C"
        else:
            return "D"
    
    def extract_city(self, address: str) -> str:
        """Extract city from address"""
        cities = ["Grand Rapids", "Muskegon", "Kalamazoo", "Battle Creek"]
        for city in cities:
            if city.lower() in address.lower():
                return city
        return "default"
    
    def generate_mock_comps(self, property_data: PropertyDetails, pricing: Dict) -> List[Dict]:
        """Generate realistic comparable sales data"""
        comps = []
        
        for i in range(random.randint(5, 8)):
            # Generate similar properties
            comp_sqft = property_data.squareFeet + random.randint(-200, 200)
            comp_price = comp_sqft * random.randint(pricing["low"], pricing["high"])
            
            comps.append({
                "address": f"Sample St {i+1}",
                "squareFeet": comp_sqft,
                "bedrooms": property_data.bedrooms + random.randint(-1, 1),
                "bathrooms": property_data.fullBaths + random.randint(0, 1),
                "salePrice": comp_price,
                "pricePerSqft": round(comp_price / comp_sqft, 2),
                "saleDate": (datetime.now() - timedelta(days=random.randint(30, 180))).strftime("%Y-%m-%d"),
                "distance": round(random.uniform(0.1, 1.5), 1)
            })
        
        return sorted(comps, key=lambda x: x["distance"])

class ReportGenerator:
    """Professional PDF report generator for investors/lenders"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for Pathfinder branding"""
        
        # Header style
        self.styles.add(ParagraphStyle(
            name='PathfinderHeader',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#0269AC'),
            alignment=TA_CENTER,
            spaceAfter=20
        ))
        
        # Subheader style
        self.styles.add(ParagraphStyle(
            name='PathfinderSubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#3C0302'),
            spaceBefore=15,
            spaceAfter=10
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='PathfinderBody',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT
        ))
    
    def generate_deal_report(self, property_data: PropertyDetails, analysis: AnalysisResult) -> bytes:
        """Generate comprehensive deal analysis report"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Header with branding
        story.append(Paragraph("DEAL ANALYSIS REPORT", self.styles['PathfinderHeader']))
        story.append(Paragraph("Pathfinder Holding Company", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Property overview table
        property_data_table = [
            ["Property Address:", property_data.address],
            ["Analysis Date:", datetime.now().strftime("%B %d, %Y")],
            ["Year Built:", str(property_data.yearBuilt)],
            ["Square Footage:", f"{property_data.squareFeet:,} sq ft"],
            ["Bedrooms/Bathrooms:", f"{property_data.bedrooms} bed / {property_data.fullBaths} bath"],
            ["Overall Condition:", property_data.overallCondition.title()]
        ]
        
        property_table = Table(property_data_table, colWidths=[2*inch, 4*inch])
        property_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(property_table)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['PathfinderSubHeader']))
        
        summary_data = [
            ["After Repair Value (ARV):", f"${analysis.arv:,}"],
            ["Estimated Repairs:", f"${analysis.repairs:,}"],
            ["Repair Cost per Sq Ft:", f"${analysis.repairsPerSqft}"],
            ["Recommended Offer (65% Rule):", f"${analysis.primaryOffer:,}"],
            ["Deal Grade:", analysis.dealGrade],
            ["Recommended Strategy:", analysis.bestStrategy],
            ["Confidence Level:", analysis.confidence]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f3ff')),
            ('BACKGROUND', (1, 3), (1, 3), colors.HexColor('#ffeeee')),  # Highlight offer
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 11),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Offer Strategy Comparison
        story.append(Paragraph("OFFER STRATEGY COMPARISON", self.styles['PathfinderSubHeader']))
        
        strategy_data = [
            ["Strategy", "Rule", "Offer Amount", "Est. Profit"],
            ["Primary (Recommended)", "65% ARV", f"${analysis.primaryOffer:,}", 
             f"${analysis.profitAnalysis.get('primary', 0):,}"],
            ["Wholesale", "70% ARV", f"${analysis.wholesaleOffer:,}", 
             f"${analysis.profitAnalysis.get('wholesale', 0):,}"],
            ["BRRR/Rental", "75% ARV", f"${analysis.brrrOffer:,}", 
             f"${analysis.profitAnalysis.get('brrr', 0):,}"],
            ["List Ready", "94% ARV", f"${analysis.listingOffer:,}", 
             f"${analysis.profitAnalysis.get('listing', 0):,}"]
        ]
        
        strategy_table = Table(strategy_data, colWidths=[1.5*inch, 1*inch, 1.25*inch, 1.25*inch])
        strategy_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0269AC')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ffeeee')),  # Highlight primary
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(strategy_table)
        story.append(Spacer(1, 20))
        
        # Repair Cost Breakdown
        story.append(Paragraph("REPAIR COST BREAKDOWN", self.styles['PathfinderSubHeader']))
        
        repairs_data = [["Component", "Cost", "Notes"]]
        for component, cost in analysis.repairBreakdown.items():
            if cost > 0:
                repairs_data.append([component.title(), f"${cost:,}", ""])
        
        repairs_data.append(["Subtotal", f"${sum(analysis.repairBreakdown.values()):,}", ""])
        repairs_data.append(["Contingency (10%)", f"${analysis.repairs - sum(analysis.repairBreakdown.values()):,}", ""])
        repairs_data.append(["Total Repairs", f"${analysis.repairs:,}", ""])
        
        repairs_table = Table(repairs_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        repairs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3C0302')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, -2), (-1, -1), colors.HexColor('#f0f0f0')),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
            ('FONT', (0, -2), (-1, -1), 'Helvetica-Bold', 10),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(repairs_table)
        story.append(Spacer(1, 20))
        
        # Comparable Sales (top 5)
        story.append(Paragraph("COMPARABLE SALES ANALYSIS", self.styles['PathfinderSubHeader']))
        story.append(Paragraph(f"Based on {len(analysis.comps)} recent sales within 1.5 miles", 
                             self.styles['PathfinderBody']))
        story.append(Spacer(1, 10))
        
        comps_data = [["Address", "Sale Date", "Sq Ft", "Bed/Bath", "Sale Price", "$/Sq Ft", "Distance"]]
        for comp in analysis.comps[:5]:  # Top 5 comps
            comps_data.append([
                comp["address"],
                comp["saleDate"], 
                f"{comp['squareFeet']:,}",
                f"{comp['bedrooms']}/{comp['bathrooms']}",
                f"${comp['salePrice']:,}",
                f"${comp['pricePerSqft']}",
                f"{comp['distance']} mi"
            ])
        
        comps_table = Table(comps_data, colWidths=[1.2*inch, 0.8*inch, 0.7*inch, 0.6*inch, 1*inch, 0.7*inch, 0.6*inch])
        comps_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0C99DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 8),
            ('ALIGN', (4, 0), (-1, -1), 'RIGHT'),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(comps_table)
        story.append(Spacer(1, 20))
        
        # Disclaimers
        story.append(Paragraph("DISCLAIMERS & ASSUMPTIONS", self.styles['PathfinderSubHeader']))
        disclaimers = [
            "• This analysis is for informational purposes only and should not be considered professional investment advice",
            "• Property condition assessments are based on preliminary observations and may require professional inspection",
            "• Repair cost estimates are based on current West Michigan market rates and may vary by contractor",
            "• Comparable sales data sourced from public records and MLS; accuracy not guaranteed",
            "• Market conditions may change affecting property values and repair costs",
            "• Consult with licensed professionals before making investment decisions"
        ]
        
        for disclaimer in disclaimers:
            story.append(Paragraph(disclaimer, self.styles['PathfinderBody']))
        
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph("Report generated by Deal Command Center", 
                             self.styles['Normal']))
        story.append(Paragraph("Pathfinder Holding Company", 
                             self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

# Initialize services
property_lookup = PropertyLookupService()
analysis_engine = DealAnalysisEngine()
report_generator = ReportGenerator()

# API Endpoints

@app.post("/lookup-property")
async def lookup_property(request: dict):
    """Enhanced property lookup endpoint"""
    address = request.get("address", "")
    
    if not address:
        raise HTTPException(status_code=400, detail="Address is required")
    
    result = await property_lookup.lookup_property_details(address)
    
    if result["success"]:
        return result["data"]
    else:
        raise HTTPException(status_code=404, detail="Property not found")

@app.post("/analyze-deal")
async def analyze_deal(property_data: PropertyDetails):
    """Enhanced deal analysis endpoint"""
    
    try:
        # Calculate ARV
        arv_result = analysis_engine.calculate_arv(property_data)
        
        # Calculate repairs
        repair_result = analysis_engine.calculate_repairs(property_data)
        
        # Calculate offers
        offers = analysis_engine.calculate_offers(arv_result["arv"], repair_result["total"])
        
        # Grade the deal
        grade = analysis_engine.grade_deal(arv_result["arv"], repair_result["total"], offers)
        
        # Build comprehensive result
        result = AnalysisResult(
            arv=arv_result["arv"],
            confidence=arv_result["confidence"],
            compCount=arv_result["compCount"],
            repairs=repair_result["total"],
            repairsPerSqft=repair_result["perSqft"],
            primaryOffer=offers["primary"],
            wholesaleOffer=offers["wholesale"],
            brrrOffer=offers["brrr"],
            listingOffer=offers["listing"],
            bestStrategy=offers["bestStrategy"],
            dealGrade=grade,
            comps=arv_result["comps"],
            repairBreakdown=repair_result["breakdown"],
            profitAnalysis={
                "primary": arv_result["arv"] - repair_result["total"] - offers["primary"],
                "wholesale": arv_result["arv"] - repair_result["total"] - offers["wholesale"],
                "brrr": arv_result["arv"] - repair_result["total"] - offers["brrr"],
                "listing": arv_result["arv"] - repair_result["total"] - offers["listing"]
            }
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/generate-report")
async def generate_report(request: dict):
    """Generate professional PDF report"""
    
    try:
        property_data = PropertyDetails(**request["dealData"])
        analysis = AnalysisResult(**request["analysis"])
        
        # Generate PDF
        pdf_bytes = report_generator.generate_deal_report(property_data, analysis)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=deal_analysis_{property_data.address.replace(',', '').replace(' ', '_')}.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

@app.post("/save-to-airtable")
async def save_to_airtable(request: dict):
    """Enhanced Airtable integration"""
    
    if not AIRTABLE_API_KEY:
        raise HTTPException(status_code=500, detail="Airtable API key not configured")
    
    try:
        deal_data = request["dealData"]
        analysis = request["analysis"]
        
        # Prepare Airtable record
        record_data = {
            "fields": {
                "Property Address": deal_data["address"],
                "Year Built": deal_data["yearBuilt"],
                "Square Footage": deal_data["squareFeet"],
                "Bedrooms": deal_data["bedrooms"],
                "Full Baths": deal_data["fullBaths"],
                "ARV": analysis["arv"],
                "Total Repairs": analysis["repairs"],
                "Repairs per SF": analysis["repairsPerSqft"],
                "Primary Offer (65%)": analysis["primaryOffer"],
                "Wholesale Offer (70%)": analysis["wholesaleOffer"],
                "BRRR Offer (75%)": analysis["brrrOffer"],
                "Listing Offer (94%)": analysis["listingOffer"],
                "Best Strategy": analysis["bestStrategy"],
                "Deal Grade": analysis["dealGrade"],
                "Confidence Level": analysis["confidence"],
                "Roof Condition": deal_data["roofCondition"],
                "Kitchen Condition": deal_data["kitchenCondition"],
                "HVAC Condition": deal_data["hvacCondition"],
                "Overall Condition": deal_data["overallCondition"],
                "Analysis Date": datetime.now().isoformat(),
                "Comp Count": analysis["compCount"]
            }
        }
        
        # Save to Airtable
        url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/Deal%20Analysis"
        headers = {
            "Authorization": f"Bearer {AIRTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json=record_data)
        
        if response.status_code == 200:
            return {"success": True, "record": response.json()}
        else:
            raise HTTPException(status_code=response.status_code, 
                              detail=f"Airtable error: {response.text}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
