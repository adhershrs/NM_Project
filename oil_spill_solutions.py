def get_oil_spill_solution(confidence):
    """
    Return appropriate oil spill solutions based on the confidence level
    of the model's prediction. Higher confidence suggests more severe action.
    
    Args:
        confidence: Float between 0 and 1 representing the model's confidence
                   that the image contains an oil spill
    
    Returns:
        Dictionary with actions and materials lists
    """
    solutions = {
        'critical': {
            'actions': [
                "URGENT: Immediate deployment of containment booms",
                "Activate full emergency response team",
                "Notify coast guard, environmental agencies, and local authorities",
                "Begin intensive skimmer operations",
                "Deploy aerial dispersant application if approved",
                "Establish wildlife rescue centers"
            ],
            'materials': [
                "Containment booms (1000m minimum)",
                "Heavy-duty oil skimmers (10+ units)",
                "Approved dispersants for large-scale application",
                "Full protective gear for 100+ workers",
                "Multiple support vessels and aircraft",
                "Wildlife rehabilitation supplies"
            ]
        },
        'high': {
            'actions': [
                "Immediate deployment of containment booms",
                "Activate emergency response team",
                "Notify coast guard and environmental agencies",
                "Begin skimmer operations",
                "Set up wildlife protection stations",
                "Establish security perimeter"
            ],
            'materials': [
                "Containment booms (500m minimum)",
                "Oil skimmers (5+ units)",
                "Biodegradable dispersants",
                "Protective gear for 50 workers",
                "Wildlife rescue equipment",
                "Communication equipment"
            ]
        },
        'medium': {
            'actions': [
                "Deploy monitoring drones",
                "Prepare cleanup crews on standby",
                "Water sampling for analysis",
                "Public advisory announcement",
                "Mobilize equipment to staging areas"
            ],
            'materials': [
                "Absorbent pads (1000 units)",
                "Portable skimmers (2-3 units)",
                "Water testing kits",
                "Safety signage",
                "Containment booms (200m on standby)"
            ]
        },
        'low': {
            'actions': [
                "Enhanced satellite monitoring",
                "Coastal patrols",
                "Review emergency protocols",
                "Public awareness campaign",
                "Continue monitoring the area"
            ],
            'materials': [
                "Monitoring buoys (5 units)",
                "Informational brochures",
                "Waterproof cameras",
                "Training materials",
                "Basic sampling equipment"
            ]
        },
        'monitoring': {
            'actions': [
                "Continue monitoring the area",
                "Schedule follow-up satellite imagery",
                "Maintain readiness of response team",
                "Document observations"
            ],
            'materials': [
                "Camera equipment",
                "GPS tracking devices",
                "Weather monitoring tools",
                "Reporting templates"
            ]
        }
    }
    
    # Provide appropriate solutions based on confidence level with more nuanced thresholds
    if confidence >= 0.85:
        return solutions['critical']
    elif confidence >= 0.65:
        return solutions['high']
    elif confidence >= 0.45:
        return solutions['medium']
    elif confidence >= 0.25:
        return solutions['low']
    else:
        return solutions['monitoring']  # Very low confidence gets minimal response