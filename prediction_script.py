import numpy as np
import requests
from geopy.geocoders import Nominatim
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2

# Load your pre-trained model
model = load_model('oil_spill_model.h5')  # Update with your model path

def get_coordinates(place_name):
    """Convert place name to latitude/longitude using OpenStreetMap's Nominatim"""
    geolocator = Nominatim(user_agent="oil_spill_detection")
    location = geolocator.geocode(place_name)
    if location:
        return location.latitude, location.longitude
    return None, None

def get_nasa_image(lat, lon, date_str):
    """Download satellite image from NASA GIBS using MODIS Terra Corrected Reflectance"""
    zoom = 7  # Zoom level (higher for more detail)
    
    # Calculate tile indices
    tile_x = int((lon + 180) / (360 / (2 ** zoom)))
    tile_y = int((90 - lat) / (180 / (2 ** (zoom - 1))))
    
    # Construct NASA GIBS URL
    layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
    url = f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/default/{date_str}/250m/{zoom}/{tile_y}/{tile_x}.jpg"
    
    # Download image
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    return None

def preprocess_image(img_content, target_size=(128, 128)):
    """Process image for model prediction"""
    img_path = "temp_image.jpg"
    with open(img_path, 'wb') as f:
        f.write(img_content)
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img) / 255.0
    return img

def predict_oil_spill(image_array):
    """Make prediction using the trained model"""
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    return prediction[0][0]

# User input
place_name = input("Enter location (e.g., Name of the sea or Ocean): ")
date_str = input("Enter date (YYYY-MM-DD): ")

# Get coordinates
lat, lon = get_coordinates(place_name)
if not lat or not lon:
    print("Location not found")
    exit()

# Download NASA satellite image
image_content = get_nasa_image(lat, lon, date_str)
if not image_content:
    print("Failed to download satellite image")
    exit()

# Preprocess and predict
processed_img = preprocess_image(image_content)
confidence = predict_oil_spill(processed_img)
result = "Oil Spill Detected" if confidence > 0.5 else "No Oil Spill Detected"

# Display results
img = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"{place_name} - {date_str}\n{result} (Confidence: {confidence:.2%})")
plt.axis('off')
plt.show()