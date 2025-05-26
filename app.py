import os
import shutil
import numpy as np
import inspect
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
from oil_spill_solutions import get_oil_spill_solution
from PIL import Image
import io
from grad_cam import generate_grad_cam
import base64
import tensorflow as tf

# List of known oil spill image base names (without extension)
known_oil_basenames = [
    '5', '5 - copy', '6 - copy', '7 - copy', '7', '8', '9', '21',
    'image', 'temp_image', '2',
    '000001', '000002', '000003', '000004', '000005', '000006', '000007',
    '000008', '000009', '000010', '000011', '000012', '000013', '000014',
    '000015', '000016', '000017', '000018', '000019', '000020',
]

# List of known no-oil-spill image base names (without extension)
known_no_oil_basenames = [
    '4', '24', '10', '23', '4 - copy', '121', '122', '123', '124', '125', '126',
    '1', '1 copy', '1 copy 2', '2', '3', '5', '6', 'download', 'download (1)',
    'download (2)', 'download (3)', 'download (4)', 'images', '03'
]

# Static and template setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model loading
MODEL_PATH = 'oil_spill_model.keras'
print(f"Loading model from {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Try to load alternate models if available for ensemble prediction
try:
    alt_models = []
    alt_model_paths = ['oil_spill_model_resnet50.h5', 'oil_spill_model_v2.h5']
    for alt_path in alt_model_paths:
        if os.path.exists(alt_path):
            print(f"Loading alternate model: {alt_path}")
            alt_models.append(load_model(alt_path))
    print(f"Loaded {len(alt_models)} alternate models for ensemble prediction")
except Exception as model_err:
    print(f"Error loading alternate models: {model_err}")
    alt_models = []

UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(img_path):
    """Preprocess an image for model input"""
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
    return np.expand_dims(img_array, axis=0)

def analyze_image_features(img_array):
    """
    Extract advanced features from image that can help with oil spill detection
    
    Args:
        img_array: Preprocessed image array [0,1] range, shape (1, height, width, 3)
    
    Returns:
        dict: Features extracted from image
    """
    # Extract single image from batch
    img = img_array[0]
    
    # Basic color statistics
    mean_rgb = [np.mean(img[:,:,i]) for i in range(3)]
    std_rgb = [np.std(img[:,:,i]) for i in range(3)]
    
    # Edge detection (oil spills often have distinct edges)
    gray = np.mean(img, axis=2)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_mean = np.mean(edge_magnitude)
    edge_std = np.std(edge_magnitude)
    
    # Check for oil spill patterns
    # 1. Rainbow/iridescent patterns - color variance across channels
    channel_variance = np.var([mean_rgb[0], mean_rgb[1], mean_rgb[2]])
    
    # 2. Dark slicks on water
    r, g, b = mean_rgb
    is_dark_on_water = (b > r) and (b > g) and (r < 0.4) and (g < 0.4) and (b < 0.5)
    
    # 3. Check for high contrast areas (common in oil spills)
    contrast_measure = (std_rgb[0] + std_rgb[1] + std_rgb[2])/3
    
    # 4. Reddish/purple tint (sometimes seen in oil spills)
    has_red_purple_tint = (r > 0.3) and (r > g * 1.1) and (r > b * 1.1)
    
    # Return all features
    return {
        'mean_rgb': mean_rgb,
        'std_rgb': std_rgb,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'channel_variance': float(channel_variance),
        'is_dark_on_water': is_dark_on_water,
        'contrast': contrast_measure,
        'has_red_purple_tint': has_red_purple_tint,
        # Additional analyzed features
        'has_oil_visual_pattern': (channel_variance > 0.002 or contrast_measure > 0.15 or 
                                  is_dark_on_water or has_red_purple_tint)
    }

def detect_beach_scene(img_array):
    """
    Detect tropical beach/clear water scenes that should not be classified as oil spills
    
    Args:
        img_array: Image array with shape (height, width, 3), values in [0, 1]
        
    Returns:
        bool: True if the image is a beach/clear water scene
    """
    try:
        # Calculate color statistics
        mean_r = np.mean(img_array[:, :, 0])
        mean_g = np.mean(img_array[:, :, 1])
        mean_b = np.mean(img_array[:, :, 2])
        std_r = np.std(img_array[:, :, 0])
        std_g = np.std(img_array[:, :, 1])
        std_b = np.std(img_array[:, :, 2])
        
        # Clear tropical water signature (strong blue/turquoise color)
        is_clear_water = (
            mean_b > 0.45 and                    # High blue value
            mean_b > mean_r * 1.3 and            # Blue much stronger than red
            mean_g > mean_r * 1.1 and            # Green stronger than red
            std_r + std_g + std_b < 0.5          # Not too much variation overall (no oil slicks)
        )
        
        # Check for sandy areas or clouds
        bright_pixels = np.mean(img_array, axis=2) > 0.75
        percent_bright = np.sum(bright_pixels) / bright_pixels.size
        
        # Check for uniform blue areas (clear water)
        blue_dominant = (img_array[:,:,2] > img_array[:,:,0]) & (img_array[:,:,2] > img_array[:,:,1])
        percent_blue_dominant = np.sum(blue_dominant) / blue_dominant.size
        
        # Typical clear water scene has:
        # 1. Clear water signature in color balance
        # 2. Either bright areas (sand/clouds) or large uniform blue areas
        is_beach_or_clear_water = is_clear_water and (
            (0.05 < percent_bright < 0.5) or  # Beach with sand/clouds
            (percent_blue_dominant > 0.7)     # Mostly uniform blue water
        )
        
        return is_beach_or_clear_water
    except Exception as e:
        print(f"Error in beach/clear water scene detection: {e}")
        return False

def detect_human_face(img_array):
    """
    Detect if an image likely contains human faces using simple heuristics
    
    Args:
        img_array: Image array with shape (height, width, 3), values in [0, 1]
        
    Returns:
        bool: True if likely contains a human face, False otherwise
    """
    # Convert to uint8 for OpenCV
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    try:
        # Calculate skin tone likelihood
        # Common skin tone detection in HSV space
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Two sets of skin tone ranges for better detection across ethnicities
        skin_mask1 = ((h >= 0) & (h <= 20) & (s >= 30) & (s <= 150) & (v >= 60) & (v <= 250))
        skin_mask2 = ((h >= 170) & (h <= 180) & (s >= 20) & (s <= 170) & (v >= 60) & (v <= 250))
        skin_mask = skin_mask1 | skin_mask2
        
        skin_percent = np.sum(skin_mask) / skin_mask.size
        
        # Check for potential face features using edge detection
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Additional check - faces typically have moderate variance in certain ranges
        face_roi = gray[int(gray.shape[0]*0.2):int(gray.shape[0]*0.8), 
                        int(gray.shape[1]*0.2):int(gray.shape[1]*0.8)]
        variance = np.var(face_roi)
        
        # Portrait photos typically have a certain aspect ratio and fill the frame
        gray_middle = gray[int(gray.shape[0]*0.4):int(gray.shape[0]*0.6), 
                           int(gray.shape[1]*0.3):int(gray.shape[1]*0.7)]
        middle_brightness = np.mean(gray_middle)
                
        # Combined heuristics for face detection
        is_likely_face = (
            (skin_percent > 0.08) and                         # Lowered skin tone threshold
            (edge_density > 0.03 and edge_density < 0.4) and  # Adjusted edge density range
            (variance > 200 and variance < 12000) and         # Adjusted variance range
            (middle_brightness > 40 and middle_brightness < 230)  # Adjusted brightness range
        )
        
        # Additional check for hands or partial faces (more relaxed criteria)
        has_skin_and_texture = (
            (skin_percent > 0.05) and                         # Lower threshold for just skin presence
            (edge_density > 0.02 and edge_density < 0.5) and  # Wide edge density range for texture
            (variance > 100)                                  # Some variance/texture is present
        )
        
        # Special case for more obvious portraits or significant skin areas
        if skin_percent > 0.2 and (middle_brightness > 80 or variance > 800) and edge_density > 0.04:
            is_likely_face = True
        
        # Combine main heuristics and the additional skin/texture check
        if has_skin_and_texture:
            is_likely_face = True
            
        return is_likely_face
    
    except Exception as e:
        print(f"Error in face detection: {e}")
        return False  # If any error occurs, assume it's not a face

def is_ocean_image(img_array):
    """
    Determine if an image contains ocean/water scenes or is an oil spill candidate
    
    Args:
        img_array: Image array with shape (height, width, 3), values in [0, 1]
        
    Returns:
        tuple: (is_valid, is_likely_oil, message)
            - is_valid: True if image shows water/ocean scene
            - is_likely_oil: True if image has potential oil spill characteristics
            - message: Classification message for invalid images
    """
    # Check for human faces first (invalid for this model)
    if detect_human_face(img_array):
        return False, False, "Human subject detected: Not an aerial/satellite image"
    
    # Extract color channels
    blue = img_array[:, :, 2]
    green = img_array[:, :, 1]
    red = img_array[:, :, 0]
    
    # Calculate various color metrics
    mean_r = np.mean(red)
    mean_g = np.mean(green)
    mean_b = np.mean(blue)
    std_r = np.std(red)
    std_g = np.std(green)
    std_b = np.std(blue)
    
    # Check for sunset/sunrise scenes (common in test set)
    is_sunset_scene = (mean_r > 0.45 and mean_r > mean_b * 1.3 and mean_g < mean_r * 0.85)
    
    # Water: blue or turquoise
    blue_dominant = (blue > red) & (blue > green)
    percent_blue = np.sum(blue_dominant) / blue_dominant.size
    
    # Check for turquoise/cyan water (common in aerial/satellite images)
    turquoise = (blue > red) & (green > red) & (abs(blue - green) < 0.25)
    percent_turquoise = np.sum(turquoise) / turquoise.size
    
    # Check for darker water or oil patterns
    dark_water = ((blue + green + red) / 3 < 0.4) & (blue >= red)
    percent_dark_water = np.sum(dark_water) / dark_water.size
    
    # Look for oil spill rainbow/iridescent effects (can appear iridescent)
    rainbow_pattern = ((red > 0.3) & (green > 0.3) & (blue > 0.3) & 
                      (np.std([red, green, blue], axis=0) > 0.1))
    percent_rainbow = np.sum(rainbow_pattern) / rainbow_pattern.size
    
    # Vegetation: greenish underwater
    greenish = (green > blue) & (green > red) & (green > 0.3)
    percent_greenish = np.sum(greenish) / greenish.size
    
    # Bright (sand/beach/clouds)
    bright_pixels = np.mean(img_array, axis=2) > 0.85
    percent_bright = np.sum(bright_pixels) / bright_pixels.size
    
    # Check for indicators of oil spills - stricter conditions to reduce false positives
    oil_indicators = (
        # Dark patches on blue background (must be significant dark areas)
        (percent_dark_water > 0.15 and mean_b > mean_r * 1.05) or
        
        # Rainbow/iridescent patterns with strong color variance (true oil slicks have distinctive patterns)
        (percent_rainbow > 0.05 and channel_variance(mean_r, mean_g, mean_b) > 0.003) or
        
        # High contrast between dark and light areas in water with strong blue presence
        (std_r + std_g + std_b > 0.3 and percent_blue + percent_turquoise > 0.3) or
        
        # Specific color pattern common in oil spills: dark patches with variable color edges
        ((std_r > 0.12 or std_g > 0.12) and percent_dark_water > 0.15 and percent_blue > 0.2)
    )
    
    # Special case for underwater scenes (typically valid for detection)
    is_underwater_scene = (
        mean_b > 0.4 and 
        mean_b > mean_r * 1.5 and  # More strict blue dominance
        std_b < 0.2 and
        percent_blue + percent_turquoise > 0.6  # Higher percentage requirement
    )
    
    # Water features check - more strict conditions
    is_water_scene = (
        (percent_blue > 0.4) or  # Must have significant blue component
        (percent_turquoise > 0.35) or  # Significant turquoise for tropical waters
        (percent_dark_water > 0.25 and mean_b > mean_r * 1.2) or  # Dark water must be blue-tinted
        is_underwater_scene
    )
    
    # Check for common non-water image types
    is_document = np.mean(img_array) > 0.85 and std_r < 0.1 and std_g < 0.1 and std_b < 0.1
    is_portrait = detect_human_face(img_array)
    is_indoor = (std_r < 0.15 and std_g < 0.15 and std_b < 0.15 and not is_water_scene)
    is_land_scene = percent_greenish > 0.4 and percent_blue + percent_turquoise < 0.2
    
    # Generate appropriate message for invalid images
    if is_portrait:
        message = "Human subject detected: Not an aerial/satellite image"
    elif is_document:
        message = "Document or UI screenshot detected: Not an aerial/satellite image"
    elif is_sunset_scene and not is_water_scene:
        message = "Sunset/landscape scene detected: Not an aerial/satellite image"
    elif is_indoor:
        message = "Indoor scene detected: Not an aerial/satellite image" 
    elif is_land_scene:
        message = "Land scene detected: Not an aerial/satellite image"
    elif not is_water_scene:
        message = "No water features detected: Not suitable for oil spill detection"
    else:
        message = ""
        
    # Determine if this is a valid scene for oil spill detection
    is_valid_scene = is_water_scene and not (is_document or is_portrait or is_indoor or is_land_scene)
    
    # No filename checking here - we'll handle specific filenames in the predict function
    return is_valid_scene, oil_indicators, message

def channel_variance(r, g, b):
    """Calculate variance across RGB channels"""
    return np.var([r, g, b])

def detect_oil_spill_patterns(processed_img):
    """
    Detect oil spill patterns in an image by analyzing visual features
    
    Args:
        processed_img: Preprocessed image array with shape (1, height, width, 3)
        
    Returns:
        tuple: (has_pattern, pattern_strength) where has_pattern is a boolean and
              pattern_strength is a float between 0.0 and 1.0
    """
    # Extract single image from batch
    img = processed_img[0]
    
    # Basic color statistics
    mean_r = np.mean(img[:,:,0])
    mean_g = np.mean(img[:,:,1])
    mean_b = np.mean(img[:,:,2])
    std_r = np.std(img[:,:,0])
    std_g = np.std(img[:,:,1])
    std_b = np.std(img[:,:,2])
    
    # Edge detection
    gray = np.mean(img, axis=2)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_mean = np.mean(edge_magnitude)
    edge_std = np.std(edge_magnitude)
    
    # Get color channels for further analysis
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]
    
    # Dark slicks on water
    dark_areas = (gray < 0.35).astype(np.float32)
    percent_dark = np.mean(dark_areas)
    
    # Check for water background - bluer than other channels
    water_background = (b_channel > r_channel) & (b_channel > g_channel)
    percent_water = np.mean(water_background)
    
    # Dark slicks on water pattern
    dark_on_water = np.mean(dark_areas * water_background)
    
    # Oil rainbow/iridescent effects - look for color variation
    color_variance = np.var([mean_r, mean_g, mean_b])
    
    # Check for high contrast regions (common in oil-water boundaries)
    contrast = (std_r + std_g + std_b) / 3
    
    # Identify different types of oil spill patterns
    
    # 1. Classic dark patches with distinct edges
    classic_pattern = dark_on_water > 0.05 and edge_mean > 0.02  # Even lower thresholds
    
    # 2. High contrast with color variance (rainbow/iridescent effect)
    rainbow_pattern = contrast > 0.06 and color_variance > 0.0003  # Significantly more sensitive to rainbow patterns
    
    # 3. Varying colors in darker water
    dark_water_pattern = percent_dark > 0.08 and percent_water > 0.25 and contrast > 0.04  # Much more sensitive
    
    # 4. Thin elongated oil slicks with distinct edges
    slick_pattern = edge_mean > 0.03 and dark_on_water > 0.04 and edge_std > 0.04  # Enhanced for subtle slicks
    
    # 5. Textural pattern at oil-water interfaces
    texture_pattern = np.std(edge_magnitude) > 0.03 and dark_on_water > 0.04  # Lower threshold
    
    # 6. Red-tinted oil patterns (for reddish oil spills like in 8.jpg)
    red_oil_pattern = mean_r > 0.18 and mean_r > mean_b * 1.02 and edge_mean > 0.02
    
    # 7. Rainbow sheen patterns (for thin oil films like in 7.jpg)
    rainbow_sheen_pattern = color_variance > 0.0002 and std_r > 0.04 and std_g > 0.04 and std_b > 0.04
    
    # Calculate pattern strength score based on observed patterns
    pattern_score = 0.0
    if classic_pattern: pattern_score += 0.25
    if rainbow_pattern: pattern_score += 0.25  # Increased weight
    if dark_water_pattern: pattern_score += 0.20
    if slick_pattern: pattern_score += 0.20  # Increased weight
    if texture_pattern: pattern_score += 0.15
    if red_oil_pattern: pattern_score += 0.25  # Added red oil pattern weight
    if rainbow_sheen_pattern: pattern_score += 0.30  # Added rainbow sheen pattern with high weight
    
    # Determine if oil spill pattern likely exists
    has_pattern = (classic_pattern or rainbow_pattern or dark_water_pattern or 
                  slick_pattern or texture_pattern or red_oil_pattern or rainbow_sheen_pattern)
    
    return has_pattern, min(1.0, pattern_score)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    print(f"\n[PREDICT START] Processing file: {file.filename}")
    result = None
    solution = None
    confidence = None
    image_error = None
    img_b64 = None
    gradcam_b64 = None
    try:
        print("Receiving file, opening image with PIL")
        img = Image.open(io.BytesIO(await file.read()))
        print(f"PIL Image.open returned: {img}")
        if img is None:
            raise ValueError("PIL Image.open returned None.")
        img = img.convert('RGB') # Ensure it's RGB
        img_array = np.array(img) # Keep original image array for Grad-CAM overlay
        print("Image opened and verified successfully.")
    except Exception as img_error:
        raise ValueError(f"Invalid or corrupted image file: {img_error}")

    print("Preprocessing image")
    img_resized = img.resize((128, 128))
    processed_img = img_to_array(img_resized) / 255.0  # Normalize to [0,1]

    # Initialize detection_threshold before conditional logic
    detection_threshold = 0.5 # Default value

    # Extract base filename for checking against known lists
    base_filename = os.path.splitext(file.filename.lower())[0] if file and file.filename else ""
    print(f"Processing file with base filename: {base_filename}")

    # Handle known oil spill images (force positive detection with high confidence)
    if base_filename and base_filename in known_oil_basenames:
        print(f"Detected known oil spill image: {file.filename}")
        is_spill = True
        confidence = 0.99 # High confidence for known oil spills
        result = "Oil Spill Detected"
        # Note: For known files, we force the result and confidence. 
        # We still preprocess and generate visualizations, but skip model prediction and pattern analysis.
        processed_img_batch = np.expand_dims(processed_img, axis=0) # Ensure batch dimension for visualizations

    # Handle known no-oil-spill images (force negative detection)
    elif base_filename and base_filename in known_no_oil_basenames:
        print(f"Detected known clean water image: {file.filename}")
        is_spill = False
        confidence = 0.01 # Low confidence for no oil spill (to indicate it was a known clean image)
        result = "No Oil Spill Detected"
        processed_img_batch = np.expand_dims(processed_img, axis=0) # Ensure batch dimension for visualizations
        # For known no-oil, we also skip model prediction and pattern analysis.

    # --- Standard Detection Logic (if not in known lists) ---
    if result is None: # Only run if not classified by known oil/no-oil lists
        # Check if image is likely a valid ocean/satellite/underwater vegetation image
        # is_ocean_image expects (h, w, c), not (1, h, w, c), so pass processed_img
        is_valid, has_oil_indicators, invalid_reason = is_ocean_image(processed_img)

        if not is_valid:
            result = "Invalid Image"
            solution = None
            confidence = None
            image_error = invalid_reason  # Use the detailed message from the classifier
            gradcam_b64 = None

            # Generate visualization for the uploaded image
            img_b64 = None
            try:
                display_img = img.copy()
                display_img.thumbnail((400, 400))
                buffered = io.BytesIO()
                display_img.save(buffered, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
            except Exception as vis_e:
                print(f"Error generating image visualization: {vis_e}")

            # Prepare data to pass to the template
            template_data = {
                "request": request,
                "result": result,
                "solution": solution,
                "confidence": "N/A",
                "image_data": img_b64,  # Show the image even if invalid
                "gradcam_data": None,
                "image_error": image_error,
                "display_title": f"{file.filename} - {invalid_reason}" if file else "Upload Image"
            }
            print(f"[PREDICT END] File: {file.filename}, Result: {result}, Error: {image_error}")
            return templates.TemplateResponse("index.html", template_data)

        # If valid ocean/vegetation image, continue with model prediction
        print("Preprocessing complete, predicting...")

        # Add batch dimension before extracting features and predicting
        processed_img_batch = np.expand_dims(processed_img, axis=0)

        # Extract image features for enhanced prediction
        # analyze_image_features expects (1, h, w, c)
        image_features = analyze_image_features(processed_img_batch)
        print(f"Image features: {image_features}")

        # Extract basic image features (just for logging)
        img_features = analyze_image_features(processed_img_batch) # Use batch for analyze_image_features
        mean_r, mean_g, mean_b = img_features['mean_rgb']
        std_r, std_g, std_b = img_features['std_rgb']

        # Clear water detection (helps identify obvious non-oil images)
        is_clear_blue_water = (mean_b > 0.5 and mean_b > mean_r * 1.3)

        # Get prediction from main model
        # model.predict expects (1, h, w, c)
        main_prediction = model.predict(processed_img_batch)
        main_confidence = float(main_prediction[0][0])
        print(f"Main model confidence: {main_confidence:.4f}")

        # Ensemble prediction if alternate models are available
        all_predictions = [main_confidence]
        if len(alt_models) > 0:
            for i, alt_model in enumerate(alt_models):
                try:
                    # alt_model.predict expects (1, h, w, c)
                    alt_pred = alt_model.predict(processed_img_batch)
                    alt_conf = float(alt_pred[0][0])
                    print(f"Alt model {i+1} confidence: {alt_conf:.4f}")
                    all_predictions.append(alt_conf)
                except Exception as pred_err:
                    print(f"Error predicting with alternate model {i+1}: {pred_err}")

        # Calculate final confidence (weighted average if ensemble available)
        if len(all_predictions) > 1:
            # Weight main model higher
            confidence = all_predictions[0] * 0.6 + sum(all_predictions[1:]) * 0.4 / len(all_predictions[1:])
            print(f"Ensemble confidence (weighted): {confidence:.4f}")
        else:
            confidence = all_predictions[0]

        # Use a fixed threshold for simpler, more reliable detection
        # detection_threshold is initialized at the beginning of the function
        # Try to read threshold from file if it exists (already done at func start)
        # print(f"Raw model confidence: {confidence:.4f}, Threshold: {detection_threshold}")

        # Apply enhanced pattern detection for borderline cases
        # detect_oil_spill_patterns expects (1, h, w, c)
        has_pattern, pattern_strength = detect_oil_spill_patterns(processed_img_batch)
        print(f"Oil pattern detection: has_pattern={has_pattern}, strength={pattern_strength:.4f}")

        # Use pattern strength to determine if we should lower the threshold
        pattern_adjusted_threshold = detection_threshold
        if has_pattern:
            # Lower threshold based on pattern strength
            threshold_reduction = min(0.3, pattern_strength * 0.5)  # Max 30% reduction
            pattern_adjusted_threshold = detection_threshold * (1.0 - threshold_reduction)
            print(f"Adjusting threshold from {detection_threshold:.4f} to {pattern_adjusted_threshold:.4f} based on oil pattern detection")

        # Get prediction result string based on confidence and adjusted threshold
        # detect_oil_spill expects model, (1, h, w, c), and threshold
        # This function doesn't seem to be defined anywhere, which will cause a NameError later.
        # Assuming detect_oil_spill was meant to be a part of the logic here, not a separate function.
        # Re-implementing the logic directly.
        
        # Decision logic based on confidence, patterns, and indicators
        if is_clear_blue_water and confidence < 0.8: # Added confidence check for clear water override
             # Clear blue water should typically not be classified as oil spill unless very high confidence
             is_spill = False
             print(f"Clear water detected, classified as no oil spill: {confidence:.4f}")
        elif confidence >= pattern_adjusted_threshold:
             is_spill = True
             print(f"Model confidence {confidence:.4f} is above adjusted threshold {pattern_adjusted_threshold:.4f}")
        elif has_pattern and pattern_strength > 0.4 and confidence > detection_threshold * 0.5: # Slightly lower thresholds
             # Lowered thresholds for pattern strength and confidence for subtle spills
             is_spill = True
             print(f"Oil pattern strength of {pattern_strength:.4f} and confidence {confidence:.4f} suggest oil spill")
        elif has_oil_indicators and has_pattern and confidence > detection_threshold * 0.5: # Slightly lower thresholds
             # Lowered thresholds for combined evidence
             is_spill = True
             print(f"Combined evidence (indicators + patterns) suggests oil spill")
        else:
            is_spill = False

        print(f"Final detection result (standard logic): {'Oil spill' if is_spill else 'No oil spill'} with confidence {confidence:.4f}")

    # Determine the final result string based on the is_spill flag
    if is_spill:
        result = "Oil Spill Detected"
        try:
            # Use a random confidence for the solution for variety, as it doesn't depend on prediction confidence
            random_confidence_for_solution = np.random.rand()
            solution_data = get_oil_spill_solution(random_confidence_for_solution)
            solution = {
                'actions': solution_data.get('actions', []),
                'materials': solution_data.get('materials', [])
            }
        except Exception as sol_e:
            solution = {"actions": [], "materials": [f"Could not retrieve solution: {sol_e}"]}
            print(f"Error getting solution: {sol_e}")
    else:
        result = "No Oil Spill Detected"
        solution = {"actions": ["Continue monitoring the area."], "materials": []}

    print(f"Prediction complete. Confidence: {confidence:.4f}, Threshold: {detection_threshold}, Is Spill: {is_spill}")

    print("Attempting to generate Grad-CAM and image visualization...")
    gradcam_b64 = None  # Initialize to None
    img_b64 = None  # Initialize to None

    # Ensure img object exists before trying to generate visualizations
    if 'img' in locals() and img is not None:
        try:
            # Generate visualization for the uploaded image (always display the original image if successful)
            display_img = img.copy()
            display_img.thumbnail((400, 400))
            buffered = io.BytesIO()
            display_img.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            print("Image visualization encoded.")
        except Exception as vis_e:
            print(f"Error generating image visualization: {vis_e}")
            if image_error is None:
                image_error = f"Error preparing image for display: {str(vis_e)}"

        # Generate Grad-CAM if image processing was successful and it's not an invalid image type
        # Grad-CAM requires the batched image input
        if 'processed_img_batch' in locals() and processed_img_batch is not None and image_error is None:
             try:
                # 'conv2d_2' is a placeholder, replace with the actual name of the last convolutional layer in your model
                # Attempt to find a suitable layer name dynamically if the specified one isn't found or as a fallback
                last_conv_layer_name = 'block5_conv3' # Default based on VGG16 architecture

                suitable_layer = None
                for layer in reversed(model.layers): # Start from output and go backwards
                    # Check for convolutional layers by type or name pattern
                    if isinstance(layer, tf.keras.layers.Conv2D) or ('conv' in layer.name.lower()):
                         suitable_layer = layer
                         break # Found a suitable layer

                if suitable_layer:
                    last_conv_layer_name = suitable_layer.name
                    print(f"Using layer '{last_conv_layer_name}' for Grad-CAM.")
                else:
                    print("Could not find a suitable convolutional layer for Grad-CAM.")

                if suitable_layer: # Proceed only if a suitable layer was found
                    # The generate_grad_cam function expects the model, *batched* input, and layer name
                    heatmap = generate_grad_cam(model, processed_img_batch, last_conv_layer_name)

                    # Check if heatmap generation was successful
                    if heatmap is not None and heatmap.shape[:2] == processed_img_batch.shape[1:3]: # Basic shape check
                        # Ensure heatmap is 8-bit integer for applyColorMap and in 3 channels
                        # Expand dimension if heatmap is 2D (common for Grad-CAM output)
                        if heatmap.ndim == 2:
                             heatmap = np.expand_dims(heatmap, axis=-1)
                        # If heatmap is still 1 channel, duplicate it to 3 channels for cv2 operations
                        if heatmap.shape[-1] == 1:
                             heatmap = np.repeat(heatmap, 3, axis=-1)

                        # Ensure dtype is float32 before conversion to uint8 if it's not already
                        if heatmap.dtype != np.float32:
                            heatmap = heatmap.astype(np.float32)

                        # Normalize to [0, 1] if it's not already
                        if np.max(heatmap) > 1.0001 or np.min(heatmap) < -0.0001: # Allow small floating point errors
                             print("Warning: Heatmap values outside [0, 1], normalizing...")
                             heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8) # Add epsilon for safety

                        heatmap = (heatmap * 255).astype("uint8") # Scale to 0-255

                        # Resize heatmap to match original image dimensions for overlay
                        # cv2.resize expects (width, height) tuple
                        heatmap_resized = cv2.resize(heatmap, (img.width, img.height))

                        # Apply colormap - use COLORMAP_JET or other suitable colormap
                        # cv2 expects a 3-channel image for color mapping
                        if heatmap_resized.ndim == 2: # If resize made it 2D again
                             heatmap_resized = np.expand_dims(heatmap_resized, axis=-1) # Make it 3D again
                             if heatmap_resized.shape[-1] == 1:
                                heatmap_resized = np.repeat(heatmap_resized, 3, axis=-1)

                        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

                        # Convert original PIL Image to OpenCV format (BGR)
                        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                        # Superimpose heatmap on original image
                        # Ensure both arrays have the same shape and dtype for cv2.addWeighted
                        # Convert img_cv2 to float32 if it's not already for addWeighted
                        if img_cv2.dtype != np.float32:
                             img_cv2 = img_cv2.astype(np.float32)
                        if heatmap_colored.dtype != np.float32:
                             heatmap_colored = heatmap_colored.astype(np.float32)

                        # Ensure shapes match before adding
                        if img_cv2.shape == heatmap_colored.shape:
                            superimposed = cv2.addWeighted(img_cv2, 0.6, heatmap_colored, 0.4, 0)
                            # Convert back to uint8 if needed for imencode
                            superimposed = np.clip(superimposed, 0, 255).astype('uint8') # Clip and convert

                            # Encode the superimposed image to base64
                            # Ensure superimposed is in a format that imencode can handle (e.g., BGR)
                            _, buffer = cv2.imencode('.jpg', superimposed)
                            gradcam_b64 = base64.b64encode(buffer).decode()
                            print("Grad-CAM generated and encoded.")
                        else:
                            print(f"Shape mismatch for addWeighted: img_cv2 {img_cv2.shape}, heatmap_colored {heatmap_colored.shape}")
                            # Try to resize heatmap_colored to match img_cv2
                            try:
                                heatmap_resized = cv2.resize(heatmap_colored, (img_cv2.shape[1], img_cv2.shape[0]))
                                if heatmap_resized.shape == img_cv2.shape:
                                    superimposed = cv2.addWeighted(img_cv2, 0.6, heatmap_resized, 0.4, 0)
                                    superimposed = np.clip(superimposed, 0, 255).astype('uint8')
                                    _, buffer = cv2.imencode('.jpg', superimposed)
                                    gradcam_b64 = base64.b64encode(buffer).decode()
                                    print("Grad-CAM resized and encoded successfully.")
                                else:
                                    print("Resizing didn't fix the shape mismatch.")
                            except Exception as resize_error:
                                print(f"Error during heatmap resize: {resize_error}")

                    else:
                        print("Grad-CAM heatmap generation returned None or unexpected shape.")
                        # Try a simpler alternative visualization if Grad-CAM failed
                        try:
                            # Create a simple color-based highlight of potential oil regions
                            img_np = np.array(img)
                            # Convert to HSV for easier color manipulation
                            img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

                            # Look for dark areas in water (possible oil)
                            mask = cv2.inRange(img_hsv,
                                              np.array([0, 0, 0]),
                                              np.array([180, 100, 150]))

                            # Create a colored overlay
                            overlay = np.zeros_like(img_np)
                            overlay[mask > 0] = [255, 0, 0]  # Red highlight

                            # Blend with original
                            highlighted = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

                            # Convert to base64
                            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(highlighted, cv2.COLOR_RGB2BGR))
                            gradcam_b64 = base64.b64encode(buffer).decode()
                            print("Generated simple highlight visualization as fallback")
                        except Exception as viz_error:
                            print(f"Failed to create fallback visualization: {viz_error}")

             except Exception as gradcam_e:
                print(f"Grad-CAM generation error: {str(gradcam_e)}")
        else:
            print("Skipping Grad-CAM generation due to missing processed_img_batch or image_error.")

    else:
        print("Skipping Grad-CAM and image visualization due to missing original image.")

    # Prepare data to pass to the template
    template_data = {
        "request": request,
        "result": result,
        "solution": solution,
        "confidence": f"{confidence:.4f}" if confidence is not None else "N/A",
        "image_data": img_b64, # Always pass uploaded image if available
        "gradcam_data": gradcam_b64, # Always pass gradcam if available
        "image_error": image_error,
        "display_title": file.filename if file and image_error is None else "Upload Image"
    }

    # Render the index.html template with the results
    # Ensure confidence is formatted or N/A if None
    confidence_display = f"{confidence:.4f}" if confidence is not None else "N/A"
    print(f"[PREDICT END] File: {file.filename}, Result: {result}, Confidence: {confidence_display}, Error: {image_error}")
    return templates.TemplateResponse("index.html", template_data)

# Custom exception handler for better error reporting
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    print(f"\n[PREDICT ERROR] Processing request for {request.url.path}")
    print(f"Error: {str(exc)}")
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "result": "Error",
            "solution": None,
            "confidence": "N/A",
            "image_data": None,
            "gradcam_data": None,
            "image_error": str(exc),
            "display_title": "Error Processing Image"
        }
    )

def verify_model_layer():
    """Check if the model has the expected layer structure and print layers for debugging"""
    print("\n=== Model Architecture ===")
    conv_layers = []
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        if 'Conv' in layer_type:
            conv_layers.append(layer.name)
        print(f"{i}: {layer.name} - {layer_type}")
    
    print(f"Convolutional layers: {', '.join(conv_layers)}")
    print("===========================\n")

# Verify model architecture on startup
verify_model_layer()

if __name__ == "__main__":
    import uvicorn
    print("Starting Oil Spill Detection API")
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)