import numpy as np
import tensorflow as tf
import cv2

def generate_grad_cam(model, img_array, layer_name='conv2d_2', pred_index=None):
    """
    Generate Grad-CAM heatmap for model visualization
    
    Args:
        model: Trained model
        img_array: Preprocessed input image (batch, height, width, channels)
        layer_name: Name of the convolutional layer to use
        pred_index: Class index for visualization (None = predicted class)
        
    Returns:
        heatmap: Normalized heatmap showing model attention
    """
    # Handle layer name not found
    try:
        target_layer = model.get_layer(layer_name)
        print(f"Using layer '{layer_name}' for Grad-CAM visualization")
    except:
        # Find the last convolutional layer if specified one isn't found
        found_conv = False
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                layer_name = layer.name
                found_conv = True
                print(f"Using fallback layer '{layer_name}' for Grad-CAM")
                break
        
        if not found_conv:
            print("No convolutional layer found for Grad-CAM")
            return None
    
    # Create a model that outputs both the convolution output and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Capture gradients with error handling
    try:
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = grad_model(img_array)
            
            # Use predicted class for visualization if not specified
            if pred_index is None:
                # For binary classification, use class 1 (oil spill class)
                if predictions.shape[1] <= 2:
                    pred_index = 0  # Focus on oil spill probability
                else:
                    pred_index = tf.argmax(predictions[0])
                    
            # Get output for target class
            class_channel = predictions[:, pred_index]

        # Extract gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Safety check for valid gradients
        if grads is None:
            print("Gradient is None. Could not compute Grad-CAM.")
            return None
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Apply gradients to convolution outputs
        conv_outputs = conv_outputs[0]
        
        # Handle case where pooled_grads is all zeros
        if tf.reduce_sum(tf.abs(pooled_grads)) < 1e-10:
            print("Pooled gradients are effectively zero. Using uniform weighting.")
            pooled_grads = tf.ones_like(pooled_grads) / pooled_grads.shape[0]
            
        # Create heatmap using matrix multiplication
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Safety check for valid heatmap
        if tf.reduce_all(tf.math.is_nan(heatmap)) or tf.reduce_all(tf.math.is_inf(heatmap)):
            print("Heatmap contains NaN or Inf values. Using fallback.")
            return None

        # Normalize heatmap
        heatmap_max = tf.math.reduce_max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / (heatmap_max + tf.keras.backend.epsilon())
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap.numpy()
        
        # Handle potential NaN values that may have occurred
        if np.isnan(heatmap).any() or np.isinf(heatmap).any():
            print("NaN or Inf values in normalized heatmap.")
            heatmap = np.nan_to_num(heatmap)
        
        # Resize to original image dimensions with error handling
        try:
            heatmap = cv2.resize(heatmap, (128, 128))
        except Exception as resize_error:
            print(f"Error resizing heatmap: {resize_error}")
            return None
        
        # Apply mild gaussian blur to smooth the heatmap
        heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
        
        # Final conversion to uint8
        heatmap = np.uint8(255 * heatmap)
        
        return heatmap
        
    except Exception as e:
        print(f"Error in Grad-CAM generation: {e}")
        return None

def generate_alternative_heatmap(img_array):
    """
    Generate an alternative visualization when Grad-CAM fails
    
    Args:
        img_array: Preprocessed image array of shape (1, height, width, channels)
        
    Returns:
        heatmap: Visualization highlighting potential oil regions
    """
    try:
        # Extract the image from batch
        img = img_array[0].copy()
        
        # Create a grayscale version
        gray = np.mean(img, axis=2)
        
        # Find dark regions (potential oil)
        dark_mask = (gray < 0.4).astype(np.float32)
        
        # Find high contrast regions
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_mask = (edges > 0.1).astype(np.float32)
        
        # Combine masks
        combined_mask = (dark_mask * 0.7 + edge_mask * 0.3)
        
        # Normalize and convert to heatmap
        combined_mask = combined_mask / np.max(combined_mask)
        heatmap = np.uint8(255 * combined_mask)
        
        # Apply color map
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return colored_heatmap
    except Exception as e:
        print(f"Error generating alternative heatmap: {e}")
        return None

def detect_oil_spill(model, processed_img, detection_threshold=0.5):
    """
    Detect oil spill in an image using the trained model
    
    Args:
        model: Trained model
        processed_img: Preprocessed input image (batch, height, width, channels)
        detection_threshold: Confidence threshold for detection
        
    Returns:
        tuple: (result_string, confidence) with detection result and confidence score
    """
    try:
        # Get model prediction
        prediction = model.predict(processed_img)
        raw_confidence = float(prediction[0][0])
        
        # Analyze image to detect false positive patterns
        img = processed_img[0]  # Get the image out of batch dimension
        
        # Look for characteristics commonly leading to false positives
        mean_blue = np.mean(img[:,:,2])
        mean_green = np.mean(img[:,:,1])
        mean_red = np.mean(img[:,:,0])
        std_red = np.std(img[:,:,0])
        std_green = np.std(img[:,:,1])
        std_blue = np.std(img[:,:,2])
        std_colors = np.std([mean_red, mean_green, mean_blue])
        
        # Calculate edge features (oil spills often have distinct edges)
        gray = np.mean(img, axis=2)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_mean = np.mean(edge_magnitude)
        
        # Check for oil spill visual patterns
        # 1. Dark slicks on water background
        dark_on_water = np.mean((img < 0.35) & (img[:,:,2] > img[:,:,0]))
        
        # 2. High contrast areas (common in oil spills)
        contrast = np.mean(std_red + std_green + std_blue)
        
        # 3. Oil rainbow/iridescent patterns
        color_variance = np.var([mean_red, mean_green, mean_blue])
        
        # Clear water signature - strong blue channel with low variance
        is_likely_clear_water = (
            mean_blue > 0.5 and  # Increased threshold for blue
            mean_blue > mean_red * 1.4 and  # More strict blue dominance
            mean_blue > mean_green * 1.2 and
            std_blue < 0.15  # Less variance in blue for clear water
        )        # Look for oil spill patterns - much more sensitive detection for higher recall
        # 1. Classic dark patches with distinct edges (most common pattern)
        classic_pattern = dark_on_water > 0.10 and edge_mean > 0.03  # Reduced thresholds
          
        # 2. High contrast with color variance (rainbow/iridescent effect) - significantly lowered thresholds
        rainbow_pattern = contrast > 0.06 and color_variance > 0.0005
        
        # 3. Varying colors in darker water areas - much more sensitive detection
        dark_water_pattern = std_red > 0.06 and std_green > 0.06 and std_blue > 0.06 and mean_blue < 0.70
        
        # 4. Thin elongated oil slicks (may have lower overall coverage but distinct pattern)
        # Check for elongated shapes using edge detection - lower thresholds
        slick_pattern = edge_mean > 0.04 and dark_on_water > 0.06
        
        # 5. Textural pattern common in oil-water interfaces - more sensitive
        texture_pattern = np.std(edge_magnitude) > 0.04 and dark_on_water > 0.07
        
        # 6. Red-tinted oil patterns (seen in some oil types) - more sensitive
        red_oil_pattern = mean_red > 0.20 and mean_red > mean_blue * 1.05 and edge_mean > 0.03
        
        # Combined detection using multiple patterns
        is_likely_oil_pattern = (
            classic_pattern or 
            rainbow_pattern or 
            dark_water_pattern or
            slick_pattern or
            texture_pattern
        )
          # Calculate pattern strength metrics to adjust confidence - increased boost values
        pattern_strength = 0
        if classic_pattern: pattern_strength += 0.18
        if rainbow_pattern: pattern_strength += 0.15  # Increased rainbow pattern boost
        if dark_water_pattern: pattern_strength += 0.15  # Increased dark water pattern boost
        if slick_pattern: pattern_strength += 0.12
        if texture_pattern: pattern_strength += 0.10
        
        # Calculate adjustments based on visual patterns
        penalty = 0
        boost = 0
        
        if is_likely_clear_water and not is_likely_oil_pattern:
            # Apply penalty for clear water with no oil patterns
            blue_dominance = mean_blue / (mean_red + 0.001)
            penalty = min(0.35, blue_dominance * 0.12)  # Increased penalty
            print(f"Clear water detected, applying confidence penalty: -{penalty:.4f}")
        
        if is_likely_oil_pattern:
            # Apply boost proportional to pattern strength
            boost = min(0.30, pattern_strength)  # Increased max boost to 0.30
            pattern_types = []
            if classic_pattern: pattern_types.append("dark patches")
            if rainbow_pattern: pattern_types.append("rainbow effect")
            if dark_water_pattern: pattern_types.append("dark water variance")
            if slick_pattern: pattern_types.append("oil slick")
            if texture_pattern: pattern_types.append("textural interface")
            
            pattern_desc = ", ".join(pattern_types)
            print(f"Oil pattern detected ({pattern_desc}), applying confidence boost: +{boost:.4f}")
          # Adjust confidence by penalty and boost
        adjusted_confidence = max(0, min(1.0, raw_confidence - penalty + boost))        # Enhanced decision logic with pattern-based detection - significantly lowered thresholds for better detection
        if is_likely_oil_pattern and adjusted_confidence >= detection_threshold * 0.5:  # Significantly lowered threshold
            # Lower threshold for images with clear oil patterns
            result = "Oil Spill Detected"
            print(f"Oil spill detected based on visual patterns with confidence {adjusted_confidence:.4f}")
        elif adjusted_confidence >= detection_threshold:
            # Standard threshold-based detection
            result = "Oil Spill Detected"
            print(f"Oil spill detected based on model confidence {adjusted_confidence:.4f}")
        elif is_likely_oil_pattern and adjusted_confidence >= detection_threshold * 0.4:  # Significantly lowered threshold
            # Even lower threshold for very strong pattern indicators
            result = "Oil Spill Detected"
            print(f"Oil spill detected based on strong visual patterns despite lower confidence {adjusted_confidence:.4f}")
        elif rainbow_pattern and adjusted_confidence >= detection_threshold * 0.35:  # Special case for rainbow pattern
            # Rainbow patterns are highly indicative of oil spills
            result = "Oil Spill Detected"
            print(f"Oil spill detected based on rainbow pattern despite low confidence {adjusted_confidence:.4f}")
        else:
            result = "No Oil Spill Detected"
            print(f"No oil spill detected, confidence {adjusted_confidence:.4f}")
        
        return result, adjusted_confidence
    except Exception as e:
        print(f"Error in oil spill detection: {e}")
        return "Error in detection", 0.0