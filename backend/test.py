import cv2
import numpy as np
import tensorflow as tf
import json

# --- CONFIGURATION ---
model_path = 'isl_cnn_model.h5'
label_path = 'model_labels.json'
CONFIDENCE_THRESHOLD = 0.50  # Kept low for testing

# 1. Load Model & Labels
print("Loading Model...")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model Loaded Successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

with open(label_path, 'r') as f:
    labels = json.load(f)
    # Ensure keys are integers
    labels = {int(k): v for k, v in labels.items()}

# 2. Setup Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame for mirror effect (natural feel)
    frame = cv2.flip(frame, 1)
    
    # Define Region of Interest (ROI) - The Green Box
    # Users must put their hand INSIDE this box
    cv2.rectangle(frame, (50, 50), (300, 300), (0, 255, 0), 2)
    roi = frame[50:300, 50:300]
    
    # 3. Preprocessing (The "Brain" part)
    try:
        # CRITICAL FIX: Convert BGR (OpenCV) to RGB (Model)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (The size MobileNet expects)
        img = cv2.resize(roi_rgb, (224, 224))
        
        # Normalize pixel values to 0-1
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        # 4. Predict
        pred = model.predict(img, verbose=0)
        class_id = np.argmax(pred)
        confidence = np.max(pred)
        
        predicted_char = labels.get(class_id, "Unknown")
        
        # 5. Display Result
        # Print to terminal for debugging
        print(f"Detected: {predicted_char} | Confidence: {confidence:.2f}")

        if confidence > CONFIDENCE_THRESHOLD:
            # Green text if confident
            text_color = (0, 255, 0)
            display_text = f"{predicted_char} ({int(confidence*100)}%)"
        else:
            # Red text if unsure
            text_color = (0, 0, 255)
            display_text = "..."

        # Draw prediction above the box
        cv2.putText(frame, display_text, (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
    except Exception as e:
        print(f"Prediction Error: {e}")

    cv2.imshow('ISL Testing Mode', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()