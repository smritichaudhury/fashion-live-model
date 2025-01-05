from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model (replace with your actual model path)
model = load_model('fashion_model.h5')

# Mapping the class indices to class names (example for a fashion dataset)
fashion_classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Initialize the webcam
camera = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """Preprocess the frame for prediction"""
    # Resize and normalize frame as required by your model
    resized_frame = cv2.resize(frame, (224, 224))  # Example size for a model
    normalized_frame = resized_frame / 255.0  # Example normalization
    return np.expand_dims(normalized_frame, axis=0)

def generate_frames():
    """Generate frames from webcam for live video feed"""
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame")
            break

        # Preprocess the frame for prediction
        processed_frame = preprocess_frame(frame)

        # Make prediction
        try:
            predictions = model.predict(processed_frame, verbose=0)
            class_idx = np.argmax(predictions)
            label = fashion_classes[class_idx]
            probability = predictions[0][class_idx] * 100  # Probability in percentage
        except Exception as e:
            print(f"Prediction error: {e}")
            break

        # Annotate the frame with label and probability
        cv2.putText(frame, f"Detected: {label} ({probability:.2f}%)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")
            continue
        
        frame = buffer.tobytes()

        # Yield the frame as a response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Home page that shows the webcam feed"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route to display the live video feed"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
