from flask import Flask, Response, render_template, jsonify
import pickle
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model from the .pkl file
pkl_model_path = "model/fashion_mnist_model.pkl"
with open(pkl_model_path, "rb") as pkl_file:
    model = pickle.load(pkl_file)

# Define class labels
fashion_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Initialize video capture
camera = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """Preprocess frame for model prediction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1))
    return reshaped

def generate_frames():
    """Generate frames from the webcam for live video feed."""
    while True:
        success, frame = camera.read()
        if not success:
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
        frame = buffer.tobytes()

        # Yield the frame as a response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route to display the live video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Main route to render the HTML page."""
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
