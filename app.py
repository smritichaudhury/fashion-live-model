from flask import Flask, Response, render_template, request, jsonify
import pickle
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

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

def preprocess_frame(frame):
    """Preprocess frame for model prediction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1))
    return reshaped

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Route to process a single frame and return prediction."""
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode the base64 image
    image_data = data['image'].split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocess and predict
    try:
        processed_frame = preprocess_frame(frame)
        predictions = model.predict(processed_frame, verbose=0)
        class_idx = np.argmax(predictions)
        label = fashion_classes[class_idx]
        probability = predictions[0][class_idx] * 100  # Probability in percentage

        return jsonify({
            "label": label,
            "probability": f"{probability:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_frames():
    """Generate frames from the webcam for live video feed."""
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        # Resize the frame to a smaller size for display
        #frame = cv2.resize(frame, (480, 640))

        # Preprocess the frame for prediction
        processed_frame = preprocess_frame(frame)

        # Make prediction
        try:
            predictions = model.predict(processed_frame, verbose=0)
            class_idx = np.argmax(predictions)
            label = fashion_classes[class_idx]
            probability = predictions[0][class_idx] * 100  # Probability in percentage

            # Annotate the frame with label and probability
            cv2.putText(frame, f"Detected: {label} ({probability:.2f}%)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Prediction error: {e}")
            break

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
