import pickle
from tensorflow.keras.models import load_model

# Load the .h5 model
model_path = "fashion_mnist_model_v2.h5"
model = load_model(model_path)

# Save the model to a .pkl file
pkl_model_path = "fashion_mnist_model.pkl"
with open(pkl_model_path, "wb") as pkl_file:
    pickle.dump(model, pkl_file)

print(f"Model saved as {pkl_model_path}")
