import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from PIL import Image

# Download Model
REPO_ID = "konerusudhir/ai-or-not-model"
model = from_pretrained_keras(REPO_ID, token=True, force_download=True)

# Read Image
file_path = "examples/1-1900923-343813.jpg"
image = Image.open(file_path)

# Convert to RGB
if image.mode != "RGB":
    image = image.convert("RGB")

# normalize to 0-1
image = np.array(image) / 255.0

# resize
resized_image = tf.image.resize_with_pad(image, 224, 224)

print(f"File: {file_path} Image shape: {resized_image.shape}")

# Predict
predictions = model(np.array([resized_image]))

# Compute AI Probability
real_probability = predictions[0][0]
ai_probability = 1 - real_probability
print(f"AI Image Prediction: {ai_probability:.2f}")
