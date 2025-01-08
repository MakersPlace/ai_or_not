# Write a Simple gradio app to take image as input run a model on it and Returnt the Probability (0 to 1) as a confidence bar

import gradio as gr
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

REPO_ID = "ai-or-not/ai-or-not-model"

model = from_pretrained_keras(REPO_ID)


# Define the function
def classify_image(array):
    # image is numpy array
    array = array / 255.0
    image = tf.image.resize_with_pad(array, 224, 224)
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    prediction = model(image)
    # there are 3 class probabilities in the model 0: "REAL", 1: "GAN", 2: "DIFFUSION"
    real = prediction[0][0]
    ai = 1 - real

    return {"REAL": real, "AI": ai}


demo = gr.Interface(fn=classify_image, inputs="image", outputs="label", examples="examples")
demo.launch(
    debug=True,
)
