## AI or Not Model Code
Given an image, this model will predict a probability that an image is AI-generated Image.  

Demo : [Huggingface Space](https://huggingface.co/spaces/konerusudhir/ai-or-not-demo)

1. It's a Tensorflow Model built using TF Dataset for data processing. 
2. Training code can run on Mac or AWS. 
3. F1 Score is 0.90 for Current Evaluation Dataset.

# Sample code
```
from PIL import Image
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

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

```
[Huggingface Model](https://huggingface.co/konerusudhir/ai-or-not-model)

[Training Report from Weights and Biases](https://wandb.ai/makersplace/open-model/reports/AI-Generated-Image-Detection-Model-Report--VmlldzoxMDkyMzYzNA?accessToken=lu9435jahbicmqtnsrnfs76ctuxzb2p3ik7xi2tgk7i7k8sn02zv60hdnqbnq145)


# Datasets
Below are the datasets used for traiing and evaluating the model. Datasets are chosen solve the problem of
identifying Low effort AI generated Crypto Art.

<table>
  <tr>
   <td><strong>Dataset Name</strong>
   </td>
   <td><strong>Description</strong>
   </td>
   <td><strong>License</strong>
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/liaopeiyuan/artbench">Artbench</a>
   </td>
   <td>Real Artistic Images 
   </td>
   <td>MIT
   </td>
  </tr>
  <tr>
   <td><a href="https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images">CIFAKE</a>
   </td>
   <td>Real from CIFAR-10 and Fake are generated using Stable Diffusion
   </td>
   <td>MIT
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/ZhendongWang6/DIRE">DIRE</a>
   </td>
   <td>Broad dataset of other Dataset
   </td>
   <td>Inherited from other datasets
   </td>
  </tr>
  <tr>
   <td><a href="https://openai.com/index/dall-e-3/">Dalle3</a>
   </td>
   <td>Generated using Dalle3 Model
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td><a href="https://huggingface.co/datasets/InfImagine/FakeImageDataset">FakeImageDataset</a>
   </td>
   <td>Benchmarking Dataset
   </td>
   <td><a href="https://www.apache.org/licenses/LICENSE-2.0">Apache-2.0</a>
   </td>
  </tr>
  <tr>
   <td><a href="https://laion.ai/blog/laion-5b/">LAION-5B</a>
   </td>
   <td>Real Artistic Images with Aesthetics score greater than 6.5 and modified before Jan 2020
   </td>
   <td><a href="https://creativecommons.org/licenses/by/4.0/">Creative Common CC-BY 4.0</a>
   </td>
  </tr>
  <tr>
   <td><a href="https://www.midjourney.com/home">Midjourney</a>
   </td>
   <td>Generated using the Midjourney Model
   </td>
   <td>N/A
   </td>
  </tr>
</table>

# Training
Please set below env variables before runnning the pipeline.py.

export WANDB_API_KEY=[API_KEY]

export WANDB_ENTITY=[entity]

export WANDB_PROJECT=[project]