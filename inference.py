import numpy as np
import os
import cv2
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

# Set environmental flags - avoids unintuitive errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' # Disable JIT compilation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # No visible GPU

# Make sure cwd is correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# File path
TFLITE_FILE_PATH = "model.tflite"
DATA_PATH = "goldfish.jpeg"

# Instantiate interpreter
interpreter = Interpreter(model_path=TFLITE_FILE_PATH)

# Get signature runner - default name since it came from pytorch
print(interpreter.get_signature_list())
signature_runner = interpreter.get_signature_runner('serving_default')

# Test the model on goldfish picture
input_data = cv2.imread(DATA_PATH)

# Ensure the image loaded correctly
cv2.imshow('Goldfish', input_data)
cv2.waitKey(0)

# Preprocess the image in the same way as the rest of the dataset for resnet18
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(224, 224), # Same size as specified when converting, if you are unsure you can check this with input_details[0]['shape']
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2]),
])

input_data = tf.expand_dims(preprocess(input_data), axis=0)
input_data = tf.reshape(input_data, [1, 3, 224, 224]) # Change channels to be second dimension

# Inspect output data
output_data = signature_runner(args_0=input_data)['output_0']

# Model yields confidence scores, softmax for probabilities
prob = tf.nn.softmax(output_data)

# Find top prob
max_val = tf.reduce_max(prob, keepdims=True)
cond = tf.equal(prob, max_val)
res = tf.where(cond)
print(res)